"""Raw dataset rows in; the processed files every other stage reads, out.

`01_build_index.sh` expects `data/processed/corpus.jsonl`. `03_eval.sh` expects
`data/processed/hotpotqa_dev.jsonl`. The library pieces existed - field maps,
hash splits, the corpus builder, the leakage and coverage checks - and nothing
composed them, so neither file could be produced. This module is the
composition, and `cli.py prepare-data` is the command.

**One pass over each dataset's rows produces both its examples and its corpus
paragraphs.** Downloading twice, once per output, would let the two disagree -
a mirror that changed between calls, a different row limit - and the coverage
check below would then compare examples against a corpus built from other
rows, and pass.

**Coverage is the gate.** Every supporting title of every example must be a
document in the corpus. A missing one caps recall below 1.0, caps accuracy
with it, and nothing else reports it - the cap looks exactly like a model that
did not learn (`build_corpus.find_missing_titles`). A dataset whose examples
carry no supporting titles is reported as *unmeasured*, never as covered.

**Leakage is reported, not fixed.** `splits.find_leakage` says why: what to do
about a near-duplicate that lands in two splits is a judgement call for the
two owners, not a default buried in a pipeline. It is written to
`leakage.json` and counted in the manifest.

**What the corpus is.** The union of the context paragraphs of every dataset
whose spec has a `context_key` - HotpotQA and 2WikiMultihopQA. The two
out-of-distribution sets ship none, so their questions are scored against a
corpus that was not built for them. The manifest records that per dataset.

The only function here that needs the network is `download_rows`. Everything
else takes rows, so the whole composition is tested offline from fixtures.

Owner: Ayesha.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping  # runtime: used in an isinstance check below
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from verisearch.data.build_corpus import (
    CorpusBuild,
    CorpusDoc,
    CorpusStats,
    build_corpus,
    find_missing_titles,
    summarize_corpus,
    write_corpus,
)
from verisearch.data.datasets import (
    DataError,
    DatasetSpec,
    QAExample,
    Split,
    row_to_example,
    write_jsonl,
)
from verisearch.data.splits import (
    DEFAULT_FRACTIONS,
    SCHEME_VERSION,
    LeakageGroup,
    SplitFractions,
    SplitReport,
    find_leakage,
    make_splits,
    summarize,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

_FROZEN = ConfigDict(frozen=True, extra="forbid")

CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_FILENAME = "manifest.json"
LEAKAGE_FILENAME = "leakage.json"

# How many missing titles the manifest lists by name. The count is always
# exact; the list is a sample for whoever has to work out why.
_MISSING_SAMPLE = 20


def examples_filename(dataset: str, split: Split) -> str:
    """`hotpotqa_dev.jsonl` - the name `03_eval.sh` reads by default."""
    return f"{dataset}_{split.value}.jsonl"


class DatasetPart(BaseModel):
    """One dataset after its single pass: its examples, and its paragraphs if any."""

    model_config = _FROZEN

    spec: DatasetSpec
    examples: tuple[QAExample, ...]
    corpus: CorpusBuild | None


class DatasetSummary(BaseModel):
    """One dataset's line in the manifest."""

    model_config = _FROZEN

    name: str
    hf_path: str
    hf_name: str | None
    hf_splits: tuple[str, ...]
    n_rows: int = Field(ge=0)
    splits: dict[str, int]
    contributes_to_corpus: bool
    # Supporting titles checked against the corpus. Zero means coverage for
    # this dataset is unmeasured - not that it is complete.
    titles_checked: int = Field(ge=0)
    titles_missing: int = Field(ge=0)

    @property
    def coverage_measured(self) -> bool:
        return self.titles_checked > 0


class Manifest(BaseModel):
    """Provenance for one `data/processed/` directory. Log it with every run."""

    model_config = _FROZEN

    built_at: str
    seed: int
    scheme_version: int
    fractions: SplitFractions
    # Rows read per Hugging Face split, or None for everything. Anything but
    # None is a smoke build, and nothing measured on it is a result.
    limit: int | None
    datasets: tuple[DatasetSummary, ...]
    split_report: SplitReport
    corpus: CorpusStats
    # Same normalised title, different text: which text a document carries
    # then depends on row order. See `CorpusBuild.n_title_conflicts`.
    corpus_title_conflicts: int = Field(ge=0)
    leakage_groups: int = Field(ge=0)
    missing_titles: int = Field(ge=0)
    missing_titles_sample: tuple[str, ...]
    files: tuple[str, ...]

    def problems(self) -> tuple[str, ...]:
        """Why these files should not be trained or evaluated on, if they should not.

        Only coverage is here. Leakage and title conflicts are reported in
        `summary` and need a decision, but they do not make the files unusable;
        a gold paragraph missing from the corpus does.
        """
        found = []
        if self.missing_titles:
            found.append(
                f"{self.missing_titles} supporting titles have no document in the corpus, so "
                f"recall is capped below 1.0 for the questions that need them"
            )
        return tuple(found)

    def summary(self) -> str:
        lines = [f"seed {self.seed}, scheme v{self.scheme_version}"]
        if self.limit is not None:
            lines.append(f"LIMITED to {self.limit} rows per split - a smoke build, not the data")
        for dataset in self.datasets:
            counts = ", ".join(f"{split}={n}" for split, n in sorted(dataset.splits.items()))
            coverage = (
                f"{dataset.titles_missing} of {dataset.titles_checked} titles missing"
                if dataset.coverage_measured
                else "coverage UNMEASURED - no supporting titles"
            )
            source = "corpus source" if dataset.contributes_to_corpus else "no paragraphs"
            lines.append(f"  {dataset.name:<16} {counts}; {source}; {coverage}")
        lines.append(
            f"corpus      {self.corpus.n_docs} docs, median {self.corpus.median_chars} chars, "
            f"p95 {self.corpus.p95_chars}, {self.corpus_title_conflicts} title conflicts"
        )
        lines.append(f"leakage     {self.leakage_groups} groups across splits (see leakage.json)")
        lines.append(f"missing     {self.missing_titles} supporting titles not in the corpus")
        return "\n".join(lines)


class Prepared(BaseModel):
    """Everything `write_prepared` writes, held together until it is written."""

    model_config = _FROZEN

    examples: tuple[QAExample, ...]
    corpus: CorpusBuild
    leakage: tuple[LeakageGroup, ...]
    manifest: Manifest


def read_dataset(rows: Iterable[Mapping[str, object]], spec: DatasetSpec) -> DatasetPart:
    """One pass over `rows`: every row becomes an example, and its paragraphs
    feed the corpus if the spec says where they are.

    The tap is what makes it one pass. `build_corpus` consumes an iterable, and
    each row is mapped to an example on its way through, so a download is read
    once and its two outputs cannot come from different rows.
    """
    examples: list[QAExample] = []

    def tap() -> Iterator[Mapping[str, object]]:
        for row in rows:
            examples.append(row_to_example(row, spec))
            yield row

    if spec.context_key is None:
        for _ in tap():
            pass
        return DatasetPart(spec=spec, examples=tuple(examples), corpus=None)
    corpus = build_corpus(tap(), spec, context_key=spec.context_key)
    return DatasetPart(spec=spec, examples=tuple(examples), corpus=corpus)


def merge_corpora(builds: Sequence[CorpusBuild]) -> CorpusBuild:
    """One corpus from several, deduplicated on `doc_id`, first dataset wins.

    The same Wikipedia title can come from HotpotQA and from 2WikiMultihopQA
    with different text. Keeping the first is deterministic for a fixed dataset
    order, but the losing text is gone - so a cross-dataset clash is counted as
    a title conflict, exactly as a clash within one dataset is.
    """
    kept: dict[str, CorpusDoc] = {}
    duplicates = 0
    conflicts = 0
    for build in builds:
        for doc in build.docs:
            existing = kept.get(doc.doc_id)
            if existing is None:
                kept[doc.doc_id] = doc
                continue
            duplicates += 1
            if existing.text != doc.text:
                conflicts += 1
    return CorpusBuild(
        docs=tuple(sorted(kept.values(), key=lambda doc: doc.doc_id)),
        n_rows=sum(build.n_rows for build in builds),
        n_paragraphs=sum(build.n_paragraphs for build in builds),
        n_duplicate_titles=sum(build.n_duplicate_titles for build in builds) + duplicates,
        n_title_conflicts=sum(build.n_title_conflicts for build in builds) + conflicts,
        n_empty_skipped=sum(build.n_empty_skipped for build in builds),
    )


def assemble(
    parts: Sequence[DatasetPart],
    *,
    seed: int,
    fractions: SplitFractions = DEFAULT_FRACTIONS,
    limit: int | None = None,
    built_at: str | None = None,
) -> Prepared:
    """Split, merge, and check. The whole of preparation once the rows are read."""
    if not parts:
        raise DataError("nothing to prepare: no datasets")
    names = [part.spec.name for part in parts]
    if len(set(names)) != len(names):
        raise DataError(f"a dataset is listed twice: {names}")

    examples = make_splits([e for part in parts for e in part.examples], seed, fractions)
    corpus = merge_corpora([part.corpus for part in parts if part.corpus is not None])
    if not corpus.docs:
        raise DataError(
            "the corpus is empty - no dataset in this build has a context_key, so there "
            "is nothing to retrieve from"
        )

    missing_all: list[str] = []
    summaries: list[DatasetSummary] = []
    for part in parts:
        own = [e for e in examples if e.dataset == part.spec.name]
        missing = find_missing_titles(own, corpus.docs)
        missing_all.extend(missing)
        summaries.append(
            DatasetSummary(
                name=part.spec.name,
                hf_path=part.spec.hf_path,
                hf_name=part.spec.hf_name,
                hf_splits=part.spec.hf_splits,
                n_rows=len(part.examples),
                splits=dict(Counter(e.split.value for e in own if e.split is not None)),
                contributes_to_corpus=part.corpus is not None,
                titles_checked=len({t for e in own for t in e.supporting_titles}),
                titles_missing=len(missing),
            )
        )

    leakage = find_leakage(examples)
    missing_sorted = tuple(sorted(set(missing_all)))
    manifest = Manifest(
        built_at=built_at or datetime.now(UTC).isoformat(timespec="seconds"),
        seed=seed,
        scheme_version=SCHEME_VERSION,
        fractions=fractions,
        limit=limit,
        datasets=tuple(summaries),
        split_report=summarize(examples, seed),
        corpus=summarize_corpus(corpus.docs),
        corpus_title_conflicts=corpus.n_title_conflicts,
        leakage_groups=len(leakage),
        missing_titles=len(missing_sorted),
        missing_titles_sample=missing_sorted[:_MISSING_SAMPLE],
        files=tuple(sorted(_planned_files(examples))),
    )
    return Prepared(examples=examples, corpus=corpus, leakage=leakage, manifest=manifest)


def _planned_files(examples: Sequence[QAExample]) -> set[str]:
    names = {CORPUS_FILENAME, MANIFEST_FILENAME, LEAKAGE_FILENAME}
    for example in examples:
        assert example.split is not None, "make_splits assigns every example a split"
        names.add(examples_filename(example.dataset, example.split))
    return names


def write_prepared(prepared: Prepared, directory: str | Path) -> tuple[Path, ...]:
    """Write one JSONL per dataset and split, the corpus, leakage and the manifest.

    The manifest goes last. A directory with a manifest in it is a finished
    build; one without is an interrupted one, and `prepare-data` refuses to
    write over a finished build without `--force`.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    by_file: dict[str, list[QAExample]] = {}
    for example in prepared.examples:
        assert example.split is not None, "make_splits assigns every example a split"
        by_file.setdefault(examples_filename(example.dataset, example.split), []).append(example)
    for name in sorted(by_file):
        write_jsonl(by_file[name], target / name)
        written.append(target / name)

    write_corpus(prepared.corpus.docs, target / CORPUS_FILENAME)
    written.append(target / CORPUS_FILENAME)

    leakage_path = target / LEAKAGE_FILENAME
    leakage_path.write_text(
        json.dumps([group.model_dump(mode="json") for group in prepared.leakage], indent=2),
        encoding="utf-8",
    )
    written.append(leakage_path)

    manifest_path = target / MANIFEST_FILENAME
    manifest_path.write_text(prepared.manifest.model_dump_json(indent=2), encoding="utf-8")
    written.append(manifest_path)
    return tuple(written)


def download_rows(  # pragma: no cover - needs the network and the [data] extra
    spec: DatasetSpec, limit: int | None = None
) -> Iterator[Mapping[str, object]]:
    """Every row of every split in `spec.hf_splits`, lazily, `limit` per split.

    Not exercised in CI, and **not yet run against the live datasets**: the
    field maps and split names in `DATASET_SPECS` are from the published
    schemas. A smoke build (`--limit 50`) is the check.

    Lazy because HotpotQA's training split with its contexts is gigabytes as
    Python objects; `read_dataset` consumes each row once and keeps only what it
    needs. Excluded from coverage as a whole, like `load_hf_dataset`: every
    statement past the import needs a download.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataError("prepare-data needs the `datasets` package: uv sync --extra data") from exc

    for hf_split in spec.hf_splits:
        # Streaming when limited, so a smoke build fetches a few rows rather
        # than the whole split. A full build downloads and caches normally.
        loaded = load_dataset(
            spec.hf_path, spec.hf_name, split=hf_split, streaming=limit is not None
        )
        for index, row in enumerate(loaded):
            if limit is not None and index >= limit:
                break
            if not isinstance(row, Mapping):
                raise DataError(f"{spec.name}/{hf_split}: row {index} is not a mapping")
            yield row
