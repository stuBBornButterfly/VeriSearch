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

**Coverage is the gate, at the level of the sentence.** Every supporting fact
names a title and a sentence. The sentence, as the example's own context
gives it, must appear in some corpus document with that title. A title being
present is not enough: the first full build had every title present and was
missing the answering sentence for 6.6% of 2WikiMultihopQA's facts, because
one version of each paragraph had been kept and the others dropped. A missing
sentence caps what the policy can find, and the cap looks exactly like a model
that did not learn. Titles are still checked, as the cheaper diagnostic. A
dataset whose examples carry no supporting facts is reported as *unmeasured*,
never as covered.

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
    extra_versions,
    find_missing_titles,
    normalize_title,
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
    # (normalised title, whitespace-normalised sentence) for every supporting
    # fact, taken from the example's own context.
    evidence: tuple[tuple[str, str], ...] = ()
    # Supporting facts that could not be resolved to a sentence in their own
    # row - an id past the end of the paragraph, a title the context does not
    # have. A defect in the data, not in the corpus, so counted, not failed.
    evidence_unreadable: int = Field(default=0, ge=0)


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
    # Supporting sentences checked, and how many are in no document with their
    # title. The gate - see the module docstring.
    evidence_checked: int = Field(ge=0)
    evidence_missing: int = Field(ge=0)
    evidence_unreadable: int = Field(ge=0)

    @property
    def coverage_measured(self) -> bool:
        return self.titles_checked > 0 or self.evidence_checked > 0


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
    # Documents beyond the first for their title - several versions of one page,
    # all kept. See `CorpusBuild`.
    corpus_extra_versions: int = Field(ge=0)
    leakage_groups: int = Field(ge=0)
    missing_titles: int = Field(ge=0)
    missing_titles_sample: tuple[str, ...]
    evidence_missing: int = Field(ge=0)
    evidence_missing_sample: tuple[str, ...]
    evidence_unreadable: int = Field(ge=0)
    files: tuple[str, ...]

    def problems(self) -> tuple[str, ...]:
        """Why these files should not be trained or evaluated on, if they should not.

        Only coverage is here. Leakage and extra versions are reported in
        `summary` and need a decision, but they do not make the files unusable;
        gold evidence missing from the corpus does.
        """
        found = []
        if self.missing_titles:
            found.append(
                f"{self.missing_titles} supporting titles have no document in the corpus, so "
                f"recall is capped below 1.0 for the questions that need them"
            )
        if self.evidence_missing:
            found.append(
                f"{self.evidence_missing} supporting sentences are in no corpus document with "
                f"their title, so the evidence for those facts cannot be retrieved"
            )
        return tuple(found)

    def summary(self) -> str:
        lines = [f"seed {self.seed}, scheme v{self.scheme_version}"]
        if self.limit is not None:
            lines.append(f"LIMITED to {self.limit} rows per split - a smoke build, not the data")
        for dataset in self.datasets:
            counts = ", ".join(f"{split}={n}" for split, n in sorted(dataset.splits.items()))
            coverage = (
                f"{dataset.titles_missing} of {dataset.titles_checked} titles and "
                f"{dataset.evidence_missing} of {dataset.evidence_checked} gold sentences "
                f"missing, {dataset.evidence_unreadable} facts unreadable"
                if dataset.coverage_measured
                else "coverage UNMEASURED - no supporting facts"
            )
            source = "corpus source" if dataset.contributes_to_corpus else "no paragraphs"
            lines.append(f"  {dataset.name:<16} {counts}; {source}; {coverage}")
        lines.append(
            f"corpus      {self.corpus.n_docs} docs, median {self.corpus.median_chars} chars, "
            f"p95 {self.corpus.p95_chars}, {self.corpus_extra_versions} extra versions of a "
            f"title kept"
        )
        lines.append(f"leakage     {self.leakage_groups} groups across splits (see leakage.json)")
        lines.append(f"missing     {self.missing_titles} supporting titles not in the corpus")
        lines.append(
            f"evidence    {self.evidence_missing} gold sentences not in the corpus, "
            f"{self.evidence_unreadable} facts unreadable in their own row"
        )
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
    evidence: list[tuple[str, str]] = []
    unreadable = 0

    def tap() -> Iterator[Mapping[str, object]]:
        nonlocal unreadable
        for row in rows:
            examples.append(row_to_example(row, spec))
            found, bad = supporting_evidence(row, spec)
            evidence.extend(found)
            unreadable += bad
            yield row

    corpus: CorpusBuild | None = None
    if spec.context_key is None:
        for _ in tap():
            pass
    else:
        corpus = build_corpus(tap(), spec, context_key=spec.context_key)
    return DatasetPart(
        spec=spec,
        examples=tuple(examples),
        corpus=corpus,
        evidence=tuple(evidence),
        evidence_unreadable=unreadable,
    )


def _sentences_by_title(value: object) -> dict[str, Sequence[object]]:
    """A row's own context as title -> sentences, in either published shape."""
    pairs: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        titles = value.get("title", [])
        groups = value.get("sentences", [])
        if not isinstance(titles, (list, tuple)) or not isinstance(groups, (list, tuple)):
            return {}
        pairs = zip(titles, groups, strict=False)
    elif isinstance(value, (list, tuple)):
        pairs = (
            (item[0], item[1])
            for item in value
            if isinstance(item, (list, tuple)) and len(item) >= 2
        )
    else:
        return {}
    found: dict[str, Sequence[object]] = {}
    for title, group in pairs:
        if isinstance(title, str) and isinstance(group, (list, tuple)):
            found.setdefault(title, group)
    return found


def _supporting_facts(value: object) -> tuple[list[tuple[object, object]], int]:
    """(title, sentence id) pairs, in either published shape, and how many items
    were not pairs at all - a flattened title list has no sentence to check."""
    if isinstance(value, Mapping):
        titles = value.get("title", [])
        ids = value.get("sent_id", [])
        if not isinstance(titles, (list, tuple)) or not isinstance(ids, (list, tuple)):
            return [], 0
        return list(zip(titles, ids, strict=False)), 0
    if not isinstance(value, (list, tuple)):
        return [], 0
    facts: list[tuple[object, object]] = []
    skipped = 0
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            facts.append((item[0], item[1]))
        else:
            skipped += 1
    return facts, skipped


def supporting_evidence(
    row: Mapping[str, object], spec: DatasetSpec
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Each supporting fact's sentence from the row's own context, and how many
    facts could not be resolved to one.

    Titles are normalised as the corpus normalises them and sentences have their
    whitespace collapsed, so the check compares content rather than spacing.
    """
    if spec.supporting_facts_key is None or spec.context_key is None:
        return (), 0
    if spec.supporting_facts_key not in row or spec.context_key not in row:
        return (), 0
    context = _sentences_by_title(row[spec.context_key])
    facts, unreadable = _supporting_facts(row[spec.supporting_facts_key])
    found: list[tuple[str, str]] = []
    for title, sent_id in facts:
        if not isinstance(title, str) or not isinstance(sent_id, int) or isinstance(sent_id, bool):
            unreadable += 1
            continue
        sentences = context.get(title)
        if sentences is None or not 0 <= sent_id < len(sentences):
            unreadable += 1
            continue
        raw = sentences[sent_id]
        sentence = " ".join(raw.split()) if isinstance(raw, str) else ""
        if not sentence:
            unreadable += 1
            continue
        found.append((normalize_title(title), sentence))
    return tuple(found), unreadable


def _evidence_index(docs: Iterable[CorpusDoc]) -> dict[str, tuple[str, ...]]:
    """Normalised title -> the text of every version.

    The text is used as built: `build_corpus` already collapses whitespace when
    it joins a paragraph's sentences, so only the sentence side needs it.
    """
    index: dict[str, list[str]] = {}
    for doc in docs:
        index.setdefault(normalize_title(doc.title), []).append(doc.text)
    return {title: tuple(texts) for title, texts in index.items()}


def missing_evidence(
    evidence: Iterable[tuple[str, str]], index: Mapping[str, Sequence[str]]
) -> list[tuple[str, str]]:
    """The facts whose sentence appears in no document carrying their title."""
    return [
        (title, sentence)
        for title, sentence in evidence
        if not any(sentence in text for text in index.get(title, ()))
    ]


def merge_corpora(builds: Sequence[CorpusBuild]) -> CorpusBuild:
    """One corpus from several, by the rule `build_corpus` applies within one.

    A document is a title and a text, so the same page from HotpotQA and from
    2WikiMultihopQA is one document when the text matches and two when it does
    not. Neither dataset's version is dropped, and the result does not depend
    on the order the datasets are listed in.
    """
    kept: dict[str, CorpusDoc] = {}
    duplicates = 0
    for build in builds:
        for doc in build.docs:
            existing = kept.get(doc.doc_id)
            if existing is None:
                kept[doc.doc_id] = doc
                continue
            duplicates += 1
            if doc.title < existing.title:
                kept[doc.doc_id] = doc
    docs = tuple(sorted(kept.values(), key=lambda doc: doc.doc_id))
    return CorpusBuild(
        docs=docs,
        n_rows=sum(build.n_rows for build in builds),
        n_paragraphs=sum(build.n_paragraphs for build in builds),
        n_duplicate_paragraphs=sum(build.n_duplicate_paragraphs for build in builds) + duplicates,
        n_extra_versions=extra_versions(docs),
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

    index = _evidence_index(corpus.docs)
    missing_all: list[str] = []
    evidence_gone: list[tuple[str, str]] = []
    summaries: list[DatasetSummary] = []
    for part in parts:
        own = [e for e in examples if e.dataset == part.spec.name]
        missing = find_missing_titles(own, corpus.docs)
        missing_all.extend(missing)
        gone = missing_evidence(part.evidence, index)
        evidence_gone.extend(gone)
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
                evidence_checked=len(part.evidence),
                evidence_missing=len(gone),
                evidence_unreadable=part.evidence_unreadable,
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
        corpus_extra_versions=corpus.n_extra_versions,
        leakage_groups=len(leakage),
        missing_titles=len(missing_sorted),
        missing_titles_sample=missing_sorted[:_MISSING_SAMPLE],
        evidence_missing=len(evidence_gone),
        evidence_missing_sample=tuple(
            f"{title}: {sentence[:80]}" for title, sentence in evidence_gone[:_MISSING_SAMPLE]
        ),
        evidence_unreadable=sum(part.evidence_unreadable for part in parts),
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

    Not exercised in CI. Run against the live datasets by a 50-row smoke build
    on 2026-10-17; a full build has not run yet.

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
        try:
            loaded = load_dataset(
                spec.hf_path, spec.hf_name, split=hf_split, streaming=limit is not None
            )
        except Exception as exc:
            # Broad on purpose, and only around this one call. `datasets` says
            # "this source cannot be read" with a dozen unrelated types - a
            # RuntimeError for a script-based mirror, a ValueError for an
            # unknown split, its own DatasetNotFoundError, ConnectionError -
            # and to this command they all mean one thing. Translated, not
            # swallowed: the type and message travel with it.
            raise DataError(
                f"{spec.name}: cannot load {spec.hf_path} split {hf_split!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        for index, row in enumerate(loaded):
            if limit is not None and index >= limit:
                break
            if not isinstance(row, Mapping):
                raise DataError(f"{spec.name}/{hf_split}: row {index} is not a mapping")
            yield row
