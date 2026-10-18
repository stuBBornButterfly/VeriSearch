"""Build the retrieval corpus from the QA datasets' own context paragraphs.

**The corpus decision, and what it costs.** README records the choice: HotpotQA
Wikipedia abstracts. The full abstracts dump is roughly 5.2M paragraphs. Indexing
it means embedding all of them and holding the index in RAM, and the hardware
report puts this machine at 15.7 GB with no training GPU. So the corpus built
here is the union of the `context` paragraphs HotpotQA already ships with each
question - two gold paragraphs plus eight distractors per example.

That is a real weakening of the task and it must be reported as one. The corpus
is guaranteed to contain the answer and is three orders of magnitude smaller than
Wikipedia, so retrieval precision measured against it is an **upper bound** on
what the same policy would achieve against the full dump. It is not a substitute
for full-corpus retrieval; it is what makes the experiment runnable at all before
there is a GPU box. Say so in the writeup next to every retrieval number.

The distractors are why this is still a real retrieval problem rather than a
lookup: eight of every ten paragraphs in context are deliberate near-misses
chosen by HotpotQA's authors to be lexically plausible and wrong.

No network here. `build_corpus` takes rows, so every test drives it from
fixtures; `build_corpus_from_hf` is the one function that downloads, and it
defers the `datasets` import exactly as `load_hf_dataset` does.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping  # runtime: used in isinstance checks below
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from verisearch.data.datasets import DataError, DatasetSpec, spec_for

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from verisearch.data.datasets import QAExample

_FROZEN = ConfigDict(frozen=True, extra="forbid")

# Must stay identical to `eval.metrics.normalize_title`. It is duplicated rather
# than imported because `data/` sits below `eval/` and must not depend on it. If
# the two ever diverge, precision@k reads zero on data that is correct, silently
# - the two docstrings and `test_title_normalisation_matches_the_eval_rule` are
# what hold them together. TODO: once both branches merge, move this into a
# shared `verisearch/text.py` and delete one copy.
_TITLE_NOISE = re.compile(r"[\s_]+")
_WHITESPACE = re.compile(r"\s+")


class CorpusDoc(BaseModel):
    """One indexable paragraph. The unit the embedder and the index see."""

    model_config = _FROZEN

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class CorpusBuild(BaseModel):
    """Corpus plus the counts that tell you whether it is the corpus you wanted.

    **A document is a title and a text, not a title.** The datasets ship the
    same Wikipedia paragraph many times and the copies are not always
    identical - 2WikiMultihopQA especially carries several versions of one
    abstract, differing in leftover markup. Keeping one version per title
    dropped the others, and with them, on the first full build, the sentence
    that answers 6.6% of 2WikiMultihopQA's supporting facts - while a
    title-level coverage check reported nothing missing. So every distinct
    version is its own document, and the corpus no longer depends on row order.

    `n_extra_versions` counts documents beyond the first for their normalised
    title. The cost it measures is retrieval's: several versions of one page can
    fill the top k.
    """

    model_config = _FROZEN

    docs: tuple[CorpusDoc, ...]
    n_rows: int = Field(ge=0)
    n_paragraphs: int = Field(ge=0)
    # Paragraphs whose title and text were both already in the corpus.
    n_duplicate_paragraphs: int = Field(ge=0)
    n_extra_versions: int = Field(ge=0)
    n_empty_skipped: int = Field(ge=0)


class CorpusStats(BaseModel):
    """Size distribution. Read this before choosing an index type.

    Percentiles use nearest-rank, not interpolation: the result is an actual
    document length rather than an average of two, which is what you want when
    the number is being used to pick a maximum sequence length.
    """

    model_config = _FROZEN

    n_docs: int = Field(ge=0)
    n_chars: int = Field(ge=0)
    min_chars: int = Field(ge=0)
    median_chars: int = Field(ge=0)
    p95_chars: int = Field(ge=0)
    max_chars: int = Field(ge=0)


def normalize_title(title: str) -> str:
    """Canonical form for comparing Wikipedia paragraph titles.

    Underscores and spaces are interchangeable in Wikipedia titles and the
    datasets are not consistent about which they use. Duplicated from
    `eval.metrics.normalize_title` on purpose - see the comment on
    `_TITLE_NOISE`.
    """
    return _TITLE_NOISE.sub(" ", title).strip().lower()


def doc_id_for(title: str, text: str) -> str:
    """Stable document id from the normalised title and the text.

    Both, because a document is a version of a paragraph: two copies of one
    page that differ in their text are two documents (see `CorpusBuild`). The
    title is normalised so that `Ada_Lovelace` and `ada lovelace` with the same
    text are one document; the text is not, because the text is what gets
    embedded and checked for evidence. "Smash Cut", a film, and "Smash cut", an
    editing technique, normalise to one title and stay two documents.

    A hash, stable across runs, processes and machines, which `hash()` is not:
    Python randomises string hashing per process unless `PYTHONHASHSEED` is
    set, and the Jenkins pipeline deliberately leaves it unset.
    """
    canonical = normalize_title(title)
    if not canonical:
        raise DataError("cannot build a doc_id from a blank title")
    return hashlib.sha256(f"{canonical}\x00{text}".encode()).hexdigest()[:16]


def paragraphs_from_context(
    value: object, key: str, spec: DatasetSpec
) -> tuple[tuple[str, str], ...]:
    """Pull `(title, text)` pairs out of a context field.

    HotpotQA and 2WikiMultihopQA publish context in the same two shapes their
    supporting facts come in: a mapping of parallel lists
    (`{"title": [...], "sentences": [[...], ...]}`) or a list of
    `[title, [sentences]]` pairs. Both appear in the wild depending on which
    mirror of the dataset you pulled, so both are handled rather than guessed at.

    Sentences are stripped, empties dropped, and the rest joined with a single
    space. The published data is inconsistent about whether a sentence carries
    its trailing space, so joining raw produces words run together in some rows
    and double spaces in others - which changes the embedding of text that is
    supposed to be identical.
    """
    pairs: list[tuple[str, str]] = []

    if isinstance(value, Mapping):
        titles = value.get("title", [])
        sentences = value.get("sentences", [])
        if not isinstance(titles, (list, tuple)) or not isinstance(sentences, (list, tuple)):
            raise DataError(f"{spec.name}: {key!r} mapping must hold 'title' and 'sentences' lists")
        if len(titles) != len(sentences):
            raise DataError(
                f"{spec.name}: {key!r} has {len(titles)} titles for "
                f"{len(sentences)} sentence groups"
            )
        for title, group in zip(titles, sentences, strict=True):
            if isinstance(title, str):
                pairs.append((title, _join_sentences(group)))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], str):
                pairs.append((item[0], _join_sentences(item[1])))
    else:
        raise DataError(
            f"{spec.name}: expected a mapping or sequence at {key!r}, got {type(value).__name__}"
        )

    return tuple(pairs)


def _require_key(row: Mapping[str, object], key: str, spec: DatasetSpec) -> object:
    """Fetch `key` or raise listing what the row actually had.

    Same contract as the private helper in `datasets.py`, repeated rather than
    imported: reaching across a module boundary for an underscore-prefixed name
    is a dependency nobody declared and the next refactor breaks it silently.
    """
    if key not in row:
        available = ", ".join(sorted(row)) or "<none>"
        raise DataError(
            f"{spec.name}: row has no key {key!r}. Keys present: {available}. "
            f"Pass context_key= if this dataset names the field differently."
        )
    return row[key]


def _join_sentences(group: object) -> str:
    """Flatten one paragraph's sentences into a single normalised string."""
    if isinstance(group, str):
        parts = [group]
    elif isinstance(group, (list, tuple)):
        parts = [item for item in group if isinstance(item, str)]
    else:
        return ""
    joined = " ".join(part.strip() for part in parts if part.strip())
    return _WHITESPACE.sub(" ", joined).strip()


def build_corpus(
    rows: Iterable[Mapping[str, object]],
    spec: DatasetSpec,
    *,
    context_key: str = "context",
) -> CorpusBuild:
    """Collapse every row's context paragraphs into a deduplicated corpus.

    Deduplication is on the normalised title **and** the text: identical copies
    collapse, distinct versions are all kept. Documents come back sorted by
    `doc_id`, and when identical text arrives under two spellings of a title
    the smaller spelling is kept - so the corpus is the same for any order of
    the same rows, and a diff between two builds means the data changed.
    """
    seen: dict[str, CorpusDoc] = {}
    n_rows = 0
    n_paragraphs = 0
    n_duplicate_paragraphs = 0
    n_empty_skipped = 0

    for row in rows:
        n_rows += 1
        context = _require_key(row, context_key, spec)
        for title, text in paragraphs_from_context(context, context_key, spec):
            n_paragraphs += 1
            canonical = normalize_title(title)
            if not canonical or not text:
                # An empty paragraph embeds to a near-zero vector, which
                # `l2_normalize` maps to zeros and which then matches every
                # query equally badly. Keep it out of the index.
                n_empty_skipped += 1
                continue
            doc_id = doc_id_for(title, text)
            existing = seen.get(doc_id)
            if existing is None:
                seen[doc_id] = CorpusDoc(doc_id=doc_id, title=title, text=text)
                continue
            n_duplicate_paragraphs += 1
            if title < existing.title:
                seen[doc_id] = CorpusDoc(doc_id=doc_id, title=title, text=text)

    docs = tuple(sorted(seen.values(), key=lambda doc: doc.doc_id))
    return CorpusBuild(
        docs=docs,
        n_rows=n_rows,
        n_paragraphs=n_paragraphs,
        n_duplicate_paragraphs=n_duplicate_paragraphs,
        n_extra_versions=extra_versions(docs),
        n_empty_skipped=n_empty_skipped,
    )


def extra_versions(docs: Iterable[CorpusDoc]) -> int:
    """Documents beyond the first for each normalised title."""
    ordered = list(docs)
    return len(ordered) - len({normalize_title(doc.title) for doc in ordered})


def summarize_corpus(docs: Sequence[CorpusDoc]) -> CorpusStats:
    """Length distribution over `title + text`, which is what gets embedded."""
    lengths = sorted(len(f"{doc.title} {doc.text}") for doc in docs)
    if not lengths:
        return CorpusStats(
            n_docs=0, n_chars=0, min_chars=0, median_chars=0, p95_chars=0, max_chars=0
        )
    return CorpusStats(
        n_docs=len(lengths),
        n_chars=sum(lengths),
        min_chars=lengths[0],
        median_chars=lengths[len(lengths) // 2],
        # Nearest rank: ceil(p * N) as a 1-indexed rank. `int(0.95 * N)` is
        # wrong and not obviously wrong - for N = 20 it indexes the 20th of
        # 20 values, so p95 silently equals max_chars and the number is
        # useless for choosing a sequence length.
        p95_chars=lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)],
        max_chars=lengths[-1],
    )


def find_missing_titles(
    examples: Iterable[QAExample], docs: Iterable[CorpusDoc]
) -> tuple[str, ...]:
    """Supporting titles that no document in the corpus provides.

    This is the gate that catches a corpus built from the wrong split. If a gold
    paragraph is absent, the policy cannot retrieve it, maximum recall is below
    1.0 and accuracy is capped - and nothing anywhere else reports that. The
    number it caps at looks exactly like a model that did not learn.

    Returns normalised titles, sorted, so the result is reproducible and can be
    diffed between builds.
    """
    present = {normalize_title(doc.title) for doc in docs}
    missing = {
        normalize_title(title)
        for example in examples
        for title in example.supporting_titles
        if normalize_title(title) and normalize_title(title) not in present
    }
    return tuple(sorted(missing))


def write_corpus(docs: Iterable[CorpusDoc], path: str | Path) -> int:
    """Write the corpus to JSONL, sorted by `doc_id`, and return the count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(docs, key=lambda doc: doc.doc_id)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for doc in ordered:
            handle.write(doc.model_dump_json())
            handle.write("\n")
    return len(ordered)


def load_corpus(path: str | Path) -> tuple[CorpusDoc, ...]:
    """Read a corpus JSONL, reporting `file:line` on a bad row."""
    target = Path(path)
    if not target.is_file():
        raise DataError(f"corpus file not found: {target}")

    docs: list[CorpusDoc] = []
    with target.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataError(f"{target}:{lineno}: not valid JSON: {exc}") from exc
            try:
                docs.append(CorpusDoc.model_validate(payload))
            except ValueError as exc:
                raise DataError(f"{target}:{lineno}: {exc}") from exc
    return tuple(docs)


def build_corpus_from_hf(  # pragma: no cover - needs the network and the [data] extra
    dataset: str = "hotpotqa",
    hf_split: str = "train",
    limit: int | None = None,
    *,
    context_key: str = "context",
) -> CorpusBuild:
    """Download one dataset and build the corpus from its context paragraphs.

    Not exercised in CI - needs the network and the `[data]` extra. Run it once,
    write the result with `write_corpus`, and read that file everywhere else.

    Excluded from coverage as a whole rather than line by line: every statement
    past the spec lookup requires a download, and pretending otherwise with
    scattered pragmas makes the report harder to read than admitting it.
    """
    spec = spec_for(dataset)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - requires the [data] extra
        raise DataError(
            "build_corpus_from_hf needs the `datasets` package: uv pip install -e '.[data]'"
        ) from exc

    loaded = load_dataset(spec.hf_path, spec.hf_name, split=hf_split)
    rows: list[Mapping[str, object]] = []
    for index, row in enumerate(loaded):
        if limit is not None and index >= limit:
            break
        if not isinstance(row, Mapping):
            raise DataError(f"{spec.name}: row {index} is not a mapping")
        rows.append(row)
    return build_corpus(rows, spec, context_key=context_key)
