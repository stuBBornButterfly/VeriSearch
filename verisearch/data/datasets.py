"""QA example representation and dataset loading.

PROJECT.md section 7 names four datasets: HotpotQA (distractor) and
2WikiMultihopQA in distribution, Musique and Bamboogle out of distribution.

Two loading paths, deliberately:

* `load_jsonl` / `write_jsonl` - local files. Deterministic, offline, and what
  every test and every training run actually reads. Snapshots are committed as
  data, so a result can be reproduced without re-downloading anything.
* `load_hf_dataset` - pulls from Hugging Face once, to produce those snapshots.
  `datasets` lives in the `[data]` extra, not the core dependencies, so the
  import is deferred into the function. The PR pipeline never installs it and
  must never need it. `data/prepare.py` is the command-line path that uses it.

The field maps in `DATASET_SPECS` were checked against the live datasets by a
50-row smoke build (`prepare-data --limit 50`) on 2026-10-17, which found two
of them wrong: the 2WikiMultihopQA mirror was a loading script `datasets` 5
will not run, and Bamboogle's rows carry no id. Both are fixed and noted at
their specs. A full build reads every row, so a key absent only from some rows
can still surface there; it raises `DataError` naming the keys the row had.

Owner: Ayesha.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping  # runtime: used in isinstance checks below
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class DataError(Exception):
    """Raised when a dataset cannot be read or a row cannot be mapped.

    PROJECT.md section 3: "Fail loud. Typed exceptions."
    """


class Split(StrEnum):
    """Which slice an example belongs to.

    OOD is not a slice of the in-distribution pool - Musique and Bamboogle are
    separate datasets, used whole, and never split.
    """

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    OOD = "ood"


class Distribution(StrEnum):
    """Whether a dataset is in or out of distribution. PROJECT.md section 7."""

    IN_DISTRIBUTION = "in_distribution"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_question(text: str) -> str:
    """Canonical form used for duplicate and leakage detection.

    Lowercases, strips accents, drops punctuation and articles, collapses
    whitespace. Deliberately aggressive: two questions that differ only in
    punctuation are the same question for leakage purposes, and a false positive
    here costs a manual look while a false negative costs a wrong headline
    number.

    This is *not* the answer normalisation used by EM and F1 - that belongs in
    `eval/metrics.py` and must follow the SQuAD convention exactly so numbers
    stay comparable with published work.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    stripped = _PUNCTUATION.sub(" ", folded.lower())
    return _WHITESPACE.sub(" ", _ARTICLES.sub(" ", stripped)).strip()


class QAExample(BaseModel):
    """One multi-hop question with its gold answers.

    DECIDE (contract session): this type crosses the boundary - Dhrubo's episode
    loop consumes examples to run rollouts - so it may belong in `types.py`
    rather than here. Kept in `data/` for now so this module does not depend on
    the unmerged contract proposal. Moving it later is a rename, not a redesign.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    # Plural because EM and F1 need alias tolerance, and Musique ships
    # `answer_aliases` alongside the primary answer.
    answers: tuple[str, ...] = Field(min_length=1)
    dataset: str = Field(min_length=1)
    distribution: Distribution
    # Gold paragraph titles where the datasets provide them. Used to measure
    # retrieval recall independently of answer correctness - a run can be right
    # for the wrong reasons and this is how that gets caught.
    supporting_titles: tuple[str, ...] = ()
    split: Split | None = None

    @property
    def normalized_question(self) -> str:
        return normalize_question(self.question)

    def with_split(self, split: Split) -> QAExample:
        return self.model_copy(update={"split": split})


class DatasetSpec(BaseModel):
    """How to read one Hugging Face dataset into `QAExample`.

    A field map rather than a hand-written adapter per dataset: when a published
    schema turns out to differ from what is recorded here, the fix is one string
    in this file instead of a new code path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    hf_path: str
    hf_name: str | None = None
    distribution: Distribution
    # None when the dataset ships no id - Bamboogle does not. The id is then
    # derived from the question text; see `row_to_example`.
    id_key: str | None
    question_key: str
    answer_key: str
    # Optional keys. Absent from the row is fine; present but of the wrong shape
    # is an error.
    aliases_key: str | None = None
    supporting_facts_key: str | None = None
    # The Hugging Face splits that together form this dataset's pool. Our own
    # train/dev/test are assigned afterwards by hash (`splits.py`), so these
    # name *what to download*, not how it is divided. Splits without public
    # answers are left out - they cannot be scored.
    hf_splits: tuple[str, ...] = Field(default=("train",), min_length=1)
    # Where the row's own paragraphs are, or None when it ships none. Every
    # dataset with a context key contributes those paragraphs to the corpus.
    # This field is the corpus decision written as data: change it here, and
    # the change is reviewed, not buried in a script.
    context_key: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        hf_path="hotpotqa/hotpot_qa",
        hf_name="distractor",
        distribution=Distribution.IN_DISTRIBUTION,
        id_key="id",
        question_key="question",
        answer_key="answer",
        supporting_facts_key="supporting_facts",
        hf_splits=("train", "validation"),
        context_key="context",
    ),
    "2wikimultihopqa": DatasetSpec(
        name="2wikimultihopqa",
        # Not xanhho/2WikiMultihopQA: that mirror is a loading script, which
        # `datasets` 5 refuses to run - found by the first live smoke build.
        # This one is plain parquet, and its split sizes match the official
        # release exactly (167,454 / 12,576 / 12,576). It carries no license
        # field on the Hub; the original release is Apache-2.0.
        hf_path="framolfese/2WikiMultihopQA",
        distribution=Distribution.IN_DISTRIBUTION,
        id_key="id",
        question_key="question",
        answer_key="answer",
        supporting_facts_key="supporting_facts",
        hf_splits=("train", "validation"),
        context_key="context",
    ),
    "musique": DatasetSpec(
        name="musique",
        hf_path="dgslibisey/MuSiQue",
        distribution=Distribution.OUT_OF_DISTRIBUTION,
        id_key="id",
        question_key="question",
        answer_key="answer",
        aliases_key="answer_aliases",
        # MuSiQue's test split has no public answers.
        hf_splits=("validation",),
    ),
    "bamboogle": DatasetSpec(
        name="bamboogle",
        hf_path="chiayewken/bamboogle",
        distribution=Distribution.OUT_OF_DISTRIBUTION,
        # The live rows carry only Question and Answer.
        id_key=None,
        question_key="Question",
        answer_key="Answer",
        hf_splits=("test",),
    ),
}


def _require(row: Mapping[str, object], key: str, spec: DatasetSpec) -> object:
    if key not in row:
        available = ", ".join(sorted(row)) or "<none>"
        raise DataError(
            f"{spec.name}: row has no key {key!r}. Keys present: {available}. "
            f"Fix the field map in DATASET_SPECS."
        )
    return row[key]


def _as_text(value: object, key: str, spec: DatasetSpec) -> str:
    if not isinstance(value, str):
        raise DataError(f"{spec.name}: expected a string at {key!r}, got {type(value).__name__}")
    return value


def _extract_titles(value: object, key: str, spec: DatasetSpec) -> tuple[str, ...]:
    """Pull paragraph titles out of a supporting-facts field.

    HotpotQA and 2WikiMultihopQA both use supporting facts, in two shapes: a
    mapping of parallel lists ({"title": [...], "sent_id": [...]}) or a list of
    [title, sent_id] pairs. Order is preserved and duplicates dropped - a title
    appears once per supporting sentence and we want paragraphs, not sentences.
    """
    titles: list[str] = []
    if isinstance(value, Mapping):
        raw_titles = value.get("title", [])
        if not isinstance(raw_titles, (list, tuple)):
            raise DataError(f"{spec.name}: {key!r}['title'] is not a sequence")
        titles = [t for t in raw_titles if isinstance(t, str)]
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                titles.append(item)
            elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                titles.append(item[0])
    else:
        raise DataError(
            f"{spec.name}: expected a mapping or sequence at {key!r}, got {type(value).__name__}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for title in titles:
        if title not in seen:
            seen.add(title)
            ordered.append(title)
    return tuple(ordered)


def _extract_aliases(value: object, key: str, spec: DatasetSpec) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str) and item)
    raise DataError(
        f"{spec.name}: expected a string or sequence at {key!r}, got {type(value).__name__}"
    )


def row_to_example(row: Mapping[str, object], spec: DatasetSpec) -> QAExample:
    """Map one raw dataset row onto `QAExample`.

    Separated from the download so the mapping logic is unit-testable offline
    against synthetic rows. PROJECT.md section 3: no network in unit tests.
    """
    question = _as_text(_require(row, spec.question_key, spec), spec.question_key, spec)
    if spec.id_key is not None:
        question_id = _as_text(_require(row, spec.id_key, spec), spec.id_key, spec)
    else:
        question_id = derived_id(question)
    answer = _as_text(_require(row, spec.answer_key, spec), spec.answer_key, spec)

    answers: list[str] = [answer] if answer else []
    if spec.aliases_key is not None and spec.aliases_key in row:
        for alias in _extract_aliases(row[spec.aliases_key], spec.aliases_key, spec):
            if alias not in answers:
                answers.append(alias)
    if not answers:
        raise DataError(f"{spec.name}: example {question_id!r} has an empty answer")

    titles: tuple[str, ...] = ()
    if spec.supporting_facts_key is not None and spec.supporting_facts_key in row:
        titles = _extract_titles(row[spec.supporting_facts_key], spec.supporting_facts_key, spec)

    return QAExample(
        question_id=f"{spec.name}-{question_id}",
        question=question,
        answers=tuple(answers),
        dataset=spec.name,
        distribution=spec.distribution,
        supporting_titles=titles,
        split=Split.OOD if spec.distribution is Distribution.OUT_OF_DISTRIBUTION else None,
    )


def derived_id(question: str) -> str:
    """An id for a dataset that ships none, from the question's exact text.

    From the text and not the row number: splits are a function of the id, so
    an id that moved when the file was reordered would move the example
    between splits. Two rows with the same question get the same id, and
    `check_unique_ids` then refuses them - which is right, since they are the
    same question twice.
    """
    return "q" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


def spec_for(dataset: str) -> DatasetSpec:
    """Look up a dataset spec by name, or raise listing the known names.

    Split out of `load_hf_dataset` so the argument check is reachable in CI. The
    download itself is not, and a function that can never run is a function
    whose validation is never tested.
    """
    spec = DATASET_SPECS.get(dataset)
    if spec is None:
        known = ", ".join(sorted(DATASET_SPECS))
        raise DataError(f"unknown dataset {dataset!r}. Known: {known}")
    return spec


def load_hf_dataset(  # pragma: no cover - needs the network and the [data] extra
    dataset: str, hf_split: str = "train", limit: int | None = None
) -> tuple[QAExample, ...]:
    """Download one dataset from Hugging Face and map it to examples.

    Not exercised in CI - it needs the network and the `[data]` extra. Use it
    once to produce a JSONL snapshot, commit the snapshot, and read that
    everywhere else.

    Excluded from coverage as a whole rather than line by line: every statement
    past the spec lookup requires a download, and scattering pragmas through the
    body makes the report harder to read than saying so once.
    """
    spec = spec_for(dataset)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - requires the [data] extra
        raise DataError(
            "load_hf_dataset needs the `datasets` package: uv pip install -e '.[data]'"
        ) from exc

    loaded = load_dataset(spec.hf_path, spec.hf_name, split=hf_split)
    examples: list[QAExample] = []
    for index, row in enumerate(loaded):
        if limit is not None and index >= limit:
            break
        if not isinstance(row, Mapping):
            raise DataError(f"{spec.name}: row {index} is not a mapping")
        examples.append(row_to_example(row, spec))
    return tuple(examples)


def load_jsonl(path: str | Path) -> tuple[QAExample, ...]:
    """Read examples from a JSONL snapshot.

    Reports the line number on a bad row. A dataset file is thousands of lines
    long and "validation error" without a position is useless.
    """
    target = Path(path)
    if not target.is_file():
        raise DataError(f"dataset file not found: {target}")

    examples: list[QAExample] = []
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
                examples.append(QAExample.model_validate(payload))
            except ValueError as exc:
                raise DataError(f"{target}:{lineno}: {exc}") from exc
    return tuple(examples)


def write_jsonl(examples: Iterable[QAExample], path: str | Path) -> int:
    """Write examples to JSONL, one per line, and return the count.

    Sorted by `question_id` so the file is byte-stable across runs: a snapshot
    that reorders itself produces a meaningless diff and hides real changes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(examples, key=lambda e: e.question_id)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for example in ordered:
            handle.write(example.model_dump_json(exclude_none=True))
            handle.write("\n")
    return len(ordered)


def check_unique_ids(examples: Sequence[QAExample]) -> None:
    """Raise if any `question_id` repeats.

    Split assignment is a function of `question_id`, so a collision silently
    puts the same example in one split twice, or two different examples in
    lockstep. Cheap to check, expensive to discover later.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for example in examples:
        if example.question_id in seen:
            duplicates.add(example.question_id)
        seen.add(example.question_id)
    if duplicates:
        listed = ", ".join(sorted(duplicates)[:5])
        raise DataError(f"{len(duplicates)} duplicate question_id(s): {listed}")
