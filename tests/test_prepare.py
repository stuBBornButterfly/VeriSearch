"""Tests for `data/prepare.py`: rows in, the processed directory out.

Every row here is a fixture in the shape the field maps expect. No network,
per section 3 - `download_rows` is the one function that fetches, and it is
replaced by these rows in the CLI tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from verisearch.data.build_corpus import CorpusBuild, CorpusDoc, load_corpus
from verisearch.data.datasets import (
    DATASET_SPECS,
    DataError,
    Distribution,
    Split,
    load_jsonl,
    spec_for,
)
from verisearch.data.prepare import (
    CORPUS_FILENAME,
    LEAKAGE_FILENAME,
    MANIFEST_FILENAME,
    DatasetPart,
    Manifest,
    assemble,
    examples_filename,
    merge_corpora,
    read_dataset,
    write_prepared,
)
from verisearch.data.splits import assign_split

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

HOTPOT = spec_for("hotpotqa")
WIKI = spec_for("2wikimultihopqa")
MUSIQUE = spec_for("musique")
BAMBOOGLE = spec_for("bamboogle")
STAMP = "2026-10-17T00:00:00+00:00"


def hotpot_row(
    index: int, *, gold: str = "Ada", context_titles: tuple[str, ...] = ()
) -> dict[str, object]:
    titles = context_titles or (gold, f"Distractor {index}")
    return {
        "id": f"h{index}",
        "question": f"Hotpot question {index}?",
        "answer": "yes",
        "supporting_facts": {"title": [gold], "sent_id": [0]},
        "context": {"title": list(titles), "sentences": [[f"About {t}."] for t in titles]},
    }


def wiki_row(index: int) -> dict[str, object]:
    return {
        "_id": f"w{index}",
        "question": f"Wiki question {index}?",
        "answer": "no",
        "supporting_facts": [[f"Wiki gold {index}", 0]],
        "context": [[f"Wiki gold {index}", ["Wiki says so."]], ["Ada", ["Wiki's Ada."]]],
    }


def musique_row(index: int) -> dict[str, object]:
    return {
        "id": f"m{index}",
        "question": f"Musique {index}?",
        "answer": "x",
        "answer_aliases": ["ex"],
    }


def bamboogle_row(index: int) -> dict[str, object]:
    return {"id": f"b{index}", "Question": f"Bamboogle {index}?", "Answer": "y"}


def once(rows: list[dict[str, object]]) -> Iterator[Mapping[str, object]]:
    """A one-shot iterator, like a download. A second pass would see nothing."""
    yield from rows


def parts(n: int = 30) -> list[DatasetPart]:
    return [
        read_dataset(once([hotpot_row(i) for i in range(n)]), HOTPOT),
        read_dataset(once([wiki_row(i) for i in range(n)]), WIKI),
        read_dataset(once([musique_row(i) for i in range(5)]), MUSIQUE),
        read_dataset(once([bamboogle_row(i) for i in range(5)]), BAMBOOGLE),
    ]


# ---------------------------------------------------------------------------
# The specs encode the corpus decision
# ---------------------------------------------------------------------------


def test_every_in_distribution_dataset_contributes_paragraphs_and_no_ood_one_does() -> None:
    """Section 7's corpus is the in-distribution datasets' own context. If an
    OOD set ever gains a context_key, that is a change to the experiment and
    this test should be changed with it, deliberately."""
    for spec in DATASET_SPECS.values():
        if spec.distribution is Distribution.IN_DISTRIBUTION:
            assert spec.context_key is not None, spec.name
        else:
            assert spec.context_key is None, spec.name
        assert spec.hf_splits, spec.name


# ---------------------------------------------------------------------------
# One pass per dataset
# ---------------------------------------------------------------------------


def test_one_pass_yields_both_the_examples_and_the_corpus() -> None:
    """Fed a one-shot iterator. If either output needed a second pass over the
    rows it would come back empty."""
    part = read_dataset(once([hotpot_row(i) for i in range(3)]), HOTPOT)
    assert len(part.examples) == 3
    assert part.corpus is not None
    assert part.corpus.n_rows == 3
    assert {doc.title for doc in part.corpus.docs} == {
        "Ada",
        "Distractor 0",
        "Distractor 1",
        "Distractor 2",
    }


def test_a_dataset_without_paragraphs_yields_examples_only() -> None:
    part = read_dataset(once([musique_row(i) for i in range(4)]), MUSIQUE)
    assert part.corpus is None
    assert [e.question_id for e in part.examples] == [f"musique-m{i}" for i in range(4)]


def test_a_row_the_field_map_cannot_read_names_the_keys_it_had() -> None:
    with pytest.raises(DataError, match="Keys present"):
        read_dataset(once([{"question": "No id?"}]), HOTPOT)


# ---------------------------------------------------------------------------
# Merging corpora
# ---------------------------------------------------------------------------


def build(*docs: tuple[str, str, str], conflicts: int = 0) -> CorpusBuild:
    return CorpusBuild(
        docs=tuple(CorpusDoc(doc_id=i, title=t, text=x) for i, t, x in docs),
        n_rows=1,
        n_paragraphs=len(docs),
        n_duplicate_titles=0,
        n_title_conflicts=conflicts,
        n_empty_skipped=0,
    )


def test_merging_keeps_the_first_dataset_text_and_counts_the_clash() -> None:
    """The same title from two datasets with different text: the second text is
    lost, so it has to show up as a conflict rather than vanish."""
    first = build(("d1", "Ada", "Hotpot's Ada."), ("d2", "Bob", "Bob."))
    second = build(("d1", "Ada", "Wiki's Ada."), ("d3", "Cy", "Cy."), conflicts=1)

    merged = merge_corpora([first, second])

    assert [doc.doc_id for doc in merged.docs] == ["d1", "d2", "d3"]
    assert merged.docs[0].text == "Hotpot's Ada."
    assert merged.n_title_conflicts == 2  # one carried in, one from the merge
    assert merged.n_duplicate_titles == 1
    assert merged.n_rows == 2


def test_merging_identical_copies_is_a_duplicate_not_a_conflict() -> None:
    merged = merge_corpora([build(("d1", "Ada", "Same.")), build(("d1", "Ada", "Same."))])
    assert merged.n_duplicate_titles == 1
    assert merged.n_title_conflicts == 0


# ---------------------------------------------------------------------------
# Assembling
# ---------------------------------------------------------------------------


def test_assembly_splits_in_distribution_and_leaves_ood_whole() -> None:
    prepared = assemble(parts(), seed=42, built_at=STAMP)
    by_dataset: dict[str, set[Split]] = {}
    for example in prepared.examples:
        assert example.split is not None
        by_dataset.setdefault(example.dataset, set()).add(example.split)
    assert by_dataset["musique"] == {Split.OOD}
    assert by_dataset["bamboogle"] == {Split.OOD}
    assert by_dataset["hotpotqa"] <= {Split.TRAIN, Split.DEV, Split.TEST}
    assert Split.OOD not in by_dataset["hotpotqa"]


def test_the_corpus_comes_only_from_datasets_with_paragraphs() -> None:
    prepared = assemble(parts(), seed=42, built_at=STAMP)
    titles = {doc.title for doc in prepared.corpus.docs}
    assert "Ada" in titles
    assert "Wiki gold 0" in titles
    assert not any("Musique" in title or "Bamboogle" in title for title in titles)


def test_full_coverage_is_measured_and_reported() -> None:
    manifest = assemble(parts(), seed=42, built_at=STAMP).manifest
    hotpot = next(d for d in manifest.datasets if d.name == "hotpotqa")
    assert hotpot.coverage_measured
    assert hotpot.titles_missing == 0
    assert manifest.missing_titles == 0
    assert manifest.problems() == ()


def test_a_dataset_with_no_supporting_titles_is_unmeasured_not_covered() -> None:
    """Zero missing out of zero checked is not coverage. Musique and Bamboogle
    are scored against a corpus that was not built for them, and the manifest
    has to say that rather than print a reassuring 0."""
    manifest = assemble(parts(), seed=42, built_at=STAMP).manifest
    for name in ("musique", "bamboogle"):
        dataset = next(d for d in manifest.datasets if d.name == name)
        assert not dataset.coverage_measured
        assert not dataset.contributes_to_corpus
    assert "coverage UNMEASURED" in manifest.summary()


def test_a_gold_paragraph_missing_from_the_corpus_is_a_problem() -> None:
    """The row's own context lacks its gold paragraph - the shape of a corpus
    built from the wrong split. Recall is capped and nothing else reports it."""
    rows = [hotpot_row(i) for i in range(5)]
    rows.append(hotpot_row(99, gold="Nowhere", context_titles=("Elsewhere",)))
    prepared = assemble([read_dataset(once(rows), HOTPOT)], seed=42, built_at=STAMP)

    assert prepared.manifest.missing_titles == 1
    assert prepared.manifest.missing_titles_sample == ("nowhere",)
    (problem,) = prepared.manifest.problems()
    assert "recall is capped" in problem


def test_near_duplicate_questions_across_splits_are_reported_not_fixed() -> None:
    """Two ids that hash into different splits, asking the same question."""
    ids = [f"h{i}" for i in range(200)]
    first = next(i for i in ids if assign_split(f"hotpotqa-{i}", 42) is Split.TRAIN)
    second = next(i for i in ids if assign_split(f"hotpotqa-{i}", 42) is Split.TEST)
    rows = [hotpot_row(0) | {"id": first}, hotpot_row(0) | {"id": second}]

    prepared = assemble([read_dataset(once(rows), HOTPOT)], seed=42, built_at=STAMP)

    assert prepared.manifest.leakage_groups == 1
    assert len(prepared.leakage) == 1
    assert len(prepared.examples) == 2  # reported, and nothing dropped


def test_a_limited_build_says_so() -> None:
    summary = assemble(parts(), seed=42, limit=50, built_at=STAMP).manifest.summary()
    assert "LIMITED to 50 rows per split" in summary


def test_assembly_is_deterministic() -> None:
    first = assemble(parts(), seed=42, built_at=STAMP)
    again = assemble(parts(), seed=42, built_at=STAMP)
    assert first == again


def test_another_seed_moves_examples_between_splits() -> None:
    def splits(seed: int) -> list[Split | None]:
        return [e.split for e in assemble(parts(), seed=seed, built_at=STAMP).examples]

    assert splits(42) != splits(7)


def test_assembly_refuses_nothing_a_repeat_or_no_corpus() -> None:
    with pytest.raises(DataError, match="no datasets"):
        assemble([], seed=42)
    part = read_dataset(once([hotpot_row(0)]), HOTPOT)
    with pytest.raises(DataError, match="listed twice"):
        assemble([part, part], seed=42)
    ood_only = [read_dataset(once([musique_row(0)]), MUSIQUE)]
    with pytest.raises(DataError, match="corpus is empty"):
        assemble(ood_only, seed=42)


def test_the_build_time_is_recorded_when_not_given() -> None:
    assert assemble(parts(), seed=42).manifest.built_at.endswith("+00:00")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_examples_files_are_named_the_way_03_eval_reads_them() -> None:
    assert examples_filename("hotpotqa", Split.DEV) == "hotpotqa_dev.jsonl"


def test_writing_produces_every_file_the_manifest_names(tmp_path: Path) -> None:
    prepared = assemble(parts(), seed=42, built_at=STAMP)

    written = write_prepared(prepared, tmp_path / "processed")

    names = sorted(path.name for path in written)
    assert names == sorted(prepared.manifest.files)
    assert written[-1].name == MANIFEST_FILENAME  # last: its presence means finished


def test_written_files_round_trip(tmp_path: Path) -> None:
    prepared = assemble(parts(), seed=42, built_at=STAMP)
    out = tmp_path / "processed"
    write_prepared(prepared, out)

    reread = [
        example
        for name in prepared.manifest.files
        if name.endswith(".jsonl") and name != CORPUS_FILENAME
        for example in load_jsonl(out / name)
    ]
    assert sorted(e.question_id for e in reread) == sorted(e.question_id for e in prepared.examples)
    assert load_corpus(out / CORPUS_FILENAME) == prepared.corpus.docs
    assert json.loads((out / LEAKAGE_FILENAME).read_text(encoding="utf-8")) == []
    manifest = Manifest.model_validate_json((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest == prepared.manifest


def test_each_file_holds_one_dataset_and_one_split(tmp_path: Path) -> None:
    prepared = assemble(parts(), seed=42, built_at=STAMP)
    out = tmp_path / "processed"
    write_prepared(prepared, out)
    for name in prepared.manifest.files:
        if not name.endswith(".jsonl") or name == CORPUS_FILENAME:
            continue
        examples = load_jsonl(out / name)
        assert {examples_filename(e.dataset, e.split) for e in examples if e.split} == {name}


def test_an_interrupted_build_leaves_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is written last, so its presence means the build finished.
    `prepare-data` refuses to overwrite a finished build - and must not mistake
    a half-written one for finished."""

    def disk_full(*args: object, **kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr("verisearch.data.prepare.write_corpus", disk_full)
    out = tmp_path / "processed"
    with pytest.raises(OSError, match="disk full"):
        write_prepared(assemble(parts(), seed=42, built_at=STAMP), out)
    assert not (out / MANIFEST_FILENAME).exists()
