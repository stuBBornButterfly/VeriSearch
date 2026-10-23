"""Tests for `data/prepare.py`: rows in, the processed directory out.

Every row here is a fixture in the shape the field maps expect. No network,
per section 3 - `download_rows` is the one function that fetches, and it is
replaced by these rows in the CLI tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from verisearch.data.build_corpus import CorpusBuild, CorpusDoc, doc_id_for, load_corpus
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
    COVERAGE_FILENAME,
    LEAKAGE_FILENAME,
    MANIFEST_FILENAME,
    CoverageGap,
    DatasetPart,
    Manifest,
    assemble,
    examples_filename,
    merge_corpora,
    read_dataset,
    supporting_evidence,
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
        "id": f"w{index}",
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
    # The live rows carry no id at all.
    return {"Question": f"Bamboogle {index}?", "Answer": "y"}


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


def build(*docs: tuple[str, str]) -> CorpusBuild:
    """A corpus of (title, text) documents with real ids."""
    made = tuple(CorpusDoc(doc_id=doc_id_for(t, x), title=t, text=x) for t, x in docs)
    return CorpusBuild(
        docs=made,
        n_rows=1,
        n_paragraphs=len(made),
        n_duplicate_paragraphs=0,
        n_extra_versions=0,
        n_empty_skipped=0,
    )


def test_merging_keeps_both_datasets_versions_of_a_page() -> None:
    """The first full build kept HotpotQA's text for a shared title and dropped
    2WikiMultihopQA's, and with it the sentence some 2Wiki questions needed."""
    hotpot = build(("Ada", "Hotpot's Ada."), ("Bob", "Bob."))
    wiki = build(("Ada", "Wiki's Ada."), ("Cy", "Cy."))

    merged = merge_corpora([hotpot, wiki])

    assert sorted(doc.text for doc in merged.docs) == [
        "Bob.",
        "Cy.",
        "Hotpot's Ada.",
        "Wiki's Ada.",
    ]
    assert merged.n_extra_versions == 1
    assert merged.n_rows == 2


def test_merging_identical_copies_is_one_document() -> None:
    merged = merge_corpora([build(("Ada", "Same.")), build(("Ada", "Same."))])
    assert len(merged.docs) == 1
    assert merged.n_duplicate_paragraphs == 1
    assert merged.n_extra_versions == 0


def test_merging_does_not_depend_on_dataset_order() -> None:
    first = build(("Ada_Lovelace", "Same."), ("Bob", "Bob."))
    second = build(("Ada Lovelace", "Same."), ("Bob", "Another Bob."))
    assert merge_corpora([first, second]).docs == merge_corpora([second, first]).docs


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
    assert "coverage UNMEASURED - no supporting facts" in manifest.summary()


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
    # And which question it caps.
    assert prepared.coverage_gaps == (
        CoverageGap(question_id="hotpotqa-h99", dataset="hotpotqa", missing_titles=("nowhere",)),
    )
    assert prepared.manifest.questions_missing_evidence == 1
    assert prepared.manifest.questions_missing_evidence_sample == ("hotpotqa-h99",)


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
    assert json.loads((out / COVERAGE_FILENAME).read_text(encoding="utf-8")) == []
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


def test_a_dataset_without_ids_gets_stable_ones_from_the_question() -> None:
    """Bamboogle ships Question and Answer only. The id comes from the text, so
    a reordered file keeps every example in the same split."""
    rows = [bamboogle_row(i) for i in range(4)]
    forward = read_dataset(once(rows), BAMBOOGLE).examples
    backward = read_dataset(once(list(reversed(rows))), BAMBOOGLE).examples
    assert sorted(e.question_id for e in forward) == sorted(e.question_id for e in backward)
    assert all(e.question_id.startswith("bamboogle-q") for e in forward)


def test_the_same_question_twice_without_ids_is_refused() -> None:
    """Same text, same derived id - and two copies of one question are a
    duplicate, not two examples."""
    rows = [bamboogle_row(0), bamboogle_row(0)]
    part = read_dataset(once(rows), BAMBOOGLE)
    with pytest.raises(DataError, match="duplicate question_id"):
        assemble([read_dataset(once([hotpot_row(0)]), HOTPOT), part], seed=42)


# ---------------------------------------------------------------------------
# The sentence-level gate
# ---------------------------------------------------------------------------


def test_every_supporting_sentence_is_checked_against_the_corpus() -> None:
    manifest = assemble(parts(), seed=42, built_at=STAMP).manifest
    hotpot = next(d for d in manifest.datasets if d.name == "hotpotqa")
    wiki = next(d for d in manifest.datasets if d.name == "2wikimultihopqa")
    assert hotpot.evidence_checked == 30
    assert wiki.evidence_checked == 30
    assert manifest.evidence_missing == 0
    assert manifest.problems() == ()


def test_a_title_present_with_its_sentence_absent_is_caught() -> None:
    """The failure the title-level check could not see: the gold title is in the
    corpus, but no document with that title holds the sentence. Built here by
    giving the examples a corpus made from other rows."""
    examples_from = read_dataset(once([hotpot_row(i) for i in range(3)]), HOTPOT)
    other_rows = [
        hotpot_row(i)
        | {"context": {"title": ["Ada", "Distractor 0"], "sentences": [["Not it."], ["No."]]}}
        for i in range(1)
    ]
    corpus_from = read_dataset(once(other_rows), HOTPOT)
    mixed = examples_from.model_copy(update={"corpus": corpus_from.corpus})

    manifest = assemble([mixed], seed=42, built_at=STAMP).manifest

    assert manifest.missing_titles == 0  # every title is there
    assert manifest.evidence_missing == 3  # and none of the sentences
    assert any("supporting sentences are in no corpus document" in p for p in manifest.problems())
    assert manifest.evidence_missing_sample[0].startswith("ada: About Ada.")


def test_a_fact_that_points_outside_its_own_row_is_unreadable_not_missing() -> None:
    """HotpotQA has a few supporting facts whose sentence id is past the end of
    the paragraph. The data is wrong, not the corpus, so it is counted apart."""
    row = hotpot_row(0) | {
        "supporting_facts": {"title": ["Ada", "Ada", "Nowhere"], "sent_id": [0, 9, 0]}
    }
    part = read_dataset(once([row]), HOTPOT)
    assert len(part.evidence) == 1
    assert part.evidence_unreadable == 2


def test_spacing_differences_are_not_missing_evidence() -> None:
    row = hotpot_row(0) | {
        "context": {"title": ["Ada", "Other"], "sentences": [["  About   Ada.  "], ["No."]]},
    }
    manifest = assemble([read_dataset(once([row]), HOTPOT)], seed=42, built_at=STAMP).manifest
    assert manifest.evidence_missing == 0


def test_list_shaped_facts_and_context_are_read_too() -> None:
    """2WikiMultihopQA's published shape: lists of pairs, not parallel lists."""
    part = read_dataset(once([wiki_row(0)]), WIKI)
    assert part.evidence == (("wiki gold 0", "Wiki says so."),)
    assert part.evidence_unreadable == 0


def test_a_flattened_title_list_has_no_sentences_to_check() -> None:
    """Some redistributions flatten supporting_facts to bare titles. Nothing to
    check a sentence against, so each counts as unreadable."""
    row = hotpot_row(0) | {"supporting_facts": ["Ada", "Distractor 0"]}
    part = read_dataset(once([row]), HOTPOT)
    assert part.evidence == ()
    assert part.evidence_unreadable == 2


def test_facts_the_field_map_cannot_read_still_fail_the_row() -> None:
    """`row_to_example` reads the same field first and refuses it, so the
    evidence reader never sees a shape the field map has not accepted."""
    for value in ({"title": "Ada", "sent_id": 0}, 7):
        row = hotpot_row(0) | {"supporting_facts": value}
        with pytest.raises(DataError):
            read_dataset(once([row]), HOTPOT)


def test_a_dataset_without_facts_or_context_contributes_no_evidence() -> None:
    part = read_dataset(once([musique_row(0)]), MUSIQUE)
    assert part.evidence == ()
    assert part.evidence_unreadable == 0


GOOD_FACTS = {"title": ["Ada"], "sent_id": [0]}
GOOD_CONTEXT = {"title": ["Ada"], "sentences": [["About Ada."]]}


@pytest.mark.parametrize(
    ("facts", "context", "unreadable"),
    [
        pytest.param(
            GOOD_FACTS, {"title": "Ada", "sentences": []}, 1, id="context-titles-not-a-list"
        ),
        pytest.param(GOOD_FACTS, 7, 1, id="context-neither-shape"),
        pytest.param(
            GOOD_FACTS, [[1, ["x"]], ["Ada", "not a list"]], 1, id="context-pairs-unusable"
        ),
        pytest.param({"title": "Ada", "sent_id": 0}, GOOD_CONTEXT, 0, id="facts-not-lists"),
        pytest.param(7, GOOD_CONTEXT, 0, id="facts-neither-shape"),
        pytest.param(
            {"title": ["Ada"], "sent_id": [True]},
            {"title": ["Ada"], "sentences": [["First.", "Second."]]},
            1,
            id="bool-is-not-an-id",  # True would otherwise index sentence 1
        ),
        pytest.param(
            {"title": ["Ada"], "sent_id": [0]},
            {"title": ["Ada"], "sentences": [[" "]]},
            1,
            id="blank-sentence",
        ),
        pytest.param(
            {"title": ["Ada"], "sent_id": [0]},
            {"title": ["Ada"], "sentences": [[5]]},
            1,
            id="sentence-not-text",
        ),
    ],
)
def test_the_evidence_reader_never_crashes_on_a_shape_it_cannot_use(
    facts: object, context: object, unreadable: int
) -> None:
    """Reached before `build_corpus` validates the context, so it must degrade to
    "nothing usable" rather than raise. `row_to_example` and `build_corpus`
    still refuse the row; this only has to not be the thing that crashes."""
    row = {
        "id": "h",
        "question": "Q?",
        "answer": "A",
        "supporting_facts": facts,
        "context": context,
    }
    found, bad = supporting_evidence(row, HOTPOT)
    assert found == ()
    assert bad == unreadable


def test_the_evidence_reader_needs_both_fields_present() -> None:
    assert supporting_evidence({"supporting_facts": GOOD_FACTS}, HOTPOT) == ((), 0)
    assert supporting_evidence(
        {"supporting_facts": GOOD_FACTS, "context": GOOD_CONTEXT}, HOTPOT
    ) == (
        (("ada", "About Ada."),),
        0,
    )


# ---------------------------------------------------------------------------
# Which questions lost evidence - Dhrubo's question on #24
# ---------------------------------------------------------------------------


def test_a_missing_sentence_names_every_question_that_needed_it() -> None:
    """Same setup as the sentence-level check above: titles present, the gold
    sentence in none of their documents. All three questions need it."""
    examples_from = read_dataset(once([hotpot_row(i) for i in range(3)]), HOTPOT)
    other_rows = [
        hotpot_row(0)
        | {"context": {"title": ["Ada", "Distractor 0"], "sentences": [["Not it."], ["No."]]}}
    ]
    mixed = examples_from.model_copy(
        update={"corpus": read_dataset(once(other_rows), HOTPOT).corpus}
    )

    prepared = assemble([mixed], seed=42, built_at=STAMP)

    assert [gap.question_id for gap in prepared.coverage_gaps] == [
        "hotpotqa-h0",
        "hotpotqa-h1",
        "hotpotqa-h2",
    ]
    for gap in prepared.coverage_gaps:
        assert gap.missing_titles == ()
        assert gap.missing_sentences == ("ada: About Ada.",)
    assert prepared.manifest.questions_missing_evidence == 3


def test_a_clean_build_has_no_gaps() -> None:
    prepared = assemble(parts(), seed=42, built_at=STAMP)
    assert prepared.coverage_gaps == ()
    assert prepared.manifest.questions_missing_evidence == 0
    assert "questions   0 missing some gold evidence" in prepared.manifest.summary()


def test_the_gaps_file_lists_every_question_with_what_it_is_missing(tmp_path: Path) -> None:
    rows = [hotpot_row(i) for i in range(3)]
    rows.append(hotpot_row(7, gold="Nowhere", context_titles=("Elsewhere",)))
    prepared = assemble([read_dataset(once(rows), HOTPOT)], seed=42, built_at=STAMP)
    out = tmp_path / "processed"
    write_prepared(prepared, out)

    gaps = json.loads((out / COVERAGE_FILENAME).read_text(encoding="utf-8"))

    assert gaps == [
        {
            "question_id": "hotpotqa-h7",
            "dataset": "hotpotqa",
            "missing_titles": ["nowhere"],
            "missing_sentences": [],
        }
    ]
    assert COVERAGE_FILENAME in prepared.manifest.files


def test_every_evidence_entry_knows_its_question() -> None:
    part = read_dataset(once([hotpot_row(i) for i in range(4)]), HOTPOT)
    assert part.evidence_question_ids == tuple(f"hotpotqa-h{i}" for i in range(4))
    assert len(part.evidence_question_ids) == len(part.evidence)


def test_question_ids_must_line_up_with_the_evidence() -> None:
    part = read_dataset(once([hotpot_row(i) for i in range(2)]), HOTPOT)
    with pytest.raises(ValueError, match="paired by position"):
        DatasetPart.model_validate(
            part.model_dump() | {"evidence_question_ids": ("h0",)},
        )


def test_a_manifest_written_before_the_gap_fields_still_loads() -> None:
    """`data/processed/manifest.json` from the full build predates them."""
    manifest = assemble(parts(), seed=42, built_at=STAMP).manifest
    older = manifest.model_dump(mode="json")
    del older["questions_missing_evidence"]
    del older["questions_missing_evidence_sample"]

    reread = Manifest.model_validate(older)

    assert reread.questions_missing_evidence == 0
    assert reread.questions_missing_evidence_sample == ()
