"""Tests for `verisearch/data/`.

The split tests carry the weight. An example drifting between train and test
between two runs inflates the test number with nothing in any log to show for
it, so the properties asserted here - determinism, and stability under dataset
growth - are the ones that keep the headline results honest.

No network anywhere (PROJECT.md section 3). The Hugging Face field maps are
exercised against synthetic rows shaped like each published schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from verisearch.data.datasets import (
    DATASET_SPECS,
    DataError,
    Distribution,
    QAExample,
    Split,
    check_unique_ids,
    load_hf_dataset,
    load_jsonl,
    normalize_question,
    row_to_example,
    write_jsonl,
)
from verisearch.data.splits import (
    DEFAULT_FRACTIONS,
    SCHEME_VERSION,
    SplitFractions,
    assign_split,
    bucket,
    find_leakage,
    make_splits,
    subsample,
    summarize,
)

if TYPE_CHECKING:
    from pathlib import Path


def _example(
    question_id: str,
    question: str = "Who directed the film?",
    *,
    answers: tuple[str, ...] = ("Peter Weir",),
    distribution: Distribution = Distribution.IN_DISTRIBUTION,
    split: Split | None = None,
) -> QAExample:
    return QAExample(
        question_id=question_id,
        question=question,
        answers=answers,
        dataset="synthetic",
        distribution=distribution,
        split=split,
    )


# ---------------------------------------------------------------------------
# Question normalisation - the basis of leakage detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("Who directed Gallipoli?", "who directed gallipoli", id="punctuation"),
        pytest.param("WHO Directed   GALLIPOLI", "who directed gallipoli", id="case-and-spacing"),
        pytest.param("Who directed the film?", "who directed film", id="articles-dropped"),
        pytest.param("Who directed a film?", "who directed film", id="indefinite-article"),
        pytest.param("Café Terrace at Night?", "cafe terrace at night", id="accents-folded"),
        pytest.param("  trailing and leading  ", "trailing and leading", id="trimmed"),
    ],
)
def test_normalize_question(raw: str, expected: str) -> None:
    assert normalize_question(raw) == expected


def test_normalization_collapses_near_duplicates() -> None:
    """Two rows asking the same thing must normalise to one key."""
    a = "Which director of the 1999 film was born in Australia?"
    b = "which director of the 1999 film was born in australia"
    assert normalize_question(a) == normalize_question(b)


# ---------------------------------------------------------------------------
# QAExample
# ---------------------------------------------------------------------------


def test_example_is_frozen() -> None:
    from pydantic import ValidationError

    example = _example("q1")
    with pytest.raises(ValidationError):
        example.question = "changed"  # type: ignore[misc]


def test_example_requires_at_least_one_answer() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QAExample(
            question_id="q1",
            question="Who?",
            answers=(),
            dataset="synthetic",
            distribution=Distribution.IN_DISTRIBUTION,
        )


def test_with_split_returns_a_copy() -> None:
    original = _example("q1")
    assigned = original.with_split(Split.TRAIN)
    assert original.split is None
    assert assigned.split is Split.TRAIN


# ---------------------------------------------------------------------------
# Hugging Face field maps, against synthetic rows
# ---------------------------------------------------------------------------


def test_hotpotqa_row_with_parallel_list_supporting_facts() -> None:
    """HotpotQA ships supporting_facts as a mapping of parallel lists."""
    row = {
        "id": "5a7a06935542990198eaf050",
        "question": "Which magazine was started first, Arthur's or First for Women?",
        "answer": "Arthur's Magazine",
        "supporting_facts": {
            "title": ["Arthur's Magazine", "Arthur's Magazine", "First for Women"],
            "sent_id": [0, 1, 0],
        },
    }
    example = row_to_example(row, DATASET_SPECS["hotpotqa"])

    assert example.question_id == "hotpotqa-5a7a06935542990198eaf050"
    assert example.answers == ("Arthur's Magazine",)
    # Deduplicated, order preserved: we want paragraphs, not sentences.
    assert example.supporting_titles == ("Arthur's Magazine", "First for Women")
    assert example.distribution is Distribution.IN_DISTRIBUTION
    assert example.split is None


def test_2wiki_row_with_pairwise_supporting_facts() -> None:
    """The other published shape: a list of [title, sent_id] pairs."""
    row = {
        "_id": "abc123",
        "question": "Where was the director of Gallipoli born?",
        "answer": "Sydney",
        "supporting_facts": [["Gallipoli (1981 film)", 0], ["Peter Weir", 2]],
    }
    example = row_to_example(row, DATASET_SPECS["2wikimultihopqa"])

    assert example.question_id == "2wikimultihopqa-abc123"
    assert example.supporting_titles == ("Gallipoli (1981 film)", "Peter Weir")


def test_musique_row_merges_answer_aliases() -> None:
    row = {
        "id": "2hop__1234",
        "question": "Who founded the company that made the Model T?",
        "answer": "Henry Ford",
        "answer_aliases": ["Ford", "Henry Ford"],
    }
    example = row_to_example(row, DATASET_SPECS["musique"])

    # Primary answer first, aliases appended, no duplicates.
    assert example.answers == ("Henry Ford", "Ford")
    assert example.distribution is Distribution.OUT_OF_DISTRIBUTION
    # OOD is never split, so it is labelled at load time.
    assert example.split is Split.OOD


def test_bamboogle_row_uses_capitalised_keys() -> None:
    row = {"id": "bb-7", "Question": "Who is older?", "Answer": "Alice"}
    example = row_to_example(row, DATASET_SPECS["bamboogle"])
    assert example.question == "Who is older?"
    assert example.answers == ("Alice",)


def test_missing_key_names_the_keys_that_were_present() -> None:
    """The error has to be actionable - the field maps are unverified."""
    row = {"identifier": "x", "question": "Who?", "answer": "A"}
    with pytest.raises(DataError, match="Keys present: answer, identifier, question"):
        row_to_example(row, DATASET_SPECS["hotpotqa"])


def test_wrong_type_is_rejected() -> None:
    row = {"id": 42, "question": "Who?", "answer": "A"}
    with pytest.raises(DataError, match="expected a string"):
        row_to_example(row, DATASET_SPECS["hotpotqa"])


def test_empty_answer_is_rejected() -> None:
    row = {"id": "x", "question": "Who?", "answer": ""}
    with pytest.raises(DataError, match="empty answer"):
        row_to_example(row, DATASET_SPECS["hotpotqa"])


def test_malformed_supporting_facts_is_rejected() -> None:
    row = {"id": "x", "question": "Who?", "answer": "A", "supporting_facts": 7}
    with pytest.raises(DataError, match="expected a mapping or sequence"):
        row_to_example(row, DATASET_SPECS["hotpotqa"])


def test_unknown_dataset_lists_the_known_ones() -> None:
    with pytest.raises(DataError, match="Known: 2wikimultihopqa, bamboogle, hotpotqa, musique"):
        load_hf_dataset("squad")


def test_every_spec_declares_a_distribution() -> None:
    """PROJECT.md section 7 splits the four datasets two and two."""
    in_dist = {
        n for n, s in DATASET_SPECS.items() if s.distribution is Distribution.IN_DISTRIBUTION
    }
    ood = {
        n for n, s in DATASET_SPECS.items() if s.distribution is Distribution.OUT_OF_DISTRIBUTION
    }
    assert in_dist == {"hotpotqa", "2wikimultihopqa"}
    assert ood == {"musique", "bamboogle"}


# ---------------------------------------------------------------------------
# JSONL round trip
# ---------------------------------------------------------------------------


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = (_example("q2"), _example("q1", split=Split.TRAIN))
    path = tmp_path / "data" / "examples.jsonl"

    written = write_jsonl(examples, path)
    loaded = load_jsonl(path)

    assert written == 2
    assert {e.question_id for e in loaded} == {"q1", "q2"}
    assert {e.question_id: e.split for e in loaded}["q1"] is Split.TRAIN


def test_write_jsonl_is_byte_stable_regardless_of_input_order(tmp_path: Path) -> None:
    """A snapshot that reorders itself produces a meaningless diff."""
    a = _example("q1")
    b = _example("q2")
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"

    write_jsonl((a, b), first)
    write_jsonl((b, a), second)

    assert first.read_bytes() == second.read_bytes()


def test_write_jsonl_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "out.jsonl"
    write_jsonl((_example("q1"),), path)
    assert path.is_file()


def test_jsonl_omits_unset_split(tmp_path: Path) -> None:
    """exclude_none keeps unassigned examples out of the split field entirely."""
    path = tmp_path / "x.jsonl"
    write_jsonl((_example("q1"),), path)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert "split" not in payload


def test_load_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_load_jsonl_reports_the_line_number_on_bad_json(tmp_path: Path) -> None:
    """A dataset file is thousands of lines long; an error without a position is useless."""
    path = tmp_path / "bad.jsonl"
    good = _example("q1").model_dump_json()
    path.write_text(f"{good}\n{{not json\n", encoding="utf-8")
    with pytest.raises(DataError, match=r"bad\.jsonl:2"):
        load_jsonl(path)


def test_load_jsonl_reports_the_line_number_on_a_bad_example(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question_id": "q1"}\n', encoding="utf-8")
    with pytest.raises(DataError, match=r"bad\.jsonl:1"):
        load_jsonl(path)


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "gappy.jsonl"
    body = _example("q1").model_dump_json()
    path.write_text(f"\n{body}\n\n", encoding="utf-8")
    assert len(load_jsonl(path)) == 1


def test_duplicate_ids_are_rejected() -> None:
    """Split assignment is a function of question_id, so a collision is silent."""
    with pytest.raises(DataError, match="duplicate question_id"):
        check_unique_ids([_example("q1"), _example("q1")])


# ---------------------------------------------------------------------------
# The split scheme
# ---------------------------------------------------------------------------


def test_bucket_is_in_range_and_deterministic() -> None:
    values = [bucket(f"q{i}", seed=42) for i in range(200)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert values == [bucket(f"q{i}", seed=42) for i in range(200)]


def test_bucket_depends_on_the_seed() -> None:
    assert bucket("q1", seed=42) != bucket("q1", seed=43)


def test_assignment_is_stable_when_the_dataset_grows() -> None:
    """The property this whole module exists for.

    Shuffle-and-slice reassigns every example the moment one row is appended,
    which silently moves examples between train and test between runs.
    """
    original = [_example(f"q{i}") for i in range(100)]
    grown = original + [_example(f"extra{i}") for i in range(400)]

    before = {e.question_id: e.split for e in make_splits(original, seed=42)}
    after = {e.question_id: e.split for e in make_splits(grown, seed=42)}

    for question_id, split in before.items():
        assert after[question_id] is split


def test_assignment_changes_with_the_seed() -> None:
    examples = [_example(f"q{i}") for i in range(200)]
    a = {e.question_id: e.split for e in make_splits(examples, seed=42)}
    b = {e.question_id: e.split for e in make_splits(examples, seed=43)}
    assert a != b


def test_fractions_are_respected_in_aggregate() -> None:
    examples = [_example(f"q{i}") for i in range(5000)]
    assigned = make_splits(examples, seed=42)
    report = summarize(assigned, seed=42)

    assert report.count(Split.TRAIN) / 5000 == pytest.approx(0.8, abs=0.025)
    assert report.count(Split.DEV) / 5000 == pytest.approx(0.05, abs=0.015)
    assert report.count(Split.TEST) / 5000 == pytest.approx(0.15, abs=0.025)
    assert report.total == 5000


def test_splits_are_disjoint_and_cover_everything() -> None:
    examples = [_example(f"q{i}") for i in range(500)]
    assigned = make_splits(examples, seed=42)
    report = summarize(assigned, seed=42)
    assert sum(report.counts.values()) == 500


def test_out_of_distribution_examples_are_never_split() -> None:
    """Musique and Bamboogle are held-out test sets in their entirety."""
    examples = [
        _example(f"ood{i}", distribution=Distribution.OUT_OF_DISTRIBUTION) for i in range(50)
    ]
    assigned = make_splits(examples, seed=42)
    assert {e.split for e in assigned} == {Split.OOD}


def test_mixed_pool_splits_only_the_in_distribution_half() -> None:
    examples = [_example(f"in{i}") for i in range(100)]
    examples += [
        _example(f"ood{i}", distribution=Distribution.OUT_OF_DISTRIBUTION) for i in range(20)
    ]
    assigned = make_splits(examples, seed=42)

    ood = [e for e in assigned if e.distribution is Distribution.OUT_OF_DISTRIBUTION]
    in_dist = [e for e in assigned if e.distribution is Distribution.IN_DISTRIBUTION]
    assert all(e.split is Split.OOD for e in ood)
    assert all(e.split is not Split.OOD for e in in_dist)


def test_make_splits_preserves_input_order() -> None:
    """So the function cannot quietly become a shuffle."""
    examples = [_example(f"q{i}") for i in range(20)]
    assigned = make_splits(examples, seed=42)
    assert [e.question_id for e in assigned] == [e.question_id for e in examples]


def test_make_splits_rejects_duplicate_ids() -> None:
    with pytest.raises(DataError, match="duplicate question_id"):
        make_splits([_example("q1"), _example("q1")], seed=42)


def test_fractions_must_sum_to_one() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=r"must sum to 1\.0"):
        SplitFractions(train=0.8, dev=0.1, test=0.2)


def test_default_fractions_sum_to_one() -> None:
    total = DEFAULT_FRACTIONS.train + DEFAULT_FRACTIONS.dev + DEFAULT_FRACTIONS.test
    assert total == pytest.approx(1.0)


def test_custom_fractions_shift_the_boundaries() -> None:
    fractions = SplitFractions(train=0.5, dev=0.25, test=0.25)
    examples = [_example(f"q{i}") for i in range(2000)]
    assigned = [assign_split(e.question_id, seed=42, fractions=fractions) for e in examples]
    assert assigned.count(Split.TRAIN) / 2000 == pytest.approx(0.5, abs=0.04)


def test_report_records_the_scheme_version() -> None:
    """A split built under a different hash scheme is not comparable, even at
    the same seed."""
    report = summarize(make_splits([_example("q1")], seed=42), seed=42)
    assert report.scheme_version == SCHEME_VERSION
    assert report.seed == 42


def test_summarize_refuses_unassigned_examples() -> None:
    with pytest.raises(DataError, match="has no split"):
        summarize([_example("q1")], seed=42)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_leakage_is_reported_across_splits() -> None:
    shared = "Which director of the 1999 film was born in Australia?"
    examples = [
        _example("a", question=shared, split=Split.TRAIN),
        _example("b", question=shared.lower().replace("?", ""), split=Split.TEST),
    ]
    groups = find_leakage(examples)

    assert len(groups) == 1
    assert groups[0].question_ids == ("a", "b")
    assert groups[0].splits == ("test", "train")


def test_duplicates_inside_one_split_are_not_leakage() -> None:
    """Two copies in train is a dedup question, not a leak."""
    examples = [
        _example("a", question="Same question?", split=Split.TRAIN),
        _example("b", question="same question", split=Split.TRAIN),
    ]
    assert find_leakage(examples) == ()


def test_no_leakage_on_distinct_questions() -> None:
    examples = [
        _example("a", question="Who directed Gallipoli?", split=Split.TRAIN),
        _example("b", question="Who scored Gallipoli?", split=Split.TEST),
    ]
    assert find_leakage(examples) == ()


def test_unassigned_examples_cannot_create_a_false_leak() -> None:
    examples = [
        _example("a", question="Same?", split=Split.TRAIN),
        _example("b", question="Same?", split=None),
    ]
    assert find_leakage(examples) == ()


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------


def test_subsample_is_deterministic() -> None:
    examples = make_splits([_example(f"q{i}") for i in range(500)], seed=42)
    first = subsample(examples, Split.TRAIN, limit=50, seed=42)
    second = subsample(examples, Split.TRAIN, limit=50, seed=42)
    assert [e.question_id for e in first] == [e.question_id for e in second]
    assert len(first) == 50
    assert all(e.split is Split.TRAIN for e in first)


def test_raising_the_limit_only_adds_examples() -> None:
    """PROJECT.md section 7 starts at 2000-5000 train examples. A 2000-example
    run must be a strict subset of a 5000-example one, or the two are not
    comparable."""
    examples = make_splits([_example(f"q{i}") for i in range(2000)], seed=42)
    small = subsample(examples, Split.TRAIN, limit=200, seed=42)
    large = subsample(examples, Split.TRAIN, limit=800, seed=42)

    assert {e.question_id for e in small} <= {e.question_id for e in large}
    assert len(large) == 800


def test_subsample_is_stable_when_the_pool_grows() -> None:
    original = [_example(f"q{i}") for i in range(500)]
    grown = original + [_example(f"extra{i}") for i in range(1500)]

    small_pool = subsample(make_splits(original, seed=42), Split.TRAIN, limit=50, seed=42)
    large_pool = subsample(make_splits(grown, seed=42), Split.TRAIN, limit=50, seed=42)

    # Not identical - new examples can out-rank old ones - but the overlap must
    # be substantial, and every selected example must still come from TRAIN.
    overlap = {e.question_id for e in small_pool} & {e.question_id for e in large_pool}
    assert len(overlap) > 0
    assert all(e.split is Split.TRAIN for e in large_pool)


def test_subsample_returns_everything_when_the_limit_exceeds_the_split() -> None:
    examples = make_splits([_example(f"q{i}") for i in range(20)], seed=42)
    train = [e for e in examples if e.split is Split.TRAIN]
    assert len(subsample(examples, Split.TRAIN, limit=10_000, seed=42)) == len(train)


def test_subsample_rejects_a_negative_limit() -> None:
    with pytest.raises(DataError, match="non-negative"):
        subsample([], Split.TRAIN, limit=-1, seed=42)
