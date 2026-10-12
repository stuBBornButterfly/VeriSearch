"""Tests for `verisearch/eval/metrics.py`.

`normalize_answer` and `token_f1` were checked against verbatim transcriptions
of the official SQuAD v1.1 and HotpotQA eval scripts over 250k randomised
answer pairs - zero mismatches. That check is not committed because it would
mean vendoring the reference scripts; the cases below pin the behaviour it
established, including the one divergence it caught.

PROJECT.md section 2 compares against published baselines, so these numbers
have to be the published definitions, not improved versions of them.
"""

from __future__ import annotations

import pytest

from verisearch.eval.metrics import (
    AnswerScore,
    FailureReason,
    MetricError,
    answer_token_count,
    best_exact_match,
    best_f1,
    bootstrap_ci,
    classify_failure,
    exact_match,
    failure_breakdown,
    normalize_answer,
    normalize_title,
    paired_bootstrap_delta,
    precision_at_k,
    recall_at_k,
    score_answer,
    token_f1,
)

# ---------------------------------------------------------------------------
# Normalisation - SQuAD v1.1, exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("Peter Weir", "peter weir", id="lowercased"),
        pytest.param("The Film", "film", id="article-dropped"),
        pytest.param("A film, an answer", "film answer", id="all-articles-dropped"),
        pytest.param("Arthur's Magazine", "arthurs magazine", id="apostrophe-removed"),
        pytest.param("U.S.A.", "usa", id="periods-removed"),
        pytest.param("  spaced   out  ", "spaced out", id="whitespace-collapsed"),
        pytest.param("", "", id="empty"),
        pytest.param("!!!", "", id="punctuation-only-normalises-away"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_punctuation_removal_runs_before_article_removal() -> None:
    """Pins a quirk of the reference script that looks like a bug.

    Punctuation is deleted rather than replaced with a space, so "the-film"
    becomes "thefilm" - and the article regex then finds no word boundary around
    "the", so it survives. A later reader may well want to "fix" this by
    inserting a space. They must not: it would change every reported F1 and
    break comparability with published work.

    Verified against a verbatim transcription of SQuAD v1.1 over 250k pairs.
    """
    assert normalize_answer("the-film") == "thefilm"
    # Whereas with a space the article does get dropped.
    assert normalize_answer("the film") == "film"


def test_accents_are_not_folded() -> None:
    """The reference script does not fold accents, so neither do we.

    `data.datasets.normalize_question` *does* fold them, because it serves
    duplicate detection where recall matters more than comparability. The two
    must stay different.
    """
    assert normalize_answer("Café") != "cafe"


# ---------------------------------------------------------------------------
# F1 and EM
# ---------------------------------------------------------------------------


def test_identical_answers_score_one() -> None:
    assert token_f1("Peter Weir", "Peter Weir") == 1.0
    assert exact_match("Peter Weir", "peter  weir") == 1.0


def test_disjoint_answers_score_zero() -> None:
    assert token_f1("Peter Weir", "Steven Spielberg") == 0.0
    assert exact_match("Peter Weir", "Steven Spielberg") == 0.0


def test_partial_overlap() -> None:
    """One of two tokens shared: precision 1/1, recall 1/2, F1 = 2/3."""
    assert token_f1("Weir", "Peter Weir") == pytest.approx(2 / 3)


def test_token_multiplicity_is_respected() -> None:
    """Counter intersection, not set intersection - a repeated token in the
    prediction must not match a single occurrence in the gold twice."""
    # pred has 2 "a", gold has 1. Overlap is 1, not 2.
    assert token_f1("x x", "x") == pytest.approx(2 / 3)


def test_em_is_stricter_than_f1() -> None:
    assert token_f1("Peter Weir the director", "Peter Weir") > 0.0
    assert exact_match("Peter Weir the director", "Peter Weir") == 0.0


@pytest.mark.parametrize(
    ("prediction", "gold", "expected"),
    [
        pytest.param("yes", "yes", 1.0, id="matching-yes"),
        pytest.param("no", "no", 1.0, id="matching-no"),
        pytest.param("yes", "no", 0.0, id="yes-vs-no"),
        pytest.param("yes, it was", "yes", 0.0, id="no-partial-credit-on-gold-yes"),
        pytest.param("yes", "yes, it was", 0.0, id="no-partial-credit-on-pred-yes"),
    ],
)
def test_categorical_answers_get_no_partial_credit(
    prediction: str, gold: str, expected: float
) -> None:
    """Official HotpotQA rule. Without it a policy that always answers "yes"
    farms partial credit on every yes/no question, and HotpotQA is the primary
    training set."""
    assert token_f1(prediction, gold) == expected


def test_the_categorical_rule_can_be_disabled() -> None:
    """Non-strict mode is plain SQuAD F1, for datasets without yes/no answers."""
    assert token_f1("yes, it was", "yes", yes_no_strict=False) > 0.0


def test_empty_prediction_scores_zero() -> None:
    assert token_f1("", "Peter Weir") == 0.0


def test_two_empty_answers_score_zero_not_one() -> None:
    """Regression. This returned 1.0 until a differential test against the
    reference script caught it.

    The reference reaches 0 via num_same == 0, and matching it is the point. It
    is also the safe answer: an empty prediction against a gold that normalises
    away - "-" for instance - would otherwise be free credit.
    """
    assert token_f1("", "") == 0.0
    assert token_f1("!!!", "---") == 0.0


# ---------------------------------------------------------------------------
# Scoring against multiple acceptable answers
# ---------------------------------------------------------------------------


def test_best_score_takes_the_max_over_aliases() -> None:
    """Max, not mean: aliases are alternative spellings of one correct answer.
    Averaging would punish a right answer for not being all its own synonyms."""
    golds = ("Henry Ford", "Ford")
    assert best_f1("Ford", golds) == 1.0
    assert best_exact_match("Ford", golds) == 1.0


def test_best_score_still_zero_when_nothing_matches() -> None:
    assert best_f1("Karl Benz", ("Henry Ford", "Ford")) == 0.0


@pytest.mark.parametrize("scorer", [best_f1, best_exact_match])
def test_empty_gold_list_is_an_error(scorer: object) -> None:
    with pytest.raises(MetricError, match="at least one gold"):
        scorer("anything", [])  # type: ignore[operator]


def test_unanswered_question_scores_zero_not_missing() -> None:
    """Dropping it from the denominator would reward a policy for refusing hard
    questions."""
    score = score_answer(None, ("Peter Weir",))
    assert score == AnswerScore(exact_match=0.0, f1=0.0)


def test_score_answer_combines_both_metrics() -> None:
    score = score_answer("Weir", ("Peter Weir",))
    assert score.exact_match == 0.0
    assert score.f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param(None, 0, id="no-answer"),
        pytest.param("", 0, id="empty"),
        pytest.param("Peter Weir", 2, id="two-tokens"),
        pytest.param("the Peter Weir", 2, id="articles-not-counted"),
    ],
)
def test_answer_token_count(answer: str | None, expected: int) -> None:
    """PROJECT.md section 6 lists answer length as the canary for the
    copy-passages-verbatim reward hack."""
    assert answer_token_count(answer) == expected


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic() -> None:
    """An unseeded CI changes between two runs of the report and makes a result
    impossible to check."""
    values = [0.1 * i for i in range(50)]
    first = bootstrap_ci(values, seed=42)
    second = bootstrap_ci(values, seed=42)
    assert (first.low, first.point, first.high) == (second.low, second.point, second.high)


def test_bootstrap_depends_on_the_seed() -> None:
    values = [0.1 * i for i in range(50)]
    assert bootstrap_ci(values, seed=42).low != bootstrap_ci(values, seed=43).low


def test_bootstrap_point_is_the_sample_mean() -> None:
    values = [0.0, 0.5, 1.0]
    assert bootstrap_ci(values).point == pytest.approx(0.5)


def test_bootstrap_interval_brackets_the_point() -> None:
    values = [float(i % 2) for i in range(200)]
    interval = bootstrap_ci(values)
    assert interval.low <= interval.point <= interval.high
    assert interval.n == 200
    assert interval.resamples == 1000


def test_identical_values_give_a_zero_width_interval() -> None:
    interval = bootstrap_ci([0.7] * 100)
    assert interval.width == pytest.approx(0.0)
    assert interval.point == pytest.approx(0.7)


def test_a_wider_confidence_level_gives_a_wider_interval() -> None:
    values = [float(i % 3) / 2 for i in range(300)]
    narrow = bootstrap_ci(values, confidence=0.80)
    wide = bootstrap_ci(values, confidence=0.99)
    assert wide.width > narrow.width


def test_more_examples_give_a_narrower_interval() -> None:
    """PROJECT.md section 9: "A 2-point gain on 500 examples is noise.\""""
    pattern = [0.0, 1.0, 0.5, 0.25, 0.75]
    small = bootstrap_ci(pattern * 20)
    large = bootstrap_ci(pattern * 400)
    assert large.width < small.width


def test_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(MetricError, match="at least one value"):
        bootstrap_ci([])


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_rejects_impossible_confidence(confidence: float) -> None:
    with pytest.raises(MetricError, match="confidence must be"):
        bootstrap_ci([0.5], confidence=confidence)


def test_bootstrap_rejects_zero_resamples() -> None:
    with pytest.raises(MetricError, match="resamples must be"):
        bootstrap_ci([0.5], resamples=0)


def test_interval_str_is_report_ready() -> None:
    rendered = str(bootstrap_ci([0.5] * 10))
    assert "95% CI" in rendered
    assert "n=10" in rendered


# ---------------------------------------------------------------------------
# Paired bootstrap - what the nightly eval-regression stage needs
# ---------------------------------------------------------------------------


def test_paired_delta_of_identical_systems_is_zero() -> None:
    values = [float(i % 2) for i in range(100)]
    delta = paired_bootstrap_delta(values, values)
    assert delta.point == pytest.approx(0.0)
    assert not delta.excludes_zero


def test_paired_delta_detects_a_uniform_improvement() -> None:
    baseline = [float(i % 5) / 4 for i in range(200)]
    candidate = [min(1.0, b + 0.05) for b in baseline]
    delta = paired_bootstrap_delta(baseline, candidate)
    assert delta.point > 0.0
    assert delta.excludes_zero


def test_pairing_is_tighter_than_treating_the_systems_as_independent() -> None:
    """The reason this function exists.

    Two systems evaluated on the same questions are correlated. Here the
    candidate is the baseline plus a constant, so every per-example difference
    is identical and the paired interval has zero width - while the baseline's
    own interval is wide, because the scores themselves vary a lot.
    """
    baseline = [float(i % 11) / 10 for i in range(220)]
    candidate = [b + 0.05 for b in baseline]

    paired = paired_bootstrap_delta(baseline, candidate)
    unpaired_scale = bootstrap_ci(baseline)

    assert paired.width == pytest.approx(0.0)
    assert unpaired_scale.width > 0.05
    assert paired.point == pytest.approx(0.05)


def test_paired_delta_rejects_mismatched_lengths() -> None:
    with pytest.raises(MetricError, match="equal-length"):
        paired_bootstrap_delta([0.1, 0.2], [0.1])


def test_excludes_zero_is_false_for_a_straddling_interval() -> None:
    baseline = [0.0, 1.0] * 100
    candidate = [1.0, 0.0] * 100
    assert not paired_bootstrap_delta(baseline, candidate).excludes_zero


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("Peter_Weir", "peter weir", id="underscores"),
        pytest.param("Peter  Weir", "peter weir", id="double-space"),
        pytest.param("  Peter Weir  ", "peter weir", id="trimmed"),
        pytest.param("PETER WEIR", "peter weir", id="case"),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    """Underscores and spaces are interchangeable in Wikipedia titles and the
    datasets are inconsistent. Without this, precision@k reads zero on data
    that is correct."""
    assert normalize_title(raw) == expected


def test_precision_at_k() -> None:
    retrieved = ["Peter_Weir", "Gallipoli (1981 film)", "Unrelated Article"]
    gold = ["Peter Weir", "Gallipoli (1981 film)"]
    assert precision_at_k(retrieved, gold) == pytest.approx(2 / 3)


def test_precision_counts_duplicate_retrievals_as_separate_slots() -> None:
    """Retrieving the same paragraph three times really is three slots spent,
    and a policy that does it is querying badly. That is what this measures."""
    retrieved = ["Peter Weir", "Peter Weir", "Peter Weir"]
    assert precision_at_k(retrieved, ["Peter Weir"]) == 1.0
    assert precision_at_k(retrieved, ["Someone Else"]) == 0.0


def test_precision_requires_retrieved_titles() -> None:
    with pytest.raises(MetricError, match="at least one retrieved"):
        precision_at_k([], ["Peter Weir"])


def test_recall_at_k_distinguishes_one_hop_from_two() -> None:
    """Precision alone cannot tell "found one of two hops" from "found both" -
    and multi-hop questions need every hop."""
    gold = ["Gallipoli (1981 film)", "Peter Weir"]
    assert recall_at_k(["Gallipoli_(1981_film)"], gold) == pytest.approx(0.5)
    assert recall_at_k(["Gallipoli (1981 film)", "Peter_Weir"], gold) == 1.0


def test_recall_deduplicates_retrieved_titles() -> None:
    gold = ["Peter Weir", "Gallipoli (1981 film)"]
    assert recall_at_k(["Peter Weir", "Peter_Weir"], gold) == pytest.approx(0.5)


def test_recall_requires_gold_titles() -> None:
    with pytest.raises(MetricError, match="at least one gold"):
        recall_at_k(["Peter Weir"], [])


# ---------------------------------------------------------------------------
# Failure breakdown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("em", "has_answer", "turn_limit", "retrieved", "expected"),
    [
        pytest.param(1.0, True, False, 3, FailureReason.CORRECT, id="correct"),
        pytest.param(0.0, True, False, 3, FailureReason.WRONG_ANSWER, id="wrong"),
        pytest.param(0.0, False, False, 3, FailureReason.NO_ANSWER, id="no-answer"),
        pytest.param(0.0, False, True, 3, FailureReason.NO_ANSWER, id="no-answer-beats-turn-limit"),
        pytest.param(0.0, True, True, 3, FailureReason.TURN_LIMIT, id="turn-limit"),
        pytest.param(0.0, True, False, 0, FailureReason.EMPTY_RETRIEVAL, id="empty-retrieval"),
        pytest.param(0.0, True, True, 0, FailureReason.EMPTY_RETRIEVAL, id="cause-beats-symptom"),
    ],
)
def test_classify_failure(
    em: float, has_answer: bool, turn_limit: bool, retrieved: int, expected: FailureReason
) -> None:
    """Precedence is cause before symptom: retrieving nothing explains a wrong
    answer, and a wrong answer does not explain retrieving nothing."""
    assert (
        classify_failure(
            exact_match_score=em,
            has_answer=has_answer,
            hit_turn_limit=turn_limit,
            retrieved_count=retrieved,
        )
        is expected
    )


def test_a_correct_answer_with_no_retrieval_is_still_correct() -> None:
    """The model answered from parametric memory. PROJECT.md section 6 lists
    that as a reward hack to watch, so it must not be hidden in a failure
    bucket - it shows up as CORRECT with avg searches trending to zero.
    """
    assert (
        classify_failure(
            exact_match_score=1.0, has_answer=True, hit_turn_limit=False, retrieved_count=0
        )
        is FailureReason.CORRECT
    )


def test_failure_breakdown_sums_to_one_and_lists_every_category() -> None:
    """A missing key in a report reads as "not measured"."""
    reasons = [
        FailureReason.CORRECT,
        FailureReason.CORRECT,
        FailureReason.WRONG_ANSWER,
        FailureReason.NO_ANSWER,
    ]
    breakdown = failure_breakdown(reasons)

    assert set(breakdown) == {r.value for r in FailureReason}
    assert sum(breakdown.values()) == pytest.approx(1.0)
    assert breakdown["correct"] == pytest.approx(0.5)
    assert breakdown["turn_limit"] == 0.0


def test_failure_breakdown_rejects_empty_input() -> None:
    with pytest.raises(MetricError, match="at least one reason"):
        failure_breakdown([])
