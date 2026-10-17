"""Tests for `eval/regression.py`, the nightly gate.

Every verdict is built from hand-chosen per-example F1 scores, so the expected
outcome can be worked out by hand: which examples got worse, by how much, and
whether 100 of them can tell that from noise.
"""

from __future__ import annotations

import pytest

from verisearch.data.datasets import Distribution, QAExample, Split
from verisearch.eval.baselines import BaselineKind, EvalError, context_for, examples_fingerprint
from verisearch.eval.harness import EvalRun, ExampleResult, aggregate
from verisearch.eval.metrics import FailureReason
from verisearch.eval.regression import (
    RegressionStatus,
    check_regression,
    regression_set,
)
from verisearch.types import Termination

N = 100
IDS = tuple(f"q{i:03d}" for i in range(N))


def row(question_id: str, f1: float) -> ExampleResult:
    return ExampleResult(
        question_id=question_id,
        answer="something",
        exact_match=1.0 if f1 == 1.0 else 0.0,
        f1=f1,
        n_searches=1,
        n_duplicate_queries=0,
        n_retrieved=3,
        answer_tokens=1,
        termination=Termination.ANSWERED,
        failure=FailureReason.CORRECT if f1 == 1.0 else FailureReason.WRONG_ANSWER,
    )


def run(
    f1s: list[float],
    *,
    name: str,
    ids: tuple[str, ...] = IDS,
    retriever: str = "retriever-fp",
    context_ids: tuple[str, ...] | None = None,
) -> EvalRun:
    rows = tuple(row(qid, f1) for qid, f1 in zip(ids, f1s, strict=True))
    context = context_for(
        context_ids if context_ids is not None else ids,
        dataset="hotpotqa",
        split="dev",
        retriever_fingerprint=retriever,
        k=3,
    )
    report = aggregate(
        rows, run_name=name, kind=BaselineKind.TRAINED, context=context, resamples=200
    )
    return EvalRun(report=report, results=rows)


def changed(*, worse: int) -> list[float]:
    """A candidate against the all-correct `REFERENCE`: `worse` questions that
    were right are now wrong, the rest unchanged."""
    return [0.0] * worse + [1.0] * (N - worse)


REFERENCE = run([1.0] * N, name="reference")


def verdict(candidate_f1s: list[float], *, max_drop: float = 0.02) -> RegressionStatus:
    candidate = run(candidate_f1s, name="nightly")
    return check_regression(REFERENCE, candidate, max_drop=max_drop, resamples=500).status


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_an_unchanged_model_holds() -> None:
    assert verdict([1.0] * N) is RegressionStatus.HELD


def test_a_drop_of_exactly_the_threshold_holds() -> None:
    """ "Has not dropped >2 pts" - two points is not more than two points."""
    assert verdict(changed(worse=2)) is RegressionStatus.HELD


def test_a_large_consistent_drop_regresses() -> None:
    """Thirty questions that used to be right are now wrong. That is breakage,
    and n=100 can see it."""
    assert verdict(changed(worse=30)) is RegressionStatus.REGRESSED


def test_a_drop_past_the_threshold_inside_the_noise_is_noise_not_failure() -> None:
    """Ten questions got worse and seven got better: a 3-point net drop that a
    paired interval on 100 examples cannot tell from zero."""
    reference = run([1.0] * 93 + [0.0] * 7, name="reference")
    candidate = run([0.0] * 10 + [1.0] * 83 + [1.0] * 7, name="nightly")
    result = check_regression(reference, candidate, max_drop=0.02, resamples=500)
    assert result.f1_delta.point == pytest.approx(-0.03)
    assert result.f1_delta.low < 0.0 < result.f1_delta.high
    assert result.status is RegressionStatus.NOISE


def test_an_improvement_holds() -> None:
    reference = run([0.0] * 50 + [1.0] * 50, name="reference")
    candidate = run([1.0] * N, name="nightly")
    result = check_regression(reference, candidate, max_drop=0.02, resamples=500)
    assert result.status is RegressionStatus.HELD
    assert result.f1_delta.point == pytest.approx(0.5)


def test_a_zero_threshold_still_needs_the_interval_to_fail() -> None:
    assert verdict(changed(worse=1), max_drop=0.0) is RegressionStatus.NOISE


@pytest.mark.parametrize("max_drop", [-0.01, 1.0, 2.0])
def test_a_threshold_outside_the_unit_interval_is_refused(max_drop: float) -> None:
    """2.0 is the CLI's "points" passed where a fraction belongs - the mistake
    this guard exists for."""
    with pytest.raises(EvalError, match="fraction"):
        check_regression(REFERENCE, run([1.0] * N, name="nightly"), max_drop=max_drop)


# ---------------------------------------------------------------------------
# Refusing comparisons that would mean nothing
# ---------------------------------------------------------------------------


def test_a_candidate_on_another_index_is_refused() -> None:
    with pytest.raises(EvalError, match="retriever_fingerprint"):
        check_regression(
            REFERENCE, run([1.0] * N, name="nightly", retriever="other"), max_drop=0.02
        )


def test_rows_that_are_not_the_questions_their_context_describes_are_refused() -> None:
    """Same count, different questions: `load_run` would accept this, and every
    delta would compare unrelated questions."""
    other = tuple(f"x{i:03d}" for i in range(N))
    impostor = run([1.0] * N, name="nightly", ids=other, context_ids=IDS)
    with pytest.raises(EvalError, match="candidate rows are not the questions"):
        check_regression(REFERENCE, impostor, max_drop=0.02)


def test_a_reference_whose_rows_do_not_match_its_context_is_refused() -> None:
    other = tuple(f"x{i:03d}" for i in range(N))
    bad_reference = run([1.0] * N, name="reference", ids=other, context_ids=IDS)
    with pytest.raises(EvalError, match="reference rows are not the questions"):
        check_regression(bad_reference, run([1.0] * N, name="nightly"), max_drop=0.02)


# ---------------------------------------------------------------------------
# The summary says what the gate could see
# ---------------------------------------------------------------------------


def test_the_summary_states_the_verdict_and_the_resolvable_drop() -> None:
    result = check_regression(
        REFERENCE, run(changed(worse=30), name="nightly"), max_drop=0.02, resamples=500
    )
    text = result.summary()
    assert "verdict             regressed" in text
    assert "threshold           -2.0 pts" in text
    assert "resolvable          about +/-" in text
    assert result.resolvable_drop == pytest.approx(result.f1_delta.width / 2)


def test_a_noise_verdict_says_it_did_not_fail_and_why() -> None:
    reference = run([1.0] * 93 + [0.0] * 7, name="reference")
    candidate = run([0.0] * 10 + [1.0] * 90, name="nightly")
    text = check_regression(reference, candidate, max_drop=0.02, resamples=500).summary()
    assert "interval includes zero" in text
    assert "inside this run's noise" in text


def test_a_tight_interval_does_not_claim_the_threshold_is_invisible() -> None:
    """An unchanged greedy model gives a zero-width interval; no "inside the
    noise" note belongs on that."""
    text = check_regression(REFERENCE, run([1.0] * N, name="nightly"), max_drop=0.02).summary()
    assert "inside this run's noise" not in text


# ---------------------------------------------------------------------------
# The fixed set
# ---------------------------------------------------------------------------


def example(question_id: str, split: Split | None = Split.DEV) -> QAExample:
    return QAExample(
        question_id=question_id,
        question=f"Question {question_id}?",
        answers=("A",),
        dataset="hotpotqa",
        distribution=Distribution.IN_DISTRIBUTION,
        split=split,
    )


POOL = tuple(example(f"d{i:04d}") for i in range(400))


def test_the_set_is_deterministic_and_in_file_order() -> None:
    first = regression_set(POOL, 100, seed=42)
    again = regression_set(POOL, 100, seed=42)
    assert first == again
    assert len(first) == 100
    assert [e.question_id for e in first] == sorted(e.question_id for e in first)


def test_a_reordered_pool_gives_the_same_set() -> None:
    assert regression_set(tuple(reversed(POOL)), 100, seed=42) == regression_set(POOL, 100, seed=42)


def test_another_seed_gives_another_set() -> None:
    assert regression_set(POOL, 100, seed=42) != regression_set(POOL, 100, seed=7)


def test_the_set_comes_from_one_split_only() -> None:
    mixed = POOL[:150] + tuple(example(f"t{i:04d}", Split.TRAIN) for i in range(300))
    chosen = regression_set(mixed, 100, seed=42)
    assert all(e.split is Split.DEV for e in chosen)


def test_too_few_examples_is_refused_rather_than_a_smaller_gate() -> None:
    with pytest.raises(EvalError, match="found 50"):
        regression_set(POOL[:50], 100, seed=42)


def test_examples_without_a_split_are_not_silently_dev() -> None:
    unsplit = tuple(example(f"u{i}", split=None) for i in range(200))
    with pytest.raises(EvalError, match="split set"):
        regression_set(unsplit, 100, seed=42)


def test_a_set_of_nothing_is_refused() -> None:
    with pytest.raises(EvalError, match="at least one"):
        regression_set(POOL, 0, seed=42)


def test_the_fingerprint_of_the_set_is_the_one_a_context_built_from_it_gets() -> None:
    chosen = regression_set(POOL, 100, seed=42)
    ids = [e.question_id for e in chosen]
    context = context_for(ids, dataset="hotpotqa", split="dev", retriever_fingerprint="r", k=3)
    assert context.examples_fingerprint == examples_fingerprint(ids)
