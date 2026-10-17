"""Tests for `eval/harness.py` and `eval/baselines.py`.

Every metric in PROJECT.md section 9 is exercised here with no model, no GPU and
no network, which is the point of `EpisodeRunner` being a protocol. `StubRunner`
below returns trajectories a test wrote by hand, so "what does precision@k
report when the policy retrieved nothing" is a fact the test states rather than
something a model happens to produce.

Resamples are kept low on purpose. `bootstrap_ci` is already tested in
`tests/test_metrics.py`; what matters here is that the right values reach it and
that the `n` it reports is the right denominator.

TODO: `make_trajectory` from `verisearch/testing/fakes.py` (#9) replaces the
builders below once that merges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from verisearch.data.datasets import Distribution, QAExample, Split
from verisearch.eval.baselines import (
    BASELINES,
    REFERENCE_BASELINE,
    BaselineKind,
    EvalContext,
    EvalError,
    baseline,
    check_comparable,
    context_for,
    examples_fingerprint,
    joint_improvement_claim,
    resolved_search_budget,
)
from verisearch.eval.harness import (
    Comparison,
    EvalRun,
    aggregate,
    compare_runs,
    evaluate,
    load_run,
    read_results,
    score_example,
    write_report,
    write_results,
)
from verisearch.eval.metrics import (
    ConfidenceInterval,
    FailureReason,
    bootstrap_ci,
    paired_bootstrap_delta,
)
from verisearch.types import (
    Claim,
    Entailment,
    FaithfulnessResult,
    Passage,
    Termination,
    Trajectory,
    Turn,
    TurnKind,
)

if TYPE_CHECKING:
    from pathlib import Path

FAST = {"resamples": 50, "seed": 7}


def make_example(
    question_id: str,
    *,
    answers: tuple[str, ...] = ("Ada Lovelace",),
    titles: tuple[str, ...] = ("Ada Lovelace", "Analytical Engine"),
) -> QAExample:
    return QAExample(
        question_id=question_id,
        question=f"Who is {question_id}?",
        answers=answers,
        dataset="hotpotqa",
        distribution=Distribution.IN_DISTRIBUTION,
        supporting_titles=titles,
        split=Split.DEV,
    )


def make_trajectory(
    example: QAExample,
    *,
    answer: str | None = "Ada Lovelace",
    queries: tuple[str, ...] = ("ada lovelace",),
    titles: tuple[str, ...] = ("Ada Lovelace",),
    termination: Termination = Termination.ANSWERED,
) -> Trajectory:
    turns: list[Turn] = []
    for query in queries:
        turns.append(Turn(kind=TurnKind.SEARCH, text=query))
        turns.append(
            Turn(
                kind=TurnKind.INFO,
                text="<info/>",
                passages=tuple(
                    Passage(doc_id=title.lower(), title=title, text=f"About {title}.")
                    for title in titles
                ),
            )
        )
    if answer is not None:
        turns.append(Turn(kind=TurnKind.ANSWER, text=answer))
    return Trajectory(
        question_id=example.question_id,
        question=example.question,
        gold_answers=example.answers,
        turns=tuple(turns),
        answer=answer,
        termination=termination,
    )


class StubRunner:
    """Returns a prepared trajectory per question id."""

    def __init__(self, trajectories: dict[str, Trajectory]) -> None:
        self._trajectories = trajectories
        self.seen: list[str] = []

    def run(self, example: QAExample) -> Trajectory:
        self.seen.append(example.question_id)
        return self._trajectories[example.question_id]


class StubScorer:
    """Unsupported rate derived from whether any passage was retrieved.

    Crude on purpose. The real NLI scorer is `faithfulness/nli.py`'s problem;
    what the harness needs is that a score reaches the right column with the
    right denominator.
    """

    def __init__(self, *, rate_with_evidence: float = 0.0) -> None:
        self._rate = rate_with_evidence
        self.calls: list[tuple[str, int]] = []

    def score(self, answer: str, passages: list[Passage]) -> FaithfulnessResult:
        self.calls.append((answer, len(passages)))
        rate = self._rate if passages else 1.0
        label = Entailment.ENTAILED if rate == 0.0 else Entailment.NEUTRAL
        return FaithfulnessResult(
            claims=(Claim(text=answer, label=label),),
            score=1.0 - rate,
        )


def make_context(
    question_ids: tuple[str, ...], *, k: int = 3, fingerprint: str = "abc123"
) -> EvalContext:
    return context_for(
        question_ids,
        dataset="hotpotqa",
        split="dev",
        retriever_fingerprint=fingerprint,
        k=k,
    )


def run_eval(
    examples: tuple[QAExample, ...],
    trajectories: dict[str, Trajectory],
    *,
    name: str,
    kind: BaselineKind = BaselineKind.TRAINED,
    scorer: StubScorer | None = None,
    fingerprint: str = "abc123",
) -> EvalRun:
    context = make_context(tuple(e.question_id for e in examples), fingerprint=fingerprint)
    return evaluate(
        StubRunner(trajectories),
        examples,
        run_name=name,
        kind=kind,
        context=context,
        scorer=scorer,
        **FAST,
    )


# ---------------------------------------------------------------------------
# baselines.py - the four specs
# ---------------------------------------------------------------------------


def test_all_five_kinds_have_a_spec_and_no_duplicates() -> None:
    kinds = [spec.kind for spec in BASELINES]
    assert set(kinds) == set(BaselineKind)
    assert len(kinds) == len(set(kinds))
    for kind in BaselineKind:
        assert baseline(kind).kind is kind


def test_the_structural_baselines_pin_their_own_search_budget() -> None:
    """0 and 1 are what *make* baselines 1 and 2 those baselines. A config that
    could override them would turn 'vanilla RAG' into something else and still
    print 'vanilla RAG' in the table."""
    assert resolved_search_budget(baseline(BaselineKind.NO_RETRIEVAL), 4) == 0
    assert resolved_search_budget(baseline(BaselineKind.VANILLA_RAG), 4) == 1
    # The iterative ones take the config's budget.
    assert resolved_search_budget(baseline(BaselineKind.PROMPTED_ITERATIVE), 4) == 4
    assert resolved_search_budget(baseline(BaselineKind.TRAINED), 6) == 6


def test_a_negative_search_budget_is_refused() -> None:
    with pytest.raises(EvalError, match="budget must be >= 0"):
        resolved_search_budget(baseline(BaselineKind.TRAINED), -1)


def test_the_reference_baseline_is_the_prompted_one() -> None:
    """Section 8 calls it 'the baseline that matters'. Named rather than indexed
    so reordering BASELINES cannot change what 'the baseline' means."""
    assert REFERENCE_BASELINE is BaselineKind.PROMPTED_ITERATIVE
    assert not baseline(REFERENCE_BASELINE).requires_checkpoint


def test_only_the_trained_and_sft_baselines_need_weights() -> None:
    needs = {spec.kind for spec in BASELINES if spec.requires_checkpoint}
    assert needs == {BaselineKind.SFT_ONLY, BaselineKind.TRAINED}


# ---------------------------------------------------------------------------
# baselines.py - comparability
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_stable_across_processes_and_sensitive_to_order() -> None:
    """`hashlib`, not `hash()`. Python randomises string hashing per process
    unless PYTHONHASHSEED is set, and this pipeline leaves it unset."""
    assert examples_fingerprint(["a", "b"]) == examples_fingerprint(["a", "b"])
    assert examples_fingerprint(["a", "b"]) != examples_fingerprint(["b", "a"])
    # Not a concatenation: ("ab",) and ("a","b") must differ.
    assert examples_fingerprint(["ab"]) != examples_fingerprint(["a", "b"])


def test_an_empty_eval_set_cannot_be_fingerprinted() -> None:
    with pytest.raises(EvalError, match="empty eval set"):
        examples_fingerprint([])


def test_duplicate_question_ids_are_refused() -> None:
    """A duplicate inflates whichever question is doubled and makes a paired
    comparison ambiguous."""
    with pytest.raises(EvalError, match="duplicate question ids"):
        make_context(("q1", "q1"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "musique"),
        ("split", "test"),
        ("k", 5),
        ("retriever_fingerprint", "deadbeef"),
    ],
)
def test_check_comparable_refuses_any_difference_and_names_it(field: str, value: object) -> None:
    """Section 2's sixth criterion. Each of these is a way the headline claim
    could be quietly comparing two different experiments."""
    reference = make_context(("q1", "q2"))
    candidate = reference.model_copy(update={field: value})
    with pytest.raises(EvalError, match=field):
        check_comparable(reference, candidate)


def test_a_different_eval_set_is_caught_by_the_fingerprint() -> None:
    reference = make_context(("q1", "q2"))
    candidate = make_context(("q1", "q3"))
    with pytest.raises(EvalError, match="examples_fingerprint"):
        check_comparable(reference, candidate)


def test_identical_contexts_compare_cleanly() -> None:
    check_comparable(make_context(("q1", "q2")), make_context(("q1", "q2")))


# ---------------------------------------------------------------------------
# baselines.py - the headline claim
# ---------------------------------------------------------------------------


def ci(point: float, low: float, high: float) -> ConfidenceInterval:
    return ConfidenceInterval(point=point, low=low, high=high, confidence=0.95, resamples=100, n=50)


def test_the_claim_holds_only_when_all_three_conditions_do() -> None:
    claim = joint_improvement_claim(
        f1_delta=ci(0.08, 0.03, 0.13),
        unsupported_rate_delta=ci(-0.10, -0.16, -0.04),
        searches_delta=ci(-0.5, -0.9, -0.1),
    )
    assert claim.holds
    assert [part.holds for part in claim.parts] == [True, True, True]
    assert "HOLDS" in claim.summary()


def test_an_accuracy_gain_that_straddles_zero_does_not_count() -> None:
    """Section 9: 'A 2-point gain on 500 examples is noise.' An interval
    containing zero is not a gain, and treating it as one is the single most
    likely way this project reports a result it cannot defend."""
    claim = joint_improvement_claim(
        f1_delta=ci(0.02, -0.01, 0.05),
        unsupported_rate_delta=ci(-0.10, -0.16, -0.04),
        searches_delta=ci(-0.5, -0.9, -0.1),
    )
    assert not claim.holds
    assert not claim.parts[0].holds
    assert "DOES NOT HOLD" in claim.summary()


def test_more_searches_breaks_the_claim_even_with_better_accuracy() -> None:
    """The 'at fewer retrievals' half. Two out of three is a weaker paper."""
    claim = joint_improvement_claim(
        f1_delta=ci(0.08, 0.03, 0.13),
        unsupported_rate_delta=ci(-0.10, -0.16, -0.04),
        searches_delta=ci(0.6, 0.2, 1.0),
    )
    assert not claim.holds
    assert [part.holds for part in claim.parts] == [True, True, False]


def test_equal_search_counts_satisfy_the_retrieval_condition() -> None:
    """Section 2 says 'lower than or equal to', so no detected difference is a
    pass here - unlike for accuracy, where it is a failure."""
    claim = joint_improvement_claim(
        f1_delta=ci(0.08, 0.03, 0.13),
        unsupported_rate_delta=ci(-0.10, -0.16, -0.04),
        searches_delta=ci(0.0, -0.3, 0.3),
    )
    assert claim.holds
    assert claim.parts[2].holds


def test_an_unchanged_hallucination_rate_breaks_the_claim() -> None:
    claim = joint_improvement_claim(
        f1_delta=ci(0.08, 0.03, 0.13),
        unsupported_rate_delta=ci(-0.01, -0.05, 0.03),
        searches_delta=ci(-0.5, -0.9, -0.1),
    )
    assert not claim.holds
    assert not claim.parts[1].holds


# ---------------------------------------------------------------------------
# harness.py - scoring one example
# ---------------------------------------------------------------------------


def test_score_example_fills_every_section_9_column() -> None:
    example = make_example("q1")
    result = score_example(
        example,
        make_trajectory(example, titles=("Ada Lovelace", "Lovelace, Texas")),
        scorer=StubScorer(),
    )

    assert result.exact_match == 1.0
    assert result.f1 == 1.0
    assert result.n_searches == 1
    assert result.n_retrieved == 2
    assert result.answer_tokens == 2
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(0.5)
    assert result.unsupported_rate == 0.0
    assert result.failure is FailureReason.CORRECT


def test_a_trajectory_for_the_wrong_question_is_refused() -> None:
    """An off-by-one in a batch, a cached result, a shuffled queue. The
    prediction would be scored against another question's gold answers, every
    number would compute, and it would read as a model that cannot answer."""
    example = make_example("q1")
    other = make_trajectory(make_example("q2"))
    with pytest.raises(EvalError, match="look like a weak model"):
        score_example(example, other)


def test_a_trajectory_carrying_different_gold_answers_is_refused() -> None:
    """The runner read a different dataset than the harness is scoring against -
    a stale snapshot, or a split regenerated between the two."""
    example = make_example("q1", answers=("Ada Lovelace",))
    stale = make_trajectory(make_example("q1", answers=("Charles Babbage",)))
    with pytest.raises(EvalError, match="read a different dataset"):
        score_example(example, stale)


def test_precision_is_none_not_zero_when_nothing_was_retrieved() -> None:
    """Baseline 1 retrieves nothing by construction. A 0.0 here would put a
    retrieval-quality number in the table for a system with no retrieval."""
    example = make_example("q1")
    result = score_example(example, make_trajectory(example, queries=()))
    assert result.precision_at_k is None
    assert result.n_retrieved == 0
    # Recall is still defined: the gold titles exist and none were found.
    assert result.recall_at_k == 0.0


def test_recall_is_none_when_the_dataset_ships_no_supporting_titles() -> None:
    """Bamboogle does not. Reporting 0.0 would read as a retrieval failure on a
    dataset that never said what the right paragraphs were."""
    example = make_example("q1", titles=())
    result = score_example(example, make_trajectory(example))
    assert result.recall_at_k is None
    assert result.precision_at_k == 0.0


def test_unsupported_rate_is_none_when_the_episode_never_answered() -> None:
    """There are no claims to check, and 0.0 would make *never answering* the
    best possible hallucination score - the degenerate policy section 6 warns
    about."""
    example = make_example("q1")
    scorer = StubScorer()
    result = score_example(
        example,
        make_trajectory(example, answer=None, termination=Termination.MAX_TURNS),
        scorer=scorer,
    )
    assert result.unsupported_rate is None
    assert scorer.calls == []


def test_an_answer_with_no_evidence_is_scored_not_skipped() -> None:
    """The most important case the metric exists for: answering from parametric
    memory means every claim is unsupported."""
    example = make_example("q1")
    result = score_example(example, make_trajectory(example, queries=()), scorer=StubScorer())
    assert result.unsupported_rate == 1.0


def test_without_a_scorer_the_column_is_none_rather_than_zero() -> None:
    example = make_example("q1")
    result = score_example(example, make_trajectory(example))
    assert result.unsupported_rate is None


def test_duplicate_queries_are_counted() -> None:
    """Feeds the -0.1 duplicate penalty in section 6."""
    example = make_example("q1")
    result = score_example(example, make_trajectory(example, queries=("ada", "ada", "babbage")))
    assert result.n_searches == 3
    assert result.n_duplicate_queries == 1


def test_a_wrong_answer_with_retrieval_is_classified_wrong_answer() -> None:
    example = make_example("q1")
    result = score_example(example, make_trajectory(example, answer="Charles Babbage"))
    assert result.exact_match == 0.0
    assert result.failure is FailureReason.WRONG_ANSWER


def test_a_correct_answer_from_no_retrieval_is_reported_correct() -> None:
    """Section 6 lists this as a reward hack to watch, so it must not be hidden
    inside a failure bucket."""
    example = make_example("q1")
    result = score_example(example, make_trajectory(example, queries=()))
    assert result.failure is FailureReason.CORRECT


def test_a_turn_limited_episode_reports_no_answer_not_turn_limit() -> None:
    """Documenting `classify_failure`'s precedence rather than working around
    it: NO_ANSWER is checked before TURN_LIMIT, so TURN_LIMIT only ever fires
    for an episode that *did* answer, was wrong, and hit the limit.

    Worth a look in review of #5 - as written, the turn_limit bucket will be
    near-empty in every report, and 'ran out of turns without answering' lands
    under no_answer."""
    example = make_example("q1")
    starved = score_example(
        example, make_trajectory(example, answer=None, termination=Termination.MAX_TURNS)
    )
    assert starved.failure is FailureReason.NO_ANSWER

    answered_but_wrong = score_example(
        example,
        make_trajectory(example, answer="Charles Babbage", termination=Termination.MAX_TURNS),
    )
    assert answered_but_wrong.failure is FailureReason.TURN_LIMIT


# ---------------------------------------------------------------------------
# harness.py - aggregating a run
# ---------------------------------------------------------------------------


def test_evaluate_produces_a_report_with_an_interval_on_every_number() -> None:
    examples = tuple(make_example(f"q{i}") for i in range(6))
    trajectories = {
        e.question_id: make_trajectory(e, answer="Ada Lovelace" if i % 2 == 0 else "Nobody")
        for i, e in enumerate(examples)
    }
    run = run_eval(examples, trajectories, name="trained", scorer=StubScorer())

    assert run.report.exact_match.point == pytest.approx(0.5)
    assert run.report.exact_match.n == 6
    assert run.report.f1.n == 6
    assert run.report.searches_per_question.point == pytest.approx(1.0)
    assert run.report.unsupported_rate is not None
    assert run.report.precision_at_k is not None
    assert sum(run.report.failure_breakdown.values()) == pytest.approx(1.0)
    assert set(run.report.failure_breakdown) == {r.value for r in FailureReason}
    assert "EM" in run.report.summary()


def test_an_aggregate_over_a_subset_reports_the_subset_size() -> None:
    """A precision@k computed over 2 of 4 examples must say n=2 rather than
    looking like a full-set number."""
    examples = tuple(make_example(f"q{i}") for i in range(4))
    trajectories = {
        e.question_id: make_trajectory(e, queries=() if i < 2 else ("ada",))
        for i, e in enumerate(examples)
    }
    run = run_eval(examples, trajectories, name="mixed")

    assert run.report.precision_at_k is not None
    assert run.report.precision_at_k.n == 2
    assert run.report.exact_match.n == 4


def test_a_run_with_no_retrieval_at_all_reports_precision_as_not_applicable() -> None:
    examples = tuple(make_example(f"q{i}") for i in range(3))
    trajectories = {e.question_id: make_trajectory(e, queries=()) for e in examples}
    run = run_eval(examples, trajectories, name="floor", kind=BaselineKind.NO_RETRIEVAL)

    assert run.report.precision_at_k is None
    assert "n/a for this run" in run.report.summary()
    assert run.report.searches_per_question.point == 0.0


def test_the_duplicate_query_episode_rate_is_a_fraction_of_episodes() -> None:
    examples = tuple(make_example(f"q{i}") for i in range(4))
    trajectories = {
        e.question_id: make_trajectory(e, queries=("ada", "ada") if i < 1 else ("ada",))
        for i, e in enumerate(examples)
    }
    run = run_eval(examples, trajectories, name="dupes")
    assert run.report.duplicate_query_episode_rate == pytest.approx(0.25)


def test_evaluating_a_different_number_of_examples_than_the_context_says_is_refused() -> None:
    """The context carries the denominators. A mismatch means the report would
    describe a set it did not measure."""
    examples = tuple(make_example(f"q{i}") for i in range(3))
    trajectories = {e.question_id: make_trajectory(e) for e in examples}
    context = make_context(("q0", "q1"))
    with pytest.raises(EvalError, match="not the eval set"):
        evaluate(
            StubRunner(trajectories),
            examples,
            run_name="x",
            kind=BaselineKind.TRAINED,
            context=context,
            **FAST,
        )


def test_an_empty_example_set_is_refused() -> None:
    with pytest.raises(EvalError, match="empty example set"):
        evaluate(
            StubRunner({}),
            [],
            run_name="x",
            kind=BaselineKind.TRAINED,
            context=make_context(("q1",)),
            **FAST,
        )


def test_aggregating_an_empty_result_set_is_refused() -> None:
    with pytest.raises(EvalError, match="empty result set"):
        aggregate([], run_name="x", kind=BaselineKind.TRAINED, context=make_context(("q1",)))


def test_aggregating_under_a_mismatched_context_is_refused() -> None:
    example = make_example("q1")
    result = score_example(example, make_trajectory(example))
    with pytest.raises(EvalError, match="denominators would be wrong"):
        aggregate(
            [result],
            run_name="x",
            kind=BaselineKind.TRAINED,
            context=make_context(("q1", "q2")),
            **FAST,
        )


def test_the_runner_sees_every_example_exactly_once_in_order() -> None:
    examples = tuple(make_example(f"q{i}") for i in range(5))
    trajectories = {e.question_id: make_trajectory(e) for e in examples}
    runner = StubRunner(trajectories)
    context = make_context(tuple(e.question_id for e in examples))
    evaluate(runner, examples, run_name="x", kind=BaselineKind.TRAINED, context=context, **FAST)
    assert runner.seen == [e.question_id for e in examples]


# ---------------------------------------------------------------------------
# harness.py - comparing two runs
# ---------------------------------------------------------------------------


def make_pair(
    *,
    candidate_answers: tuple[str, ...],
    reference_answers: tuple[str, ...],
    candidate_queries: int = 1,
    reference_queries: int = 2,
    fingerprint: str = "abc123",
) -> tuple[EvalRun, EvalRun]:
    examples = tuple(make_example(f"q{i}") for i in range(len(candidate_answers)))
    reference = run_eval(
        examples,
        {
            e.question_id: make_trajectory(
                e,
                answer=reference_answers[i],
                queries=tuple(f"r{j}" for j in range(reference_queries)),
            )
            for i, e in enumerate(examples)
        },
        name="prompted",
        kind=BaselineKind.PROMPTED_ITERATIVE,
        scorer=StubScorer(rate_with_evidence=0.4),
        fingerprint=fingerprint,
    )
    candidate = run_eval(
        examples,
        {
            e.question_id: make_trajectory(
                e,
                answer=candidate_answers[i],
                queries=tuple(f"c{j}" for j in range(candidate_queries)),
            )
            for i, e in enumerate(examples)
        },
        name="trained",
        kind=BaselineKind.TRAINED,
        scorer=StubScorer(rate_with_evidence=0.0),
    )
    return reference, candidate


def test_compare_runs_produces_paired_deltas_and_the_claim() -> None:
    gold = ("Ada Lovelace",) * 8
    reference, candidate = make_pair(
        candidate_answers=gold,
        reference_answers=("Nobody",) * 4 + gold[:4],
    )
    comparison = compare_runs(reference, candidate, **FAST)

    assert isinstance(comparison, Comparison)
    assert comparison.n_paired == 8
    assert comparison.f1_delta.point > 0.0
    assert comparison.searches_delta.point < 0.0
    assert comparison.unsupported_rate_delta is not None
    assert comparison.unsupported_rate_delta.point < 0.0
    assert comparison.claim is not None
    assert comparison.claim.holds
    assert "vs prompted" in comparison.summary()


def test_comparing_runs_from_different_eval_sets_is_refused() -> None:
    gold = ("Ada Lovelace",) * 4
    reference, _ = make_pair(candidate_answers=gold, reference_answers=gold)
    other = run_eval(
        tuple(make_example(f"z{i}") for i in range(4)),
        {f"z{i}": make_trajectory(make_example(f"z{i}")) for i in range(4)},
        name="elsewhere",
    )
    with pytest.raises(EvalError, match="not comparable"):
        compare_runs(reference, other, **FAST)


def test_comparing_across_a_rebuilt_index_is_refused() -> None:
    """The failure this is all for: the index gets rebuilt between two runs and
    the comparison still prints."""
    gold = ("Ada Lovelace",) * 4
    reference, _ = make_pair(
        candidate_answers=gold, reference_answers=gold, fingerprint="old-index"
    )
    _, candidate = make_pair(candidate_answers=gold, reference_answers=gold)
    with pytest.raises(EvalError, match="retriever_fingerprint"):
        compare_runs(reference, candidate, **FAST)


def test_rows_that_do_not_line_up_are_refused_even_with_a_matching_context() -> None:
    """A results file sorted after being written passes the fingerprint check
    and fails this one. Pairing positionally across a reordering compares each
    question against a different one and still returns a narrow interval."""
    gold = ("Ada Lovelace",) * 4
    reference, candidate = make_pair(candidate_answers=gold, reference_answers=gold)
    shuffled = candidate.model_copy(update={"results": tuple(reversed(candidate.results))})
    with pytest.raises(EvalError, match="first differ at position 0"):
        compare_runs(reference, shuffled, **FAST)


def test_the_unsupported_delta_is_over_the_subset_where_both_answered() -> None:
    """The caveat for the writeup: a system that answers less often is being
    compared on the subset it chose to answer. `n_paired_unsupported` makes the
    gap visible rather than implied."""
    examples = tuple(make_example(f"q{i}") for i in range(4))
    silent = run_eval(
        examples,
        {
            e.question_id: make_trajectory(e, answer=None if i < 2 else "Ada Lovelace")
            for i, e in enumerate(examples)
        },
        name="half-silent",
        scorer=StubScorer(rate_with_evidence=0.5),
    )
    talkative = run_eval(
        examples,
        {e.question_id: make_trajectory(e) for e in examples},
        name="always-answers",
        scorer=StubScorer(),
    )
    comparison = compare_runs(silent, talkative, **FAST)

    assert comparison.n_paired == 4
    assert comparison.n_paired_unsupported == 2
    assert "both answered" in comparison.summary()


def test_no_claim_is_made_without_a_faithfulness_delta() -> None:
    """A two-part claim labelled as the headline result would be the most
    expensive kind of convenience in this project."""
    examples = tuple(make_example(f"q{i}") for i in range(3))
    trajectories = {e.question_id: make_trajectory(e) for e in examples}
    reference = run_eval(examples, trajectories, name="a")
    candidate = run_eval(examples, trajectories, name="b")
    comparison = compare_runs(reference, candidate, **FAST)

    assert comparison.unsupported_rate_delta is None
    assert comparison.n_paired_unsupported == 0
    assert comparison.claim is None
    assert "n/a - no example where both answered" in comparison.summary()


def test_the_paired_delta_matches_calling_the_metric_directly() -> None:
    """The harness must not be doing its own arithmetic on top of #5."""
    gold = ("Ada Lovelace",) * 6
    reference, candidate = make_pair(
        candidate_answers=gold, reference_answers=("Nobody",) * 3 + gold[:3]
    )
    expected = paired_bootstrap_delta(
        [r.f1 for r in reference.results],
        [r.f1 for r in candidate.results],
        **FAST,
    )
    assert compare_runs(reference, candidate, **FAST).f1_delta == expected


# ---------------------------------------------------------------------------
# harness.py - persistence
# ---------------------------------------------------------------------------


def test_results_round_trip_through_jsonl_in_run_order(tmp_path: Path) -> None:
    examples = tuple(make_example(f"q{i}") for i in range(5))
    run = run_eval(
        examples,
        {e.question_id: make_trajectory(e) for e in examples},
        name="x",
        scorer=StubScorer(),
    )
    path = tmp_path / "nested" / "results.jsonl"
    assert write_results(run.results, path) == 5
    assert read_results(path) == run.results


def test_a_report_and_its_rows_round_trip_together(tmp_path: Path) -> None:
    examples = tuple(make_example(f"q{i}") for i in range(3))
    run = run_eval(examples, {e.question_id: make_trajectory(e) for e in examples}, name="x")
    write_report(run.report, tmp_path / "report.json")
    write_results(run.results, tmp_path / "results.jsonl")

    loaded = load_run(tmp_path / "report.json", tmp_path / "results.jsonl")
    assert loaded == run


def test_a_report_paired_with_the_wrong_rows_is_refused(tmp_path: Path) -> None:
    """Comparing a report against someone else's rows would produce deltas
    against numbers the report never described."""
    examples = tuple(make_example(f"q{i}") for i in range(3))
    run = run_eval(examples, {e.question_id: make_trajectory(e) for e in examples}, name="x")
    write_report(run.report, tmp_path / "report.json")
    write_results(run.results[:2], tmp_path / "results.jsonl")

    with pytest.raises(EvalError, match="not from the same run"):
        load_run(tmp_path / "report.json", tmp_path / "results.jsonl")


def test_writing_no_results_is_an_error(tmp_path: Path) -> None:
    """An empty eval file is a silent failure: every later step reads it, finds
    nothing, and reports nothing rather than failing."""
    with pytest.raises(EvalError, match="silent failure"):
        write_results([], tmp_path / "empty.jsonl")


def test_reading_a_missing_or_malformed_results_file_says_which(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="no results file at"):
        read_results(tmp_path / "absent.jsonl")

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question_id":"q1"}\n', encoding="utf-8")
    with pytest.raises(EvalError, match=r"bad\.jsonl:1"):
        read_results(bad)

    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n   \n", encoding="utf-8")
    with pytest.raises(EvalError, match="contains no results"):
        read_results(blank)


def test_reading_a_missing_or_malformed_report_says_which(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="no report at"):
        load_run(tmp_path / "absent.json", tmp_path / "absent.jsonl")

    bad = tmp_path / "report.json"
    bad.write_text('{"run_name":"x"}', encoding="utf-8")
    with pytest.raises(EvalError, match="not an eval report"):
        load_run(bad, tmp_path / "absent.jsonl")


def test_blank_lines_in_a_results_file_are_skipped(tmp_path: Path) -> None:
    example = make_example("q1")
    result = score_example(example, make_trajectory(example))
    path = tmp_path / "results.jsonl"
    path.write_text(f"{result.model_dump_json()}\n\n   \n", encoding="utf-8")
    assert read_results(path) == (result,)


def test_the_report_carries_the_context_so_a_number_is_traceable(tmp_path: Path) -> None:
    """Section 15: a reported number has to be traceable to the run behind it. A
    report that does not say which eval set and which retriever produced it is
    not a result, it is a number."""
    examples = tuple(make_example(f"q{i}") for i in range(3))
    run = run_eval(examples, {e.question_id: make_trajectory(e) for e in examples}, name="x")
    write_report(run.report, tmp_path / "report.json")

    text = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "retriever_fingerprint" in text
    assert "examples_fingerprint" in text
    assert run.report.context.retriever_fingerprint in text


def test_bootstrap_is_seeded_so_two_runs_of_the_report_agree() -> None:
    """An unseeded CI changes between two runs of the report and makes a result
    impossible to check."""
    examples = tuple(make_example(f"q{i}") for i in range(6))
    trajectories = {
        e.question_id: make_trajectory(e, answer="Ada Lovelace" if i % 2 else "Nobody")
        for i, e in enumerate(examples)
    }
    first = run_eval(examples, trajectories, name="x")
    second = run_eval(examples, trajectories, name="x")
    assert first.report == second.report
    assert first.report.f1 == bootstrap_ci([r.f1 for r in first.results], **FAST)
