"""Reward tests. PROJECT.md section 3: table-driven, one per component, plus the
properties section 6 says the reward must have.

The last section is the one that matters most: correctness dominance, and the
hacks section 6 predicts priced the way it says they should be.
"""

from __future__ import annotations

import pytest

from verisearch.config import PenaltyConfig, RewardConfig
from verisearch.rewards.compose import RewardSpec, compose_reward, n_empty_queries
from verisearch.rewards.correctness import Correctness
from verisearch.rewards.faithfulness import faithfulness_reward, score_from
from verisearch.rewards.format import FormatValid, is_well_formed
from verisearch.rewards.retrieval_cost import retrieval_cost
from verisearch.testing import DEFAULT_CORPUS, FakeFaithfulnessScorer, make_trajectory
from verisearch.types import (
    Claim,
    Entailment,
    FaithfulnessResult,
    ParseFailure,
    Termination,
    Trajectory,
    Turn,
    TurnKind,
)

BUDGET = 4


def weights(**overrides: float) -> RewardConfig:
    """Section 6's weights, which are also `configs/base.yaml`'s."""
    base: dict[str, object] = {
        "w_correctness": 1.0,
        "w_faithfulness": 0.5,
        "w_format": 0.2,
        "lambda_retrieval_cost": 0.05,
        "penalties": PenaltyConfig(duplicate_query=-0.1, no_answer=-0.5, empty_query=-0.05),
    }
    return RewardConfig.model_validate(base | overrides)


def spec(**overrides: float) -> RewardSpec:
    return RewardSpec(weights(**overrides), search_budget=BUDGET)


def answered(answer: str | None = "Ada Lovelace") -> Trajectory:
    """A no-search trajectory with the given answer. The simplest shape there is."""
    return make_trajectory(answer=answer)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        pytest.param("Ada Lovelace", 1.0, id="exact"),
        pytest.param("ada lovelace", 1.0, id="normalised-exact"),
        pytest.param("The Ada Lovelace", 1.0, id="articles-dropped"),
        pytest.param("Charles Babbage", 0.0, id="wrong"),
        pytest.param(None, 0.0, id="unanswered"),
    ],
)
def test_correctness_on_whole_answers(answer: str | None, expected: float) -> None:
    assert Correctness()(answered(answer)) == pytest.approx(expected)


def test_a_partial_answer_scores_between() -> None:
    """F1 pays for overlap, and the EM bonus is the gap to 1.0."""
    score = Correctness()(answered("Ada Byron Lovelace"))

    assert 0.0 < score < 1.0


@pytest.mark.parametrize("em_bonus", [0.0, 0.2, 1.0])
def test_an_exact_answer_scores_one_at_every_em_bonus(em_bonus: float) -> None:
    assert Correctness(em_bonus=em_bonus)(answered()) == pytest.approx(1.0)


def test_the_em_bonus_only_changes_near_misses() -> None:
    traj = answered("Ada Byron Lovelace")

    assert Correctness(em_bonus=0.0)(traj) > Correctness(em_bonus=0.5)(traj)


@pytest.mark.parametrize("em_bonus", [-0.01, 1.01])
def test_an_out_of_range_em_bonus_is_refused(em_bonus: float) -> None:
    with pytest.raises(ValueError, match="em_bonus"):
        Correctness(em_bonus=em_bonus)


def test_correctness_uses_the_shared_f1_not_its_own() -> None:
    """Agreed on #5: one implementation. If `rewards` ever grew its own F1, the
    reward would optimise a number the report does not publish."""
    from verisearch.eval.metrics import score_answer

    traj = answered("Ada Byron")
    expected = score_answer("Ada Byron", traj.gold_answers)

    assert Correctness(em_bonus=0.0)(traj) == pytest.approx(expected.f1)


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def _result(*labels: Entailment) -> FaithfulnessResult:
    claims = tuple(Claim(text=f"claim {i}", label=label) for i, label in enumerate(labels))
    entailed = sum(1 for label in labels if label is Entailment.ENTAILED)
    return FaithfulnessResult(claims=claims, score=entailed / len(labels) if labels else 0.0)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(None, 0.0, id="never-scored"),
        pytest.param(_result(), 0.0, id="no-claims"),
        pytest.param(_result(Entailment.ENTAILED), 1.0, id="all-entailed"),
        pytest.param(_result(Entailment.NEUTRAL), 0.0, id="unsupported"),
        pytest.param(_result(Entailment.ENTAILED, Entailment.NEUTRAL), 0.5, id="half"),
    ],
)
def test_faithfulness_reward(result: FaithfulnessResult | None, expected: float) -> None:
    assert faithfulness_reward(result) == pytest.approx(expected)


def test_an_unmeasured_trajectory_scores_zero_while_the_metric_abstains() -> None:
    """The asymmetry is deliberate (#2): a metric may abstain, a reward may not.
    `unsupported_rate` is None so the headline number cannot be flattered by a
    vacuous answer; the reward is 0.0 so the policy is not paid for one."""
    empty = _result()

    assert empty.unsupported_rate is None
    assert faithfulness_reward(empty) == 0.0


def test_scoring_runs_against_the_passages_the_policy_saw() -> None:
    traj = make_trajectory(
        searches=("Ada Lovelace",),
        passages_per_search=[DEFAULT_CORPUS[:1]],
        answer="Ada Lovelace was an English mathematician.",
    )

    result = score_from(traj, FakeFaithfulnessScorer())

    assert result is not None
    assert result.claims[0].supporting_doc_ids == ("d1",)
    assert result.score == pytest.approx(1.0)


def test_an_answer_with_no_passages_behind_it_is_unsupported() -> None:
    """The no-search hack, measured: answer from parametric memory and the
    faithfulness term pays nothing, because nothing was retrieved."""
    traj = make_trajectory(searches=(), answer="Ada Lovelace was an English mathematician.")

    result = score_from(traj, FakeFaithfulnessScorer())

    assert result is not None
    assert result.score == pytest.approx(0.0)
    assert faithfulness_reward(result) == 0.0


def test_an_unanswered_episode_is_not_scored_at_all() -> None:
    """The -0.5 no-answer penalty already prices it; scoring would charge twice."""
    assert score_from(answered(None), FakeFaithfulnessScorer()) is None


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_a_clean_two_hop_trajectory_is_well_formed() -> None:
    traj = make_trajectory(searches=("one", "two"), passages_per_search=[DEFAULT_CORPUS[:1]] * 2)

    assert FormatValid(search_budget=BUDGET)(traj) == 1.0


@pytest.mark.parametrize(
    ("traj", "reason"),
    [
        pytest.param(answered(None), "no answer", id="unanswered"),
        pytest.param(
            make_trajectory(answer="   ", termination=Termination.ANSWERED),
            "blank answer",
            id="blank-answer",
        ),
        pytest.param(
            Trajectory(
                question_id="q",
                question="q?",
                gold_answers=("a",),
                turns=(Turn(kind=TurnKind.THINK, text="broken"),),
                answer=None,
                termination=Termination.PARSE_ERROR,
                parse_failure=ParseFailure.NESTED_TAG,
            ),
            "parse failure",
            id="parse-error",
        ),
        pytest.param(
            make_trajectory(searches=tuple(f"q{i}" for i in range(BUDGET + 1))),
            "over budget",
            id="over-budget",
        ),
        pytest.param(
            Trajectory(
                question_id="q",
                question="q?",
                gold_answers=("a",),
                turns=(
                    Turn(kind=TurnKind.SEARCH, text="q"),
                    Turn(kind=TurnKind.ANSWER, text="a"),
                ),
                answer="a",
                termination=Termination.ANSWERED,
            ),
            "search never served",
            id="unserved-search",
        ),
    ],
)
def test_format_is_zero_when_a_structural_requirement_fails(traj: Trajectory, reason: str) -> None:
    assert FormatValid(search_budget=BUDGET)(traj) == 0.0, reason


def test_format_is_binary() -> None:
    """Section 6 keeps this weight small; a graded score here would compete with
    correctness and produce beautifully formatted wrong answers."""
    scores = {
        FormatValid(search_budget=BUDGET)(traj)
        for traj in (answered(), answered(None), make_trajectory(searches=("one",)))
    }

    assert scores <= {0.0, 1.0}


def test_a_budget_under_one_is_refused() -> None:
    with pytest.raises(ValueError, match="search_budget"):
        FormatValid(search_budget=0)


# ---------------------------------------------------------------------------
# Retrieval cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_searches", "expected"),
    [(0, 0.0), (1, 0.25), (4, 1.0), (6, 1.5)],
)
def test_retrieval_cost_is_a_normalised_count(n_searches: int, expected: float) -> None:
    """Past the budget it keeps counting: over-searching must not look free on
    the frontier's cost axis."""
    traj = make_trajectory(searches=tuple(f"q{i}" for i in range(n_searches)))

    assert retrieval_cost(traj, search_budget=BUDGET) == pytest.approx(expected)


def test_a_blank_query_still_costs() -> None:
    traj = make_trajectory(searches=("", "Ada Lovelace"))

    assert retrieval_cost(traj, search_budget=BUDGET) == pytest.approx(0.5)
    assert n_empty_queries(traj) == 1


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_the_total_is_the_sum_of_its_parts() -> None:
    traj = make_trajectory(searches=("one",), passages_per_search=[DEFAULT_CORPUS[:1]])

    breakdown = compose_reward(traj, spec(), faithfulness=_result(Entailment.ENTAILED))

    assert breakdown.total == pytest.approx(
        sum(breakdown.components.values()) + sum(breakdown.penalties.values())
    )


def test_every_term_in_section_6_appears_separately() -> None:
    """Per-component values have to survive composition or a reward hack is
    invisible in the mean."""
    breakdown = compose_reward(answered(), spec())

    assert set(breakdown.components) == {
        "correctness",
        "faithfulness",
        "format",
        "retrieval_cost",
    }
    assert set(breakdown.penalties) == {"duplicate_query", "empty_query", "no_answer"}


def test_a_perfect_grounded_answer_scores_the_weighted_sum() -> None:
    traj = make_trajectory(searches=("one",), passages_per_search=[DEFAULT_CORPUS[:1]])

    breakdown = compose_reward(traj, spec(), faithfulness=_result(Entailment.ENTAILED))

    # 1.0 correctness + 0.5 faithfulness + 0.2 format - 0.05 * (1/4) cost
    assert breakdown.total == pytest.approx(1.0 + 0.5 + 0.2 - 0.0125)


@pytest.mark.parametrize(
    ("traj", "penalty", "expected"),
    [
        pytest.param(
            make_trajectory(searches=("Ada Lovelace", "Ada Lovelace ")),
            "duplicate_query",
            -0.1,
            id="one-duplicate",
        ),
        pytest.param(
            make_trajectory(searches=("", " ")), "empty_query", -0.1, id="two-blank-queries"
        ),
        pytest.param(answered(None), "no_answer", -0.5, id="no-answer"),
    ],
)
def test_penalties_are_priced_as_section_6_says(
    traj: Trajectory, penalty: str, expected: float
) -> None:
    breakdown = compose_reward(traj, spec())

    assert breakdown.penalties[penalty] == pytest.approx(expected)


def test_the_duplicate_penalty_survives_a_trailing_space() -> None:
    """The hack pre-registered in failure_cases.md. `n_duplicate_queries`
    normalises, so re-wording whitespace does not dodge the charge."""
    traj = make_trajectory(searches=("Sam Mendes birthplace", "sam mendes birthplace!"))

    assert compose_reward(traj, spec()).penalties["duplicate_query"] == pytest.approx(-0.1)


def test_a_parse_error_is_charged_for_not_answering() -> None:
    traj = Trajectory(
        question_id="q",
        question="q?",
        gold_answers=("a",),
        turns=(Turn(kind=TurnKind.THINK, text="broken"),),
        answer=None,
        termination=Termination.PARSE_ERROR,
        parse_failure=ParseFailure.UNCLOSED_TAG,
    )

    breakdown = compose_reward(traj, spec())

    assert breakdown.penalties["no_answer"] == pytest.approx(-0.5)
    assert breakdown.total < 0.0


def test_lambda_zero_removes_the_cost_term_entirely() -> None:
    """Sweep point 1 of 4. The frontier is produced by changing this number and
    nothing else."""
    traj = make_trajectory(searches=("one", "two"))

    breakdown = compose_reward(traj, spec(lambda_retrieval_cost=0.0))

    assert breakdown.components["retrieval_cost"] == 0.0


@pytest.mark.parametrize("lambda_value", [0.0, 0.02, 0.05, 0.1])
def test_a_higher_lambda_never_raises_the_total(lambda_value: float) -> None:
    traj = make_trajectory(searches=("one", "two"))

    at_zero = compose_reward(traj, spec(lambda_retrieval_cost=0.0)).total
    at_value = compose_reward(traj, spec(lambda_retrieval_cost=lambda_value)).total

    assert at_value <= at_zero


# ---------------------------------------------------------------------------
# The properties section 6 requires
# ---------------------------------------------------------------------------


def test_correctness_dominates_searching_beats_guessing() -> None:
    """Section 6: *"Verify a no-search policy scores worse than a searching one,
    empirically, before training."* Here with the fake scorer; D3 repeats it on
    real rollouts at P2."""
    scorer = FakeFaithfulnessScorer()
    grounded = make_trajectory(
        searches=("Ada Lovelace",),
        passages_per_search=[DEFAULT_CORPUS[:1]],
        answer="Ada Lovelace was an English mathematician.",
    )
    from_memory = make_trajectory(searches=(), answer="Ada Lovelace was an English mathematician.")

    searched = compose_reward(grounded, spec(), faithfulness=score_from(grounded, scorer)).total
    guessed = compose_reward(
        from_memory, spec(), faithfulness=score_from(from_memory, scorer)
    ).total

    assert searched > guessed


def test_a_right_answer_beats_a_well_formatted_wrong_one() -> None:
    """Section 6: large format rewards produce beautifully formatted wrong
    answers. At `w_f = 0.2` they cannot outscore being right."""
    right = compose_reward(answered("Ada Lovelace"), spec()).total
    wrong = compose_reward(answered("Charles Babbage"), spec()).total

    assert right > wrong


def test_searching_within_budget_is_cheaper_than_exhausting_it() -> None:
    one = compose_reward(make_trajectory(searches=("one",)), spec()).total
    four = compose_reward(make_trajectory(searches=tuple("abcd")), spec()).total

    assert one > four


def test_an_empty_trajectory_scores_without_raising() -> None:
    """Degenerate rollouts happen at step 0 and the trainer must get a number,
    not an exception."""
    traj = Trajectory(
        question_id="q",
        question="q?",
        gold_answers=("a",),
        turns=(),
        answer=None,
        termination=Termination.MAX_TURNS,
    )

    assert compose_reward(traj, spec()).total == pytest.approx(-0.5)


def test_the_reward_is_deterministic() -> None:
    """Section 3: pure, no globals. Same trajectory, same number, always."""
    traj = make_trajectory(searches=("one",), passages_per_search=[DEFAULT_CORPUS[:1]])
    result = _result(Entailment.ENTAILED, Entailment.NEUTRAL)

    first = compose_reward(traj, spec(), faithfulness=result)
    second = compose_reward(traj, spec(), faithfulness=result)

    assert first.model_dump() == second.model_dump()


def test_the_component_form_matches_the_function() -> None:
    """`RetrievalCost` exists so the term can be logged per rollout alongside
    the others (section 9); it must not drift from the function."""
    from verisearch.rewards.retrieval_cost import RetrievalCost

    traj = make_trajectory(searches=("one", "two"))

    assert RetrievalCost(search_budget=BUDGET)(traj) == pytest.approx(
        retrieval_cost(traj, search_budget=BUDGET)
    )
    assert RetrievalCost(search_budget=BUDGET).name == "retrieval_cost"


@pytest.mark.parametrize("budget", [0, -1])
def test_a_budget_under_one_is_refused_everywhere(budget: int) -> None:
    """Dividing by it would make the cost axis infinite rather than wrong, which
    is the kind of number that reaches a plot before anyone notices."""
    with pytest.raises(ValueError, match="search_budget"):
        retrieval_cost(make_trajectory(), search_budget=budget)
    with pytest.raises(ValueError, match="search_budget"):
        RewardSpec(weights(), search_budget=budget)


def test_the_spec_takes_the_em_bonus_when_one_is_given() -> None:
    traj = answered("Ada Byron Lovelace")

    with_default = compose_reward(traj, spec()).components["correctness"]
    with_none = compose_reward(
        traj, RewardSpec(weights(), search_budget=BUDGET, em_bonus=0.0)
    ).components["correctness"]

    assert with_none > with_default


def test_an_answer_that_also_recorded_a_parse_failure_is_not_well_formed() -> None:
    """A contradictory trajectory: it answered, and it recorded a malformed turn.

    The episode loop cannot produce this today - it only sets `parse_failure` on
    the fatal path - but `Trajectory` permits it, and a non-fatal failure that
    starts being recorded (the -0.05 empty query is the candidate) would land
    here. Refusing it keeps "well formed" meaning what it says.
    """
    traj = answered().model_copy(update={"parse_failure": ParseFailure.EMPTY_QUERY})

    assert is_well_formed(traj, search_budget=BUDGET) is False
