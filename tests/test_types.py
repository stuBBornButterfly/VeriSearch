"""Tests for the proposed shared contract.

These pin down what the computed properties in `verisearch/types.py` actually
mean. They are part of the P1 proposal: if the contract session changes the
contract, these change with it. Table-driven per PROJECT.md section 3.
"""

from __future__ import annotations

import collections.abc
import typing
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from verisearch.types import (
    Claim,
    Entailment,
    FaithfulnessResult,
    FaithfulnessScorer,
    ParseFailure,
    Passage,
    RewardBreakdown,
    ScoredTrajectory,
    Termination,
    Trajectory,
    Turn,
    TurnKind,
)


def _passage(doc_id: str = "d1") -> Passage:
    return Passage(doc_id=doc_id, title=f"Title {doc_id}", text=f"Text of {doc_id}")


def _trajectory(*turns: Turn, answer: str | None = "Peter Weir") -> Trajectory:
    return Trajectory(
        question_id="hpqa-0001",
        question="Which director of the 1999 Best Picture winner was born in Australia?",
        gold_answers=("Peter Weir",),
        turns=turns,
        answer=answer,
        termination=Termination.ANSWERED,
    )


# ---------------------------------------------------------------------------
# Turn.generated - the loss-masking distinction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (TurnKind.THINK, True),
        (TurnKind.SEARCH, True),
        (TurnKind.ANSWER, True),
        (TurnKind.UNPARSED, True),
        (TurnKind.INFO, False),
    ],
)
def test_generated_is_false_only_for_injected_info(kind: TurnKind, expected: bool) -> None:
    """INFO is injected by the environment, so it must not carry loss."""
    assert Turn(kind=kind, text="x").generated is expected


def test_every_turn_kind_is_covered_by_the_masking_rule() -> None:
    """A new TurnKind must force a decision about whether it carries loss.

    If someone adds a kind and forgets to classify it, the parametrize above
    silently stops being exhaustive. This test fails instead.
    """
    classified = {
        TurnKind.THINK,
        TurnKind.SEARCH,
        TurnKind.ANSWER,
        TurnKind.UNPARSED,
        TurnKind.INFO,
    }
    assert set(TurnKind) == classified


def test_a_generation_that_parsed_to_nothing_is_representable() -> None:
    """The case UNPARSED exists for: the policy answered in bare prose.

    Its tokens carry loss, it is not counted as a search, and the trajectory
    says why it went nowhere. Mirrors what the episode loop records on #28.
    """
    traj = Trajectory(
        question_id="hpqa-0003",
        question="Who directed the film?",
        gold_answers=("Peter Weir",),
        turns=(Turn(kind=TurnKind.UNPARSED, text="Peter Weir, I think.", token_ids=(1, 2, 3)),),
        answer=None,
        termination=Termination.PARSE_ERROR,
        parse_failure=ParseFailure.NO_ANSWER,
    )

    assert traj.turns[0].generated is True
    assert traj.n_searches == 0
    assert traj.retrieved_passages == ()


# ---------------------------------------------------------------------------
# FaithfulnessResult.unsupported_rate - the headline metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param((Entailment.ENTAILED,), 0.0, id="all-entailed"),
        pytest.param((Entailment.NEUTRAL,), 1.0, id="all-neutral"),
        pytest.param((Entailment.CONTRADICTED,), 1.0, id="all-contradicted"),
        pytest.param((Entailment.ENTAILED, Entailment.NEUTRAL), 0.5, id="half-unsupported"),
        pytest.param(
            (Entailment.ENTAILED, Entailment.NEUTRAL, Entailment.CONTRADICTED),
            pytest.approx(2 / 3),
            id="neutral-and-contradicted-both-count",
        ),
    ],
)
def test_unsupported_rate(labels: tuple[Entailment, ...], expected: float) -> None:
    """Anything not ENTAILED is unsupported. NEUTRAL is not a free pass."""
    result = FaithfulnessResult(
        claims=tuple(Claim(text=f"claim {i}", label=lab) for i, lab in enumerate(labels)),
        score=0.0,
    )
    assert result.unsupported_rate == expected


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_faithfulness_score_is_bounded(score: float) -> None:
    with pytest.raises(ValidationError):
        FaithfulnessResult(claims=(), score=score)


# ---------------------------------------------------------------------------
# Trajectory - search accounting that the reward depends on
# ---------------------------------------------------------------------------


def test_search_accounting_on_a_two_hop_episode() -> None:
    traj = _trajectory(
        Turn(kind=TurnKind.THINK, text="Need the film first."),
        Turn(kind=TurnKind.SEARCH, text="1999 Best Picture winner"),
        Turn(kind=TurnKind.INFO, text="...", passages=(_passage("d1"), _passage("d2"))),
        Turn(kind=TurnKind.THINK, text="Now the director's birthplace."),
        Turn(kind=TurnKind.SEARCH, text="American Beauty director birthplace"),
        Turn(kind=TurnKind.INFO, text="...", passages=(_passage("d3"),)),
        Turn(kind=TurnKind.ANSWER, text="Peter Weir"),
    )

    assert traj.n_searches == 2
    assert traj.search_queries == (
        "1999 Best Picture winner",
        "American Beauty director birthplace",
    )
    assert traj.n_duplicate_queries == 0
    assert [p.doc_id for p in traj.retrieved_passages] == ["d1", "d2", "d3"]


@pytest.mark.parametrize(
    ("queries", "n_searches", "n_duplicates"),
    [
        pytest.param((), 0, 0, id="no-searches"),
        pytest.param(("a",), 1, 0, id="single"),
        pytest.param(("a", "b"), 2, 0, id="distinct"),
        pytest.param(("a", "a"), 2, 1, id="one-repeat"),
        pytest.param(("a", "a", "a"), 3, 2, id="two-repeats"),
        pytest.param(("a", "b", "a", "b"), 4, 2, id="interleaved-repeats"),
    ],
)
def test_duplicate_query_penalty_input(
    queries: tuple[str, ...], n_searches: int, n_duplicates: int
) -> None:
    """Feeds the -0.1 duplicate-query penalty in PROJECT.md section 6."""
    traj = _trajectory(*(Turn(kind=TurnKind.SEARCH, text=q) for q in queries))
    assert traj.n_searches == n_searches
    assert traj.n_duplicate_queries == n_duplicates


def test_retrieved_passages_is_empty_when_nothing_was_retrieved() -> None:
    """A no-search policy sees no passages, so faithfulness has nothing to score."""
    traj = _trajectory(Turn(kind=TurnKind.ANSWER, text="Peter Weir"))
    assert traj.retrieved_passages == ()
    assert traj.n_searches == 0


def test_unanswered_trajectory_is_representable() -> None:
    """Needed for the -0.5 no-answer penalty."""
    traj = _trajectory(Turn(kind=TurnKind.THINK, text="..."), answer=None)
    assert traj.answer is None


# ---------------------------------------------------------------------------
# Immutability and strictness. PROJECT.md section 3.
# ---------------------------------------------------------------------------


def test_models_are_frozen() -> None:
    passage = _passage()
    with pytest.raises(ValidationError):
        passage.text = "mutated"  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    """extra="forbid" - a typo in a field name is an error, not a silent no-op."""
    with pytest.raises(ValidationError):
        Passage(doc_id="d1", title="t", text="x", scor=0.9)  # type: ignore[call-arg]


def test_passage_score_is_optional_so_fakes_can_omit_it() -> None:
    assert _passage().score is None
    assert Passage(doc_id="d1", title="t", text="x", score=0.42).score == 0.42


# ---------------------------------------------------------------------------
# Reward containers
# ---------------------------------------------------------------------------


def test_reward_breakdown_keeps_components_separable() -> None:
    """Per-component values must survive composition - they are how reward hacks
    get spotted. PROJECT.md section 6."""
    breakdown = RewardBreakdown(
        components={"correctness": 1.0, "faithfulness": 0.5, "format": 0.2},
        penalties={"duplicate_query": -0.1},
        total=1.6,
    )
    assert breakdown.components["faithfulness"] == 0.5
    assert breakdown.penalties["duplicate_query"] == -0.1


def test_scored_trajectory_composes_without_faithfulness() -> None:
    """Faithfulness is offline, so a freshly scored trajectory may not have it yet."""
    scored = ScoredTrajectory(
        trajectory=_trajectory(Turn(kind=TurnKind.ANSWER, text="Peter Weir")),
        reward=RewardBreakdown(components={"correctness": 1.0}, total=1.0),
    )
    assert scored.faithfulness is None
    assert scored.trajectory.answer == "Peter Weir"


# ---------------------------------------------------------------------------
# Protocol conformance - structural, checked by mypy, demonstrated here
# ---------------------------------------------------------------------------


def test_a_minimal_fake_satisfies_the_retriever_shape() -> None:
    """Proves Dhrubo can stand up a fake retriever in three lines and never wait
    on the FAISS index. PROJECT.md section 3, dependency inversion."""

    class FakeRetriever:
        def search(self, query: str, k: int) -> list[Passage]:
            return [_passage(f"fake-{i}") for i in range(k)]

    from verisearch.types import Retriever

    retriever: Retriever = FakeRetriever()
    assert len(retriever.search("anything", k=3)) == 3


# ---------------------------------------------------------------------------
# Review findings on PR #2. Each test names the finding it closes.
# ---------------------------------------------------------------------------


def _ended(termination: Termination, **extra: object) -> Trajectory:
    """A trajectory that ended some way other than by answering."""
    return Trajectory(
        question_id="hpqa-0002",
        question="Who directed the film?",
        gold_answers=("Peter Weir",),
        turns=(Turn(kind=TurnKind.THINK, text="..."),),
        answer=None,
        termination=termination,
        **extra,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("failure", list(ParseFailure))
def test_every_parse_failure_category_can_reach_a_reader(failure: ParseFailure) -> None:
    """Finding 1. The enum existed and no model had a field of that type.

    A category that cannot be stored cannot be counted, and section 9 asks for a
    failure breakdown while section 3 asks for a parser test per category.
    """
    traj = _ended(Termination.PARSE_ERROR, parse_failure=failure)
    assert traj.parse_failure is failure


def test_a_parse_error_must_name_which_failure() -> None:
    """Finding 1, the half that keeps the field from going unused again.

    `Termination.PARSE_ERROR` on its own says parsing failed and nothing more.
    """
    with pytest.raises(ValidationError, match="parse_failure"):
        _ended(Termination.PARSE_ERROR)


@pytest.mark.parametrize(
    "termination",
    [Termination.ANSWERED, Termination.MAX_TURNS, Termination.SEARCH_BUDGET_EXHAUSTED],
)
def test_an_episode_that_parsed_carries_no_parse_failure(termination: Termination) -> None:
    """The validator constrains one termination, not every trajectory."""
    assert _ended(termination).parse_failure is None


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        pytest.param("Sam Mendes birthplace", "Sam Mendes birthplace ", True, id="trailing-space"),
        pytest.param("Sam Mendes birthplace", "sam mendes birthplace", True, id="case"),
        pytest.param("Sam Mendes birthplace", "Sam Mendes' birthplace", True, id="punctuation"),
        pytest.param("birthplace of director", "birthplace of the director", True, id="article"),
        pytest.param("Pedro Almodovar films", "Pedro Almodóvar films", True, id="accent"),
        pytest.param("a  b", "a b", True, id="inner-whitespace"),
        pytest.param("Sam Mendes birthplace", "Sam Mendes birth year", False, id="different"),
        pytest.param("1999 winner", "1998 winner", False, id="digits-matter"),
    ],
)
def test_duplicate_queries_are_counted_on_a_normalised_form(
    first: str, second: str, *, same: bool
) -> None:
    """Finding 2. Raw string comparison makes the -0.1 penalty dodgeable.

    The hack is cheap enough for a policy to find by accident, and it would show
    up as a duplicate-query rate that falls during training while the behaviour
    behind it does not change at all.
    """
    traj = _trajectory(
        Turn(kind=TurnKind.SEARCH, text=first),
        Turn(kind=TurnKind.SEARCH, text=second),
    )
    assert traj.n_searches == 2
    assert traj.n_duplicate_queries == (1 if same else 0)


def test_normalisation_does_not_collapse_queries_that_differ_in_content() -> None:
    """The guard on the other side: over-normalising charges a real refinement.

    A false positive costs one unfair -0.1, which is the trade the aggressive
    rule accepts - but only for queries that differ in form, never in content.
    """
    traj = _trajectory(
        Turn(kind=TurnKind.SEARCH, text="American Beauty director"),
        Turn(kind=TurnKind.SEARCH, text="American Beauty director birthplace"),
        Turn(kind=TurnKind.SEARCH, text="American Beauty release year"),
    )
    assert traj.n_duplicate_queries == 0


def test_a_scorer_can_take_the_trajectorys_own_passages() -> None:
    """Finding 4. `retrieved_passages` is a tuple and the parameter took a list.

    This is the call every consumer of the contract makes, and before the change
    it needed a `list(...)` that existed only to satisfy the signature.
    """

    class TupleScorer:
        def score(self, answer: str, passages: Sequence[Passage]) -> FaithfulnessResult:
            label = Entailment.ENTAILED if passages else Entailment.NEUTRAL
            return FaithfulnessResult(
                claims=(Claim(text=answer, label=label),),
                score=1.0 if passages else 0.0,
            )

    scorer: FaithfulnessScorer = TupleScorer()
    traj = _trajectory(
        Turn(kind=TurnKind.INFO, text="...", passages=(_passage("d1"),)),
        Turn(kind=TurnKind.ANSWER, text="Peter Weir"),
    )

    result = scorer.score("Peter Weir", traj.retrieved_passages)
    assert result.score == 1.0


def test_the_faithfulness_protocol_accepts_any_sequence() -> None:
    """Finding 4, pinned against the annotation so a revert is caught here."""
    hints = typing.get_type_hints(FaithfulnessScorer.score)
    assert typing.get_origin(hints["passages"]) is collections.abc.Sequence


@pytest.mark.parametrize("field", ["components", "penalties"])
def test_reward_mappings_are_declared_immutable(field: str) -> None:
    """Finding 5. `frozen=True` stops `breakdown.components = {}` and not
    `breakdown.components["correctness"] = 9.0`.

    Pydantic still hands back a plain dict at runtime, so this is enforced by
    `mypy --strict` and not by an exception - which is exactly why the guarantee
    is pinned to the annotation here. Asserting on the runtime type would assert
    the opposite of what is wanted.
    """
    annotation = RewardBreakdown.model_fields[field].annotation
    assert typing.get_origin(annotation) is collections.abc.Mapping, (
        f"RewardBreakdown.{field} is declared {annotation}, which lets a caller "
        "mutate a frozen model's contents"
    )


def test_reward_breakdown_still_accepts_a_plain_dict() -> None:
    """The annotation change must not cost callers anything at the call site."""
    breakdown = RewardBreakdown(
        components={"correctness": 1.0, "faithfulness": 0.5},
        penalties={"duplicate_query": -0.1},
        total=1.4,
    )
    assert breakdown.components["correctness"] == 1.0
    assert breakdown.penalties["duplicate_query"] == -0.1


def test_unsupported_rate_is_none_when_there_is_nothing_to_check() -> None:
    """Finding 3. Not 0.0, which is this metric's *best* possible value.

    An answer with no checkable claims is unmeasured, not perfectly faithful.
    Returning 0.0 would put a policy that learned to answer in unfalsifiable
    language at the top of section 9's headline hallucination number, and would
    contradict `score`, which calls that same trajectory 0.0 - one trajectory
    described as both the worst and the best possible.
    """
    result = FaithfulnessResult(claims=(), score=0.0)
    assert result.unsupported_rate is None


def test_an_answer_with_claims_still_reports_a_rate() -> None:
    """None means 'nothing to check', never 'nothing was unsupported'."""
    supported = FaithfulnessResult(
        claims=(Claim(text="checkable", label=Entailment.ENTAILED),), score=1.0
    )
    assert supported.unsupported_rate == 0.0
