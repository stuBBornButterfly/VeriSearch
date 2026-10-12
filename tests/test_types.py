"""Tests for the proposed shared contract.

These pin down what the computed properties in `verisearch/types.py` actually
mean. They are part of the P1 proposal: if the contract session changes the
contract, these change with it. Table-driven per PROJECT.md section 3.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from verisearch.types import (
    Claim,
    Entailment,
    FaithfulnessResult,
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
    classified = {TurnKind.THINK, TurnKind.SEARCH, TurnKind.ANSWER, TurnKind.INFO}
    assert set(TurnKind) == classified


# ---------------------------------------------------------------------------
# FaithfulnessResult.unsupported_rate - the headline metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        pytest.param((), 0.0, id="no-claims"),
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
