"""Faithfulness: the fraction of the answer's claims the retrieved passages support.

Owner: Dhrubo, consuming Ayesha's `FaithfulnessScorer`. PROJECT.md section 6:
range [0, 1], weight `w_a = 0.5`. The anti-hallucination lever, and the term most
comparable work skips.

**Why this is not a `RewardComponent`.** The protocol is
`__call__(traj) -> float`, and faithfulness cannot be computed from a trajectory
alone - it needs NLI output that a `Trajectory` does not carry, produced offline
against the passages the policy actually saw. KICKOFF.md is explicit that no LLM
judge runs in the training hot loop. So the dependency enters `compose` as data
(`FaithfulnessResult | None`), exactly as `ScoredTrajectory` stores it, and this
module stays a pure function of that data.

`score_from` exists so the scorer can be driven over a finished group of rollouts
in one place, between rollout and reward.
"""

from __future__ import annotations

from verisearch.types import FaithfulnessResult, FaithfulnessScorer, Trajectory

__all__ = ["faithfulness_reward", "score_from"]


def faithfulness_reward(result: FaithfulnessResult | None) -> float:
    """The reward term. `None` - never scored - is 0.0.

    `None` arrives two ways, and both deserve 0.0 rather than a free pass: the
    episode never answered, or the scorer has not run yet. Treating "unmeasured"
    as "faithful" would let a policy collect `w_a` for producing nothing
    checkable, which is the safe harbour argued down on #2 when `score` for an
    empty claim set was fixed at 0.0.

    Note the asymmetry with `FaithfulnessResult.unsupported_rate`, which returns
    `None` for an unmeasured trajectory so the headline metric cannot be
    flattered by vacuous answers. A metric may abstain; a reward may not - the
    policy gets a number either way, and 0.0 is the honest one.
    """
    return 0.0 if result is None else result.score


def score_from(
    traj: Trajectory,
    scorer: FaithfulnessScorer,
) -> FaithfulnessResult | None:
    """Score one finished trajectory offline. `None` when there is nothing to score.

    Scored against `traj.retrieved_passages` - what was in context when the
    answer was written - and never against the gold passages, which is what makes
    the term verifiable rather than a second correctness signal.

    An unanswered episode returns `None`: the no-answer penalty in section 6
    already prices it at -0.5, and asking an NLI model about a missing answer
    would charge it twice.
    """
    if traj.answer is None:
        return None
    return scorer.score(traj.answer, traj.retrieved_passages)
