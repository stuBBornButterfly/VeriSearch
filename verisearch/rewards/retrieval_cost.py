"""Retrieval cost: the dial the whole experiment turns on.

Owner: Dhrubo. PROJECT.md section 6: `searches / SEARCH_BUDGET`, subtracted, with
`lambda` swept over {0.0, 0.02, 0.05, 0.1}. Range [0, inf): a policy that blows
through its budget pays more than 1.0, which is intended - the frontier's cost
axis has to keep counting past the budget or over-searching looks free.

KICKOFF.md: lambda is the experiment, not a hyperparameter. Each value is a point
on the accuracy-versus-retrieval frontier, so this function stays a plain
normalised count with no shaping, no discount for cache hits and no forgiveness
for empty queries. Anything cleverer here would change what the frontier means.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisearch.types import Trajectory

__all__ = ["RetrievalCost", "retrieval_cost"]


def retrieval_cost(traj: Trajectory, *, search_budget: int) -> float:
    """Searches normalised by the budget. Every search counts, including the
    blank ones: the call was made and the policy spent a turn on it."""
    if search_budget < 1:
        msg = f"search_budget must be at least 1, got {search_budget}"
        raise ValueError(msg)
    return traj.n_searches / search_budget


@dataclass(frozen=True, slots=True)
class RetrievalCost:
    """`RewardComponent` form. The weight (`lambda`) is applied by `compose`."""

    search_budget: int
    name: str = "retrieval_cost"

    def __call__(self, traj: Trajectory) -> float:
        return retrieval_cost(traj, search_budget=self.search_budget)
