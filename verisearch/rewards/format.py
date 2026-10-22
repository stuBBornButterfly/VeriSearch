"""Format validity: well-formed tags, an answer present, terminated in budget.

Owner: Dhrubo. PROJECT.md section 6: range {0, 1}, weight `w_f = 0.2`, and the
weight stays small on purpose - "large format rewards produce beautifully
formatted wrong answers".

Binary, not graded. A partial-credit format score would be a second correctness
signal competing with the real one, and the thing being rewarded here is simply
"this trajectory is usable", which is a yes or no.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisearch.types import Termination, Trajectory, TurnKind

__all__ = ["FormatValid", "is_well_formed"]


def is_well_formed(traj: Trajectory, *, search_budget: int) -> bool:
    """True when every structural requirement in section 6 holds.

    Five conditions, each one a failure mode seen in the P1 spike or named in
    section 6:

    1. the episode ended by answering
    2. nothing was left unparsed
    3. the answer is not blank
    4. it searched within budget
    5. every search it made got its passages back

    Condition 5 is what makes this more than a restatement of `termination`: a
    `SEARCH` turn with no `INFO` after it means the episode stopped mid-search,
    so the answer - if any - was written without the evidence it asked for.
    """
    if traj.termination is not Termination.ANSWERED:
        return False
    if traj.parse_failure is not None:
        return False
    if not (traj.answer or "").strip():
        return False
    if traj.n_searches > search_budget:
        return False
    return _every_search_was_served(traj)


def _every_search_was_served(traj: Trajectory) -> bool:
    kinds = [turn.kind for turn in traj.turns]
    return all(
        index + 1 < len(kinds) and kinds[index + 1] is TurnKind.INFO
        for index, kind in enumerate(kinds)
        if kind is TurnKind.SEARCH
    )


@dataclass(frozen=True, slots=True)
class FormatValid:
    """`RewardComponent` wrapper around `is_well_formed`."""

    search_budget: int
    name: str = "format"

    def __post_init__(self) -> None:
        if self.search_budget < 1:
            msg = f"search_budget must be at least 1, got {self.search_budget}"
            raise ValueError(msg)

    def __call__(self, traj: Trajectory) -> float:
        return 1.0 if is_well_formed(traj, search_budget=self.search_budget) else 0.0
