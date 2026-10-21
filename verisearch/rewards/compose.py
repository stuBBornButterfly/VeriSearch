"""Composition: the one place the reward formula lives.

Owner: Dhrubo. PROJECT.md section 6:

    R = w_c * correctness
      + w_a * faithfulness
      + w_f * format_valid
      - lambda * (searches / SEARCH_BUDGET)
      - P * penalties

Penalties: duplicate query -0.1 each, no answer by max turns -0.5, empty search
query -0.05 each.

Every term is kept separately in `RewardBreakdown`, because a scalar cannot show
"faithfulness went flat while correctness rose" and section 6 makes documenting
reward hacks a deliverable. The breakdown is what reaches W&B and the trajectory
dump.

Pure: no I/O, no globals, deterministic (section 3). The one input that needs the
outside world - the NLI result - arrives as data.
"""

from __future__ import annotations

from verisearch.config import RewardConfig
from verisearch.rewards.correctness import Correctness
from verisearch.rewards.faithfulness import faithfulness_reward
from verisearch.rewards.format import FormatValid
from verisearch.rewards.retrieval_cost import retrieval_cost
from verisearch.types import FaithfulnessResult, RewardBreakdown, Termination, Trajectory, TurnKind

__all__ = ["RewardSpec", "compose_reward", "n_empty_queries"]


def n_empty_queries(traj: Trajectory) -> int:
    """Searches whose query was blank. Priced at -0.05 each in section 6.

    The episode loop serves them as a search with no hits rather than refusing
    them, so they cost budget; this is what makes them cost reward too.
    """
    return sum(1 for turn in traj.turns if turn.kind is TurnKind.SEARCH and not turn.text.strip())


class RewardSpec:
    """The weights, the budget, and the components built from them.

    Built once per run from `RewardConfig` and `env.search_budget` so that the
    weights the trainer applies are the weights in the YAML. `lambda` arrives
    through `RewardConfig.lambda_retrieval_cost`, which `Config.with_lambda`
    rewrites for each sweep point - so a frontier run changes this one number and
    nothing else.
    """

    def __init__(
        self, reward: RewardConfig, *, search_budget: int, em_bonus: float | None = None
    ) -> None:
        if search_budget < 1:
            msg = f"search_budget must be at least 1, got {search_budget}"
            raise ValueError(msg)
        self.reward = reward
        self.search_budget = search_budget
        self.correctness = Correctness() if em_bonus is None else Correctness(em_bonus=em_bonus)
        self.format = FormatValid(search_budget=search_budget)


def compose_reward(
    traj: Trajectory,
    spec: RewardSpec,
    *,
    faithfulness: FaithfulnessResult | None = None,
) -> RewardBreakdown:
    """Score one trajectory. Weighted terms in `components`, subtractions in `penalties`.

    `total` is the sum of both maps, so a reader can check the arithmetic without
    knowing the formula - and a component that stops being applied shows up as a
    total that no longer matches its parts.
    """
    weights = spec.reward
    components = {
        "correctness": weights.w_correctness * spec.correctness(traj),
        "faithfulness": weights.w_faithfulness * faithfulness_reward(faithfulness),
        "format": weights.w_format * spec.format(traj),
        "retrieval_cost": -weights.lambda_retrieval_cost
        * retrieval_cost(traj, search_budget=spec.search_budget),
    }

    penalty_weights = weights.penalties
    penalties = {
        "duplicate_query": penalty_weights.duplicate_query * traj.n_duplicate_queries,
        "empty_query": penalty_weights.empty_query * n_empty_queries(traj),
        "no_answer": penalty_weights.no_answer * float(_never_answered(traj)),
    }

    total = sum(components.values()) + sum(penalties.values())
    return RewardBreakdown(components=components, penalties=penalties, total=total)


def _never_answered(traj: Trajectory) -> bool:
    """True when the episode produced no answer, however it ended.

    Keyed on the answer rather than on `termination is MAX_TURNS`: a parse error
    and an exhausted search budget also leave the question unanswered, and
    section 6's -0.5 is for not answering, not for one particular way of failing
    to.
    """
    return not (traj.termination is Termination.ANSWERED and bool((traj.answer or "").strip()))
