"""Correctness: token F1 against the gold answers, plus an EM bonus.

Owner: Dhrubo. PROJECT.md section 6: range [0, 1], weight `w_c = 1.0`, and it
must dominate - a no-search policy has to score worse than a searching one.

**One F1 in the project, not two.** The scoring primitives come from
`eval.metrics`, which is Ayesha's. Agreed on #5: if the reward computed its own
F1, training would optimise one number while the report claimed another, and
nothing would fail - the gap is invisible. `yes_no_strict` is the concrete case:
a reward paying partial credit for "yes-ish" answers that eval scores zero.

That makes `rewards/` import from `eval/`, which crosses the ownership line in
section 0. Taken deliberately: `eval.metrics` is pure, so the import costs the
"no I/O, no globals, deterministic" rule in section 3 nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from verisearch.eval.metrics import score_answer
from verisearch.types import Trajectory

__all__ = ["Correctness"]

#: How much of the score an exact match buys over a token-level match.
#:
#: Not a config field yet, because `RewardConfig` has none and that file is
#: Ayesha's: flagged on #3 to move there once the value has been swept. Kept as
#: a constructor argument meanwhile so a test can vary it without patching.
#:
#: 0.2 so that EM is worth having without making near-misses worthless: an
#: answer with F1 0.8 scores 0.64 and an exact one 1.0.
DEFAULT_EM_BONUS = 0.2


@dataclass(frozen=True, slots=True)
class Correctness:
    """`RewardComponent`: `(1 - b) * F1 + b * EM`, which stays inside [0, 1]."""

    em_bonus: float = DEFAULT_EM_BONUS
    yes_no_strict: bool = True
    name: str = "correctness"

    def __post_init__(self) -> None:
        if not 0.0 <= self.em_bonus <= 1.0:
            msg = f"em_bonus must be in [0, 1], got {self.em_bonus}"
            raise ValueError(msg)

    def __call__(self, traj: Trajectory) -> float:
        """Score the trajectory's answer. An unanswered episode scores 0.0.

        `score_answer` treats `None` as wrong rather than missing, which is the
        behaviour the reward wants: dropping unanswered episodes would pay a
        policy for refusing hard questions.
        """
        score = score_answer(traj.answer, traj.gold_answers, yes_no_strict=self.yes_no_strict)
        return (1.0 - self.em_bonus) * score.f1 + self.em_bonus * score.exact_match
