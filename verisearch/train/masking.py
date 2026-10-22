"""Loss masking: which tokens the policy wrote, and therefore which carry loss.

Owner: Dhrubo. The most load-bearing 60 lines in the RL half.

Retrieved passages are spliced into the model's context, but the model did not
write them. If those tokens carry loss, the policy learns to predict its own
retrieval results - it learns to hallucinate evidence. Training curves look
perfect and every number downstream is garbage. KICKOFF.md says to read that
twice; `tests/test_masking.py` is blocking in CI because of it.

This module is a pure function of a `Trajectory`, which is the whole reason
`Turn.token_ids` exists in the contract: the blocking test needs no tokenizer, no
model and no GPU, so it runs in the `[dev]`-only pipeline that gates every PR.

The mask is consumed two ways, and they must agree:

- TRL's `rollout_func` returns it as `env_mask`, which the trainer folds into
  `loss_mask = completion_mask * tool_mask` (`grpo_trainer.py:2293-2295, 2519`)
- a hand-rolled loop would multiply it into the per-token log-probs directly

Convention, fixed here and matching TRL's: **1 = the policy wrote this token and
it carries loss, 0 = the environment injected it and it must not.**
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from verisearch.types import Trajectory, Turn, TurnKind

__all__ = ["MaskingError", "Span", "completion_token_ids", "loss_mask", "spans"]

POLICY_TOKEN = 1
INJECTED_TOKEN = 0


class MaskingError(Exception):
    """Raised when a trajectory cannot be masked. Never guess - fail loud."""


class Span(BaseModel):
    """One turn's half-open range in the completion, with its mask value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: TurnKind
    start: int
    stop: int
    generated: bool

    @property
    def length(self) -> int:
        return self.stop - self.start


def _tokenized_turns(traj: Trajectory) -> list[Turn]:
    """The turns that carry ids, with the two unsafe shapes refused.

    A trajectory is either tokenized (training) or not (read for eval, text
    only). Tokenized does *not* mean every turn carries ids: one policy
    generation can cover a `<think>` and a `<search>`, and the episode loop
    attaches that generation's ids to the acting turn, because the policy never
    said where the boundary between them was. So a `<think>` turn without ids is
    normal and its tokens are already in the acting turn's range.

    An `INFO` turn without ids is a different matter, and it is the dangerous
    direction: the passages were in the model's context, so their tokens are in
    the real completion, but they would be absent from
    `completion_token_ids` - which shifts every later token left and quietly
    aligns the mask with the wrong tokens. That is the bug this module exists to
    prevent, so it raises rather than guesses.
    """
    with_ids = [turn for turn in traj.turns if turn.token_ids]
    if not with_ids:
        return []

    unmasked_injection = [
        turn
        for turn in traj.turns
        if turn.kind is TurnKind.INFO and turn.text and not turn.token_ids
    ]
    if unmasked_injection:
        msg = (
            f"trajectory {traj.question_id} is partly tokenized: {len(unmasked_injection)} INFO "
            "turn(s) carry text but no token ids while policy turns do. Those passages are in "
            "the model's context, so leaving them out of the completion shifts every later "
            "token and aligns the mask with the wrong tokens - silently. Tokenize the whole "
            "trajectory or none of it."
        )
        raise MaskingError(msg)

    if all(turn.kind is TurnKind.INFO for turn in with_ids):
        msg = (
            f"trajectory {traj.question_id} has token ids on its INFO turns only, so the "
            "completion would be entirely injected text and the loss mask entirely zero. "
            "The policy's own generations must carry their ids."
        )
        raise MaskingError(msg)

    return with_ids


def spans(traj: Trajectory) -> tuple[Span, ...]:
    """Per-turn ranges over the concatenated completion, in generation order.

    Useful on its own: a span table is how you eyeball a masking bug against a
    decoded trajectory, which section 15 asks for after every significant change.
    """
    result: list[Span] = []
    cursor = 0
    for turn in _tokenized_turns(traj):
        stop = cursor + len(turn.token_ids)
        result.append(Span(kind=turn.kind, start=cursor, stop=stop, generated=turn.generated))
        cursor = stop
    return tuple(result)


def completion_token_ids(traj: Trajectory) -> tuple[int, ...]:
    """Every token of the completion, in order. Prompt tokens are not included."""
    return tuple(token for turn in _tokenized_turns(traj) for token in turn.token_ids)


def loss_mask(traj: Trajectory) -> tuple[int, ...]:
    """The mask for `completion_token_ids(traj)`, same length, 1 or 0 per token.

    `Turn.generated` is the single source of the distinction - it is False for
    exactly `TurnKind.INFO` - so adding a turn kind forces a decision in the
    contract rather than leaving a hole here.
    """
    return tuple(
        POLICY_TOKEN if turn.generated else INJECTED_TOKEN
        for turn in _tokenized_turns(traj)
        for _ in turn.token_ids
    )
