"""Batched multi-turn rollout: the function TRL calls instead of generating itself.

Owner: Dhrubo. PROJECT.md section 5, "the hard part": standard GRPO is
single-turn - prompt in, completion out - and ours generates, pauses for
retrieval, continues, all as one trajectory.

TRL's hook makes that ours to drive. `GRPOTrainer(rollout_func=...)` hands us the
prompts for this step and expects back `prompt_ids`, `completion_ids` and
`logprobs` (`trl/trainer/grpo_trainer.py:258, 2242`); any extra key is forwarded
to the reward functions. We also return `env_mask`, which TRL folds into
`loss_mask = completion_mask * tool_mask` (`2293-2295, 2519`) - that is the
masking guarantee, and it is a mask we build.

**Nothing in this module imports `trl`, `torch` or `vllm`.** The generator and
the tokenizer arrive as protocols, so the whole payload-assembly path - the part
that can silently misalign a mask - is unit-tested on CPU in the `[dev]`-only
pipeline that gates every PR. `train/grpo.py` is the only place that touches the
trainer.

Two invariants, both asserted rather than assumed:

1. the completion stream and the mask agree with `train.masking`, which is what
   `tests/test_masking.py` blocks on
2. every policy token carries a sampling log-probability and every injected
   token carries `None` - TRL neutralises those in the importance-sampling
   correction (`2683-2687`), and a number there would be a log-probability the
   policy never produced
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from verisearch.env.episode import EpisodeConfig, Generation, Question, run_episode
from verisearch.env.protocol import render_prompt
from verisearch.train.masking import INJECTED_TOKEN, POLICY_TOKEN, completion_token_ids, loss_mask
from verisearch.types import Retriever, Trajectory

__all__ = [
    "RolloutError",
    "RolloutPayload",
    "TokenGenerator",
    "Tokenizer",
    "VllmTurnPolicy",
    "build_payload",
    "rollout_group",
]


class RolloutError(Exception):
    """Raised when a rollout cannot be assembled into a trainable payload."""


class Tokenizer(Protocol):
    """Enough of a tokenizer for one episode. `transformers` satisfies it."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class TokenGenerator(Protocol):
    """TRL's `vllm_generation.generate`, narrowed to what a turn needs.

    Token ids in, token ids out: the prompt is already tokenized and the
    completion comes back as ids with their sampling log-probabilities, so no
    step in the loop re-tokenizes decoded text.
    """

    def generate(
        self,
        prompts: list[list[int]],
        images: None,
        num_generations: int,
    ) -> tuple[
        list[list[int]],
        list[list[int]],
        list[list[list[float | None]]] | None,
        object,
    ]: ...


class VllmTurnPolicy:
    """A `Policy` over a `TokenGenerator`, recording what it generated.

    One generator call per turn, one completion per call: the trainer already
    repeated each prompt `num_generations` times (`RepeatSampler`,
    `grpo_trainer.py:1281`) and colocate generation uses `n=1`
    (`generation/vllm_generation.py:663`), so a rollout is one episode and the
    group is formed by the prompts TRL hands us. Confirmed by the P1 spike run.

    The records are kept because `Trajectory` has nowhere to put a
    log-probability - and should not: it is a training artefact, not something
    eval reads.
    """

    def __init__(self, generator: TokenGenerator, tokenizer: Tokenizer) -> None:
        self._generator = generator
        self._tokenizer = tokenizer
        self.records: list[Generation] = []

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        """Generate one turn. `stop` and `max_new_tokens` are the generator's own
        configuration in TRL (`SamplingParams` / `max_completion_length`), so they
        are checked here rather than silently ignored."""
        if not stop:
            msg = "the generator must be configured with at least one stop sequence"
            raise RolloutError(msg)
        if max_new_tokens < 1:
            msg = f"max_new_tokens must be positive, got {max_new_tokens}"
            raise RolloutError(msg)

        prompt_ids = self._tokenizer.encode(prompt)
        _, completions, logprobs, _ = self._generator.generate(
            [prompt_ids], None, num_generations=1
        )
        if len(completions) != 1:
            msg = f"expected one completion per turn, got {len(completions)}"
            raise RolloutError(msg)

        token_ids = tuple(completions[0])
        generation = Generation(
            text=self._tokenizer.decode(token_ids),
            token_ids=token_ids,
            logprobs=_turn_logprobs(logprobs, token_ids),
        )
        self.records.append(generation)
        return generation


def _turn_logprobs(
    logprobs: list[list[list[float | None]]] | None,
    token_ids: tuple[int, ...],
) -> tuple[float | None, ...]:
    """Keep the sampled token's log-probability, which is vLLM's top-1.

    TRL does the same reduction on its own path (`grpo_trainer.py:1841`). A
    missing or short row is an error rather than a pad: a silently zero-filled
    log-probability is an importance-sampling ratio of 1 on a token that was
    never scored.
    """
    if logprobs is None:
        msg = "the generator returned no logprobs; GRPO needs them for the IS ratio"
        raise RolloutError(msg)
    row = logprobs[0]
    if len(row) != len(token_ids):
        msg = f"logprobs cover {len(row)} tokens but the completion has {len(token_ids)}"
        raise RolloutError(msg)
    return tuple(entry[0] if entry else None for entry in row)


class RolloutPayload(BaseModel):
    """One trainable rollout: the trajectory, its token stream, mask and logprobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trajectory: Trajectory
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    env_mask: tuple[int, ...]
    logprobs: tuple[float | None, ...]


def build_payload(
    trajectory: Trajectory,
    prompt_ids: Sequence[int],
    records: Sequence[Generation],
) -> RolloutPayload:
    """Assemble one rollout, cross-checking against `train.masking`.

    The mask is built here from the turns and then compared with `loss_mask`,
    which is the function `tests/test_masking.py` blocks on. Two independent
    paths to the same mask is the cheapest guard there is against the bug that
    does not show up in a training curve.
    """
    completion: list[int] = []
    mask: list[int] = []
    logprobs: list[float | None] = []
    queue = list(records)

    for turn in trajectory.turns:
        if not turn.token_ids:
            continue
        completion.extend(turn.token_ids)
        if not turn.generated:
            mask.extend([INJECTED_TOKEN] * len(turn.token_ids))
            logprobs.extend([None] * len(turn.token_ids))
            continue
        if not queue:
            msg = (
                f"trajectory {trajectory.question_id} has more generated turns with token ids "
                "than the policy recorded generations"
            )
            raise RolloutError(msg)
        record = queue.pop(0)
        if record.token_ids != turn.token_ids:
            msg = (
                f"trajectory {trajectory.question_id}: a generated turn's ids do not match the "
                "generation they came from, so the mask would describe different tokens"
            )
            raise RolloutError(msg)
        if len(record.logprobs) != len(record.token_ids):
            msg = (
                f"trajectory {trajectory.question_id}: {len(record.logprobs)} logprobs for "
                f"{len(record.token_ids)} generated tokens"
            )
            raise RolloutError(msg)
        mask.extend([POLICY_TOKEN] * len(turn.token_ids))
        logprobs.extend(record.logprobs)

    if not completion:
        # TRL indexes `ids[-1]` on every completion to decide whether it was
        # truncated (`grpo_trainer.py:2325`), so an empty one is an IndexError
        # inside the trainer rather than a bad rollout here. The episode loop
        # keeps even an unparseable generation, so reaching this means the
        # generator produced no tokens at all - a misconfiguration, not a
        # policy behaviour.
        msg = (
            f"trajectory {trajectory.question_id} has an empty completion: the generator "
            "returned no tokens for any turn, so there is nothing to train on"
        )
        raise RolloutError(msg)

    _check_against_masking(trajectory, completion, mask)
    return RolloutPayload(
        trajectory=trajectory,
        prompt_ids=tuple(prompt_ids),
        completion_ids=tuple(completion),
        env_mask=tuple(mask),
        logprobs=tuple(logprobs),
    )


def _check_against_masking(
    trajectory: Trajectory, completion: Sequence[int], mask: Sequence[int]
) -> None:
    if tuple(completion) != completion_token_ids(trajectory):
        msg = (
            f"trajectory {trajectory.question_id}: the assembled completion disagrees with "
            "train.masking.completion_token_ids"
        )
        raise RolloutError(msg)
    if tuple(mask) != loss_mask(trajectory):
        msg = (
            f"trajectory {trajectory.question_id}: the assembled env_mask disagrees with "
            "train.masking.loss_mask - one of the two is wrong and neither failure is visible "
            "in a training curve"
        )
        raise RolloutError(msg)


def rollout_group(
    questions: Sequence[Question],
    retriever: Retriever,
    config: EpisodeConfig,
    *,
    policy_factory: Callable[[], VllmTurnPolicy],
    tokenizer: Tokenizer,
) -> list[RolloutPayload]:
    """Run one episode per question and return their payloads, in order.

    The prompts TRL hands `rollout_func` are already repeated `num_generations`
    times, so "one group" here is simply the slice it gave us: equal questions
    land next to each other and the advantage is computed over them by the
    trainer. A fresh policy per rollout keeps each one's generation records its
    own - sharing a policy across rollouts is how a mask ends up describing
    another rollout's tokens.
    """
    payloads: list[RolloutPayload] = []
    for question in questions:
        policy = policy_factory()
        trajectory = run_episode(question, policy, retriever, config, tokenizer=tokenizer)
        prompt_ids = tokenizer.encode(
            render_prompt(question.question, search_budget=config.search_budget, top_k=config.top_k)
        )
        payloads.append(build_payload(trajectory, prompt_ids, policy.records))
    return payloads
