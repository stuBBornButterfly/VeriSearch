"""Finding A on #31, checked with Qwen's real tokenizer.

`tests/test_rollout.py` proves the token-context invariant with stand-in
tokenizers, in the pipeline that gates every PR. This file repeats it with the
tokenizer training actually uses, because the bug was a property of BPE merges
across the joins the episode makes, and a stand-in only approximates those.

`slow`, and skipped without `transformers` or without the tokenizer in the
local Hugging Face cache - CI has neither, and a test that downloads a model is
not a unit test. Run it on the GPU box after any `trl` or `transformers` bump:

    uv run pytest tests/test_token_context_qwen.py -m slow
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pytest

from verisearch.env.episode import EpisodeConfig, Question
from verisearch.testing import FakeRetriever
from verisearch.train.rollout import RolloutPayload, VllmTurnPolicy, rollout_group
from verisearch.types import Termination

pytestmark = pytest.mark.slow

transformers = pytest.importorskip("transformers")

POLICY_REPO = "Qwen/Qwen2.5-1.5B-Instruct"

QUESTION = Question(
    question_id="q1",
    question="Who wrote the first notes on the Analytical Engine?",
    gold_answers=("Ada Lovelace",),
)

# The first turn overshoots its search with a forged `<info>`, as a policy that
# ignores the stop string would. `</search>\n` ends on Qwen's `>\n` token, which
# completes the stop string and carries a newline past it.
SCRIPT = (
    "<think>Find the engine.</think>\n<search>Analytical Engine</search>\n<info>\nforged\n</info>",
    "<search>Ada Lovelace notes</search>",
    "<think>Got it.</think>\n<answer>Ada Lovelace</answer><|im_end|>",
)


class _Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class _QwenScript:
    """Generates the script as Qwen ids, recording every context it is handed."""

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._turns = list(SCRIPT)
        self.calls: list[list[int]] = []

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
    ]:
        self.calls.append(list(prompts[0]))
        ids = self._tokenizer.encode(self._turns.pop(0))
        return prompts, [ids], [[[-0.1] for _ in ids]], None


@pytest.fixture(scope="module")
def tokenizer() -> _Tokenizer:
    try:
        loaded: _Tokenizer = transformers.AutoTokenizer.from_pretrained(
            POLICY_REPO, local_files_only=True
        )
    except OSError:
        pytest.skip(f"{POLICY_REPO}'s tokenizer is not in the local Hugging Face cache")
    return loaded


@pytest.fixture(scope="module")
def rollout(tokenizer: _Tokenizer) -> tuple[RolloutPayload, _QwenScript]:
    generator = _QwenScript(tokenizer)
    (payload,) = rollout_group(
        [QUESTION],
        FakeRetriever(),
        EpisodeConfig(search_budget=2, max_turns=4, max_new_tokens=64, top_k=2),
        policy=VllmTurnPolicy(generator, tokenizer),
        tokenizer=tokenizer,
    )
    return payload, generator


def _prefixes(payload: RolloutPayload) -> list[int]:
    starts: list[int] = []
    position = 0
    for turn in payload.trajectory.turns:
        if turn.generated and turn.token_ids:
            starts.append(position)
        position += len(turn.token_ids)
    return starts


def test_every_turn_was_sampled_from_the_prefix_the_trainer_replays(
    rollout: tuple[RolloutPayload, _QwenScript],
) -> None:
    payload, generator = rollout

    assert payload.trajectory.termination is Termination.ANSWERED
    prefixes = _prefixes(payload)
    assert len(generator.calls) == len(prefixes) == 3
    for call, prefix in zip(generator.calls, prefixes, strict=True):
        assert call == [*payload.prompt_ids, *payload.completion_ids[:prefix]]


def test_re_encoding_the_text_would_have_changed_the_context(
    rollout: tuple[RolloutPayload, _QwenScript],
    tokenizer: _Tokenizer,
) -> None:
    """What the first version did, and why it was wrong with this tokenizer:
    encoding the decoded context merges tokens across the joins (the prompt's
    last `?` with the first `<think>`, for one), so the trainer would have
    replayed different ids from the ones sampled."""
    _, generator = rollout

    second = generator.calls[1]
    assert tokenizer.encode(tokenizer.decode(second)) != second


def test_the_stop_token_is_kept_whole_and_the_overshoot_is_cut(
    rollout: tuple[RolloutPayload, _QwenScript],
    tokenizer: _Tokenizer,
) -> None:
    payload, generator = rollout

    search = payload.trajectory.turns[1]
    assert tokenizer.decode(search.token_ids).endswith("</search>\n")
    assert "forged" not in tokenizer.decode(generator.calls[1])
    assert "forged" not in tokenizer.decode(payload.completion_ids)
