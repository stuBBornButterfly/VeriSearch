"""**Blocking.** PROJECT.md section 3 and KICKOFF.md: injected `<info>` tokens
must carry zero loss weight.

Unmasked, the policy learns to predict its own retrieval results. It is invisible
in training curves and it poisons every number downstream, so this file is
mandatory, runs in the `[dev]`-only PR pipeline, and needs no tokenizer, no model
and no GPU - which is the argument that put `Turn.token_ids` in the contract.

Two layers, deliberately:

1. against `make_trajectory`, which is what the rest of the RL half builds on
2. against a trajectory produced by the real episode loop, so the thing actually
   trained on is the thing asserted about
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from verisearch.env.episode import EpisodeConfig, Generation, Question, run_episode
from verisearch.testing import DEFAULT_CORPUS, FakeRetriever, make_trajectory
from verisearch.train.masking import (
    INJECTED_TOKEN,
    POLICY_TOKEN,
    MaskingError,
    completion_token_ids,
    loss_mask,
    spans,
)
from verisearch.types import Termination, Trajectory, Turn, TurnKind

TOKENS = 4


def _traj(*, searches: Sequence[str] = ("Ada Lovelace",), answer: str | None = "Ada Lovelace"):
    return make_trajectory(
        searches=searches,
        passages_per_search=[DEFAULT_CORPUS[:2] for _ in searches],
        answer=answer,
        tokens_per_turn=TOKENS,
    )


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------


def test_injected_tokens_carry_zero_and_policy_tokens_carry_one() -> None:
    traj = _traj()
    mask = loss_mask(traj)

    assert len(mask) == len(completion_token_ids(traj))
    for span in spans(traj):
        expected = INJECTED_TOKEN if span.kind is TurnKind.INFO else POLICY_TOKEN
        assert set(mask[span.start : span.stop]) == {expected}, span


@pytest.mark.parametrize("n_searches", [0, 1, 2, 4])
def test_the_number_of_zeros_is_exactly_the_injected_token_count(n_searches: int) -> None:
    traj = _traj(searches=tuple(f"query {i}" for i in range(n_searches)))

    injected = sum(len(t.token_ids) for t in traj.turns if t.kind is TurnKind.INFO)
    assert mask_zeros(loss_mask(traj)) == injected == n_searches * TOKENS


def mask_zeros(mask: Sequence[int]) -> int:
    return sum(1 for value in mask if value == INJECTED_TOKEN)


def test_zeros_cover_the_info_spans_and_nothing_else() -> None:
    """A mask of the right *size* in the wrong *place* trains on injected text
    and still passes a count check. `make_trajectory` gives every turn a
    distinct id range so position can be asserted, not just arithmetic."""
    traj = _traj(searches=("one", "two"))
    ids = completion_token_ids(traj)
    mask = loss_mask(traj)

    injected_ids = {
        token for turn in traj.turns if turn.kind is TurnKind.INFO for token in turn.token_ids
    }
    masked_out = {token for token, value in zip(ids, mask, strict=True) if value == INJECTED_TOKEN}

    assert masked_out == injected_ids


def test_mask_and_ids_stay_aligned_turn_by_turn() -> None:
    traj = _traj(searches=("one", "two"))
    ids = completion_token_ids(traj)

    cursor = 0
    for turn in traj.turns:
        assert ids[cursor : cursor + len(turn.token_ids)] == turn.token_ids
        cursor += len(turn.token_ids)
    assert cursor == len(ids)


# ---------------------------------------------------------------------------
# Shapes that must not surprise the trainer
# ---------------------------------------------------------------------------


def test_a_no_search_trajectory_is_all_ones() -> None:
    """The no-search policy is a real policy - section 6 requires checking it
    scores worse - so its mask has to be well-defined, not a special case."""
    mask = loss_mask(_traj(searches=()))

    assert set(mask) == {POLICY_TOKEN}


def test_an_unanswered_trajectory_still_masks() -> None:
    traj = _traj(answer=None)

    assert mask_zeros(loss_mask(traj)) == TOKENS
    assert traj.termination is Termination.MAX_TURNS


def test_an_untokenized_trajectory_masks_to_nothing() -> None:
    """P1 reads trajectories with text only. Masking them is meaningless rather
    than wrong, and an empty mask is what a trainer would ignore."""
    traj = make_trajectory(searches=("Ada Lovelace",))

    assert loss_mask(traj) == ()
    assert completion_token_ids(traj) == ()
    assert spans(traj) == ()


def test_a_partly_tokenized_trajectory_is_refused() -> None:
    """The dangerous state: the mask would line up with the wrong tokens and
    nothing would complain. Fail loud instead (section 3)."""
    traj = _traj()
    crippled = traj.model_copy(
        update={
            "turns": tuple(
                turn.model_copy(update={"token_ids": ()}) if turn.kind is TurnKind.INFO else turn
                for turn in traj.turns
            )
        }
    )

    with pytest.raises(MaskingError, match="partly tokenized"):
        loss_mask(crippled)


def test_a_think_turn_without_ids_is_not_partial_tokenization() -> None:
    """One generation can cover `<think>` and `<search>`; the loop attaches its
    ids to the acting turn, so a think turn without ids is the normal shape and
    rejecting it would refuse every real trajectory."""
    traj = _traj()
    with_blank = traj.model_copy(
        update={"turns": (Turn(kind=TurnKind.THINK, text=""), *traj.turns)}
    )

    assert len(loss_mask(with_blank)) == len(loss_mask(traj))


# ---------------------------------------------------------------------------
# Against the real episode loop
# ---------------------------------------------------------------------------


class TwoHopPolicy:
    """Searches twice, then answers. Returns ids, like vLLM will."""

    def __init__(self) -> None:
        self._script = [
            "<think>Find the engine.</think><search>Analytical Engine</search>",
            "<think>Now the author.</think><search>Ada Lovelace notes</search>",
            "<answer>Ada Lovelace</answer>",
        ]
        self._turn = 0

    def generate(self, prompt: str, *, stop: Sequence[str], max_new_tokens: int) -> Generation:
        assert prompt
        assert stop
        assert max_new_tokens > 0
        text = self._script[self._turn]
        base = 1000 * (self._turn + 1)
        self._turn += 1
        return Generation(text=text, token_ids=tuple(range(base, base + 6)))


class WordLengthTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


def _real_trajectory() -> Trajectory:
    return run_episode(
        Question(
            question_id="q1",
            question="Who wrote the first notes on the Analytical Engine?",
            gold_answers=("Ada Lovelace",),
        ),
        TwoHopPolicy(),
        FakeRetriever(),
        EpisodeConfig(search_budget=2, max_turns=4, max_new_tokens=64, top_k=2),
        tokenizer=WordLengthTokenizer(),
    )


def test_a_real_two_hop_trajectory_masks_only_the_injected_passages() -> None:
    traj = _real_trajectory()
    mask = loss_mask(traj)
    ids = completion_token_ids(traj)

    assert traj.termination is Termination.ANSWERED
    assert len(mask) == len(ids) > 0
    for span in spans(traj):
        expected = INJECTED_TOKEN if span.kind is TurnKind.INFO else POLICY_TOKEN
        assert set(mask[span.start : span.stop]) == {expected}, span


def test_passage_text_never_appears_in_a_trained_token() -> None:
    """End to end, in the units the trainer sees: every token that came from the
    retriever's output is masked out, and every token the policy emitted is not."""
    traj = _real_trajectory()
    mask = loss_mask(traj)
    ids = completion_token_ids(traj)

    trained = {token for token, value in zip(ids, mask, strict=True) if value == POLICY_TOKEN}
    injected = {
        token for turn in traj.turns if turn.kind is TurnKind.INFO for token in turn.token_ids
    }

    assert trained.isdisjoint(injected)
    assert all(token >= 1000 for token in trained), "policy ids start at 1000 in this script"


def test_the_trained_fraction_is_reported_so_a_regression_is_visible() -> None:
    """If this ratio ever reaches 1.0 on a trajectory with searches, masking is
    off and the model is being trained on its own retrieval results."""
    traj = _real_trajectory()
    mask = loss_mask(traj)

    assert 0.0 < sum(mask) / len(mask) < 1.0
