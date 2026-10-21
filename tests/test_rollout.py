"""Rollout tests. CPU, no `trl`, no `torch`, no GPU.

The payload this builds is what TRL trains on, so the parts that can silently
misalign a mask - the completion stream, `env_mask`, and which tokens carry a
sampling log-probability - are asserted here, in the pipeline that gates every
PR. `tests/test_masking.py` proves the mask builder; this proves the payload
agrees with it.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from verisearch.config import PenaltyConfig, RewardConfig
from verisearch.env.episode import EpisodeConfig, Generation, Question
from verisearch.env.protocol import render_prompt
from verisearch.rewards.compose import RewardSpec
from verisearch.testing import DEFAULT_CORPUS, FakeRetriever, make_trajectory
from verisearch.train.grpo import (
    TrainError,
    TrainSummary,
    _as_float,
    _question_for,
    load_questions,
    main,
    rewards_for_trajectories,
    summarize_history,
)
from verisearch.train.masking import completion_token_ids, loss_mask, spans
from verisearch.train.rollout import (
    RolloutError,
    VllmTurnPolicy,
    build_payload,
    rollout_group,
)
from verisearch.types import Termination, TurnKind

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTION = Question(
    question_id="q1",
    question="Who wrote the first notes on the Analytical Engine?",
    gold_answers=("Ada Lovelace",),
)


class FakeTokenizer:
    """Word-length ids. Deterministic, and distinct enough to spot misalignment."""

    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(token) for token in token_ids)


class FakeGenerator:
    """Stands in for TRL's `vllm_generation`: token ids in, ids and logprobs out."""

    def __init__(self, *turns: str, logprobs: bool = True) -> None:
        self._turns = list(turns)
        self._logprobs = logprobs
        self.calls: list[list[list[int]]] = []

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
        assert images is None
        assert num_generations == 1
        self.calls.append(prompts)
        text = self._turns.pop(0)
        ids = [ord(char) for char in text]
        rows: list[list[list[float | None]]] | None = None
        if self._logprobs:
            rows = [[[-0.5] for _ in ids]]
        return prompts, [ids], rows, None


class TextGenerator(FakeGenerator):
    """A generator whose decoded text is the script, so the parser sees real tags."""

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
        assert images is None
        assert num_generations == 1
        self.calls.append(prompts)
        text = self._turns.pop(0)
        ids = [ord(char) for char in text]
        return prompts, [ids], [[[-0.25] for _ in ids]], None


class CharTokenizer:
    """Round-trips `TextGenerator`'s ids, so a turn's text is what the policy wrote."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token) for token in token_ids)


def config(**overrides: int) -> EpisodeConfig:
    base = {"search_budget": 2, "max_turns": 4, "max_new_tokens": 64, "top_k": 2}
    return EpisodeConfig.model_validate(base | overrides)


def spec() -> RewardSpec:
    return RewardSpec(
        RewardConfig(
            w_correctness=1.0,
            w_faithfulness=0.5,
            w_format=0.2,
            lambda_retrieval_cost=0.05,
            penalties=PenaltyConfig(duplicate_query=-0.1, no_answer=-0.5, empty_query=-0.05),
        ),
        search_budget=2,
    )


# ---------------------------------------------------------------------------
# VllmTurnPolicy
# ---------------------------------------------------------------------------


def test_the_policy_returns_ids_and_logprobs_together() -> None:
    policy = VllmTurnPolicy(FakeGenerator("hi"), FakeTokenizer())

    generation = policy.generate("prompt text", stop=("</search>",), max_new_tokens=8)

    assert generation.token_ids == (ord("h"), ord("i"))
    assert generation.logprobs == (-0.5, -0.5)
    assert policy.records == [generation]


def test_the_prompt_is_tokenized_before_it_reaches_the_generator() -> None:
    """Token ids in, token ids out. No step in the loop re-tokenizes text."""
    generator = FakeGenerator("hi")
    policy = VllmTurnPolicy(generator, FakeTokenizer())

    policy.generate("two words", stop=("</search>",), max_new_tokens=8)

    assert generator.calls == [[[3, 5]]]


def test_a_generator_without_logprobs_is_refused() -> None:
    """GRPO needs them for the importance-sampling ratio. Zero-filling would be
    a ratio of 1 on a token that was never scored."""
    policy = VllmTurnPolicy(FakeGenerator("hi", logprobs=False), FakeTokenizer())

    with pytest.raises(RolloutError, match="no logprobs"):
        policy.generate("prompt", stop=("</search>",), max_new_tokens=8)


@pytest.mark.parametrize(
    ("stop", "max_new_tokens", "match"),
    [
        pytest.param((), 8, "stop sequence", id="no-stop"),
        pytest.param(("</search>",), 0, "max_new_tokens", id="no-budget"),
    ],
)
def test_a_misconfigured_generator_is_refused(
    stop: tuple[str, ...], max_new_tokens: int, match: str
) -> None:
    policy = VllmTurnPolicy(FakeGenerator("hi"), FakeTokenizer())

    with pytest.raises(RolloutError, match=match):
        policy.generate("prompt", stop=stop, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def _records(traj: object) -> list[Generation]:
    """The generations a policy would have recorded for this trajectory."""
    return [
        Generation(
            text=turn.text,
            token_ids=turn.token_ids,
            logprobs=tuple(-0.1 for _ in turn.token_ids),
        )
        for turn in traj.turns  # type: ignore[attr-defined]
        if turn.token_ids and turn.generated
    ]


def test_the_payload_agrees_with_the_mask_builder() -> None:
    """Two independent paths to the same mask. The cheapest guard there is
    against a bug that does not show up in a training curve."""
    traj = make_trajectory(
        searches=("one", "two"),
        passages_per_search=[DEFAULT_CORPUS[:1]] * 2,
        tokens_per_turn=3,
    )

    payload = build_payload(traj, (1, 2, 3), _records(traj))

    assert payload.completion_ids == completion_token_ids(traj)
    assert payload.env_mask == loss_mask(traj)
    assert payload.prompt_ids == (1, 2, 3)


def test_injected_tokens_carry_no_logprob_and_policy_tokens_do() -> None:
    """TRL neutralises `None` in the IS correction (`grpo_trainer.py:2683-2687`).
    A number there would be a log-probability the policy never produced."""
    traj = make_trajectory(
        searches=("one",), passages_per_search=[DEFAULT_CORPUS[:1]], tokens_per_turn=2
    )

    payload = build_payload(traj, (1,), _records(traj))

    assert len(payload.logprobs) == len(payload.completion_ids)
    for value, mask in zip(payload.logprobs, payload.env_mask, strict=True):
        assert (value is None) == (mask == 0)


def test_a_mismatched_generation_is_refused() -> None:
    """If a record's ids are not the turn's ids, the mask describes different
    tokens than the ones being trained on."""
    traj = make_trajectory(searches=("one",), tokens_per_turn=2)
    wrong = [record.model_copy(update={"token_ids": (999, 998)}) for record in _records(traj)]

    with pytest.raises(RolloutError, match="do not match"):
        build_payload(traj, (1,), wrong)


def test_too_few_records_is_refused() -> None:
    traj = make_trajectory(searches=("one", "two"), tokens_per_turn=2)

    with pytest.raises(RolloutError, match="more generated turns"):
        build_payload(traj, (1,), _records(traj)[:1])


def test_a_record_missing_logprobs_is_refused() -> None:
    traj = make_trajectory(searches=("one",), tokens_per_turn=2)
    stripped = [record.model_copy(update={"logprobs": ()}) for record in _records(traj)]

    with pytest.raises(RolloutError, match="logprobs for"):
        build_payload(traj, (1,), stripped)


# ---------------------------------------------------------------------------
# rollout_group, against the real episode loop
# ---------------------------------------------------------------------------


def test_a_group_is_one_episode_per_prompt() -> None:
    """TRL repeats each prompt `num_generations` times before calling us, so a
    rollout is one episode and the group is the slice we were handed."""
    questions = [QUESTION] * 4
    payloads = rollout_group(
        questions,
        FakeRetriever(),
        config(),
        policy_factory=lambda: VllmTurnPolicy(
            TextGenerator("<answer>Ada Lovelace</answer>"), CharTokenizer()
        ),
        tokenizer=CharTokenizer(),
    )

    assert len(payloads) == 4
    assert all(p.trajectory.termination is Termination.ANSWERED for p in payloads)
    assert all(p.trajectory.answer == "Ada Lovelace" for p in payloads)


def test_a_real_two_hop_rollout_masks_only_the_injected_passages() -> None:
    payloads = rollout_group(
        [QUESTION],
        FakeRetriever(),
        config(),
        policy_factory=lambda: VllmTurnPolicy(
            TextGenerator(
                "<search>Analytical Engine</search>",
                "<search>Ada Lovelace notes</search>",
                "<answer>Ada Lovelace</answer>",
            ),
            CharTokenizer(),
        ),
        tokenizer=CharTokenizer(),
    )
    payload = payloads[0]
    traj = payload.trajectory

    assert traj.n_searches == 2
    assert payload.env_mask == loss_mask(traj)
    # Positional, not by id value: this tokenizer is per-character, so the
    # policy's text and a passage's text legitimately share ids (a space is 32
    # either way). What must hold is that the zeros cover the INFO spans and
    # nothing else - a mask of the right size in the wrong place would train on
    # injected text and still pass a count check.
    for span in spans(traj):
        expected = 0 if span.kind is TurnKind.INFO else 1
        assert set(payload.env_mask[span.start : span.stop]) == {expected}, span
    assert 0.0 < sum(payload.env_mask) / len(payload.env_mask) < 1.0


def test_each_rollout_gets_its_own_policy() -> None:
    """Sharing one policy across rollouts is how a mask ends up describing
    another rollout's tokens."""
    made: list[VllmTurnPolicy] = []

    def factory() -> VllmTurnPolicy:
        policy = VllmTurnPolicy(TextGenerator("<answer>Ada Lovelace</answer>"), CharTokenizer())
        made.append(policy)
        return policy

    rollout_group(
        [QUESTION, QUESTION],
        FakeRetriever(),
        config(),
        policy_factory=factory,
        tokenizer=CharTokenizer(),
    )

    assert len(made) == 2
    assert all(len(policy.records) == 1 for policy in made)


# ---------------------------------------------------------------------------
# The entrypoint's importable parts
# ---------------------------------------------------------------------------


def test_rewards_are_scored_per_rollout() -> None:
    right = make_trajectory(answer="Ada Lovelace")
    wrong = make_trajectory(answer="Charles Babbage")

    scores = rewards_for_trajectories([right, wrong], spec())

    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_questions_load_from_the_dataset_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {"question_id": f"q{i}", "question": f"Question {i}?", "answers": ["a", "b"]}
            )
            for i in range(3)
        ),
        encoding="utf-8",
    )

    questions = load_questions(path, limit=2)

    assert [q.question_id for q in questions] == ["q0", "q1"]
    assert questions[0].gold_answers == ("a", "b")


def test_an_empty_question_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(TrainError, match="no questions"):
        load_questions(path)


# ---------------------------------------------------------------------------
# The script contract with Jenkins
# ---------------------------------------------------------------------------


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/02_train.sh", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHON": sys.executable, "HOME": str(REPO_ROOT)},
    )


@pytest.mark.integration
def test_the_nightly_invocation_works() -> None:
    """The exact command in the Jenkinsfile's `smoke-train` stage, plus `--plan`
    so it stops before the trainer. If this breaks, the nightly build on main
    goes red with an error that looks like a training failure."""
    result = _run_script("--smoke", "--steps", "5", "--plan")

    assert result.returncode == 0, result.stderr
    assert "steps        5" in result.stdout
    assert "retriever    FakeRetriever" in result.stdout
    assert "stopping before the trainer" in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize(
    ("args", "match"),
    [
        pytest.param(("--bogus",), "unknown flag", id="unknown-flag"),
        pytest.param(("--steps",), "positive integer", id="steps-without-a-value"),
        pytest.param(("--steps", "many"), "positive integer", id="steps-not-a-number"),
    ],
)
def test_a_bad_invocation_fails_with_the_reason(args: tuple[str, ...], match: str) -> None:
    result = _run_script(*args)

    assert result.returncode == 2
    assert match in result.stderr


# ---------------------------------------------------------------------------
# The entrypoint's CLI, in plan mode
# ---------------------------------------------------------------------------


def test_plan_mode_prints_the_run_and_changes_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--plan` is the only path through `main` that needs no GPU, and it is how
    a sweep point gets checked before it books hours of GPU time."""
    code = main(["--smoke", "--steps", "5", "--plan"])

    out = capsys.readouterr().out
    assert code == 0
    assert "steps        5" in out
    assert "retriever    FakeRetriever" in out
    assert "questions    5" in out
    assert "stopping before the trainer" in out


def test_the_lambda_flag_overrides_the_config(capsys: pytest.CaptureFixture[str]) -> None:
    """One sweep point differs from the next by this number and nothing else."""
    assert main(["--smoke", "--plan", "--lambda-cost", "0.02"]) == 0

    assert "lambda       0.02" in capsys.readouterr().out


def test_plan_mode_reports_the_kl_beta_that_will_be_used(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """TRL defaults `beta` to 0.0 - no KL term and no reference model - while
    section 5 makes it the main anti-hacking guard. Printing it is how a run
    that silently lost its KL penalty gets noticed in the log."""
    main(["--smoke", "--plan"])

    assert "kl beta      0.04" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("loss", "finite"),
    [(0.42, True), (-1.5, True), (float("nan"), False), (float("inf"), False)],
)
def test_the_summary_decides_the_nightly_gate(loss: float, finite: bool) -> None:
    """`smoke-train` asserts the loss is finite (section 10). A NaN loss on step
    one is the signal that a GRPO run has gone wrong in a way worth stopping for."""
    summary = TrainSummary(steps=5, final_loss=loss, mean_reward=0.1, output_dir="out")

    assert summary.loss_is_finite is finite


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(1, 1.0, id="int"),
        pytest.param(0.5, 0.5, id="float"),
        pytest.param(True, None, id="bool-is-not-a-number"),
        pytest.param("0.5", None, id="string"),
        pytest.param(None, None, id="missing"),
    ],
)
def test_a_log_entry_that_is_not_a_number_becomes_nan(
    value: object, expected: float | None
) -> None:
    """Rather than a number nobody measured. Section 15: never fabricate a figure -
    which includes not coercing a string out of a log into a reported loss."""
    result = _as_float(value)

    if expected is None:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_a_prompt_maps_to_exactly_one_question() -> None:
    """Rollouts are matched to their gold answers by the rendered prompt. A
    near-miss would score a rollout against another question's answers and
    nothing would raise, so the lookup is exact."""
    questions = [
        Question(question_id="a", question="Who wrote it?", gold_answers=("Ada",)),
        Question(question_id="b", question="Who wrote it, really?", gold_answers=("Grace",)),
    ]
    by_prompt = {render_prompt(q.question, search_budget=2, top_k=2): q for q in questions}

    for question in questions:
        prompt = render_prompt(question.question, search_budget=2, top_k=2)
        assert _question_for(prompt, by_prompt).question_id == question.question_id


def test_an_unknown_prompt_is_refused() -> None:
    with pytest.raises(TrainError, match="not one this run rendered"):
        _question_for("something else entirely", {})


def test_an_empty_completion_is_refused_with_our_own_message() -> None:
    """Rather than an IndexError from inside TRL, which is where this surfaced."""
    traj = make_trajectory(searches=(), answer=None, tokens_per_turn=0)
    empty = traj.model_copy(update={"turns": ()})

    with pytest.raises(RolloutError, match="empty completion"):
        build_payload(empty, (1, 2), [])


def test_the_summary_prefers_the_per_step_loss() -> None:
    history = [{"loss": 0.4, "reward": -0.5}, {"loss": 0.2, "reward": -0.1}]

    summary = summarize_history(history, 2, Path("out"))

    assert summary.final_loss == pytest.approx(0.2)
    assert summary.mean_reward == pytest.approx(-0.3)
    assert summary.loss_is_finite


def test_the_summary_falls_back_to_the_final_train_loss() -> None:
    """A run too short to log a step still has a verdict. The first real smoke
    run failed its own gate on this: logging_steps defaulted to 500, two steps
    logged no per-step loss, and an empty history reported NaN while the loop had
    run fine. A gate that fails for its own bookkeeping is worse than no gate."""
    history = [{"train_runtime": 6.1, "train_loss": 0.0, "reward": -0.5}]

    summary = summarize_history(history, 2, Path("out"))

    assert summary.final_loss == pytest.approx(0.0)
    assert summary.loss_is_finite
    assert summary.mean_reward == pytest.approx(-0.5)


def test_a_history_with_nothing_usable_is_nan_not_zero() -> None:
    """NaN fails the nightly gate, which is the right outcome for a run that
    reported nothing: section 15 forbids inventing a number, and 0.0 here would
    be a passing grade nobody measured."""
    summary = summarize_history([{"epoch": 0.4}], 2, Path("out"))

    assert math.isnan(summary.final_loss)
    assert summary.loss_is_finite is False
