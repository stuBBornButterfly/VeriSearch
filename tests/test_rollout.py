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
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from verisearch.config import Config, PenaltyConfig, RewardConfig, load_config
from verisearch.env.episode import EpisodeConfig, Generation, Question, TokenEpisode
from verisearch.env.protocol import render_prompt
from verisearch.rewards.compose import RewardSpec, compose_reward
from verisearch.testing import (
    DEFAULT_CORPUS,
    FakeFaithfulnessScorer,
    FakeRetriever,
    make_trajectory,
)
from verisearch.train import grpo
from verisearch.train.grpo import (
    RESOLVED_CONFIG_FILENAME,
    TrainError,
    TrainSummary,
    _as_float,
    _module,
    _question_for,
    index_prompts,
    load_question_set,
    load_questions,
    main,
    rewards_for_trajectories,
    summarize_history,
    write_resolved_config,
)
from verisearch.train.masking import completion_token_ids, loss_mask, spans
from verisearch.train.rollout import (
    RolloutError,
    RolloutPayload,
    VllmTurnPolicy,
    build_payload,
    rollout_group,
)
from verisearch.types import Termination, Trajectory, TurnKind

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTION = Question(
    question_id="q1",
    question="Who wrote the first notes on the Analytical Engine?",
    gold_answers=("Ada Lovelace",),
)


class CharTokenizer:
    """One id per character, so decoding is exact."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token) for token in token_ids)


class MergingTokenizer(CharTokenizer):
    """Greedy longest match over a few multi-character pieces, like BPE merges.

    The pieces straddle the joins the episode makes - `>` then a newline, a
    newline then `<` - so encoding a decoded transcript does *not* give back the
    ids it was spliced from. That is the property that made finding A on #31
    possible, and this tokenizer is how the tests below would catch it coming
    back: a re-encoded context differs from the spliced one.
    """

    PIECES = (">\n<", ">\n", "\n<")

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        position = 0
        while position < len(text):
            piece = next((p for p in self.PIECES if text.startswith(p, position)), None)
            if piece is None:
                ids.append(ord(text[position]))
                position += 1
            else:
                ids.append(1_000_000 + self.PIECES.index(piece))
                position += len(piece)
        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(
            self.PIECES[token - 1_000_000] if token >= 1_000_000 else chr(token)
            for token in token_ids
        )


class ScriptGenerator:
    """Stands in for TRL's `vllm_generation`: token ids in, ids and logprobs out.

    Each turn's text is encoded with the run's tokenizer, so the ids decode back
    to the script and the parser sees real tags. Records every context it is
    handed - the thing finding A was about.
    """

    def __init__(
        self, *turns: str, tokenizer: CharTokenizer | None = None, logprobs: bool = True
    ) -> None:
        self._turns = list(turns)
        self._tokenizer = tokenizer or CharTokenizer()
        self._logprobs = logprobs
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
        assert images is None
        assert num_generations == 1
        assert len(prompts) == 1
        self.calls.append(list(prompts[0]))
        ids = self._tokenizer.encode(self._turns.pop(0))
        rows: list[list[list[float | None]]] | None = None
        if self._logprobs:
            rows = [[[-0.25] for _ in ids]]
        return prompts, [ids], rows, None


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


def _rollout(
    *turns: str, tokenizer: CharTokenizer | None = None
) -> tuple[RolloutPayload, ScriptGenerator]:
    tokenizer = tokenizer or CharTokenizer()
    generator = ScriptGenerator(*turns, tokenizer=tokenizer)
    (payload,) = rollout_group(
        [QUESTION],
        FakeRetriever(),
        config(),
        policy=VllmTurnPolicy(generator, tokenizer),
        tokenizer=tokenizer,
    )
    return payload, generator


def _prefixes(payload: RolloutPayload) -> list[int]:
    """Where each generated turn starts in the completion, in generation order."""
    starts: list[int] = []
    position = 0
    for turn in payload.trajectory.turns:
        if turn.generated and turn.token_ids:
            starts.append(position)
        position += len(turn.token_ids)
    return starts


# ---------------------------------------------------------------------------
# VllmTurnPolicy
# ---------------------------------------------------------------------------


def test_the_policy_returns_ids_and_logprobs_together() -> None:
    policy = VllmTurnPolicy(ScriptGenerator("hi"), CharTokenizer())

    generation = policy.generate_ids((1, 2), stop=("</search>",), max_new_tokens=8)

    assert generation.text == "hi"
    assert generation.token_ids == (ord("h"), ord("i"))
    assert generation.logprobs == (-0.25, -0.25)


def test_the_context_reaches_the_generator_as_the_ids_it_was_given() -> None:
    """Token ids in, token ids out. Nothing re-tokenises text on the way."""
    generator = ScriptGenerator("hi")
    policy = VllmTurnPolicy(generator, CharTokenizer())

    policy.generate_ids((3, 5, 8), stop=("</search>",), max_new_tokens=8)

    assert generator.calls == [[3, 5, 8]]


def test_a_generator_without_logprobs_is_refused() -> None:
    """GRPO needs them for the importance-sampling ratio. Zero-filling would be
    a ratio of 1 on a token that was never scored."""
    policy = VllmTurnPolicy(ScriptGenerator("hi", logprobs=False), CharTokenizer())

    with pytest.raises(RolloutError, match="no logprobs"):
        policy.generate_ids((1,), stop=("</search>",), max_new_tokens=8)


def test_a_logprob_row_of_the_wrong_length_is_refused() -> None:
    class Short(ScriptGenerator):
        def generate(
            self, prompts: list[list[int]], images: None, num_generations: int
        ) -> tuple[list[list[int]], list[list[int]], list[list[list[float | None]]] | None, object]:
            return prompts, [[104, 105]], [[[-0.1]]], None

    with pytest.raises(RolloutError, match="logprobs cover 1 tokens"):
        VllmTurnPolicy(Short(), CharTokenizer()).generate_ids(
            (1,), stop=("</search>",), max_new_tokens=8
        )


def test_more_than_one_completion_per_turn_is_refused() -> None:
    class Two(ScriptGenerator):
        def generate(
            self, prompts: list[list[int]], images: None, num_generations: int
        ) -> tuple[list[list[int]], list[list[int]], list[list[list[float | None]]] | None, object]:
            return prompts, [[104], [105]], [[[-0.1]], [[-0.1]]], None

    with pytest.raises(RolloutError, match="one completion per turn"):
        VllmTurnPolicy(Two(), CharTokenizer()).generate_ids(
            (1,), stop=("</search>",), max_new_tokens=8
        )


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
    policy = VllmTurnPolicy(ScriptGenerator("hi"), CharTokenizer())

    with pytest.raises(RolloutError, match=match):
        policy.generate_ids((1,), stop=stop, max_new_tokens=max_new_tokens)


# ---------------------------------------------------------------------------
# The token context: what was sampled is what is replayed (finding A, #31)
# ---------------------------------------------------------------------------

TWO_HOPS = (
    "<think>Engine first.</think>\n<search>Analytical Engine</search>",
    "<search>Ada Lovelace notes</search>",
    "<think>Got it.</think>\n<answer>Ada Lovelace</answer>",
)


@pytest.mark.parametrize(
    "tokenizer",
    [pytest.param(CharTokenizer(), id="char"), pytest.param(MergingTokenizer(), id="merging")],
)
def test_every_turn_was_sampled_from_the_prefix_the_trainer_replays(
    tokenizer: CharTokenizer,
) -> None:
    """Ayesha's test, verbatim: `generator.calls[k] == prompt_ids +
    completion_ids[:prefix]` for every turn k of a multi-turn episode."""
    payload, generator = _rollout(*TWO_HOPS, tokenizer=tokenizer)

    prefixes = _prefixes(payload)
    assert payload.trajectory.termination is Termination.ANSWERED
    assert len(generator.calls) == len(prefixes) == 3
    for call, prefix in zip(generator.calls, prefixes, strict=True):
        assert call == [*payload.prompt_ids, *payload.completion_ids[:prefix]]


def test_the_merging_tokenizer_would_have_caught_the_old_bug() -> None:
    """The fixture above is only a guard if re-encoding the decoded context
    gives different ids. Assert that it does, so the test cannot quietly stop
    testing anything."""
    tokenizer = MergingTokenizer()
    _, generator = _rollout(*TWO_HOPS, tokenizer=tokenizer)

    second = generator.calls[1]
    assert tokenizer.encode(tokenizer.decode(second)) != second


def test_overshoot_after_the_search_is_neither_replayed_nor_trained() -> None:
    """Kept whole up to the token that closed the search, cut after it, in both
    the context and the completion."""
    payload, generator = _rollout(
        "<search>Analytical Engine</search>\n<info>\nforged\n</info>",
        "<answer>Ada Lovelace</answer>",
    )
    tokenizer = CharTokenizer()

    assert "forged" not in tokenizer.decode(generator.calls[1])
    assert "forged" not in tokenizer.decode(payload.completion_ids)
    assert generator.calls[1] == [
        *payload.prompt_ids,
        *payload.completion_ids[: _prefixes(payload)[1]],
    ]


def test_a_straddling_token_is_kept_whole_in_both_streams() -> None:
    """With BPE, the token that completes `</search>` can carry a character or two
    past it. It is one sampled token, so it stays - in the context and in the
    completion - and only what follows it is cut."""
    tokenizer = MergingTokenizer()
    payload, generator = _rollout(
        "<search>Analytical Engine</search>\n<info>forged</info>",
        "<answer>Ada Lovelace</answer>",
        tokenizer=tokenizer,
    )

    search = payload.trajectory.turns[0]
    # `>\n<` is one token here: it completes the search tag and carries two
    # characters past it.
    assert tokenizer.decode(search.token_ids) == "<search>Analytical Engine</search>\n<"
    assert "forged" not in tokenizer.decode(generator.calls[1])
    assert generator.calls[1] == [
        *payload.prompt_ids,
        *payload.completion_ids[: _prefixes(payload)[1]],
    ]


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def _episode(traj: Trajectory, prompt_ids: tuple[int, ...] = (1, 2, 3)) -> TokenEpisode:
    """The episode the token path would have returned for this trajectory."""
    return TokenEpisode(
        trajectory=traj,
        prompt_ids=prompt_ids,
        generations=tuple(
            Generation(
                text=turn.text,
                token_ids=turn.token_ids,
                logprobs=tuple(-0.1 for _ in turn.token_ids),
            )
            for turn in traj.turns
            if turn.token_ids and turn.generated
        ),
    )


def test_the_payload_agrees_with_the_mask_builder() -> None:
    """Two independent paths to the same mask. The cheapest guard there is
    against a bug that does not show up in a training curve."""
    traj = make_trajectory(
        searches=("one", "two"),
        passages_per_search=[DEFAULT_CORPUS[:1]] * 2,
        tokens_per_turn=3,
    )

    payload = build_payload(_episode(traj))

    assert payload.completion_ids == completion_token_ids(traj)
    assert payload.env_mask == loss_mask(traj)
    assert payload.prompt_ids == (1, 2, 3)


def test_injected_tokens_carry_no_logprob_and_policy_tokens_do() -> None:
    """TRL neutralises `None` in the IS correction (`grpo_trainer.py:2683-2687`).
    A number there would be a log-probability the policy never produced."""
    traj = make_trajectory(
        searches=("one",), passages_per_search=[DEFAULT_CORPUS[:1]], tokens_per_turn=2
    )

    payload = build_payload(_episode(traj))

    assert len(payload.logprobs) == len(payload.completion_ids)
    for value, mask in zip(payload.logprobs, payload.env_mask, strict=True):
        assert (value is None) == (mask == 0)


def test_a_mismatched_generation_is_refused() -> None:
    """If a generation's ids are not the turn's ids, the mask describes different
    tokens than the ones being trained on."""
    episode = _episode(make_trajectory(searches=("one",), tokens_per_turn=2))
    wrong = tuple(g.model_copy(update={"token_ids": (999, 998)}) for g in episode.generations)

    with pytest.raises(RolloutError, match="do not match"):
        build_payload(episode.model_copy(update={"generations": wrong}))


def test_too_few_generations_is_refused() -> None:
    episode = _episode(make_trajectory(searches=("one", "two"), tokens_per_turn=2))

    with pytest.raises(RolloutError, match="more generated turns"):
        build_payload(episode.model_copy(update={"generations": episode.generations[:1]}))


def test_a_generation_that_belongs_to_no_turn_is_refused() -> None:
    episode = _episode(make_trajectory(searches=("one",), tokens_per_turn=2))
    extra = (*episode.generations, Generation(text="x", token_ids=(7,), logprobs=(-0.1,)))

    with pytest.raises(RolloutError, match="belong to no turn"):
        build_payload(episode.model_copy(update={"generations": extra}))


def test_a_generation_missing_logprobs_is_refused() -> None:
    episode = _episode(make_trajectory(searches=("one",), tokens_per_turn=2))
    stripped = tuple(g.model_copy(update={"logprobs": ()}) for g in episode.generations)

    with pytest.raises(RolloutError, match="logprobs for"):
        build_payload(episode.model_copy(update={"generations": stripped}))


def test_a_payload_that_disagrees_with_the_mask_builder_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-check itself: if `train.masking` and the payload ever disagree,
    neither failure would show in a training curve, so the rollout stops."""
    from verisearch.train import rollout

    episode = _episode(make_trajectory(searches=("one",), tokens_per_turn=2))
    monkeypatch.setattr(rollout, "loss_mask", lambda traj: (1,) * len(completion_token_ids(traj)))
    with pytest.raises(RolloutError, match="env_mask disagrees"):
        build_payload(episode)

    monkeypatch.setattr(rollout, "completion_token_ids", lambda traj: (0,))
    with pytest.raises(RolloutError, match="completion disagrees"):
        build_payload(episode)


# ---------------------------------------------------------------------------
# rollout_group, against the real episode loop
# ---------------------------------------------------------------------------


def test_a_group_is_one_episode_per_prompt() -> None:
    """TRL repeats each prompt `num_generations` times before calling us, so a
    rollout is one episode and the group is the slice we were handed."""
    tokenizer = CharTokenizer()
    payloads = rollout_group(
        [QUESTION] * 4,
        FakeRetriever(),
        config(),
        policy=VllmTurnPolicy(ScriptGenerator(*["<answer>Ada Lovelace</answer>"] * 4), tokenizer),
        tokenizer=tokenizer,
    )

    assert len(payloads) == 4
    assert all(p.trajectory.termination is Termination.ANSWERED for p in payloads)
    assert all(p.trajectory.answer == "Ada Lovelace" for p in payloads)


def test_a_real_two_hop_rollout_masks_only_the_injected_passages() -> None:
    payload, _ = _rollout(*TWO_HOPS)
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


def test_the_prompt_ids_are_the_ones_the_first_turn_was_sampled_from() -> None:
    payload, generator = _rollout("<answer>Ada Lovelace</answer>")

    prompt = render_prompt(QUESTION.question, search_budget=2, top_k=2)
    assert payload.prompt_ids == tuple(CharTokenizer().encode(prompt))
    assert generator.calls[0] == list(payload.prompt_ids)


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


#: Git Bash, where `bash` on PATH is not the shell we want.
_GIT_BASH = (
    Path("C:/Program Files/Git/bin/bash.exe"),
    Path("C:/Program Files (x86)/Git/bin/bash.exe"),
)


def _posix_bash() -> str | None:
    """A bash that can run a script in this repo, or None if there is none.

    On Windows, `bash` on PATH is usually `C:/Windows/System32/bash.exe` - the
    WSL launcher - which runs in a different filesystem where neither the repo
    path nor the venv exists. That is why these tests failed on the Jenkins
    laptop while behaving correctly under Git Bash (A2 on the remaining-work
    list). Resolve the real shell instead of trusting the name.
    """
    found = shutil.which("bash")
    if found is not None and "system32" not in found.replace("\\", "/").lower():
        return found
    return next((str(path) for path in _GIT_BASH if path.is_file()), None)


def _run_script(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run `scripts/02_train.sh` the way Jenkins does: through bash, not `uv run`.

    The environment is inherited rather than replaced - the previous version set
    `PATH` to a POSIX-only value, which is meaningless on Windows - with `PYTHON`
    pointing at the interpreter running the tests, so the script does not have to
    guess, and W&B forced offline so a test never touches the network.
    """
    bash = _posix_bash()
    if bash is None:
        # Not a skip to make CI green (section 10 forbids that): the script under
        # test is a shell script, and a machine with no POSIX shell cannot run
        # it at all. Same category as the `gpu` marker.
        pytest.skip("no POSIX bash on this machine; Git Bash or a Unix shell is needed")
    return subprocess.run(
        [bash, "scripts/02_train.sh", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHON": Path(sys.executable).as_posix(),
            "WANDB_MODE": "offline",
            **env,
        },
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
def test_the_script_takes_a_sampled_set_of_question_files(tmp_path: Path) -> None:
    """Section 7's mix: several files, the same number drawn from each."""
    hotpot = _question_file(tmp_path / "h.jsonl", "h", 10)
    wiki = _question_file(tmp_path / "w.jsonl", "w", 10)

    result = _run_script("--plan", QUESTIONS=f"{hotpot.as_posix()} {wiki.as_posix()}", PER_FILE="3")

    assert result.returncode == 0, result.stderr
    assert "questions    6" in result.stdout


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
        build_payload(TokenEpisode(trajectory=empty, prompt_ids=(1, 2), generations=()))


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


# ---------------------------------------------------------------------------
# The training set (section 7: HotpotQA plus 2WikiMultihopQA, sampled)
# ---------------------------------------------------------------------------


def _question_file(path: Path, prefix: str, n: int, *, answers: bool = True) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "question_id": f"{prefix}{i}",
                    "question": f"{prefix} question {i}?",
                    "answers": ["a"] if answers else [],
                }
            )
            + "\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )
    return path


def test_a_set_reads_every_file_in_order(tmp_path: Path) -> None:
    hotpot = _question_file(tmp_path / "h.jsonl", "h", 3)
    wiki = _question_file(tmp_path / "w.jsonl", "w", 2)

    questions = load_question_set([hotpot, wiki], seed=42)

    assert [q.question_id for q in questions] == ["h0", "h1", "h2", "w0", "w1"]


def test_per_file_draws_the_same_number_from_each_file_whatever_their_sizes(
    tmp_path: Path,
) -> None:
    hotpot = _question_file(tmp_path / "h.jsonl", "h", 50)
    wiki = _question_file(tmp_path / "w.jsonl", "w", 500)

    questions = load_question_set([hotpot, wiki], per_file=10, seed=42)

    assert sum(q.question_id.startswith("h") for q in questions) == 10
    assert sum(q.question_id.startswith("w") for q in questions) == 10


def test_the_draw_is_a_function_of_the_seed(tmp_path: Path) -> None:
    """A rerun of a sweep point has to train on the same questions."""
    source = _question_file(tmp_path / "h.jsonl", "h", 200)

    first = load_question_set([source], per_file=20, seed=42)
    again = load_question_set([source], per_file=20, seed=42)
    other = load_question_set([source], per_file=20, seed=7)

    assert first == again
    assert first != other


def test_limit_keeps_the_first_n_of_the_set(tmp_path: Path) -> None:
    """The overfit-20 run: the same twenty, every time."""
    source = _question_file(tmp_path / "h.jsonl", "h", 30)

    assert [q.question_id for q in load_question_set([source], seed=1, limit=2)] == ["h0", "h1"]


@pytest.mark.parametrize(
    ("per_file", "match"),
    [pytest.param(0, "at least 1", id="zero"), pytest.param(99, "fewer than", id="too-many")],
)
def test_an_impossible_draw_is_refused(tmp_path: Path, per_file: int, match: str) -> None:
    source = _question_file(tmp_path / "h.jsonl", "h", 5)

    with pytest.raises(TrainError, match=match):
        load_question_set([source], per_file=per_file, seed=1)


def test_a_set_with_no_files_or_an_empty_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")

    with pytest.raises(TrainError, match="no question files"):
        load_question_set([], seed=1)
    with pytest.raises(TrainError, match="no questions"):
        load_question_set([empty], seed=1)


def test_a_question_without_a_gold_answer_is_refused(tmp_path: Path) -> None:
    """`Question` allows none for the policy server; training cannot score it."""
    source = _question_file(tmp_path / "h.jsonl", "h", 2, answers=False)

    with pytest.raises(TrainError, match="h0 has no gold answer"):
        load_question_set([source], seed=1)
    with pytest.raises(TrainError, match="no gold answer"):
        load_questions(source)


def test_same_text_with_different_answers_is_refused() -> None:
    """One would be scored against the other's gold answers."""
    questions = [
        Question(question_id="a", question="Who wrote it?", gold_answers=("Ada",)),
        Question(question_id="b", question="Who wrote it?", gold_answers=("Grace",)),
    ]

    with pytest.raises(TrainError, match="a and b have the same text"):
        index_prompts(questions, load_config())


def test_an_exact_duplicate_is_kept_once() -> None:
    questions = [
        Question(question_id="a", question="Who wrote it?", gold_answers=("Ada",)),
        Question(question_id="b", question="Who wrote it?", gold_answers=("Ada",)),
    ]

    by_prompt = index_prompts(questions, load_config())

    assert [q.question_id for q in by_prompt.values()] == ["a"]


# ---------------------------------------------------------------------------
# The run directory describes itself (#34)
# ---------------------------------------------------------------------------


def test_the_resolved_config_round_trips(tmp_path: Path) -> None:
    config = load_config().with_lambda(0.02)

    path = write_resolved_config(config, tmp_path / "run")

    assert path == tmp_path / "run" / RESOLVED_CONFIG_FILENAME
    reread = Config.model_validate_json(path.read_text(encoding="utf-8"))
    assert reread == config
    assert reread.reward.lambda_retrieval_cost == 0.02
    assert reread.env.search_budget == config.env.search_budget


def _fake_training(
    loss: float,
) -> tuple[list[tuple[Config, Path]], object]:
    seen: list[tuple[Config, Path]] = []

    def run_training(
        config: Config,
        questions: Sequence[Question],
        retriever: object,
        *,
        output_dir: Path,
        max_steps: int,
    ) -> TrainSummary:
        assert questions
        # Written before the trainer starts, so a run that dies still says
        # what it was.
        assert (output_dir / RESOLVED_CONFIG_FILENAME).is_file()
        seen.append((config, output_dir))
        return TrainSummary(
            steps=max_steps, final_loss=loss, mean_reward=-0.5, output_dir=str(output_dir)
        )

    return seen, run_training


def test_a_run_writes_its_config_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen, fake = _fake_training(0.1)
    monkeypatch.setattr(grpo, "run_training", fake)
    out = tmp_path / "lambda-0.02"

    code = main(["--smoke", "--steps", "1", "--out", str(out), "--lambda-cost", "0.02"])

    assert code == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    written = Config.model_validate_json((out / RESOLVED_CONFIG_FILENAME).read_text("utf-8"))
    assert written.reward.lambda_retrieval_cost == 0.02
    assert seen == [(written, out)]


def test_a_non_finite_loss_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, fake = _fake_training(float("nan"))
    monkeypatch.setattr(grpo, "run_training", fake)

    assert main(["--smoke", "--steps", "1", "--out", str(tmp_path / "run")]) == 1
    assert "FAIL: the loss is not finite" in capsys.readouterr().out


def test_plan_mode_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "run"

    assert main(["--smoke", "--plan", "--out", str(out)]) == 0
    assert not out.exists()


def test_a_real_run_reads_the_question_set_and_the_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The non-smoke path, stopped at `--plan`: two files, sampled per file."""
    hotpot = _question_file(tmp_path / "h.jsonl", "h", 30)
    wiki = _question_file(tmp_path / "w.jsonl", "w", 30)

    code = main(["--plan", "--questions", str(hotpot), str(wiki), "--per-file", "5"])

    out = capsys.readouterr().out
    assert code == 0
    assert "questions    10" in out
    assert "retriever    HttpRetriever" in out
    assert "faithfulness off" in out
    assert f"config       {load_config().fingerprint}" in out


# ---------------------------------------------------------------------------
# Faithfulness in the reward (finding B on #31)
# ---------------------------------------------------------------------------


def test_without_a_scorer_faithfulness_contributes_nothing() -> None:
    """Finding B as it stands: the term is 0.0 for every rollout, so `w_a` never
    reaches the policy. With a scorer it is `w_a` times the score."""
    supported = make_trajectory(
        searches=("Ada Lovelace",), passages_per_search=[DEFAULT_CORPUS[:2]], answer="Ada Lovelace"
    )
    scorer = FakeFaithfulnessScorer()

    (without,) = rewards_for_trajectories([supported], spec())
    (with_scorer,) = rewards_for_trajectories([supported], spec(), scorer=scorer)

    assert without == compose_reward(supported, spec()).total
    score = scorer.score("Ada Lovelace", supported.retrieved_passages).score
    assert with_scorer - without == pytest.approx(0.5 * score)
    assert score > 0.0


def test_with_a_scorer_a_supported_answer_beats_an_unsupported_one() -> None:
    """Same correctness, same searches; only what the passages say differs."""
    passages = [DEFAULT_CORPUS[:2]]
    supported = make_trajectory(
        searches=("Ada Lovelace",), passages_per_search=passages, answer="Ada Lovelace"
    )
    unsupported = make_trajectory(
        searches=("Ada Lovelace",),
        passages_per_search=passages,
        answer="Ada Lovelace, who sailed to Neptune",
    )
    scorer = FakeFaithfulnessScorer()

    scored = rewards_for_trajectories([supported, unsupported], spec(), scorer=scorer)
    unscored = rewards_for_trajectories([supported, unsupported], spec())

    assert scored[0] - scored[1] > unscored[0] - unscored[1]


def test_train_only_dependencies_import_by_name() -> None:
    """`trl`, `torch` and `peft` are imported through this, lazily, so a CPU-only
    install can still import the module and run everything above."""
    assert _module("json") is json
    with pytest.raises(TrainError, match="needs the \\[train\\] extra"):
        _module("verisearch_no_such_module")
