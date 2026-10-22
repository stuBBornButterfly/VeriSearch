"""GRPO training entrypoint. The only module in the project that touches `trl`.

Owner: Dhrubo. PROJECT.md section 11, P3: overfit 20 questions, then a full 1.5B
run that beats the base model on held-out F1.

Route and reasons: `writeup/rl_method.md`. In one line - TRL's `GRPOTrainer` with
our `rollout_func` and our `env_mask`, because that gives us the multi-turn loop
and per-token mask control while TRL keeps weight sync, the
importance-sampling correction and the loss variants.

**`trl`, `torch` and `peft` are imported lazily, inside functions.** They live in
the `[train]` extra, which the PR pipeline never installs (section 10), so
importing them at module scope would turn `mypy` and collection red on every
CPU-only run. Everything that can be tested without a GPU lives in
`train/rollout.py` and `train/masking.py` instead, and is.

Config comes from YAML and nothing here invents a number: weights and budget from
`RewardConfig` and `EnvConfig`, KL beta and group size from `TrainConfig`, LoRA
from `LoraConfig`. `lambda` arrives through `Config.with_lambda`, which is how one
sweep point differs from the next.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from verisearch.config import Config, load_config
from verisearch.env.episode import EpisodeConfig, Question
from verisearch.env.protocol import check_stop_sequences, render_prompt
from verisearch.rewards.compose import RewardSpec, compose_reward
from verisearch.train.rollout import (
    TokenGenerator,
    Tokenizer,
    VllmTurnPolicy,
    rollout_group,
)
from verisearch.types import Retriever, Trajectory

__all__ = [
    "TrainError",
    "TrainSummary",
    "load_questions",
    "main",
    "rewards_for_trajectories",
    "summarize_history",
]

#: TRL folds our `env_mask` into the loss mask under this key.
ENV_MASK_KEY = "env_mask"


# TRL ships no type stubs, so `GRPOTrainer` and friends are untyped to mypy. These
# protocols describe the three attributes this module touches, which keeps the
# file free of `# type: ignore` - success criterion 9 is zero ignores outside
# pyproject.toml, and an ignore here would hide a real API change at the next
# `trl` bump.


class _GeneratorHolder(Protocol):
    """What `rollout_func` needs from the trainer it is handed."""

    @property
    def vllm_generation(self) -> TokenGenerator: ...


class _TrainerArgs(Protocol):
    beta: float


class _TrainerState(Protocol):
    log_history: Sequence[Mapping[str, object]]


class _Trainer(Protocol):
    args: _TrainerArgs
    state: _TrainerState

    def train(self) -> None: ...


class TrainError(Exception):
    """Raised when training cannot start, or finished in a state worth refusing."""


class TrainSummary(BaseModel):
    """What a finished run reports. Checked by the nightly smoke stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: int
    final_loss: float
    mean_reward: float
    output_dir: str

    @property
    def loss_is_finite(self) -> bool:
        """The nightly `smoke-train` stage asserts exactly this (section 10)."""
        return math.isfinite(self.final_loss)


def _module(name: str) -> ModuleType:
    """Import a `[train]`-only dependency, with an error that says what to install."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - requires the [train] extra to be absent
        msg = (
            f"{name} is not installed. Training needs the [train] extra and a GPU: "
            "uv sync --extra dev --extra train"
        )
        raise TrainError(msg) from exc


def load_questions(path: str | Path, *, limit: int | None = None) -> list[Question]:
    """Read questions from the JSONL `data.write_jsonl` produces.

    Only the three fields an episode needs are taken, so this stays readable
    against `QAExample` without importing the dataset loader's own validation.
    """
    questions: list[Question] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            questions.append(
                Question(
                    question_id=str(row["question_id"]),
                    question=str(row["question"]),
                    gold_answers=tuple(str(answer) for answer in row["answers"]),
                )
            )
            if limit is not None and len(questions) >= limit:
                break
    if not questions:
        msg = f"{path} produced no questions"
        raise TrainError(msg)
    return questions


def rewards_for_trajectories(
    trajectories: Sequence[Trajectory],
    spec: RewardSpec,
) -> list[float]:
    """Score a group of rollouts. The reward function TRL calls, minus the plumbing.

    Faithfulness is absent here: it is scored offline against the retrieved
    passages, never inside the hot loop (KICKOFF.md), so the term enters at P2's
    scoring pass rather than during generation. `compose_reward` keeps it as its
    own component, so a run without it is readable as such rather than looking
    like a faithfulness of zero that someone measured.
    """
    return [compose_reward(traj, spec).total for traj in trajectories]


def run_training(  # pragma: no cover - needs the [train] extra and a GPU
    config: Config,
    questions: Sequence[Question],
    retriever: Retriever,
    *,
    output_dir: Path,
    max_steps: int,
    tokenizer: Tokenizer | None = None,
) -> TrainSummary:
    """Wire TRL to our rollout and our reward, run, and report.

    Deliberately one function: the trainer objects are untyped to `mypy` (no
    stubs for `trl`), so keeping them inside a single call means nothing
    annotated ever returns one.
    """
    trl = _module("trl")
    peft = _module("peft")
    transformers = _module("transformers")

    check_stop_sequences(config.env.stop_sequences)
    episode_config = EpisodeConfig.from_config(config)
    spec = RewardSpec(config.reward, search_budget=config.env.search_budget)

    hf_tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(
        config.model.policy_repo, padding_side="left"
    )

    # Rollout: TRL hands us the prompts for this step, already repeated
    # num_generations times, and expects token ids back.
    # Keyed on the rendered prompt, which is exactly what the dataset holds:
    # with `rollout_func`, TRL forwards the dataset's prompts unchanged
    # (`grpo_trainer.py:2239`). Matching on a substring instead would let two
    # questions that share a prefix swap rollouts, and the reward would then be
    # scored against the wrong gold answers - wrong, and silent.
    by_prompt = {
        render_prompt(
            question.question,
            search_budget=config.env.search_budget,
            top_k=config.retrieval.top_k,
        ): question
        for question in questions
    }

    def rollout_func(prompts: list[str], trainer: _GeneratorHolder) -> dict[str, object]:
        generator = trainer.vllm_generation
        batch = [_question_for(prompt, by_prompt) for prompt in prompts]
        payloads = rollout_group(
            batch,
            retriever,
            episode_config,
            policy_factory=lambda: VllmTurnPolicy(generator, hf_tokenizer),
            tokenizer=hf_tokenizer,
        )
        return {
            "prompt_ids": [list(p.prompt_ids) for p in payloads],
            "completion_ids": [list(p.completion_ids) for p in payloads],
            "logprobs": [list(p.logprobs) for p in payloads],
            ENV_MASK_KEY: [list(p.env_mask) for p in payloads],
            "trajectory": [p.trajectory.model_dump_json() for p in payloads],
        }

    def reward_func(
        completions: list[str], trajectory: list[str], **_kwargs: object
    ) -> list[float]:
        del completions  # the trajectory is the richer view of the same rollout
        return rewards_for_trajectories(
            [Trajectory.model_validate_json(payload) for payload in trajectory], spec
        )

    trainer: _Trainer = trl.GRPOTrainer(
        model=config.model.policy_repo,
        reward_funcs=reward_func,
        args=trl.GRPOConfig(
            output_dir=str(output_dir),
            max_steps=max_steps,
            num_generations=config.train.group_size,
            per_device_train_batch_size=config.train.group_size,
            learning_rate=config.train.learning_rate,
            # TRL defaults beta to 0.0, which loads no reference model at all.
            # PROJECT.md section 5 makes KL the main anti-hacking guard, so this
            # one is asserted below rather than trusted.
            beta=config.train.kl_beta,
            bf16=config.model.dtype == "bfloat16",
            max_completion_length=config.env.max_new_tokens,
            use_vllm=True,
            vllm_mode="colocate",
            generation_kwargs={
                "stop": list(config.env.stop_sequences),
                "include_stop_str_in_output": True,
            },
            # Every step, not the transformers default of 500. A smoke run is 5
            # steps and a dev run 150, so the default logs nothing at all - and
            # the per-step series in section 9 (reward per component, within-group
            # std, KL, completion length, searches per rollout) is the only way a
            # reward hack is visible while it is happening.
            logging_steps=1,
            seed=config.seed,
            report_to="wandb" if config.logging.wandb_project else "none",
            run_name=f"lambda-{config.reward.lambda_retrieval_cost}",
        ),
        train_dataset=_dataset(questions, config),
        processing_class=hf_tokenizer,
        peft_config=peft.LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            task_type="CAUSAL_LM",
        ),
        rollout_func=rollout_func,
    )
    if trainer.args.beta != config.train.kl_beta:
        msg = f"KL beta did not reach the trainer: {trainer.args.beta} != {config.train.kl_beta}"
        raise TrainError(msg)

    trainer.train()
    return _summarize(trainer, max_steps, output_dir)


def _question_for(prompt: str, by_prompt: Mapping[str, Question]) -> Question:
    """The question a prompt came from. Exact match, or fail.

    A near-miss here is the worst kind of bug this file can have: the rollout
    would be scored against another question's gold answers, every metric would
    look plausible, and nothing would raise.
    """
    question = by_prompt.get(prompt)
    if question is None:
        msg = (
            "a prompt TRL passed to rollout_func is not one this run rendered, so the rollout "
            "cannot be matched to its gold answers"
        )
        raise TrainError(msg)
    return question


def _dataset(  # pragma: no cover - needs the [train] extra
    questions: Sequence[Question], config: Config
) -> object:
    datasets = _module("datasets")
    return datasets.Dataset.from_dict(
        {
            "prompt": [
                render_prompt(
                    question.question,
                    search_budget=config.env.search_budget,
                    top_k=config.retrieval.top_k,
                )
                for question in questions
            ]
        }
    )


def _as_float(value: object) -> float:
    """A log-history value as a float, or NaN. Never a guess dressed as a number."""
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, int | float):
        return float(value)
    return math.nan


def _last(history: Sequence[Mapping[str, object]], key: str) -> float:
    """The last finite value logged under `key`, or NaN."""
    values = [_as_float(entry[key]) for entry in history if key in entry]
    finite = [value for value in values if math.isfinite(value)]
    return finite[-1] if finite else math.nan


def _mean(history: Sequence[Mapping[str, object]], key: str) -> float:
    values = [_as_float(entry[key]) for entry in history if key in entry]
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def summarize_history(
    history: Sequence[Mapping[str, object]], max_steps: int, output_dir: Path
) -> TrainSummary:
    """Reduce a trainer's log history to the run's verdict.

    Reads the per-step `loss` series when there is one, and falls back to the
    final `train_loss` summary when the run was too short to log a step.

    The fallback is here because the first real smoke run reported a NaN loss
    from an empty history and failed its own gate while the loop had in fact run
    fine: `logging_steps` defaulted to 500, so two steps logged no per-step loss
    at all. A gate that fails for its own bookkeeping reasons is worse than no
    gate, because the next person to see it red will stop believing it.
    """
    final_loss = _last(history, "loss")
    if not math.isfinite(final_loss):
        final_loss = _last(history, "train_loss")
    return TrainSummary(
        steps=max_steps,
        final_loss=final_loss,
        mean_reward=_mean(history, "reward"),
        output_dir=str(output_dir),
    )


def _summarize(  # pragma: no cover - needs a finished trainer
    trainer: _Trainer, max_steps: int, output_dir: Path
) -> TrainSummary:
    return summarize_history(trainer.state.log_history, max_steps, output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m verisearch.train.grpo`. Called by `scripts/02_train.sh`."""
    parser = argparse.ArgumentParser(description="GRPO training for VeriSearch.")
    parser.add_argument("--config", action="append", default=[], help="YAML config, layered")
    parser.add_argument("--questions", default="data/processed/hotpotqa_train.jsonl")
    parser.add_argument("--out", default="checkpoints/grpo")
    parser.add_argument("--steps", type=int, default=None, help="overrides train.max_steps")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    parser.add_argument(
        "--lambda-cost", type=float, default=None, help="sweep point; overrides the config"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run against the fake retriever; asserts the loss is finite",
    )
    parser.add_argument("--plan", action="store_true", help="print the plan and change nothing")
    args = parser.parse_args(argv)

    config = load_config(*args.config)
    if args.lambda_cost is not None:
        config = config.with_lambda(args.lambda_cost)
    max_steps = args.steps if args.steps is not None else config.train.max_steps

    if args.smoke:
        from verisearch.testing import FakeRetriever

        retriever: Retriever = FakeRetriever()
        questions = _smoke_questions()
    else:
        from verisearch.retrieval.client import HttpRetriever

        retriever = HttpRetriever(config.retrieval.service_url)
        questions = load_questions(args.questions, limit=args.limit)

    print(f"policy       {config.model.policy_repo}")
    print(f"questions    {len(questions)}")
    print(f"steps        {max_steps}")
    print(f"group size   {config.train.group_size}")
    print(f"kl beta      {config.train.kl_beta}")
    print(f"lambda       {config.reward.lambda_retrieval_cost}")
    print(f"retriever    {type(retriever).__name__}")
    print(f"output       {args.out}")
    if args.plan:
        print("--plan given, stopping before the trainer")
        return 0

    summary = run_training(
        config,
        questions,
        retriever,
        output_dir=Path(args.out),
        max_steps=max_steps,
    )
    print(f"final loss   {summary.final_loss}")
    print(f"mean reward  {summary.mean_reward}")
    if not summary.loss_is_finite:
        print("FAIL: the loss is not finite")
        return 1
    print("RESULT: PASS")
    return 0


def _smoke_questions() -> list[Question]:
    """Five questions from the fake corpus. Enough to exercise the loop, not to learn."""
    return [
        Question(
            question_id=f"smoke-{index}",
            question=question,
            gold_answers=(answer,),
        )
        for index, (question, answer) in enumerate(
            (
                ("Who wrote the first notes on the Analytical Engine?", "Ada Lovelace"),
                ("Who designed the Analytical Engine?", "Charles Babbage"),
                ("What nationality was Ada Lovelace?", "English"),
                ("What was Charles Babbage's profession?", "polymath"),
                ("What did Ada Lovelace write notes about?", "the Analytical Engine"),
            )
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main())
