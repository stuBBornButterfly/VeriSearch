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

**A run directory describes itself.** Before the trainer starts, the fully
resolved `Config` is written to `<out>/resolved_config.json`; when it finishes,
the LoRA adapter is saved to `<out>/adapter/` in the layout `vllm serve
--lora-modules` loads. Eval reads the trained lambda and the search budget from
the first file rather than from a flag (#34), so a checkpoint cannot be placed
at the wrong point on the frontier.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from verisearch.config import Config, load_config
from verisearch.env.episode import EpisodeConfig, Question
from verisearch.env.protocol import check_stop_sequences, render_prompt
from verisearch.rewards.compose import RewardSpec, compose_reward
from verisearch.rewards.faithfulness import score_from
from verisearch.train.rollout import (
    TokenGenerator,
    Tokenizer,
    VllmTurnPolicy,
    rollout_group,
)
from verisearch.types import FaithfulnessScorer, Retriever, Trajectory

__all__ = [
    "ADAPTER_DIRNAME",
    "RESOLVED_CONFIG_FILENAME",
    "TrainError",
    "TrainSummary",
    "index_prompts",
    "load_question_set",
    "load_questions",
    "main",
    "rewards_for_trajectories",
    "summarize_history",
    "write_resolved_config",
]

#: TRL folds our `env_mask` into the loss mask under this key.
ENV_MASK_KEY = "env_mask"

#: The resolved config, written into `--out` before the trainer starts. Not
#: `config.json`: that name is a model config to `transformers`, and a tool that
#: calls `from_pretrained` on the run directory would try to read this as one.
RESOLVED_CONFIG_FILENAME = "resolved_config.json"

#: vLLM's context window, in tokens. Left unset, TRL takes the model's own
#: (32,768 for Qwen2.5) and vLLM refuses to start at its default 0.3 memory
#: share on a 16 GB card: the first 1.5B smoke run asked for 0.88 GiB of KV cache
#: and had 0.63. An episode's context is bounded well below this: the prompt
#: (~150 tokens), at most `max_turns` generations of `max_new_tokens` (6 x 512 in
#: base.yaml) and at most `search_budget` injected blocks of `top_k` passages
#: (4 x 3 corpus paragraphs, typically 100-300 tokens each) - about 7,000 tokens
#: in the worst case. Belongs in `EnvConfig` (Ayesha's file); a constant here
#: until it is.
CONTEXT_TOKENS = 8192

#: Rollouts per forward pass. One group of `group_size` rollouts is still one
#: optimizer step - it is split into micro-batches with gradient accumulation,
#: and TRL computes the advantage over the whole group before splitting. Not an
#: experimental setting: the gradient is the same up to float error.
#:
#: Measured down to 1 on the 16 GB card. Eight multi-turn rollouts per pass ran
#: the first 1.5B smoke run out of memory in `selective_log_softmax` on step 1;
#: two per pass ran two steps and then out of memory in the no-grad log-prob
#: pass on step 3, at a 15.6 GB peak. Full-vocabulary logits are 151,936 entries
#: per token, and a multi-turn completion is hundreds to thousands of tokens.
MICRO_BATCH = 1

#: Where the final LoRA adapter goes, under `--out`. `adapter_config.json` and
#: `adapter_model.safetensors`, which is what `vllm serve --lora-modules` loads.
ADAPTER_DIRNAME = "adapter"


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

    def save_model(self, output_dir: str) -> None: ...


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
    A question with no gold answer is refused: the reward cannot score it, and
    `Question` itself allows one only for the policy server.
    """
    questions: list[Question] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            questions.append(_question_from_row(json.loads(line), path))
            if limit is not None and len(questions) >= limit:
                break
    if not questions:
        msg = f"{path} produced no questions"
        raise TrainError(msg)
    return questions


def load_question_set(
    paths: Sequence[str | Path],
    *,
    per_file: int | None = None,
    seed: int,
    limit: int | None = None,
) -> list[Question]:
    """The training set: one or more question files, optionally sampled.

    PROJECT.md section 7 trains on HotpotQA plus 2WikiMultihopQA and starts at
    2,000-5,000 examples, against files of 78,335 and 144,123. `per_file` draws
    that many from each file, uniformly and seeded, so the mix is the same
    whatever the files' sizes and a rerun draws the same questions. `limit` then
    keeps the first N of the result, in file order - the overfit-20 run's
    "these twenty, every time".
    """
    if not paths:
        msg = "no question files given"
        raise TrainError(msg)
    if per_file is not None and per_file < 1:
        msg = f"per_file must be at least 1, got {per_file}"
        raise TrainError(msg)
    rng = random.Random(seed)
    questions: list[Question] = []
    for path in paths:
        lines = [
            line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if per_file is not None:
            if per_file > len(lines):
                msg = f"{path} has {len(lines)} questions, fewer than per_file={per_file}"
                raise TrainError(msg)
            # Sorted indices keep the file's own order inside the sample, so the
            # draw is a pure function of the seed and the file's contents.
            lines = [lines[index] for index in sorted(rng.sample(range(len(lines)), per_file))]
        if not lines:
            msg = f"{path} produced no questions"
            raise TrainError(msg)
        questions.extend(_question_from_row(json.loads(line), path) for line in lines)
    return questions[:limit] if limit is not None else questions


def _question_from_row(row: Mapping[str, object], path: str | Path) -> Question:
    answers = row["answers"]
    gold = tuple(str(answer) for answer in answers) if isinstance(answers, list) else ()
    if not gold:
        msg = f"{path}: question {row['question_id']} has no gold answer, so no reward can score it"
        raise TrainError(msg)
    return Question(
        question_id=str(row["question_id"]),
        question=str(row["question"]),
        gold_answers=gold,
    )


def write_resolved_config(config: Config, output_dir: Path) -> Path:
    """Write the config a run actually used next to its checkpoints.

    Eval reads `reward.lambda_retrieval_cost` from here to set the report's
    `trained_lambda`, and `env.search_budget` to check it is evaluating under
    the budget the policy trained with - never from a flag that could be typed
    wrong (#34). Written before training starts, so a run that dies still says
    what it was.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RESOLVED_CONFIG_FILENAME
    path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def rewards_for_trajectories(
    trajectories: Sequence[Trajectory],
    spec: RewardSpec,
    *,
    scorer: FaithfulnessScorer | None = None,
) -> list[float]:
    """Score a group of rollouts. The reward function TRL calls, minus the plumbing.

    With `scorer=None` the faithfulness term is 0.0 for every rollout, which is
    finding B on #31: `w_a * faithfulness` then never reaches the policy. With a
    scorer, every answered rollout is scored against the passages it retrieved,
    once per step, after generation - outside the generation loop, which is what
    "not in the hot loop" protects. Which of the two a run used is printed at
    startup and recorded in `rl_method.md`.
    """
    return [
        compose_reward(
            traj,
            spec,
            faithfulness=score_from(traj, scorer) if scorer is not None else None,
        ).total
        for traj in trajectories
    ]


def run_training(  # pragma: no cover - needs the [train] extra and a GPU
    config: Config,
    questions: Sequence[Question],
    retriever: Retriever,
    *,
    output_dir: Path,
    max_steps: int,
    tokenizer: Tokenizer | None = None,
    scorer: FaithfulnessScorer | None = None,
) -> TrainSummary:
    """Wire TRL to our rollout and our reward, run, and report.

    Deliberately one function: the trainer objects are untyped to `mypy` (no
    stubs for `trl`), so keeping them inside a single call means nothing
    annotated ever returns one.
    """
    # Before torch initialises CUDA. The second 1.5B smoke run died with 616 MB
    # reserved but unallocated; expandable segments let the allocator reuse
    # that instead of fragmenting. Safe with vLLM because sleep mode is off
    # (`rl_method.md`, consequence 3), and the caller's own setting wins.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    trl = _module("trl")
    peft = _module("peft")
    transformers = _module("transformers")

    check_stop_sequences(config.env.stop_sequences)
    if config.train.group_size % MICRO_BATCH:
        msg = f"group_size {config.train.group_size} must be a multiple of {MICRO_BATCH}"
        raise TrainError(msg)
    episode_config = EpisodeConfig.from_config(config)
    spec = RewardSpec(config.reward, search_budget=config.env.search_budget)

    hf_tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(
        config.model.policy_repo, padding_side="left"
    )

    # Rollout: TRL hands us the prompts for this step, already repeated
    # num_generations times, and expects token ids back.
    by_prompt = index_prompts(questions, config)

    def rollout_func(prompts: list[str], trainer: _GeneratorHolder) -> dict[str, object]:
        batch = [_question_for(prompt, by_prompt) for prompt in prompts]
        payloads = rollout_group(
            batch,
            retriever,
            episode_config,
            policy=VllmTurnPolicy(trainer.vllm_generation, hf_tokenizer),
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
            [Trajectory.model_validate_json(payload) for payload in trajectory],
            spec,
            scorer=scorer,
        )

    # `report_to="wandb"` reads the project and entity from the environment, and
    # without them a run lands in the `huggingface` project of whoever is logged
    # in. The config is the source of truth (H3).
    if config.logging.wandb_project:
        os.environ["WANDB_PROJECT"] = config.logging.wandb_project
        if config.logging.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.logging.wandb_entity

    trainer: _Trainer = trl.GRPOTrainer(
        model=config.model.policy_repo,
        reward_funcs=reward_func,
        args=trl.GRPOConfig(
            output_dir=str(output_dir),
            max_steps=max_steps,
            num_generations=config.train.group_size,
            per_device_train_batch_size=MICRO_BATCH,
            gradient_accumulation_steps=config.train.group_size // MICRO_BATCH,
            # Activations recomputed in the backward pass instead of held: a
            # multi-turn completion is thousands of tokens.
            gradient_checkpointing=True,
            learning_rate=config.train.learning_rate,
            # TRL defaults beta to 0.0, which loads no reference model at all.
            # PROJECT.md section 5 makes KL the main anti-hacking guard, so this
            # one is asserted below rather than trusted.
            beta=config.train.kl_beta,
            bf16=config.model.dtype == "bfloat16",
            max_completion_length=config.env.max_new_tokens,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_max_model_length=CONTEXT_TOKENS,
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
    trainer.save_model(str(output_dir / ADAPTER_DIRNAME))
    return _summarize(trainer, max_steps, output_dir)


def index_prompts(questions: Sequence[Question], config: Config) -> dict[str, Question]:
    """Map each rendered prompt to its question, refusing an ambiguous one.

    Keyed on the rendered prompt, which is exactly what the dataset holds: with
    `rollout_func`, TRL forwards the dataset's prompts unchanged
    (`grpo_trainer.py:2239`). Matching on a substring would let two questions
    that share a prefix swap rollouts. Two questions with the same text and
    different gold answers are the same failure from the other side - one
    would be scored against the other's answers - so they are refused here.
    Exact duplicates are harmless and kept once.
    """
    by_prompt: dict[str, Question] = {}
    for question in questions:
        prompt = render_prompt(
            question.question,
            search_budget=config.env.search_budget,
            top_k=config.retrieval.top_k,
        )
        seen = by_prompt.get(prompt)
        if seen is not None and set(seen.gold_answers) != set(question.gold_answers):
            msg = (
                f"questions {seen.question_id} and {question.question_id} have the same text "
                "but different gold answers, so a rollout could not be matched to its answers"
            )
            raise TrainError(msg)
        by_prompt.setdefault(prompt, question)
    return by_prompt


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
    parser.add_argument(
        "--questions",
        nargs="+",
        default=["data/processed/hotpotqa_train.jsonl"],
        help="one or more question JSONL files",
    )
    parser.add_argument("--out", default="checkpoints/grpo")
    parser.add_argument("--steps", type=int, default=None, help="overrides train.max_steps")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    parser.add_argument(
        "--per-file",
        type=int,
        default=None,
        help="draw N questions from each file, seeded by the config's seed",
    )
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
        questions = load_question_set(
            args.questions, per_file=args.per_file, seed=config.seed, limit=args.limit
        )

    # Section 15: a number traces back to the exact config that produced it.
    print(f"config       {config.fingerprint}")
    print(f"policy       {config.model.policy_repo}")
    print(f"questions    {len(questions)}")
    print(f"steps        {max_steps}")
    print(f"group size   {config.train.group_size}")
    print(f"kl beta      {config.train.kl_beta}")
    print(f"lambda       {config.reward.lambda_retrieval_cost}")
    print(f"retriever    {type(retriever).__name__}")
    # Finding B on #31 is open until both owners pick an option; until then the
    # run says so, so no reader mistakes a 0.0 faithfulness term for a measured one.
    print("faithfulness off - not scored during training (finding B on #31)")
    print(f"output       {args.out}")
    if args.plan:
        print("--plan given, stopping before the trainer")
        return 0

    print(f"wrote        {write_resolved_config(config, Path(args.out))}")
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
