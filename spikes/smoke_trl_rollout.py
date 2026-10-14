"""Spike smoke run: TRL GRPOTrainer with a multi-turn rollout_func and env_mask.

Throwaway. Answers the open TRL questions in spikes/NOTES.md by running, not reading:
  1. Does rollout_func receive prompts already repeated num_generations times?
  2. Does vLLM stop on "</search>" and keep the stop string in the returned token ids?
  3. Do injected <info> tokens reach the loss with mask 0 and get exactly zero gradient?
  4. Is the loss finite with None logprobs on injected tokens?
  5. clipped_ratio when answers end on the model's EOS.
  6. GPU memory on this card.

Stops only on "</search>". Answers end on EOS so TRL's truncation check (last token must be
EOS or pad, grpo_trainer.py:2325) behaves.

Usage:
    .venv/bin/python spikes/smoke_trl_rollout.py
    .venv/bin/python spikes/smoke_trl_rollout.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-generations 8 --max-tokens-per-turn 512 --no-grad-check
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer

SEARCH_BUDGET = 2
MAX_TURNS = 3
TOP_K = 2
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

INSTRUCTIONS = (
    "Answer the question. To look something up, write <search>query</search> and stop. "
    "Results come back inside <info></info>. You may search at most twice. "
    "Give the final answer as <answer>text</answer>.\n\nQuestion: "
)
CORPUS = (
    "Gladiator won the Academy Award for Best Picture for films released in 2000.",
    "American Beauty won the Academy Award for Best Picture for films released in 1999.",
    "American Beauty was directed by Sam Mendes.",
    "Sam Mendes was born in Reading, Berkshire, England.",
    "The Matrix was directed by the Wachowskis and released in 1999.",
    "Russell Crowe, the star of Gladiator, was born in Wellington, New Zealand.",
)
QUESTIONS = (
    "Where was the director of the 1999 Best Picture winner born?",
    "Where was the star of the 2000 Best Picture winner born?",
)


def fake_search(query: str, k: int = TOP_K) -> list[str]:
    """Rank the toy corpus by word overlap. Deterministic, no network."""
    words = set(query.lower().split())
    return sorted(CORPUS, key=lambda p: -len(words & set(p.lower().split())))[:k]


@dataclass
class RolloutStats:
    """Facts from the latest rollout_func call, reset on every call."""

    prompts_seen: int = 0
    prompts_grouped: bool = False
    stop_turns: int = 0
    stop_str_in_ids: int = 0
    searches: int = 0
    injected_tokens: int = 0
    sample: str = ""


def make_rollout_func(
    tokenizer: PreTrainedTokenizerBase, group_size: int, stats: RolloutStats
) -> Callable[[list[str], GRPOTrainer], dict[str, object]]:
    end_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}

    def rollout(prompts: list[str], trainer: GRPOTrainer) -> dict[str, object]:
        n = len(prompts)
        stats.prompts_seen = n
        stats.prompts_grouped = all(p == prompts[i - i % group_size] for i, p in enumerate(prompts))
        stats.stop_turns = stats.stop_str_in_ids = stats.searches = stats.injected_tokens = 0

        prompt_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTRUCTIONS + p}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            for p in prompts
        ]
        completion_ids: list[list[int]] = [[] for _ in range(n)]
        env_mask: list[list[int]] = [[] for _ in range(n)]
        logprobs: list[list[float | None]] = [[] for _ in range(n)]
        searches = [0] * n
        active = list(range(n))

        for turn in range(MAX_TURNS):
            if not active:
                break
            contexts = [prompt_ids[i] + completion_ids[i] for i in active]
            _, turn_ids, turn_logprobs, _ = trainer.vllm_generation.generate(
                prompts=contexts, images=None, num_generations=1
            )
            still_searching = []
            for i, raw_ids, lps in zip(active, turn_ids, turn_logprobs, strict=True):
                ids = list(raw_ids)
                completion_ids[i] += ids
                env_mask[i] += [1] * len(ids)
                logprobs[i] += [lp[0] for lp in lps]
                text = tokenizer.decode(ids)

                stopped_on_string = (
                    bool(ids)
                    and ids[-1] not in end_ids
                    and len(ids) < trainer.args.max_completion_length
                )
                if stopped_on_string:
                    stats.stop_turns += 1
                    stats.stop_str_in_ids += text.endswith("</search>")

                queries = SEARCH_RE.findall(text)
                if (
                    queries
                    and text.endswith("</search>")
                    and searches[i] < SEARCH_BUDGET
                    and turn < MAX_TURNS - 1
                ):
                    searches[i] += 1
                    info = (
                        "\n<info>\n" + "\n".join(fake_search(queries[-1].strip())) + "\n</info>\n"
                    )
                    info_ids = tokenizer.encode(info, add_special_tokens=False)
                    completion_ids[i] += info_ids
                    env_mask[i] += [0] * len(info_ids)
                    logprobs[i] += [None] * len(info_ids)
                    stats.injected_tokens += len(info_ids)
                    still_searching.append(i)
            active = still_searching

        stats.searches = sum(searches)
        stats.sample = tokenizer.decode(completion_ids[0])
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_mask": env_mask,
            "num_searches": searches,
        }

    return rollout


def smoke_reward(completions: list[str], num_searches: list[int], **_kwargs: object) -> list[float]:
    """Smoke-only reward. The length term guarantees within-group variance."""
    return [
        float("<answer>" in c and "</answer>" in c) + 0.1 * s + (len(c) % 11) / 110
        for c, s in zip(completions, num_searches, strict=True)
    ]


class CheckedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that reports mask passthrough, zero-grad on injected tokens, and memory."""

    def __init__(
        self, *args: object, stats: RolloutStats, grad_check: bool, **kwargs: object
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stats = stats
        self.grad_check = grad_check
        self._logits: list[torch.Tensor] = []
        self._masks: tuple[torch.Tensor, torch.Tensor | None] = (torch.empty(0), None)
        self._smoke_step = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._masks = (inputs["completion_mask"], inputs.get("tool_mask"))
        if not self.grad_check:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        lm_head = next(m for name, m in model.named_modules() if name.endswith("lm_head"))

        def keep(_module: object, _args: object, output: torch.Tensor) -> None:
            if output.requires_grad:
                output.retain_grad()
                self._logits.append(output)

        handle = lm_head.register_forward_hook(keep)
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            handle.remove()

    def training_step(self, model, inputs, num_items_in_batch):
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._smoke_step += 1
        s = self.stats
        completion_mask, tool_mask = self._masks

        print(f"\n===== smoke step {self._smoke_step} =====")
        if self._smoke_step == 1:
            print(f"--- sample trajectory 0 ---\n{s.sample[:1500]}\n---")
        print(f"prompts per call: {s.prompts_seen}, repeated in groups: {s.prompts_grouped}")
        print(
            f"turns stopped on a string: {s.stop_turns}, decoding to </search>: {s.stop_str_in_ids}"
        )
        print(f"searches: {s.searches}")
        print(f"loss: {loss.item():.6f}, finite: {math.isfinite(loss.item())}")
        print(f"tool_mask reached loss inputs: {tool_mask is not None}")

        if tool_mask is not None:
            injected = (completion_mask == 1) & (tool_mask == 0)
            policy = (completion_mask == 1) & (tool_mask == 1)
            print(
                f"injected tokens: rollout={s.injected_tokens}, loss mask zeros={int(injected.sum())}"
            )
            if self.grad_check and self._logits:
                c = completion_mask.size(1)
                if any(t.grad is None for t in self._logits):
                    print("grad check: logits grad missing")
                else:
                    grad = torch.cat(
                        [t.grad[:, -(c + 1) : -1].abs().amax(dim=-1).float() for t in self._logits]
                    )
                    worst = grad[injected].max().item() if injected.any() else float("nan")
                    print(f"max |grad| on injected tokens: {worst} (must be 0.0; nan = none)")
                    print(
                        f"policy tokens with nonzero grad: {(grad[policy] > 0).float().mean().item():.1%}"
                    )
        self._logits.clear()

        free, total = torch.cuda.mem_get_info()
        print(
            f"GPU device used: {(total - free) / 2**30:.2f} / {total / 2**30:.2f} GiB, "
            f"torch peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
        return loss


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-tokens-per-turn", type=int, default=128)
    parser.add_argument("--no-grad-check", action="store_true", help="skip the logits-grad check")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    stats = RolloutStats()
    config = GRPOConfig(
        output_dir="outputs/spike_trl_smoke",
        max_steps=args.steps,
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        beta=0.04,
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16"},
        max_completion_length=args.max_tokens_per_turn,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=4096,
        generation_kwargs={"stop": ["</search>"], "include_stop_str_in_output": True},
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        seed=42,
    )
    trainer = CheckedGRPOTrainer(
        model=args.model,
        reward_funcs=smoke_reward,
        args=config,
        train_dataset=Dataset.from_dict({"prompt": list(QUESTIONS)}),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
        rollout_func=make_rollout_func(tokenizer, args.num_generations, stats),
        stats=stats,
        grad_check=not args.no_grad_check,
    )
    trainer.train()


if __name__ == "__main__":
    main()
