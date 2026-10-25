# RL method

Owner: Tahsinul Haque Dhrubo. Decisions for the RL half, with the evidence behind them.

Every claim here either cites a line in an installed dependency or a logged run in `outputs/`.
Nothing in this file is an estimate presented as a measurement.

---

## 1. Multi-turn RL route: TRL 1.13.0 `rollout_func`

**Decision: TRL `GRPOTrainer` with a custom `rollout_func` and our own `env_mask`.
Hand-rolled GRPO stays the documented fallback.**

Card: `[P1] Spike: evaluate 3 multi-turn RL routes, 2-day timebox`. Full working notes,
including the routes not taken, are in [`spikes/NOTES.md`](../spikes/NOTES.md).

### Why

The spike asked two questions of each route: can we drive a multi-turn rollout ourselves, and
do we control which tokens carry loss. TRL answers both, and the second one is answered by a
run rather than by reading the source.

1. **Custom generation is a supported hook.** `GRPOTrainer(rollout_func=...)` takes
   `(prompts, trainer) -> dict` and expects `prompt_ids`, `completion_ids`, `logprobs`
   (`trl/trainer/grpo_trainer.py:258, 2242`). The multi-turn loop is ours: we call
   `trainer.vllm_generation.generate(...)` once per turn, splice the retrieved `<info>` token
   ids into the completion, and continue. Token-in-token-out, no re-tokenisation.
2. **We supply the loss mask directly.** The rollout returns `env_mask` (1 = policy token,
   0 = injected); TRL folds it into `loss_mask = completion_mask * tool_mask`
   (`2293-2295, 2519, 2986, 3120`). This is the thing KICKOFF.md says to read twice, and it is
   not a side effect of some chat-template bookkeeping - it is a mask we construct and can
   unit-test on a plain data structure.
3. **It runs on the one GPU we have.** `vllm_mode="colocate"` keeps vLLM in-process and shares
   the card, and TRL syncs LoRA weights to vLLM before each step's rollout (`2233-2236`).
4. **Measured, not assumed.** `spikes/smoke_trl_rollout.py`, Qwen2.5-0.5B-Instruct, group 4,
   2 GRPO steps, fake retriever, KL beta 0.04, on the RTX 5060 Ti, 2026-10-14. The log lives at
   `outputs/smoke_0.5b.log`, which is gitignored - it is attached to the Trello card, and the
   numbers below are transcribed in [`spikes/NOTES.md`](../spikes/NOTES.md).

| Check | Result |
|---|---|
| `rollout_func` receives prompts already repeated `num_generations` times | True, both steps |
| `env_mask` survives into the loss inputs as `tool_mask` | True |
| Injected token count, rollout vs loss mask zeros | 90 vs 90, then 45 vs 45 |
| **Max abs gradient on injected tokens (`lm_head` hook, after backward)** | **0.0, both steps** |
| Policy tokens with non-zero gradient | 100% |
| Loss finite with `None` logprobs on injected tokens | True (-0.2643, then 0.6227) |
| GPU, device total | 7.85 then 8.78 GiB of 15.47 GiB |

The gradient row is the one that matters. The failure mode KICKOFF.md warns about - the model
learning to predict its own retrieval results - is measured at exactly zero here, with a hook
on the lm_head output rather than by trusting a mask count.

### What this costs

- **Both hooks are experimental** and warn on use (`577-592`): "may change or be removed at any
  time". Mitigation: pin `trl` exactly in `[train]`, and `tests/test_masking.py` asserts on
  *our* mask builder, not on TRL, so a TRL upgrade breaks a dependency and not the proof.
- **A new dependency**, and `pyproject.toml` is Ayesha's. Needs a relock PR she approves.
- **Two defaults must be overridden.** `beta` defaults to 0.0, i.e. no KL term and no reference
  model (`grpo_config.py:676`), while PROJECT.md section 5 makes KL beta 0.04 the main
  anti-hacking guard. `loss_type` defaults to `"dapo"` and `scale_rewards` to `"group"`
  (`796, 784`); PROJECT.md specifies neither, so the choice gets recorded in `configs/`.

### When to fall back to hand-rolling

Not a matter of taste. Any one of these triggers it:

1. A `trl` upgrade we need for another reason removes or changes `rollout_func` / `env_mask`.
2. The zero-gradient check above stops passing and the cause is inside TRL.
3. We need a loss variant TRL does not implement.

The fallback is scoped in `spikes/NOTES.md`: roughly 300-500 lines beyond what we write anyway,
because the parser, the episode loop and the mask builder are ours on every route. That is the
real finding of the spike - choosing TRL removes the GRPO machinery, not the RL core.

### Routes not taken

- **verl 0.9.0** - the most mature option and the one with our exact task in its lineage
  (Search-R1). Ruled out on hardware and control: multi-turn is documented on the SGLang
  backend (not in our stack), single-GPU is not a documented path (Ray + FSDP/Megatron), the
  mask is built internally from chat messages, and our `<search>` tag protocol is off the
  documented path. verl-omni #618 reports LoRA weight sync failing on one GPU with
  `torch 2.13.0+cu130` - our exact build.
- **ART / verifiers / SkyRL** - verifiers delegates training elsewhere, ART represents
  trajectories as chat messages with no documented token-level loss control, SkyRL is the
  smallest community of the three. Details in the notes.
- **TRL's built-in `tools` / `environment_factory`** - the trainer would run the tool loop and
  mask tool output itself (`1990, 2193`), but it requires chat-template tool calling (`618`),
  which turns trajectories into JSON tool calls instead of the `<search>` tags in PROJECT.md
  section 1. That is a protocol change and Ayesha's call, so it was not taken unilaterally.

### Consequences for the environment, carried into P1

1. **Stop sequences.** `configs/base.yaml` lists `</search>` and `</answer>`. TRL marks a
   completion truncated when its last token is not EOS or pad (`2325`). Stopping on `</answer>`
   therefore reports every finished episode as clipped, and `mask_truncated_completions=True`
   would zero their loss. Plan: stop only on `</search>`, let EOS end the answer. With that,
   `completions/clipped_ratio` reads as "fraction of rollouts that never answered" - the smoke
   run showed 0.5 and 0.75 with a 0.5B model that mostly ran out of turns. **Done:** #20 removed
   `</answer>` from `base.yaml` and is on `main`, and `check_stop_sequences` (#27) refuses a
   config that puts it back.
2. **Tag detection is token-boundary sensitive.** In the smoke run, 4 turns stopped on the stop
   string but only 2 decoded to a string *ending* in `</search>`. The likely mechanism: vLLM
   truncates the returned text at the stop string, but the returned token ids include the whole
   final token, which can carry trailing characters. So the parser must detect the tag by
   containment on a stripped string, never by `endswith`, or it silently drops searches.
   **Confirmed and handled:** the parser finds the action by position (#27), and with Qwen's
   tokenizer `</search>\n` does end on a single `>\n` token. The episode keeps that token whole
   and cuts whatever follows it (section 2).
3. **No vLLM sleep mode during rollouts.** Each `generate()` call sleeps at level 2 and discards
   the weights (`generation/vllm_generation.py:750`), and the next call re-pushes them (`589`).
   With a multi-turn loop that is one full weight sync per turn.
4. **Log-prob memory is the scaling wall, not the weights.** Full-vocab logits are
   ~0.6 MB/token in fp32 at `vocab_size=151936`; a group of 8 three-thousand-token trajectories
   in one pass is ~14.6 GB. Log-probs get computed chunked. Arithmetic, not a measurement - the
   1.5B run will settle it.

### The loop, run end to end

`scripts/02_train.sh --smoke --steps 5` - the exact command the nightly `smoke-train` stage
runs - on Qwen2.5-0.5B with the fake retriever, 2026-10-21:

| | |
|---|---|
| verdict | `RESULT: PASS` |
| final loss | 5.788e-05, finite |
| mean reward | -0.50 |
| step time | ~2.2 s |
| KL from reference | 0.0014 |
| `reward_std` / `frac_reward_zero_std` | 0 / 1 |

The last row is the interesting one and it is not good news: every rollout hit the no-answer
penalty and nothing else, so every advantage was zero and the steps moved almost nothing. A
0.5B policy with a 128-token turn budget never produced a well-formed answer. That is what a
dead reward looks like, which is exactly what P2's exit criterion exists to rule out at 1.5B -
so the run proves the wiring, not the method.

Two bugs of mine surfaced only by running it, both written up in
[`failure_cases.md`](failure_cases.md): generations that parsed to no turns were dropped
instead of trained on, and the nightly gate failed on its own bookkeeping while the loop was
fine. Neither was visible from reading the code, which is the argument for running five steps
before writing five hundred lines more.

### The loop at 1.5B

The same command on Qwen2.5-1.5B-Instruct, the dev policy and what the nightly stage runs by
default, 2026-10-25, after the fixes in section 3. Log: `outputs/smoke_1.5b.log`.

| | |
|---|---|
| verdict | 5 steps, final loss -1.007, finite |
| mean reward per step | 0.178, -0.371, -0.222, -0.344, -0.247 |
| `reward_std` per step | 0.62, 0.37, 0.38, 0.34, 0.36 |
| `frac_reward_zero_std` | 0 on every step |
| KL from reference | 0 rising to 0.0022 |
| policy tokens per completion | mean 233-295, max 579 |
| `completions/clipped_ratio` | 0.25-0.625 |
| step time | 21-25 s |
| peak GPU memory | 14.5 GB of 16 (sampled with `nvidia-smi`, other processes included) |

Unlike 0.5B, no group was degenerate: every step had reward spread, so every step moved the
policy. That's the first sign the reward can carry a signal at this size. It's still on the fake
corpus and five questions, so it says nothing about P2's 15-60% spread criterion, which needs
real retrieval. The 14.5 GB peak is the number B7 needs: 1.5B fits, with little room, and 3B
at bf16 beside colocated vLLM does not.

### Hardware this runs on

RTX 5060 Ti 16 GB (Blackwell, cc 12.0), driver 580.173.02, torch 2.13.0+cu130, vLLM 0.27.0,
15.3 GB RAM, 245 GB free. `scripts/00_setup_check.py` reads `RESULT: PASS` as of 2026-10-14.
1.5B with group 8 is the dev path here; 3B at bf16 does not fit alongside vLLM and is a P4
decision (4-bit NF4 or a rented card).


---

## 2. The token context: what was sampled is what is replayed

Finding A, Ayesha's review of #31. **Fixed on #28 and #31.**

**The bug.** The first episode loop re-encoded the decoded transcript every turn. That
transcript had the overshoot after `</search>` cut and a newline added on each side of the
`<info>` block. But the payload the trainer replays is `prompt_ids` + the raw generated ids +
`encode(info)`. From turn 2 on, the model was trained on a context it never sampled from, and
the importance-sampling ratio compared vLLM's log-probabilities with the trainer's across two
different contexts. The mask was right throughout and `test_masking.py` passed; this sat
alongside it.

**How big it was.** With Qwen2.5's tokenizer, after one search, re-encoding the decoded
context gives 218 ids where the spliced context has 220. The first difference is at the join
between the prompt's final `?` and the first `<think>` (`tests/test_token_context_qwen.py`).

**The fix.**
- The episode has two entry points over one loop. `run_episode` is the text path, for eval and
  serving. `run_token_episode` is the token path, for training: the context is a running list
  of ids (the prompt, then each generation's ids, then each injected block's ids), and the
  policy is handed that list. No string goes in, so nothing re-tokenises.
- The separators belong to the injected text (`render_injection`), so an `INFO` turn's text
  and ids describe the same characters.
- Overshoot after `</search>` is cut from both streams, at the token that closed the search.
  That token is kept whole, because it was sampled as a unit; with Qwen it's `>\n`.
- A generation whose text isn't the decoding of its ids, or that has no tokens or misaligned
  logprobs, is refused.

**The test.** Ayesha's, as she wrote it: `generator.calls[k] == prompt_ids +
completion_ids[:prefix]` for every turn of a multi-turn episode. It runs with a per-character
tokenizer and with one whose merges straddle the joins, and the test also asserts that
re-encoding gives different ids there, so the fixture can't quietly stop testing anything. The
Qwen version runs the same check with the real tokenizer and is `slow`. Run it after any `trl`
or `transformers` bump.

**What eval should know.** Eval's text path still re-tokenises: `vllm serve` encodes the
prompt string fresh each turn. That doesn't matter for the base model. For trained
checkpoints, vLLM's `/v1/completions` accepts a token-id prompt and returns ids with
`return_token_ids`, so eval can use the token path too. Raised on #38.

## 3. TRL settings, and what they do with a multi-turn completion

**`max_completion_length` is a per-turn limit, and long completions are trained on whole.**
With `rollout_func`, TRL 1.13 takes `completion_ids` as returned (`grpo_trainer.py:2241-2247`),
pads them to the longest in the batch (`2474-2484`) and builds `loss_mask = completion_mask *
tool_mask` (`2519`). Nothing truncates to `max_completion_length` and nothing raises.
The setting reaches only two places:
- vLLM's `max_tokens` on each `generate` call (`generation/vllm_generation.py:624, 669`),
  which is the per-turn limit we want. It's set from `env.max_new_tokens`.
- The `dr_grpo` loss normaliser (`3257`). We use the default `dapo`, which normalises by
  the number of loss tokens.

Measured: the 1.5B smoke run had completions of up to 579 policy tokens against a limit of 512,
and trained on all of them.

**`mask_truncated_completions` must stay `False`** (the default, `grpo_config.py:831`). TRL calls
a completion truncated when its last token isn't EOS or pad. An episode that ends by exhausting
its budget, or with an unclosed tag, ends that way. Masking those would zero the loss on exactly
the rollouts the no-answer penalty exists to push down.

**`vllm_max_model_length=8192`.** Left unset, TRL takes the model's own window (32,768 for
Qwen2.5), and vLLM refused to start at its default 0.3 memory share: 0.88 GiB of KV cache
needed, 0.63 available (`outputs/smoke_1.5b_oom_step1.log`, the first attempt). An episode's
worst case is about 7,000 tokens: the prompt, plus 6 turns of 512, plus 4 blocks of 3
passages. This belongs in `EnvConfig`, which is Ayesha's, and is a constant in
`train/grpo.py` until it moves there.

**One rollout per forward pass.** `per_device_train_batch_size=1`, with gradient accumulation
over the group and gradient checkpointing. The group is still one optimizer step, and TRL
computes the advantage over all of it before splitting. Eight rollouts per pass ran out of
memory on step 1, and two per pass on step 3 (`outputs/smoke_1.5b_oom_step3.log`, peak
15.6 GB). Full-vocabulary logits are 151,936 entries per token. `expandable_segments` is set
too, because the second failure had 616 MB reserved but unallocated.

**Importance sampling is sequence-level** (`vllm_importance_sampling_mode="sequence_mask"`,
the default). The 1.5B smoke run measured a per-token sampling log-prob difference of 0.020-0.029
on average. The sequence-level ratio averaged 0.83-1.02, with a maximum of 2.57. Worth watching
in long runs: a ratio drifting from 1 means vLLM and the trainer disagree on the policy.

**Recorded defaults:** `loss_type="dapo"`, `scale_rewards="group"`, KL `beta=0.04` (asserted
after construction, because TRL's default is 0.0), `logging_steps=1`.

**Prompts per step: one, today.** `group_size=8` with 8 rollouts per optimizer step is one
question per step, so 1,000 steps see 1,000 questions once each. Whether that is enough
signal per step is a hyperparameter to settle before the full 1.5B run (decision comment on
#31). It isn't in `TrainConfig` yet.

## 4. The reward as built, and where it departs from PROJECT.md

- **The -0.5 no-answer penalty applies to any unanswered episode**, not only one that ran out
  of turns. Section 6 writes "no answer by max turns". `_never_answered` in
  `rewards/compose.py` also charges an episode that ended on a parse error or an exhausted
  budget. Reason: those are the same failure from the policy's side, and the reward must not
  make one of them the cheaper way to fail. Ayesha agreed on #30.
- **The EM bonus is 0.2, a constant** (`rewards/correctness.py`), applied as `(1 - b) * F1 +
  b * EM`. It isn't in `RewardConfig`, which is Ayesha's file; she offered to add the field.
- **Faithfulness is not yet in the training reward.** See section 5.

## 5. Faithfulness in the training reward

Finding B, Ayesha's review of #31. **Open: a decision for both of us, before the sweep.**

**What happens today.** `rewards_for_trajectories` scores with no `FaithfulnessScorer`, so
`w_a * faithfulness` is 0.0 for every rollout and the policy is never rewarded for it. Section
1's thesis and success criterion 2 both depend on it. The plumbing exists:
`rewards_for_trajectories(..., scorer=)` and `run_training(..., scorer=)` take one, and a run
prints "faithfulness off" until something passes one. What's missing is a scorer that
implements the protocol (#7).

**Measured, on this card,** with `cross-encoder/nli-deberta-v3-base` as a stand-in for
`base.yaml`'s checkpoint (same DeBERTa-v3-base backbone, 184M parameters, fp32 as #7 loads it).
Scripts: `spikes/measure_nli_memory.py` and `spikes/measure_nli_cpu.py`. Logs:
`outputs/nli_memory.log` and `outputs/nli_cpu.log`.

| | GPU | CPU (6 threads) |
|---|---|---|
| weights | 0.69 GiB | - |
| batch of 16 pairs at ~495 tokens | 2.03 GiB peak, 0.46 s | - |
| realistic step: 48 pairs at ~200 tokens | - | ~5.3 s |
| worst case: 288 pairs at 512 tokens | ~5.5-8.3 s | ~120 s |

Here "realistic" means 8 rollouts x 1 claim x up to 6 passages. "Worst case" means 3 claims
and 12 passages per rollout.

**The options, with those numbers:**
1. **Score inside the reward function, on the GPU.** Fast enough, at a few seconds against a
   22 s step. But training already peaks at 14.5 GB of 16. It could fit because the reward
   runs between generation and the log-prob passes, when the training activations aren't
   live; that needs fp16 weights, a batch of 16, and an emptied cache afterwards. Not proven
   until a smoke run does it.
2. **Score inside the reward function, on the CPU.** No GPU memory at all, about +25% step
   time at realistic sizes, and much worse in the worst case.
3. **Score asynchronously, with a lag.** Rewards for step t arrive at t+k. That adds staleness
   and the most code, and it is not needed if 1 or 2 works.
4. **Train without it and say so.** That makes criterion 2 a weaker claim.

**My recommendation:** 1, falling back to 2. Once #7 implements the protocol, do one smoke run
with the scorer on the GPU. If its peak stays under 16 GB, keep it; if not, move the scorer to
the CPU and accept the step time. Either way the decision and the measured step time go here
before the sweep.
