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
   would zero their loss. Plan: stop only on `</search>`, let EOS end the answer. Raised on the
   config PR. With that, `completions/clipped_ratio` reads as "fraction of rollouts that never
   answered" - the smoke run showed 0.5 and 0.75 with a 0.5B model that mostly ran out of turns.
2. **Tag detection is token-boundary sensitive.** In the smoke run, 4 turns stopped on the stop
   string but only 2 decoded to a string *ending* in `</search>`. The likely mechanism: vLLM
   truncates the returned text at the stop string, but the returned token ids include the whole
   final token, which can carry trailing characters. So the parser must detect the tag by
   containment on a stripped string, never by `endswith`, or it silently drops searches. To be
   confirmed against `env/parser.py` when it lands, with a test either way.
3. **No vLLM sleep mode during rollouts.** Each `generate()` call sleeps at level 2 and discards
   the weights (`generation/vllm_generation.py:750`), and the next call re-pushes them (`589`).
   With a multi-turn loop that is one full weight sync per turn.
4. **Log-prob memory is the scaling wall, not the weights.** Full-vocab logits are
   ~0.6 MB/token in fp32 at `vocab_size=151936`; a group of 8 three-thousand-token trajectories
   in one pass is ~14.6 GB. Log-probs get computed chunked. Arithmetic, not a measurement - the
   1.5B run will settle it.

### Hardware this runs on

RTX 5060 Ti 16 GB (Blackwell, cc 12.0), driver 580.173.02, torch 2.13.0+cu130, vLLM 0.27.0,
15.3 GB RAM, 245 GB free. `scripts/00_setup_check.py` reads `RESULT: PASS` as of 2026-10-14.
1.5B with group 8 is the dev path here; 3B at bf16 does not fit alongside vLLM and is a P4
decision (4-bit NF4 or a rented card).
