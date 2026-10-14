# P1 spike: multi-turn RL routes

Card: `[P1] Spike: evaluate 3 multi-turn RL routes, 2-day timebox`.

Timeline, recorded honestly: card received 2026-09-17, evaluation done the same day,
then the card sat idle until 2026-10-14 when the smoke run and the write-up were finished.
The 2-day timebox was not exceeded in working time; it was exceeded in wall-clock, which is
worse for the project and is on me. The decision below did not change in between.

Decision and reasons: [writeup/rl_method.md](../writeup/rl_method.md).

## Environment

- RTX 5060 Ti 16 GB (Blackwell, cc 12.0), driver 580.173.02, 15.3 GB RAM, 12 cores, 245 GB free
- Python 3.12.3, torch 2.13.0+cu130, transformers 5.15.0, vllm 0.27.0, peft 0.20.0, trl 1.13.0
- `scripts/00_setup_check.py` on 2026-10-14: **RESULT: PASS**, `cuda_available=True`
- Note: PR #8 adds `.python-version` = 3.11 and CI pins 3.11. This machine runs 3.12, so code
  must stay 3.11-compatible (no PEP 695 syntax, no `typing.override`).

## Route 1: TRL 1.13.0

Supports custom multi-turn generation and per-token loss masking. Both hooks experimental.
Line refs are trl/trainer/grpo_trainer.py unless noted.

### Custom generation: yes

- `GRPOTrainer(rollout_func=...)`, `(prompts, trainer) -> dict` (258)
- Must return `prompt_ids`, `completion_ids`, `logprobs`; extra keys go to reward funcs (2242)
- With `use_vllm`, weights are synced to vLLM before each step's rollout_func call (2233-2236)
- Loop can call `trainer.vllm_generation.generate(prompt_ids, None, n)` (generation/vllm_generation.py:550)
- `generation_kwargs` merge into vLLM `SamplingParams`, so `stop=["</search>"]` passes through
  in colocate mode (vllm_generation.py:672, 683)
- The docstring says rollout_func receives prompts "with no duplication", but `RepeatSampler`
  (1281) repeats each prompt `num_generations` times and colocate generation uses `n=1`
  (vllm_generation.py:663). So one completion per prompt entry is what the trainer expects.
  Verified by the smoke run, see below.

### Loss masking: yes, full control

- Return `env_mask` (1 = policy token, 0 = injected) from rollout_func (2293-2295)
- Becomes `tool_mask`; `loss_mask = completion_mask * tool_mask` (2519, 2986, 3120)
- Completion-length metrics count only mask=1 tokens (2300)
- Injected tokens can carry `None` logprobs: masked first, then `nan_to_num` in the
  importance-sampling correction (2683-2687)

### Built-in alternative: `tools` / `environment_factory`

- Trainer runs the tool loop and masks tool output itself (1990, 2193)
- Requires chat-template tool calling (618): trajectories become JSON tool calls,
  not the `<search>` tags in PROJECT.md section 1. Format change = Ayesha's call. Not taken.

### Single GPU

- `vllm_mode='colocate'` default, `vllm_gpu_memory_utilization=0.3`
- **Do not enable sleep mode with a multi-turn rollout.** Each `generate()` call sleeps at
  level 2 and discards the weights (vllm_generation.py:750); the next call re-pushes them
  (589). One turn = one full weight sync.

### Defaults that conflict with PROJECT.md (grpo_config.py)

- `beta` defaults to 0.0: no KL term, no reference model loaded (676). PROJECT.md section 5
  wants KL beta 0.04 as the main anti-hacking guard -> must be set explicitly; costs memory.
- `loss_type` defaults to `"dapo"` (796), `scale_rewards` to `"group"` (784). PROJECT.md does
  not specify either -> choose deliberately and record the choice in configs/.

### Consequence for `configs/base.yaml` (PR #3 / #12, Ayesha's file)

`env.stop_sequences` currently lists `</search>` **and** `</answer>`. TRL counts a completion
as truncated when its last token is not EOS or pad (2325). Stopping on `</answer>` therefore
reports every finished episode as clipped, and `mask_truncated_completions=True` would zero
the loss on all of them. Either stop only on `</search>` and let EOS end the answer, or append
EOS to the completion in the rollout. Raised on the config PR.

### Risks

- Hooks are experimental and warn on use (577-592) -> pin `trl` exactly
- `trl` is not in pyproject.toml / uv.lock -> adding it is Ayesha's change
- `max_completion_length` applies per `generate()` call; the total budget across turns is ours
- Memory: vLLM colocate + 1.5B LoRA training in 16 GB
- Token-in-token-out: `completion_ids` = generated ids + tokenized `<info>` ids. Never
  re-tokenize decoded text, or `env_mask` lines up with the wrong tokens.

### Smoke run (`spikes/smoke_trl_rollout.py`, 2026-10-14)

Qwen2.5-0.5B-Instruct, group 4, 2 steps, fake retriever, `env_mask` on injected `<info>`, KL
beta 0.04, plus a hook on `lm_head` that checks the gradient at injected positions after
backward. Log: `outputs/smoke_0.5b.log`.

| Check | Step 1 | Step 2 |
|---|---|---|
| prompts per rollout_func call / repeated in groups | 4 / True | 4 / True |
| `tool_mask` present in loss inputs | True | True |
| injected tokens: rollout vs loss-mask zeros | 90 vs 90 | 45 vs 45 |
| max abs gradient on injected tokens | **0.0** | **0.0** |
| policy tokens with non-zero gradient | 100% | 100% |
| loss (finite) | -0.2643 | 0.6227 |
| GPU device used | 7.85 GiB | 8.78 GiB |
| TRL `completions/clipped_ratio` | 0.50 | 0.75 |
| TRL `completions/mean_length` (mask=1 only) | 21 | 48 |

Two things the run taught that reading could not:

- **Tag detection must not use `endswith`.** 4 turns stopped on the stop string but only 2
  decoded to a string ending in `</search>`. Likely mechanism: vLLM truncates the returned
  *text* at the stop string while the returned token ids include the whole final token, which
  can carry trailing characters. Detect by containment on a stripped string. The loop here used
  `endswith` and therefore dropped real searches.
- **clipped_ratio is an "never answered" gauge, not an error.** With answers ending on EOS, a
  clipped completion is one that ran out of turns while still searching. 0.5 and 0.75 here are
  a 0.5B model behaving like a 0.5B model, not a bug.

The 0.5B policy answered in Chinese (`<answer>美国</answer>`). Irrelevant to the mechanics being
tested, and a reminder that the dev policy is 1.5B for a reason.

## Route 2: agentic-RL library

### Activity snapshot (GitHub API, 2026-09-17)

| repo | stars | last push | open issues | latest release |
|---|---|---|---|---|
| volcengine/verl | 23471 | 0d | 1228 | v0.9.0 (34d) |
| PeterGriffinJin/Search-R1 | 5429 | 307d | 36 | none |
| OpenPipe/ART | 10738 | 0d | 142 | v0.5.19 (34d) |
| PrimeIntellect-ai/verifiers | 4627 | 0d | 189 | v0.3.1 (23d) |
| NovaSky-AI/SkyRL | 2318 | 0d | 467 | skyrl-v0.3.0 (62d) |

### Shortlist cut (README scan)

- Search-R1: same task, built on verl; paper introduces retrieved-token masking. Stale -> reference only.
- verifiers: environment library; training delegated to prime-rl / hosted. Not a trainer -> out.
- ART: GRPO + LoRA, local GPU or serverless. Trajectories are chat messages; token-level
  loss control not documented -> weak fit for our tag protocol.
- SkyRL: multi-turn, cites a Search-R1 port. Smaller community, 467 open issues.
- verl: most mature, Search-R1 lineage -> evaluated below.

### verl v0.9.0

- Maturity: highest of the candidates. Regular releases, large community.
- Maintenance: active (pushed today, release 34d ago), 1228 open issues.
- Multi-turn: yes, documented on the SGLang rollout backend. vLLM async tool calling is a
  prototype. SGLang is not in our stack.
- Tool format: docs show JSON tool calls via chat template only. `<search>` tags undocumented.
- Loss masking: yes but internal. Mask comes from delta tokenization of chat messages;
  no per-token mask we supply directly.
- Single GPU: not a documented path. Built on Ray + FSDP/Megatron. LoRA supported (peft).
  Signal: verl-omni #618, LoRA weight sync fails on 1 GPU (FSDP NO_SHARD),
  torch 2.13.0+cu130 = our build. Open, fix proposed.
- Risks: Ray + FSDP + SGLang on one 16 GB Blackwell card with 15 GB RAM; less masking
  control; our tag protocol is off the documented path.

## Route 3: hand-rolled GRPO

Line counts are estimates (code only, no tests/docstrings), not measurements.

### Needed on every route (TRL rollout_func needs these too)

| Module | Job | Est. lines |
|---|---|---|
| env/protocol.py | prompt template, stop sequences | 40-60 |
| env/parser.py | tag parsing, malformed cases | 80-120 |
| env/episode.py + train/rollout.py | batched multi-turn vLLM loop, token-in-token-out, env_mask, logprobs, budgets | 150-220 |
| train/masking.py | build + validate loss mask | 30-50 |
| Subtotal | | 300-450 |

### Hand-rolled only (TRL provides these)

| Component | Job | Est. lines |
|---|---|---|
| Weight sync | LoRA -> vLLM each step | 40-80 |
| Memory handoff | vLLM sleep/wake, gpu util split, one GPU | 20-40 |
| Log-probs | policy / old / ref (ref = PEFT disable_adapter, no 2nd copy), chunked | 60-100 |
| Advantages | group mean/std, zero-std groups | 20-30 |
| Loss | clipped ratio, KL (k3), token-level normalization | 40-70 |
| Train loop | optimizer, grad accum, clipping, LoRA checkpoints, W&B metrics (PROJECT.md s9), seeding | 120-180 |
| Subtotal | | 300-500 |

PROJECT.md's "~300 lines" matches the GRPO core alone. Full hand-rolled route ~600-950 lines.
At the ~400-line PR limit, the hand-rolled-only part is 1-2 extra PRs before the overfit test runs.

### Memory arithmetic

Qwen2.5-1.5B config: hidden 1536, 28 layers, 2 KV heads, head_dim 128, vocab 151936, tied
embeddings. Back-of-envelope, measure before trusting.

- Weights bf16: ~1.54B params -> ~3.1 GB per copy. HF policy + vLLM = ~6.2 GB
- LoRA r=16 on 7 projections: ~18.5M params; fp32 weights+grads+Adam ~0.3 GB
- KV cache bf16: ~28 KB/token (GQA). 8 x 4k tokens ~0.9 GB. Not the bottleneck.
- Logits: vocab 151936 x 4 bytes (fp32) ~0.6 MB/token. At ~3k-token trajectories, one group
  of 8 in a single pass is ~14.6 GB -> log-probs must be chunked.
- Activations under gradient checkpointing: unmeasured. CPU RAM peak with two runtimes: unmeasured.

### Risks (ranked)

1. Silent loss/masking bugs: curves look normal. Guard: test_masking.py, a zero-grad check on
   injected tokens, overfit-20.
2. Stale weight sync: vLLM samples from an old policy, silently off-policy. Guard: after sync,
   vLLM token logprobs ~= HF logprobs on the same sequence.
3. Logits memory (above): OOM on the first real batch if not chunked.
4. vLLM vs HF logprob mismatch -> noisy ratios. Recompute old logprobs with HF (extra forward)
   or add IS correction (TRL has this built in).
5. One 16 GB GPU + 15.3 GB RAM holding two model runtimes.
6. Schedule: extra PRs before P3's overfit test; P3 slips.

### What hand-rolled buys

No experimental API, no new dependency, every masking line reviewable in our repo,
learning value.
