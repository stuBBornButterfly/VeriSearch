# VeriSearch

**RL-trained retrieval reasoner with a faithfulness reward.**

Train a small LLM with GRPO to interleave reasoning and retrieval: the model decides *when* to search, *what* to search for, and *when it has enough evidence* to answer. Rewarded on verifiable correctness, citation faithfulness, and retrieval cost.

Two-person research project. This file is the complete brief and the working agreement.

---

## 0. Team, ownership, process

### Roles

| Person | Role | Share |
|---|---|---|
| Ayesha Siddika | Project lead, research partner, reviewer, merger | 70% |
| Tahsinul Haque Dhrubo | Research engineer, contributor | 30% |

Ayesha leads. She sets priority, assigns work, reviews and merges all code.

### Ownership split

**Dhrubo (30%) — RL core.** This is the entire scope. Do not pick up work outside it without a Trello card assigned by Ayesha.

- `verisearch/rewards/` — all reward components and composition
- `verisearch/train/` — GRPO loop, rollout generation, token masking
- `verisearch/env/` — episode loop, output parser, search-turn protocol
- Reward design experiments and the lambda sweep
- `writeup/rl_method.md` and `writeup/failure_cases.md`

**Ayesha (70%) — everything else.**

- `verisearch/retrieval/` — corpus, FAISS index, embedding service
- `verisearch/data/` — dataset loaders, splits, preprocessing
- `verisearch/eval/` — harness, baselines, metrics, statistics, plots
- `verisearch/serve/` — vLLM serving and load benchmarks
- `verisearch/faithfulness/` — NLI scorer service (Dhrubo consumes it, does not build it)
- Jenkins CI/CD pipeline, entire ownership
- Repo setup, tooling config, type-check config, release
- `README.md`, `writeup/report.md`, results tables

**Rule:** if a task is not on a Trello card assigned to Dhrubo, Dhrubo does not compute it. Ask, do not assume.

### Trello board

Board: `VeriSearch`. Lists in order:

`Backlog` → `Ready` → `In progress` → `In review (PR open)` → `Blocked` → `Done`

Card template:

```
Title:      [PHASE-N] Short imperative task
Assignee:   Ayesha | Dhrubo
Labels:     rl-core | retrieval | eval | infra | ci | writeup
Checklist:  acceptance criteria, one line each
Link:       PR URL once opened
```

Rules:
- No work starts without a card in `Ready` assigned to you.
- Move your own card to `In progress` when you start, `In review` when the PR opens.
- Only Ayesha moves cards to `Done`.
- Blocked more than 24h → move to `Blocked`, comment the reason, tag Ayesha.

### Git and PR workflow

- `main` is protected. No direct pushes by anyone.
- Branch naming: `<initials>/<phase>-<slug>`. Example `thd/p3-grpo-loss-masking`.
- Dhrubo opens PRs. Ayesha reviews and merges. Never self-merge.
- One card, one branch, one PR. Keep PRs under ~400 changed lines.
- PR must be green in Jenkins before review. Red PR is not reviewable.
- Conventional Commits: `feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`.

PR template (`.github/pull_request_template.md`):

```markdown
## Trello
<card link>

## What
One paragraph.

## Why
One paragraph.

## How to verify
Exact commands.

## Checklist
- [ ] mypy --strict passes
- [ ] ruff passes
- [ ] tests added for new logic
- [ ] no fabricated numbers
- [ ] public functions typed and docstringed
```

---

## 1. What the project is

The model is trained to produce this trajectory:

```
<think> reasoning </think>
<search> query </search>
→ environment injects <info> retrieved passages </info>
<think> reasoning </think>
<search> refined query </search>
→ environment injects <info> passages </info>
<answer> final answer </answer>
```

RL teaches the retrieval policy. The model learns to stop searching when evidence suffices, to refine queries after weak results, and to ground claims in what it actually retrieved.

### Contribution over prior work

Standard RL-for-search work rewards answer correctness only. VeriSearch adds two terms:

1. **Faithfulness reward.** Claims in the answer must be entailed by retrieved passages. Directly penalizes hallucination.
2. **Retrieval cost penalty.** Charges per search call, so the policy learns evidence sufficiency rather than searching maximally.

The headline claim is a joint improvement: higher multi-hop accuracy **and** lower unsupported-claim rate **at fewer retrievals** than a prompted RAG baseline.

---

## 2. Success criteria

Binary. All must be true.

- [ ] Trained checkpoint beats base model and vanilla RAG on held-out multi-hop F1.
- [ ] Unsupported-claim rate strictly lower than the prompted RAG baseline.
- [ ] Average retrievals per question lower than or equal to the prompted baseline.
- [ ] Results hold on an out-of-distribution set never trained on.
- [ ] All accuracy numbers carry 95% bootstrap confidence intervals.
- [ ] 4 baselines evaluated on identical eval sets with the identical retriever.
- [ ] Lambda sweep of at least 4 points, plotted as an accuracy vs retrieval-count frontier.
- [ ] Jenkins pipeline green on `main`, all stages.
- [ ] `mypy --strict` clean across the package, zero ignores outside `pyproject.toml`.
- [ ] Report published with plots, 3+ documented failure cases, W&B run links.

### Non-goals

Beating frontier models. Novel algorithms. Web UI. Production hardening. Multi-tool agents, that is a different project.

---

## 3. Engineering standards

Enforced in CI. A PR violating any of these does not merge.

### Type safety

- Python 3.11+. `mypy --strict` on the whole package.
- **No bare `Any`.** If unavoidable, `# type: ignore[code]` with a comment explaining why.
- `from __future__ import annotations` in every module.
- Public function signatures fully annotated, including return types.
- Data crossing a module boundary is a **Pydantic v2 model** or a frozen dataclass, never a raw dict.
- Enums for closed sets: tool names, turn types, failure reasons.
- `TypedDict` only for external JSON shapes at the boundary. Convert to models immediately.
- Numeric arrays: `numpy.typing.NDArray[np.float32]`, not `np.ndarray`.

`pyproject.toml` mypy config:

```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true
disallow_any_explicit = true
disallow_any_generics = true
plugins = ["pydantic.mypy"]
```

### Design principles

- **Single responsibility.** One module, one job. Reward components do not know about training. Training does not know about FAISS.
- **Dependency inversion.** Training depends on a `Retriever` protocol, not on a FAISS class. Enables fakes in tests.
- **Protocols over inheritance.** `typing.Protocol` for `Retriever`, `RewardComponent`, `FaithfulnessScorer`.
- **Pure reward functions.** No I/O, no globals, deterministic. Trivially testable.
- **Config as data.** Pydantic settings from YAML. No magic numbers in code.
- **Fail loud.** Typed exceptions. Never a silent `except Exception: pass`.
- **Immutability by default.** Frozen dataclasses for trajectories and rollouts.

### Core contracts

These interfaces are the boundary between Dhrubo's and Ayesha's code. Agree them in Phase 1, then treat as stable.

```python
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Passage]: ...

class FaithfulnessScorer(Protocol):
    def score(self, answer: str, passages: list[Passage]) -> FaithfulnessResult: ...

class RewardComponent(Protocol):
    name: str
    def __call__(self, traj: Trajectory) -> float: ...
```

Both sides code against these from day one. Dhrubo uses a fake `Retriever` until Ayesha's is ready. Neither side blocks the other.

### Testing

- `pytest`, coverage floor 80% on `rewards/`, `env/`, `eval/metrics/`.
- Unit tests for every reward component, table-driven.
- Parser tests must cover malformed output: unclosed tags, empty query, no answer, nested tags.
- **`test_masking.py` is mandatory and blocking.** Asserts injected `<info>` tokens carry zero loss weight.
- Integration test: 5 questions end to end with a fake retriever, CPU only, under 60 seconds.
- No network in unit tests.

---

## 4. Repository structure

```
VeriSearch/
├── README.md
├── PROJECT.md
├── Jenkinsfile                      # Ayesha
├── pyproject.toml
├── .github/pull_request_template.md
├── configs/
│   ├── base.yaml
│   ├── dev_1.5b.yaml
│   ├── train_3b.yaml
│   └── sweep_lambda.yaml
├── verisearch/
│   ├── types.py                     # shared models, enums, protocols  [SHARED, agreed in P1]
│   ├── config.py                    # Pydantic settings                [Ayesha]
│   ├── env/                                                          # [Dhrubo]
│   │   ├── episode.py               # rollout loop, turn control
│   │   ├── parser.py                # <think>/<search>/<answer> parsing
│   │   └── protocol.py              # prompt template, stop tokens
│   ├── rewards/                                                      # [Dhrubo]
│   │   ├── correctness.py           # EM / F1
│   │   ├── faithfulness.py          # consumes FaithfulnessScorer
│   │   ├── format.py
│   │   ├── retrieval_cost.py
│   │   └── compose.py
│   ├── train/                                                        # [Dhrubo]
│   │   ├── grpo.py                  # entrypoint
│   │   ├── rollout.py               # batched vLLM generation
│   │   └── masking.py               # loss masking for injected tokens
│   ├── retrieval/                                                    # [Ayesha]
│   │   ├── index.py                 # FAISS build and load
│   │   ├── embedder.py
│   │   ├── service.py               # HTTP retrieval service
│   │   └── client.py                # implements Retriever protocol
│   ├── faithfulness/                                                 # [Ayesha]
│   │   ├── nli.py                   # NLI entailment scorer
│   │   └── claims.py                # answer → atomic claims
│   ├── data/                                                         # [Ayesha]
│   │   ├── datasets.py
│   │   ├── splits.py
│   │   └── build_corpus.py
│   ├── eval/                                                         # [Ayesha]
│   │   ├── harness.py
│   │   ├── baselines.py
│   │   ├── metrics.py               # F1, EM, CIs, bootstrap
│   │   └── frontier.py              # sweep plots
│   ├── serve/                                                        # [Ayesha]
│   │   ├── server.py
│   │   └── benchmark.py
│   └── testing/                                                      # [Ayesha]
│       └── fakes.py                 # fake Retriever, fake scorer, make_trajectory
├── scripts/
│   ├── 00_setup_check.py
│   ├── 01_build_index.sh
│   ├── 02_train.sh
│   ├── 03_eval.sh
│   ├── 04_sweep.sh
│   └── 05_serve_bench.sh
├── tests/
│   ├── conftest.py                  # fixtures over verisearch/testing's fakes
│   ├── test_parser.py               # Dhrubo
│   ├── test_rewards.py              # Dhrubo
│   ├── test_masking.py              # Dhrubo, blocking
│   ├── test_episode.py              # Dhrubo
│   ├── test_retrieval.py            # Ayesha
│   ├── test_metrics.py              # Ayesha
│   └── test_integration.py          # Ayesha
├── results/{raw,tables,figures}/
└── writeup/
    ├── report.md                    # Ayesha
    ├── rl_method.md                 # Dhrubo
    └── failure_cases.md             # Dhrubo
```

---

## 5. Technical choices

| Item | Choice | Note |
|---|---|---|
| Policy, dev | `Qwen2.5-1.5B-Instruct` | All debugging here |
| Policy, final | `Qwen2.5-3B-Instruct` | Only after a full run succeeds |
| Algorithm | GRPO | No value network, fits 16GB |
| Adapter | LoRA r=16, alpha=32, dropout=0.05 | Attention + MLP projections |
| Precision | bf16, 4-bit base if needed | NF4 |
| Group size | 8 dev, 8–16 final | |
| KL beta | 0.04 start, tune | Main anti-hacking guard |
| Rollouts | vLLM | Dominates wall-clock, do not skip |
| Embedder | `BAAI/bge-small-en-v1.5` | |
| Index | FAISS HNSW | Memory-mapped |
| NLI | A small entailment model, e.g. DeBERTa-v3 MNLI class | Runs offline, not in the hot loop |
| Logging | Weights & Biases | Public runs |

### The hard part

Multi-turn RL is not the default path in most RL libraries. Standard GRPO assumes single-turn: prompt in, completion out. Here the model generates, pauses for retrieval, continues, all as one trajectory.

Three routes. Dhrubo evaluates all three against installed versions in Phase 1, timebox 2 days, then commits:

1. TRL with a custom rollout hook, if the installed version supports supplying your own generation function.
2. A purpose-built agentic-RL library, evaluated for maturity and single-GPU support at build time.
3. Hand-rolled GRPO, roughly 300 lines. Most control, best learning value. Default fallback.

**Blocking detail:** injected `<info>` tokens were not generated by the policy and must be masked out of the loss. Unmasked, the model learns to predict its own retrieval results. Invisible in training curves. `test_masking.py` exists for this.

---

## 6. Reward specification

```
R = w_c · correctness
  + w_a · faithfulness
  + w_f · format_valid
  − λ  · (searches / SEARCH_BUDGET)
  − P  · penalties
```

| Term | Definition | Range | Weight |
|---|---|---|---|
| correctness | Token F1 vs gold, plus EM bonus | [0,1] | `w_c = 1.0` |
| faithfulness | Fraction of atomic claims entailed by retrieved passages | [0,1] | `w_a = 0.5` |
| format_valid | Well-formed tags, answer present, terminated in budget | {0,1} | `w_f = 0.2` |
| retrieval cost | Search count normalized, `SEARCH_BUDGET = 4` | [0,∞) | `λ` swept |

Penalties: duplicate query `−0.1` each, no answer by max turns `−0.5`, empty search query `−0.05`.

### Design rules

- **Correctness must dominate.** Verify a no-search policy scores worse than a searching one, empirically, before training.
- **Faithfulness is computed offline** against retrieved passages. It is verifiable, not an LLM judge in the hot loop.
- **Lambda is the experiment, not a hyperparameter.** Sweep `{0.0, 0.02, 0.05, 0.1}`. Each value is a frontier point.
- Format reward stays small. Large format rewards produce beautifully formatted wrong answers.

### Anti-hacking guards

| Hack | Guard | Metric to watch |
|---|---|---|
| Never search, answer from parametric memory | Correctness dominance + faithfulness needs passages | avg searches → 0 |
| Copy passages verbatim to farm faithfulness | Correctness gates it; add length penalty | answer length ↑ |
| Search once, ignore results | Faithfulness requires entailment | faithfulness flat |
| Degenerate repetition | KL penalty + repetition penalty at generation | completion length ↑ |

If hacking occurs, **do not just patch it. Document it.** It is the most valuable content in the writeup.

---

## 7. Data

| Split | Source | Owner |
|---|---|---|
| Train | HotpotQA (distractor) + 2WikiMultihopQA | Ayesha |
| Dev | Held-out slice | Ayesha |
| Test, in-distribution | Held-out slice | Ayesha |
| Test, OOD | Musique + Bamboogle | Ayesha |
| Corpus | Wikipedia paragraph dump | Ayesha |

Start with 2000–5000 train examples. RL needs far less data than SFT and each example is expensive.

---

## 8. Baselines

All on identical eval sets with the identical retriever. Owner: Ayesha.

1. **Base model, no retrieval.** Floor.
2. **Vanilla RAG.** Single retrieval on the question, then answer.
3. **Prompted iterative RAG.** Same interleaved protocol, careful prompt, no training. **The baseline that matters.**
4. **SFT-only.** Fine-tuned on successful baseline-3 trajectories. Separates "RL helped" from "training helped."

---

## 9. Metrics

**Training** (Dhrubo logs): mean reward, per-component reward, within-group reward std, KL from reference, completion length, searches per rollout, duplicate-query rate, grad norm.

**Evaluation** (Ayesha owns):

| Metric | Note |
|---|---|
| Multi-hop F1 and EM | 95% bootstrap CI, 1000 resamples |
| Unsupported-claim rate | Headline hallucination metric |
| Avg searches per question | Cost axis of the frontier |
| Retrieval precision@k | Is the policy querying well |
| Answer length | Hack canary |
| Failure breakdown | wrong / no answer / turn limit / empty retrieval |

**Serving** (Ayesha): throughput req/s, p50 p95 p99, at concurrency 1, 8, 32.

Every reported number carries a CI. A 2-point gain on 500 examples is noise.

---

## 10. Jenkins pipeline

**Owner: Ayesha. Sole owner.** Dhrubo does not modify `Jenkinsfile`.

Triggers: every PR, every push to `main`, nightly on `main`.

| Stage | Runs | Blocking |
|---|---|---|
| `lint` | `ruff check`, `ruff format --check` | yes |
| `typecheck` | `mypy --strict verisearch/` | yes |
| `unit` | `pytest tests/ -m "not gpu"`, coverage gate 80% | yes |
| `integration` | 5-question end-to-end, fake retriever, CPU | yes |
| `smoke-train` | 5 GRPO steps on 1.5B, tiny slice, asserts loss is finite | nightly only |
| `eval-regression` | Fixed 100-example eval, asserts F1 has not dropped >2 pts | nightly only |
| `artifact` | Publish coverage + eval report | `main` only |

Rules:
- Red pipeline blocks review. Do not ask for review on a red PR.
- `main` must never be red. A red `main` is the top priority for both people.
- No `@Ignore`, no skipped tests to make CI green.

---

## 11. Phases

Exit criteria are hard. No phase starts before the previous one passes, verified by running something.

| Phase | Scope | Dhrubo | Ayesha | Exit criterion |
|---|---|---|---|---|
| **P0** Setup | Repo, tooling, CI skeleton, hardware check | Review | Own | Jenkins green on an empty repo, `mypy --strict` clean, hardware report in README |
| **P1** Contracts + env | `types.py` agreed, parser, episode loop, retrieval index | env/ + parser | retrieval/ + data/ + fakes | Untrained base model runs 100 questions with real retrieval, produces typed trajectory JSONL |
| **P2** Reward | All reward components + faithfulness scorer | rewards/ | faithfulness/ | Reward distribution over base rollouts non-degenerate (15–60% non-trivial). 30 trajectories hand-checked and agreed by both. |
| **P3** Training | GRPO loop, masking, overfit test | Own entirely | eval harness + baselines | Overfits 20 questions. Then a full 1.5B run improves eval F1 over base. |
| **P4** Sweep | 4-point lambda sweep on 3B, frontier plot | Run sweeps | Evaluate + plot | Frontier plot exists, one trained point beats baseline 3 on F1 and unsupported-claim rate at ≤ its search count |
| **P5** Serve | vLLM serving + load benchmark | — | Own | Benchmark table with real numbers at 3 concurrency levels |
| **P6** Writeup | Report, failure cases, README | `rl_method.md`, `failure_cases.md` | `report.md`, README, tables | A stranger understands the result from the README in 60 seconds |

Realistic total: 6 to 8 weeks part-time across two people. P1 and P2 take longer than feels reasonable. Normal.

---

## 12. Risks

| Risk | Signal | Mitigation |
|---|---|---|
| Multi-turn RL library friction | Days lost on API fights | 2-day timebox in P1, then hand-roll |
| Contract churn between owners | Rework in both halves | Freeze `types.py` at end of P1, changes need a PR both approve |
| Reward all zeros | Flat reward from step 0 | Caught at P2 exit. Easier tasks or bigger base model |
| OOM at 3B | CUDA OOM | Stay 1.5B, cut group size, gradient checkpointing, 4-bit |
| Rollouts too slow | Hours per step | vLLM from day one, shrink dev eval set |
| Reward hacking | Searches → 0, length explodes, F1 flat | Section 6 guards. Document, do not just patch. |
| One person blocks the other | Idle Trello cards | Protocols + fakes from P1. Neither side ever waits on the other's implementation. |

---

## 13. Target resume bullets

Write only from logged numbers. Placeholders marked. Credit the collaboration honestly.

> Co-developed an RL-trained retrieval agent, owning the GRPO training loop and reward design: trained a 3B LLM to interleave reasoning and retrieval, raising multi-hop QA F1 by **X** points over a prompted iterative-RAG baseline while cutting unsupported claims by **Y%** at fewer retrievals per question.

> Designed a composite reward combining verifiable correctness, NLI-based citation faithfulness, and a retrieval-cost penalty; swept the cost coefficient to produce an accuracy-versus-retrieval Pareto frontier and documented **N** reward-hacking behaviors with mitigations.

> Delivered under a strict-typed Python codebase (`mypy --strict`, Pydantic contracts, 80% coverage gate) with Jenkins CI enforcing lint, type, test, and nightly eval-regression stages.

---

## 14. Kickoff prompt for a new Claude Code session

Open a session in this folder and paste:

```
Read PROJECT.md in full. It is the complete brief and working agreement.

Context: two-person project. Ayesha Siddika leads and owns 70%. I am Dhrubo,
I own the RL core only (30%): verisearch/env, verisearch/rewards,
verisearch/train, and my two writeup files. I open PRs, Ayesha reviews and
merges. Work is tracked on Trello. Jenkins is hers, I never touch it.

Rules for you:
- Do not write code outside my ownership boundary in Section 0. If a task
  needs Ayesha's code, stub it against the protocol in Section 3 and say so.
- Strict type safety, Section 3. mypy --strict must pass. No bare Any.
- Phases are sequential. Do not start the next before the exit criterion passes.
- Never fabricate a benchmark number.
- Answer me short and on point. No long preamble, no restating what I said.

Start with P1, my half only:
1. Propose verisearch/types.py: the shared models, enums, and protocols from
   Section 3. This is the contract with Ayesha, so flag every decision she
   needs to approve.
2. Implement verisearch/env/parser.py with full typing and table-driven tests
   covering the malformed cases in Section 3.
3. Implement verisearch/env/episode.py against the Retriever protocol, using
   a fake retriever so it runs without her code.
4. Give me the Trello cards to create for this work, one line each.

Stop there. Do not touch retrieval, data, eval, serve, or Jenkinsfile.
```

---

## 15. Working rules

1. Cards before code. No card, no work.
2. Ownership boundaries are hard. Outside your half, open a card, do not open an editor.
3. Never self-merge.
4. Never fabricate a number. Every figure comes from a logged run.
5. Red CI is not reviewable. Fix before requesting review.
6. Freeze `types.py` after P1. Changes need both approvals.
7. Read raw trajectories after every significant change. Metrics hide behavior.
8. Log surprises to `failure_cases.md` immediately. You will forget them.
9. Small first: 1.5B before 3B, 20 questions before 2000, 100 eval before 500.
