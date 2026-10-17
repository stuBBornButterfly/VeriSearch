# VeriSearch — results

RL-trained retrieval reasoning with a faithfulness reward. Two-person project;
see [Contributions](#contributions).

> ## Status: skeleton. No number in this file has been measured.
>
> Every result cell reads `{{?}}`. The tables, their rows and their columns are
> fixed now — while the metric schema is being designed — so that filling them
> in later is transcription and not design. PROJECT.md section 15 rule 4: *never
> fabricate a number*. A table that already exists is the cheapest way to keep
> that rule, because the alternative at write-up time is building the table
> around whichever numbers happen to be on hand.
>
> **Before this file is published, `grep '{{' writeup/report.md` must return
> nothing.** A leftover `{{?}}` is an unmeasured claim.
>
> Skeleton written 2026-10-12. Nothing below is a result.

---

## How every table in this file gets filled

Not by hand. Each table has a command above it that prints exactly those rows,
and the rule is paste, never retype. The commands:

```bash
# Once per experiment. Pins the eval set and the retriever every run shares.
CONTEXT_ONLY=1 scripts/03_eval.sh

# Once per system in the baseline table.
RUN_NAME=prompted KIND=prompted_iterative RESULTS=results/raw/prompted.jsonl scripts/03_eval.sh

# The frontier table and figure, from the sweep manifest.
FIGURE=results/figures/frontier.png scripts/04_sweep.sh

# Serving, at concurrency 1/8/32.
scripts/05_serve_bench.sh
```

`EvalReport.summary()`, `Comparison.summary()`, `frontier_table()` and
`BenchReport.table()` are the four functions that produce these numbers. If a
row here has no function behind it, that row is the bug.

`tests/test_report.py` checks this file against the code: every metric field on
`EvalReport`, every `BaselineKind`, every `FailureReason`, every `Comparison`
delta and every lambda in `configs/sweep_lambda.yaml` must have a row or a
column here. Add a metric to the model and this file fails until it has a home.
Rename one and it fails until the row is renamed with it. Rows are tied to field
names by `field:` HTML comments, which carry the coupling without rendering.

---

## 1. The claim

One sentence, and all three clauses have to hold at once:

> A GRPO-trained policy with a faithfulness reward and a retrieval-cost penalty
> reaches **higher** multi-hop F1 and a **lower** unsupported-claim rate at **no
> more** retrievals per question than a prompted iterative-RAG baseline.

Two out of three is a different and much weaker result. `JointClaim` in
`verisearch/eval/baselines.py` evaluates the conjunction and will not report a
partial pass as a pass; `compare_runs` fills it from paired bootstrap deltas.

### Verdict

<!-- generated: the `claim` block of `Comparison.summary()`. Paste it, do not summarise it. -->

```
{{claim-verdict}}
```

| Clause | Required | Delta (95% CI) | Holds |
|---|---|---|---|
| Accuracy up <!--field:f1_delta--> | interval entirely above 0 | `{{?}}` | `{{?}}` |
| Unsupported claims down <!--field:unsupported_rate_delta--> | interval entirely below 0 | `{{?}}` | `{{?}}` |
| Retrievals not up <!--field:searches_delta--> | interval not entirely above 0 | `{{?}}` | `{{?}}` |
| Exact match <!--field:exact_match_delta--> | reported, not required | `{{?}}` | — |

The three clauses are not symmetric on purpose. Accuracy and faithfulness have
to *move* — an interval straddling zero is not an improvement. Retrievals only
have to not get worse, because section 2 asks for "lower than or equal to".

Reference system: `prompted_iterative` (`REFERENCE_BASELINE`). Paired on
`{{n-paired}}` examples; the unsupported-rate delta is scoped to the
`{{n-paired-unsupported}}` examples where **both** systems answered, since an
unanswered question has no claims to check and dropping it from one side only
would bias the delta.

---

## 2. Setup

Nothing in this section is a result, so it can be filled in now and is.

| | |
|---|---|
| Dev policy | `Qwen2.5-1.5B-Instruct` |
| Final policy | `Qwen2.5-3B-Instruct` |
| Algorithm | GRPO, group 8, no value network, KL beta 0.04 |
| Adapter | LoRA r=16 alpha=32 dropout=0.05, attention + MLP |
| Embedder | `BAAI/bge-small-en-v1.5`, 384 dims |
| Index | FAISS HNSW |
| NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, entailment threshold 0.5 |
| Reward | `1.0*correctness + 0.5*faithfulness + 0.2*format - lambda*(searches/4) - penalties` |
| Search budget | 4 |
| Training data | HotpotQA (distractor) + 2WikiMultihopQA |
| In-distribution test | held-out slice |
| Out-of-distribution test | Musique + Bamboogle, never trained on |
| Hardware | `{{hardware}}` — see README, and it is not settled yet |

### The eval context

Section 8 requires every baseline on identical eval sets with the identical
retriever. That is a promise until it is a checked fact, so it is one: every run
loads the same `context.json`, and `check_comparable` refuses two runs whose
contexts differ in any field.

<!-- generated: the `n=... k=... retriever=...` line of EvalReport.summary(). -->

| Field | Value |
|---|---|
| dataset / split | `{{?}}` |
| n examples | `{{?}}` |
| examples fingerprint | `{{?}}` |
| retriever fingerprint | `{{?}}` |
| k | `{{?}}` |

The retriever fingerprint is read out of the index's own `meta.json` by
`verisearch/cli.py eval-context`, and there is no flag to override it. A number
in this report that was measured against a different index is not comparable
with the rest of the table, and the fingerprint is how that gets caught instead
of assumed.

---

## 3. Baselines

The four systems of section 8 plus the system under test, each evaluated by the
same harness under the same context.

| System | What it isolates | Max searches |
|---|---|---|
| `no_retrieval` | the floor — how much of this is answerable from parametric memory | 0 |
| `vanilla_rag` | what one retrieval on the raw question buys over none | 1 |
| `prompted_iterative` | **whether RL beat a careful prompt.** The whole claim rests here | budget |
| `sft_only` | RL helped, as distinct from training on anything at all helped | budget |
| `trained` | the system under test | budget |

`no_retrieval` and `vanilla_rag` have structural search limits — 0 and 1 are
what those baselines *are*, not settings, and `resolved_search_budget` will not
let a config change them.

---

## 4. Main results, in distribution

<!-- generated: EvalReport.summary() per run. One column per `--kind`. -->

Every cell is a point estimate with a 95% bootstrap interval, 1000 resamples.

| Metric | `no_retrieval` | `vanilla_rag` | `prompted_iterative` | `sft_only` | `trained` |
|---|---|---|---|---|---|
| Multi-hop F1 <!--field:f1--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Exact match <!--field:exact_match--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Unsupported-claim rate <!--field:unsupported_rate--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Searches per question <!--field:searches_per_question--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Retrieval precision@k <!--field:precision_at_k--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Retrieval recall@k <!--field:recall_at_k--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Answer tokens <!--field:answer_tokens--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Duplicate-query episode rate <!--field:duplicate_query_episode_rate--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |

Reading notes, written before the numbers so they cannot be chosen to suit them:

- **Unsupported-claim rate is the headline hallucination metric**, and it is
  `n/a` rather than `0.0` for a run that never answered. A zero there would read
  as perfect faithfulness and mean the opposite.
- **Precision@k is `n/a` for `no_retrieval`** — it retrieved nothing, so the
  metric is undefined, not zero.
- **Recall@k is `n/a` on Bamboogle**, which ships no supporting titles.
- **Answer tokens is a hack canary**, not a quality measure. Section 6: copying
  passages verbatim farms the faithfulness reward, and the signal is length
  climbing while F1 stays flat.

### Failure breakdown

<!-- generated: the `failures` line of EvalReport.summary(). Shares, not counts. -->

| Reason <!--field:failure_breakdown--> | `no_retrieval` | `vanilla_rag` | `prompted_iterative` | `sft_only` | `trained` |
|---|---|---|---|---|---|
| `correct` <!--reason:correct--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| `wrong_answer` <!--reason:wrong_answer--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| `no_answer` <!--reason:no_answer--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| `turn_limit` <!--reason:turn_limit--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| `empty_retrieval` <!--reason:empty_retrieval--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |

**`turn_limit` will be near-empty, and that is a precedence artefact, not a
finding.** `classify_failure` checks `empty_retrieval`, then `no_answer`, then
`turn_limit`; a trajectory that ran out of turns also has no answer, so it is
counted under `no_answer`. The buckets are causes, ordered — the turn limit is
usually the symptom of one of the two above it. Do not report a near-zero
`turn_limit` as evidence the budget was generous.

### The headline comparison

<!-- generated: Comparison.summary() for trained vs prompted_iterative. -->

```
{{comparison-summary}}
```

Paired per example, not two independent intervals compared by eye. Two
overlapping marginal intervals can still hide a consistent per-question
difference, and two separated ones can be an artifact of which questions each
system happened to answer.

---

## 5. Out of distribution

Section 2 requires the result to hold on a set never trained on. Same metrics,
same harness, Musique + Bamboogle. A result that holds only in distribution is a
result about HotpotQA's distractor construction.

<!-- generated: EvalReport.summary() per run, OOD context. -->

| Metric | `no_retrieval` | `vanilla_rag` | `prompted_iterative` | `sft_only` | `trained` |
|---|---|---|---|---|---|
| Multi-hop F1 <!--field:f1--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Exact match <!--field:exact_match--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Unsupported-claim rate <!--field:unsupported_rate--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Searches per question <!--field:searches_per_question--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Retrieval precision@k <!--field:precision_at_k--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Retrieval recall@k <!--field:recall_at_k--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Answer tokens <!--field:answer_tokens--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| Duplicate-query episode rate <!--field:duplicate_query_episode_rate--> | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |

The OOD context is a **different** `context.json` with a different examples
fingerprint, so `check_comparable` will refuse to compare an OOD run against an
in-distribution one. That refusal is the feature. OOD numbers go in this table
and nowhere else.

| Field | Value |
|---|---|
| dataset / split | `{{?}}` |
| n examples | `{{?}}` |
| examples fingerprint | `{{?}}` |
| retriever fingerprint | `{{?}}` |

The retriever fingerprint should match section 2's. The *corpus* is shared; only
the questions change.

---

## 6. The lambda frontier

Section 6: lambda is the experiment, not a hyperparameter. Four runs differing
in exactly one float, from `configs/sweep_lambda.yaml`.

### Runs that must exist

A sweep where one point crashed leaves three reports, and three points draw a
curve that looks exactly like a complete one. `load_sweep_results` reads the
manifest, not the directory, and names the missing lambda rather than plotting
what it found.

| lambda | run name | report | W&B |
|---|---|---|---|
| 0 <!--lambda:0--> | `lambda_frontier-lam0` | `{{?}}` | `{{?}}` |
| 0.02 <!--lambda:0.02--> | `lambda_frontier-lam0.02` | `{{?}}` | `{{?}}` |
| 0.05 <!--lambda:0.05--> | `lambda_frontier-lam0.05` | `{{?}}` | `{{?}}` |
| 0.1 <!--lambda:0.1--> | `lambda_frontier-lam0.1` | `{{?}}` | `{{?}}` |

lambda=0 cannot be dropped for budget. Every other point is read relative to the
un-penalised run, and `SweepSpec` refuses a manifest without it. If the GPU
budget will not stretch to four 3B runs, cut from the middle and keep 0 and 0.1.

### Frontier

<!-- generated: `frontier_table()` via `scripts/04_sweep.sh`, which writes results/frontier.md. Paste the file. -->

{{frontier-table}}

![Accuracy versus retrieval-count frontier](../results/figures/frontier.png)

The path is the default `scripts/04_sweep.sh` writes to, so the image resolves
the moment the sweep runs and shows as broken until then. That is the intended
state - a placeholder here would have to be remembered, and a broken image
cannot be overlooked.

Accuracy on the figure is **F1, not EM** — EM on multi-hop answers is a coarse,
noisy axis, and a frontier drawn on it moves in steps. Every point in the sweep
is plotted, including dominated ones, marked. A frontier plot that silently omits
the settings that lost is a figure that cannot be argued with.

### Diagnostics

<!-- generated: `SweepDiagnostics.summary()`, included in the table block above. -->

```
{{sweep-diagnostics}}
```

Read this before the figure. The premise is that charging per search makes the
policy search less; if searches did not fall as lambda rose, there is no
trade-off and the figure is a flat scatter presented as a curve.
`verisearch/cli.py frontier` exits 1 in that case — the table is still written,
because the numbers are the evidence for the verdict, but the exit code stops the
sweep script from handing it to a figure.

**A dominated point with overlapping intervals is reported as `contested`, not
as dominated.** Dominance on point estimates alone is not a conclusion the data
carries.

---

## 7. Reward hacking observed

Section 6 lists the four hacks predicted and the guard against each. What
actually happened is in **[`failure_cases.md`](failure_cases.md)** (Dhrubo's
file — this section links, it does not restate), and section 6's instruction is
to document rather than patch.

| Predicted hack | Metric to watch | Observed? |
|---|---|---|
| Never search, answer from memory | searches per question goes to 0 | `{{?}}` |
| Copy passages verbatim to farm faithfulness | answer length climbs | `{{?}}` |
| Search once, ignore results | faithfulness flat | `{{?}}` |
| Degenerate repetition | completion length climbs | `{{?}}` |

Section 2 requires 3+ documented failure cases. Count so far: `{{n-failure-cases}}`.

---

## 8. Serving

Section 9's serving metrics, P5: the trained policy behind vLLM, measured end to
end at three concurrency levels. One request is one question and one whole
episode - every generate call, every retrieval round trip - because that is the
system this project built. A `/v1/completions` benchmark would time the part
vLLM already benchmarks.

<!-- generated: scripts/05_serve_bench.sh (BenchReport.table) -->

| Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Answered | Failed |
|---|---|---|---|---|---|---|
| 1 | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| 8 | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |
| 32 | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` | `{{?}}` |

`{{serve-footer}}`

How to read it:

- **Every cell is a point estimate and its 95% interval.** Latency percentiles
  use a percentile bootstrap over requests; throughput uses batch means over
  slices of the measured window, because requests are not independent samples
  of a rate.
- **A percentile needs a tail.** p99 is reported only with at least five
  requests above it, so 500 per level; below that the cell says `n/a` rather
  than printing a maximum under another name.
- **Answered** is the fraction of successful requests whose episode reached
  `<answer>`. A policy that gives up early is fast, and this column is what
  stops that from reading as good latency.
- **Failed** is counted, never dropped. Latency is computed over successful
  requests only, so a row with any failure is not this table, and the script
  exits 1 on it.
- **Throughput is biased low.** Warmup requests still in flight when the window
  opens finish uncounted, and the window closes on a drain. Both edges cost
  completions; neither adds any.

A latency number is not comparable across machines, so the hardware line is part
of the result and not a footnote. The script prints it under the table, read
from the policy server's `/health` - the machine that answered, not the one the
script was launched from.

---

## 9. Success criteria

Section 2 is binary — all of these must be true, and this is the audited list,
not a summary of one.

| Criterion | State | Evidence |
|---|---|---|
| Trained checkpoint beats base and vanilla RAG on held-out F1 | not measured | section 4 |
| Unsupported-claim rate strictly below prompted RAG | not measured | section 1 |
| Searches per question at or below the prompted baseline | not measured | section 1 |
| Holds on an OOD set never trained on | not measured | section 5 |
| Every accuracy number carries a 95% bootstrap CI | **met in code** | `bootstrap_ci`, 1000 resamples, and `EvalReport` has no bare float accuracy field |
| 4 baselines on identical eval sets with the identical retriever | **enforced in code**, not yet run | `EvalContext` + `check_comparable` |
| Lambda sweep, 4+ points, plotted as a frontier | not measured | section 6 |
| Jenkins green on `main`, all stages | not yet — `main` is still the P0 scaffold | section 10 of PROJECT.md |
| `mypy --strict` clean, zero ignores outside `pyproject.toml` | **met** | CI typecheck stage |
| Report published with plots, 3+ failure cases, W&B links | this file, unfilled | sections 6, 7, 11 |

Two of these are already true because they are properties of the code rather
than of a result, and they are marked that way rather than left blank: the CI
requirement is structural — there is no code path that reports an accuracy
without an interval — and the identical-retriever requirement is a fingerprint
check that exists and passes its tests.

---

## 10. Threats to validity

Written now, before any number exists, because the list is more honest this way.

**The eval set may be too small to resolve the claim.** At n=500 with roughly
+/-0.02 F1 intervals, a 0.03 F1 gap is not decisive — the intervals overlap. If
the sweep comes back looking like that, the fix is more eval examples, not a
better plot and not a weaker word than "beats". This is already visible in the
frontier tests, where a 0.03 gap between adjacent lambdas is correctly reported
as `contested`.

**Faithfulness is NLI entailment, not truth.** A claim entailed by a retrieved
passage is supported by that passage; if the passage is wrong, the claim scores
as faithful. The metric measures grounding, and this report should say grounding.

**`unsupported_rate` is only defined where a system answered.** Comparing it
across systems that answered different subsets is the trap `compare_runs` scopes
around; the count behind the delta is reported for exactly that reason.

**The reward penalises searches, and the frontier axis is searches.** A policy
can lower the cost axis without learning evidence sufficiency — by answering
sooner and worse. That is why accuracy and the unsupported rate are both on the
frontier table: a point that buys fewer searches while hallucinating more is not
on the frontier this project is claiming.

**Precision@k is title-level.** `normalize_title` matching counts a retrieved
passage as relevant when its title is a gold supporting title, which over-credits
a hit on the right article at the wrong paragraph.

**One seed.** `{{n-seeds}}` seeds per configuration. A single-seed RL result is a
sample of one from a high-variance process; if only one is affordable, the report
says one.

---

## 11. W&B runs

Section 2 requires run links. Public project: `{{wandb-project}}`.

| Run | Purpose | Link |
|---|---|---|
| `{{?}}` | base model eval | `{{?}}` |
| `{{?}}` | vanilla RAG eval | `{{?}}` |
| `{{?}}` | prompted iterative eval | `{{?}}` |
| `{{?}}` | SFT-only train + eval | `{{?}}` |
| `{{?}}` | trained policy, final | `{{?}}` |
| `{{?}}` | sweep, lambda=0 | `{{?}}` |
| `{{?}}` | sweep, lambda=0.02 | `{{?}}` |
| `{{?}}` | sweep, lambda=0.05 | `{{?}}` |
| `{{?}}` | sweep, lambda=0.1 | `{{?}}` |

A run whose config fingerprint is not in this table did not produce a number in
this report. `Config.fingerprint` is serialised into run metadata so the mapping
is checkable after the fact.

---

## 12. Reproduction

```bash
uv pip install -e '.[dev,retrieval,plot]'

scripts/01_build_index.sh --plan      # prints the fingerprint, writes nothing
scripts/01_build_index.sh

CONTEXT_ONLY=1 scripts/03_eval.sh     # once; every run below shares this file

RUN_NAME=base KIND=no_retrieval       RESULTS=results/raw/base.jsonl     scripts/03_eval.sh
RUN_NAME=vanilla KIND=vanilla_rag     RESULTS=results/raw/vanilla.jsonl  scripts/03_eval.sh
RUN_NAME=prompted KIND=prompted_iterative RESULTS=results/raw/prompted.jsonl scripts/03_eval.sh
RUN_NAME=sft KIND=sft_only            RESULTS=results/raw/sft.jsonl      scripts/03_eval.sh
RUN_NAME=trained KIND=trained         RESULTS=results/raw/trained.jsonl  scripts/03_eval.sh

FIGURE=results/figures/frontier.png scripts/04_sweep.sh
scripts/05_serve_bench.sh
```

Training is `scripts/02_train.sh` and needs a GPU; `results/raw/*.jsonl` comes
from the harness running a policy. The index build and the sweep both honour
`--plan`, which runs everything up to the first write and stops.

Every fingerprint in section 2 is reproducible from this sequence. If a rebuilt
index produces a different retriever fingerprint, the numbers in this report were
measured against something else, and the report is stale rather than merely old.

---

## Contributions

| | |
|---|---|
| Ayesha Siddika (70%) | project lead; retrieval, data, eval harness, baselines, metrics, frontier, serving, faithfulness scorer, CI, this report |
| Tahsinul Haque Dhrubo (30%) | RL core: episode loop, parser, reward components, GRPO training loop and masking; `rl_method.md` and `failure_cases.md` |

Method detail for the RL half is in **[`rl_method.md`](rl_method.md)**. Reviews
and merges: Ayesha. The ownership split is PROJECT.md section 0 and is reported
here because a joint result with an unstated split is not a reportable one.
