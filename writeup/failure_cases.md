# Failure cases

Owner: Tahsinul Haque Dhrubo. Reward hacks, degenerate policies and surprises, written down the
day they are seen.

PROJECT.md section 6: *"If hacking occurs, do not just patch it. Document it."* A silently
patched hack is worth nothing to anyone; a documented one with its mitigation is the most
valuable content in the report.

**Status: no reward hack yet.** That needs a training run long enough for the policy to find
one, and P3 has not run at length. Three findings from the first runs of the loop are recorded
below, because the same rule applies to them: write it down the day it happens.

## 1. Unparseable generations trained on nothing

- **Seen:** 2026-10-21, the first real run of `scripts/02_train.sh --smoke`
  (Qwen2.5-0.5B, group 4, fake retriever, 2 steps). No W&B link worth having: it crashed on
  step 1.
- **Symptom:** `IndexError: list index out of range` raised inside TRL, at
  `grpo_trainer.py:2325`, where it indexes `ids[-1]` on every completion to decide whether the
  rollout was truncated.
- **Diagnosis:** mine, not TRL's. The 0.5B policy answered in bare prose - no tags at all - so
  `parse_turn` returned no turns, and the episode loop attached that generation's token ids to
  the last turn, of which there were none. The tokens were dropped and the rollout reached the
  trainer with an empty completion.
- **Why it matters past the crash:** those tokens are exactly what GRPO has to push down. A
  rollout that fails to produce a usable trajectory is the negative example the policy learns
  from. Dropping it silently would have trained on nothing while the no-answer penalty still
  reported -0.5 - a quiet hole in the training signal instead of a traceback. The crash was the
  lucky outcome.
- **Mitigation:** a generation that parses to no turns is kept as a `THINK` turn carrying its
  ids, so it carries loss, with `Trajectory.parse_failure` recording why it went nowhere
  (`env/episode.py`). `build_payload` separately refuses an empty completion with its own
  message rather than letting it reach TRL (`train/rollout.py`). Both tested.
- **Open question:** `THINK` is the closest honest kind for "the policy wrote something that
  asked for nothing", but `TurnKind.UNPARSED` would read better in section 9's failure
  breakdown. Raised on #27 - `TurnKind` is shared, so it needs both approvals.
- **Status:** mitigated.

## 2. The nightly gate failed for its own bookkeeping

- **Seen:** 2026-10-21, the second run, immediately after fixing case 1.
- **Symptom:** `final loss nan` / `FAIL: the loss is not finite`, while the loop had in fact
  run fine - 2 GRPO steps in 6 seconds, rollouts generated, rewards computed, importance
  sampling ratios between 0.58 and 1.34.
- **Diagnosis:** `logging_steps` was left at the transformers default of 500, so a 2-step run
  logged no per-step `loss` at all, and my summary read an empty history as NaN. The nightly
  `smoke-train` stage asserts the loss is finite, so a correct run would have reported red.
- **Why it matters:** a gate that fails for its own reasons is worse than no gate. The next
  person to see it red stops believing it, and then it stops catching the thing it exists for.
- **Mitigation:** `logging_steps=1` - a smoke run is 5 steps and a dev run 150, so the default
  logs nothing, and the per-step series in section 9 is the only way a reward hack is visible
  while it happens. `summarize_history` reads the per-step `loss` series and falls back to the
  final `train_loss`; a history with nothing usable stays NaN and still fails, because 0.0
  would be a passing grade nobody measured.
- **Status:** mitigated.

## 3. Not a hack, but the shape of one: zero-variance groups

- **Seen:** 2026-10-21, the same smoke runs.
- **Symptom:** `reward -0.5` for all four rollouts in the group, `reward_std 0`,
  `frac_reward_zero_std 1`, `train_loss 0`. Every rollout hit the no-answer penalty and nothing
  else, so GRPO's advantage - `(r_i - mean(r)) / std(r)` - was zero for every rollout and the
  step moved nothing.
- **Diagnosis:** expected at 0.5B with a 128-token turn budget: the policy never produced a
  well-formed answer, so every rollout scored identically. Not a bug and not a hack - but it is
  what a *dead* reward looks like, and P2's exit criterion exists to rule it out (spread 15-60%
  non-trivial).
- **What to watch:** `frac_reward_zero_std` is the canary. At 1.0 the run is burning GPU time
  to learn nothing, whatever the loss curve says.
- **Status:** expected for this configuration; the measurement at P2 is what decides whether it
  persists at 1.5B.

## Format for each case

```
### N. Short name

- Seen: date, which run (W&B link), which step
- Symptom: the metric that moved, with numbers
- Trajectory: the raw excerpt that shows it
- Diagnosis: why the policy found this profitable
- Mitigation: what changed, and the measurement that shows it worked
- Status: open | mitigated | accepted
```

## Watchlist

The hacks PROJECT.md section 6 predicts, with the metric that catches each. Pre-registering
them so the first one is recognised rather than explained away.

| Predicted hack | Canary metric |
|---|---|
| Never search, answer from parametric memory | average searches per rollout falls toward 0 |
| Copy passages verbatim to farm faithfulness | answer length rises; `answer_token_count` from `eval/metrics.py` |
| Search once, ignore the results | faithfulness goes flat while correctness moves |
| Degenerate repetition | completion length rises, KL from reference rises |
| Dodge the duplicate-query penalty by re-wording | duplicate rate near zero while queries are near-identical |

The last row is not in PROJECT.md. It comes from reading the proposed contract: duplicate
queries are counted by exact string equality, so `"Paris"` and `"Paris "` are two distinct
queries. Raised on the types PR; if normalisation does not land, this is the first hack to
expect.
