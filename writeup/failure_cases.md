# Failure cases

Owner: Tahsinul Haque Dhrubo. Reward hacks, degenerate policies and surprises, written down the
day they are seen.

PROJECT.md section 6: *"If hacking occurs, do not just patch it. Document it."* A silently
patched hack is worth nothing to anyone; a documented one with its mitigation is the most
valuable content in the report.

**Status: no reward hack yet.** That needs a training run long enough for the policy to find
one, and P3 has not run at length. Five findings from building and first running the loop are
recorded below, because the same rule applies to them: write it down the day it happens.

**None of the five counts toward success criterion 10's "3+ failure cases".** Cases 1, 2, 4
and 5 are engineering failures of mine; case 3 is what a dead reward looks like, not a hack.
The cases that count have to come from real training, starting with the first long 1.5B run.

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
- **Mitigation:** a generation that parses to no turns is kept as a `TurnKind.UNPARSED` turn
  carrying its ids, so it carries loss, with `Trajectory.parse_failure` recording why it went
  nowhere (`env/episode.py`). `UNPARSED` joined the frozen contract on #2 and is on `main`, so
  a trajectory dump says the policy ignored the format rather than that it reasoned and gave up.
  `build_payload` separately refuses an empty completion with its own message rather than
  letting it reach TRL (`train/rollout.py`). Both tested.
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

## 4. Training replayed a context the model never sampled from

- **Seen:** 2026-10-23, in Ayesha's review of #31. Not found by a run, and no run would have
  shown it. She reproduced it with a character-level tokenizer and a recording generator.
- **Symptom:** none in any metric. On turn 2, the policy was sampled from
  `</search>\n<info>\n...\n</info>\n` and the trainer replayed `</search><info>\n...\n</info>`.
- **Diagnosis:** mine. The episode loop rebuilt the transcript from decoded text: the overshoot
  after `</search>` was cut and newlines were added around `<info>`, and the policy re-encoded
  that string every turn. The payload spliced the raw generated ids and `encode(info)` instead.
  With Qwen2.5's tokenizer, the context after one search was 218 ids one way and 220 the other,
  and the first difference was already at the prompt's join with `<think>`.
- **Why it matters:** every token from turn 2 on was trained on a context it wasn't sampled from,
  and the importance-sampling ratio compared log-probabilities across two contexts. The loss
  mask was right throughout, which is why the blocking masking test passed. This is the class of
  bug the project worries about most: wrong, silent, and invisible in a training curve.
- **Mitigation:** the token ids are the source of truth. Training's episode keeps a running id
  context and hands the policy that list, never a string; the separators are part of the
  injected text, and overshoot is cut from both streams at the token that closed the search
  (`rl_method.md`, section 2). The test asserts `generator.calls[k] == prompt_ids +
  completion_ids[:prefix]` for every turn, with a stand-in tokenizer in CI and Qwen's real one in
  `tests/test_token_context_qwen.py`.
- **Status:** mitigated, on #28 and #31.

## 5. The 1.5B smoke run did not fit, three times

- **Seen:** 2026-10-25, the first runs of `scripts/02_train.sh --smoke --steps 5` on Qwen2.5-1.5B,
  the size the nightly stage uses by default. Logs: `outputs/smoke_1.5b_oom_step1.log`,
  `outputs/smoke_1.5b_oom_step3.log`, then `outputs/smoke_1.5b.log` for the run that passed.
- **Symptom, in order:**
  1. vLLM refused to start: 0.88 GiB of KV cache needed for a 32,768-token window, 0.63 GiB
     available.
  2. `torch.OutOfMemoryError` in `selective_log_softmax` on step 1, with eight rollouts per
     forward pass.
  3. `torch.OutOfMemoryError` in the no-grad log-prob pass on step 3, with two per pass, after
     two good steps, at a 15.6 GB peak.
- **Diagnosis:** the earlier smoke runs used 0.5B with short turns. At 1.5B, vLLM's window
  defaulted to the model's maximum, and full-vocabulary logits for several multi-turn
  completions in one pass don't fit beside colocated vLLM on 16 GB.
- **Mitigation:** `vllm_max_model_length=8192` (an episode's worst case is about 7,000 tokens),
  one rollout per forward pass with gradient accumulation over the group, gradient
  checkpointing, and `expandable_segments`. Then: 5 steps, ~22 s/step, peak 14.5 GB. The
  gradient is unchanged: the group is still one optimizer step.
- **What to watch:** 14.5 of 16 GB on the fake corpus. Real passages are longer, so real
  contexts are too. Peak memory goes in the first real run's report, and it decides B7.
- **Status:** mitigated.

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

The last row is not in PROJECT.md. It came from reading the proposed contract, where
duplicate queries were counted by exact string equality, so `"Paris"` and `"Paris "` were
two distinct queries. Normalisation landed on #2 (`_normalize_query` in `types.py`), which
closes whitespace and case. A paraphrase is still a new query, and that's what the row now
watches.
