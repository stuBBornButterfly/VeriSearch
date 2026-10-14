# Failure cases

Owner: Tahsinul Haque Dhrubo. Reward hacks, degenerate policies and surprises, written down the
day they are seen.

PROJECT.md section 6: *"If hacking occurs, do not just patch it. Document it."* A silently
patched hack is worth nothing to anyone; a documented one with its mitigation is the most
valuable content in the report.

**Status: empty. No training run has happened yet** (P3 has not started). This file exists so
the link from `report.md` resolves and so there is somewhere to write the first case the hour it
appears, rather than reconstructing it later from memory.

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
