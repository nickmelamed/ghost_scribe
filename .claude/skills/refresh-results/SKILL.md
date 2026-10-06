---
name: refresh-results
description: Regenerate Stage-1 models, thresholds, and the results tables, then check reported numbers. Only the owner starts this.
disable-model-invocation: true
---

Regenerate everything under `results/` from the current processed data and
Stage-2 predictions. This overwrites `results/metrics_comparison.md` and
`results/phase*_metrics.json`.

1. Before running, copy the current `results/` files to
   `results/_before/` (gitignored scratch, not committed) so the before and
   after can be compared and both recorded.
2. Run from the repo root, one step at a time, and show the tail of each:
   - `python -m stage1.features`
   - `python -m stage1.baseline_logreg`
   - `python -m stage1.baseline_xgb`
   - `python -m stage1.escalation_policy`
   - `python -m evaluation.evaluate`
   - `python -m evaluation.evaluate_combined`
3. `evaluate_combined` needs full escalated Stage-2 predictions for test and
   esl_eval, not the sample runs. If they are missing, stop and point to
   `/train-stage2`.
4. Run `make numbers` and show the output.
5. Show a before and after table of the headline metrics, the escalation
   rate, and the fairness gap. Report every change honestly, including
   worse numbers. Do not round, and do not hand-edit any result.
