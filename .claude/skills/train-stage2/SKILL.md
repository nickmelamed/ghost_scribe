---
name: train-stage2
description: Rebuild the escalated Stage-2 data, run the LoRA fine-tune, then run inference and confidence scoring. Only the owner starts this.
disable-model-invocation: true
---

Retrain Stage 2 from the current Stage-1 artifacts. This is a long run on
the local machine (Apple MPS) and overwrites the adapter.

1. Confirm `stage1/models/` has `xgb.joblib` and `escalation_thresholds.json`,
   and that `data/processed/*_features.csv` exist. If not, stop and say which.
2. Confirm the tree is clean and name the current adapter directory. Ask
   before overwriting `stage2/adapters/qwen2.5-3b-instruct-lora`, and offer to
   move it aside first.
3. Run from the repo root, in the `ghost` venv, one step at a time and show
   the tail of each output:
   - `python -m stage2.prepare_finetune_data`
   - `python -m stage2.finetune_lora`
   - `python -m stage2.infer --split val`
   - `python -m stage2.infer --split test`
   - `python -m stage2.infer --split esl_eval`
   - `python -m stage2.confidence val`
   - `python -m stage2.confidence test`
4. Fine-tune on escalated training cases only, and never use the test split
   to tune anything. Count unparseable outputs and report them.
5. Report the escalated counts per split and any fallbacks. Do not edit
   `results/` here. Use `/refresh-results` for that.
