---
name: round2
description: Run flywheel round 2 after the HITL pass. Retrains both stages with the new source and the reviewed essays, then reports and plots. Only the owner starts this.
disable-model-invocation: true
---

Run round 2 end to end. Expect about 7 hours of LoRA training plus about 2
hours of inference on this machine, with most of the 64 GB of memory in use.
Confirm the machine is free first. Nothing here changes the round-1 adapter,
thresholds, or the round-1 files in `results/`.

1. Confirm the HITL pass is finished. `hitl/reviews.jsonl` must cover the queue
   built by `stage2.build_review_queue`. Run
   `python -m hitl.apply_corrections --source human` and show the counts. Never
   use the LLM reviewer's file as a stand-in for human review.
2. Run `python -m evaluation.round2 prepare` and show the counts: reviewed
   essays, how many reviewer labels differ from the true label, rows removed as
   near-duplicates, the new thresholds, and the escalated training rows. It
   stops with an error if the corrections file is missing, so do not pass
   `--no-hitl`.
3. Start the retrain in the background with output going to a log outside the
   repo:
   `python -m stage2.finetune_lora --data-dir experiments/round2/stage2_data --adapter-dir experiments/round2/adapters/lora`
   Report the step speed after a few minutes. It resumes from the newest
   checkpoint if interrupted.
4. When training exits cleanly, run inference for each of `val_r2`, `test_r1`,
   `test_new_source`, `unseen_source`, and `esl_eval`:
   `python -m stage2.infer --split <group> --data-dir experiments/round2/stage2_data --adapter-dir experiments/round2/adapters/lora --out-dir experiments/round2/predictions`
   Then run confidence scoring for `val_r2` and `test_r1`:
   `python -m stage2.confidence <group> --data-dir experiments/round2/stage2_data --adapter-dir experiments/round2/adapters/lora --out-dir experiments/round2/confidence`
   Report any outputs that did not parse.
5. Run the round-1 baselines on the two new-source groups, with the round-1
   adapter:
   `python -m stage2.infer --split <test_new_source or unseen_source> --data-dir experiments/round2/r1_stage2_data --adapter-dir stage2/adapters/qwen2.5-3b-instruct-lora --out-dir experiments/round2/r1_predictions`
6. Run `python -m evaluation.round2 report`, then
   `python -m drift.monitor --tag round2 --models-dir experiments/round2`, then
   `python -m viz.trend_plots`. Show the tables.
7. Report the direction of every change as measured, including anything that
   got worse. Then update docs/SPEC.md, docs/DECISIONS.md, PROGRESS.md and the
   README numbers, and run `make numbers`. Do not edit a result by hand.
