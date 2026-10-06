---
name: prompt-holdout
description: Retrain both stages without held-out essay prompts and compare held-out and seen prompts. Only the owner starts this.
disable-model-invocation: true
---

Run the prompt-held-out evaluation. The LoRA retrain takes about 4.5 hours on
this machine and holds most of the 64 GB of memory, so confirm the machine is
free before starting. Nothing here touches the main-run adapter, thresholds,
or `results/` files other than `results/prompt_holdout.{md,json}`.

1. Confirm `data/processed/train_features.csv`, `val_features.csv`, and
   `test_features.csv` exist and that `data.dedup` has been run. Confirm no
   `stage2.finetune_lora` process is already running. If the experiment folder
   `experiments/prompt_holdout/` exists, ask before replacing it.
2. Run `python -m evaluation.prompt_holdout prepare`. Show the held-out
   prompts, the row counts, the recalibrated thresholds, and the escalated
   counts. The prompts come from a seeded draw, so do not change the seed to
   get a different result.
3. Start the retrain in the background, with output going to a log outside the
   repo:
   `python -m stage2.finetune_lora --data-dir experiments/prompt_holdout/stage2_data --adapter-dir experiments/prompt_holdout/adapters/lora`
   Report the step speed after a few minutes. It resumes from the newest
   checkpoint if it is interrupted.
4. When training exits cleanly, run inference on both groups with
   `python -m stage2.infer --split <group> --data-dir experiments/prompt_holdout/stage2_data --adapter-dir experiments/prompt_holdout/adapters/lora --out-dir experiments/prompt_holdout/predictions`
   for `test_heldout_prompts` and `test_seen_prompts`. Report any outputs that
   did not parse.
5. Run `python -m evaluation.prompt_holdout report` and show the tables. State
   the held-out and seen numbers as measured, with the intervals, including
   any that look worse. Do not describe a difference as meaningful without
   saying how many rows it rests on.
6. Record the result in docs/SPEC.md next to the earlier findings. This is one
   draw of a few prompts, so say so.
