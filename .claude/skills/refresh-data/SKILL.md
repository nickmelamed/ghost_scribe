---
name: refresh-data
description: Download the datasets, rebuild the splits and summary, and re-extract features. Only the owner starts this.
disable-model-invocation: true
---

Rebuild the processed data from the raw files. This rewrites the splits, so
every downstream metric changes. Raw essays and processed files are
gitignored. Do not print essay text.

1. Run `python data/download.py`. If Kaggle credentials are missing, stop
   and show the instructions the script prints. Do not read or print the
   credentials file.
2. Confirm `data/raw/ellipse_corpus.csv` exists. If not, run
   `python -m fairness.data_prep_esl` and follow its instructions.
3. Run `python -m data.preprocess` and show `dataset_summary.md`. Check that
   `falcon_180b_v1` and `llama_70b_v1` appear only in `held_out.csv`.
4. Run `python -m fairness.data_prep_esl`, then `python -m stage1.features`.
5. Show split sizes and per-source counts, and say how they differ from the
   previous summary. Then point the owner to `/refresh-results` and
   `/train-stage2`, since both are now stale.
