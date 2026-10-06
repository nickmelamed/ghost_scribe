---
paths:
  - "**/*.py"
---

# Python

- Modules import each other as packages (`from stage1.features import ...`),
  so run them from the repo root with `python -m`. Do not add `sys.path`
  edits.
- Build file paths from `Path(__file__)` like the existing modules, never
  from the working directory.
- Use `random_state=42` wherever a seed is taken, so splits and models match
  the recorded results.
- Do not print or log essay text, and keep raw and processed data out of
  git. The data is gitignored on purpose.
- Keep feature columns in `stage1.features.FEATURE_COLUMNS` order. Models
  and tests depend on that order.
- Treat label 0 as human and label 1 as AI-generated everywhere.
- Any change to splits, thresholds, or evaluation code changes the reported
  numbers. Regenerate `results/` and record the before and after.
- Docstrings follow `.claude/rules/writing-style.md`.
