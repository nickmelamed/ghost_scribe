# Progress

Current phase and what is left. Update it at the end of each phase.

"Build phase" means a phase of the original spec in docs/SPEC.md. The
cleanup phases below are separate.

## Now

- [ ] Review and merge `chore/agent-standards`, then decide the order of the
  behavior changes in "Next".

## Next

Behavior changes agreed in Phase 1. Each changes reported numbers, so
regenerate `results/` and record the before and after (`/refresh-results`).
The characterization tests are the before-state, and any test that changes
should change on purpose.

- [ ] Fix test-set contamination. `stage2/build_review_queue.py` sets the
  20% human-review gate from the escalated test split. Calibrate it on val
  instead. See docs/SPEC.md 6.1 and D13.
- [ ] Remove near-duplicates across splits (proposed threshold: at least 80%
  of 8-word shingles shared with a train essay) and the 2 exact-text
  overlaps. Log removed rows per source and split.
- [ ] Add a prompt-held-out evaluation and report it next to the main
  result. Re-check the 99.3% Stage-2 accuracy on escalated test rows.
- [ ] Redo the HITL pass from scratch on a fresh queue (`/hitl-round`).
  Archive the unfinished `hitl/reviews.jsonl` first.
- [ ] Build phases 6, 7 and 8 from docs/SPEC.md: round-2 retraining with a
  held-out source and the corrected labels, drift monitoring, trend plots,
  and the full README with the ethics section.

## Done

- [x] Build phases 0 to 5 of the original spec (data, Stage 1, fairness,
  Stage 2, combined evaluation, review app), as of commit 8b3fe44.
- [x] Agent standards v2.0.0 installed, CLAUDE.md split into docs/SPEC.md and
  a short rule list, hooks smoke-tested.
- [x] Repo-wide style cleanup of comments and docs. Code unchanged.
- [x] 82 characterization tests, `pyproject.toml`, ruff and mypy clean.
  `make agent-check` runs in the stop gate.
- [x] Repo skills: train-stage2, refresh-results, hitl-round, refresh-data.

## Open questions for the owner

Carried over from the Phase 0 assessment.

- Data, processed splits, adapters, and review files are gitignored and exist
  only on this machine. There are no data checksums, so a result cannot be
  reproduced from the repo alone. `data/checksums.txt` is already a protected
  path but does not exist yet.
- CI: the project-checks step in `.github/workflows/agent-checks.yml` is
  still commented out. Enabling it installs torch through requirements.txt.
- `pre-commit` is configured but not installed as a git hook (`.git/hooks`
  has none). Run `make setup` to install it.
- `.claude/style-ignore` skips the two generated tables, because their text
  comes from string literals in the code that writes them.
- The standards scripts in `scripts/agent/` are excluded from ruff and mypy.
- Stage 2 is 99.3% on escalated test rows versus 84.4% for Stage 1. Near
  duplicates do not explain it, but all 15 prompts appear in train and test,
  so it is unconfirmed until the prompt-held-out evaluation runs.
- Mutation testing (`mutmut`) and coverage have not been run.
- Model code, the Streamlit app, and the evaluation entry points have no
  tests because they need models, a browser, or write real result files.
