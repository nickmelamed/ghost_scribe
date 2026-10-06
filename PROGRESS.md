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

- [ ] Redo the HITL pass from scratch on a fresh queue (`/hitl-round`).
  Archive the unfinished `hitl/reviews.jsonl` first.
- [ ] Build phases 6, 7 and 8 from docs/SPEC.md: round-2 retraining with a
  held-out source and the corrected labels, drift monitoring, trend plots,
  and the full README with the ethics section.

## Done

- [x] Round 2 pipeline (`evaluation/round2.py`) and `round2` skill, built and
  tested. It needs the HITL corrections and about 9 hours of compute to run.

- [x] Phase 7 drift monitoring (`drift/monitor.py`, `results/drift_round1.md`).
  llama_70b_v1 shifts Stage 1's behavior and falcon_180b_v1 mostly does not.
  Rerun with `--tag round2 --models-dir ...` after round 2.

- [x] Prompt-held-out evaluation (`evaluation/prompt_holdout.py`). Stage-2
  accuracy on escalated rows was 0.9944 on unseen prompts versus 0.9931 on
  seen ones, so the 99.3% holds. Stage 1 and the combined pipeline are worse
  on unseen prompts, with a combined human false-positive rate of 0.0087
  against 0.0032. See docs/SPEC.md 6.6 and `results/prompt_holdout.md`.

- [x] Remove near-duplicates from val, test and ESL (`data/dedup.py`). 228,
  297 and 546 rows, all human. Headline metrics barely moved, the ESL gap
  rose from +0.0176 to +0.0199, and 8.4% of the ESL set had been in train.
  Thresholds are frozen until round 2. See docs/SPEC.md 6.4, D20 and D21.

- [x] Fix test-set contamination. The human-review gate is now calibrated on
  val (41.6%, was 20% set from test). Queue is 838 of 1,280 val essays, was
  676. Details in docs/SPEC.md 6.1 and D13, numbers in
  `results/review_gate.json`.

- [x] Build phases 0 to 5 of the original spec (data, Stage 1, fairness,
  Stage 2, combined evaluation, review app), as of commit 8b3fe44.
- [x] Agent standards v2.0.0 installed, CLAUDE.md split into docs/SPEC.md and
  a short rule list, hooks smoke-tested.
- [x] Repo-wide style cleanup of comments and docs. Code unchanged.
- [x] 82 characterization tests, `pyproject.toml`, ruff and mypy clean.
  `make agent-check` runs in the stop gate.
- [x] Repo skills: train-stage2, refresh-results, hitl-round, refresh-data,
  prompt-holdout.

## Open questions for the owner

- Duplicate `word_count` and `char_count` columns in the feature files mean
  the models use the preprocessing word count, not the feature function's.
  Results are consistent, but fixing it means retraining. See docs/SPEC.md
  6.5.
- Recalibrate the escalation thresholds and retrain Stage 2 in round 2, since
  they are frozen on the pre-dedup val.

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
- Stage 2 is 99.3% on escalated test rows versus 84.4% for Stage 1. Neither
  near-duplicates nor prompt memorization explain it. It was not tested on
  unseen generators, which are reserved for round 2. The prompt-held-out run
  is a single draw of 3 prompts.
- Stage 1 and the combined pipeline degrade on unseen prompts, mostly through
  more false positives on human essays. Worth addressing in round 2.
- Mutation testing (`mutmut`) and coverage have not been run.
- Model code, the Streamlit app, and the evaluation entry points have no
  tests because they need models, a browser, or write real result files.
