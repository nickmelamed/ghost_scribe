# Progress

Current state and what is left. Update it at the end of each piece of work.
For picking the project up in a new session, start with HANDOFF.md.

"Build phase" means a phase of the original spec in docs/SPEC.md. The
cleanup work is separate.

## Now

- [ ] The HITL pass, then round 2. Everything else is built, tested and
  documented. The review is the owner's.

## Next

- [ ] Redo the HITL pass from scratch on a fresh queue (`/hitl-round`, owner
  only). It archives the unfinished `hitl/reviews.jsonl`, shows the queue of
  823 essays from the cleaned val split, and starts the review app. The app
  pre-selects Stage 2's suggestion, which the owner chose to keep, so a reviewer
  who submits without changing anything records agreement. D27 has the reason
  and the risk.
- [ ] Run round 2 (`/round2`, owner only). It refuses to start without the
  reviewed corrections. Expect about 7 hours of LoRA training and about 2 hours
  of inference on this machine.
- [ ] After round 2: fill the TBD cells in the README from `results/round2.md`,
  record the result in docs/SPEC.md and docs/DECISIONS.md, rerun
  `python -m drift.monitor --tag round2 --models-dir experiments/round2` and
  `python -m viz.trend_plots`, then run `make numbers` and `make agent-check`.
- [ ] Review and merge the branch stack in order, see HANDOFF.md. No pull
  requests are open.

## Done

- [x] Phase 8: trend plots, cost per decision, and the full README with the
  ethics and limitations section. `make numbers` checks every README number
  against `results/*.json`. Round-2 cells are TBD.
- [x] Round 2 pipeline (`evaluation/round2.py`) and `round2` skill, built and
  tested. A dry run on the real data without corrections works.
- [x] Phase 7 drift monitoring (`drift/monitor.py`, `results/drift_round1.md`).
  llama_70b_v1 shifts Stage 1's behavior and falcon_180b_v1 mostly does not.
- [x] Prompt-held-out evaluation (`evaluation/prompt_holdout.py`). Stage-2
  accuracy on escalated rows was 0.9944 on unseen prompts versus 0.9931 on seen
  ones, so the 99.3% holds. Stage 1 and the combined pipeline are worse on
  unseen prompts. See docs/SPEC.md 6.6 and `results/prompt_holdout.md`.
- [x] Near-duplicates removed from val, test and ESL (`data/dedup.py`). Headline
  metrics barely moved, the ESL gap rose from +0.0176 to +0.0199, and 546 ESL
  essays had been in train. See docs/SPEC.md 6.4, D20 and D21.
- [x] Test-set contamination fixed. The human-review gate is calibrated on val
  and test only checks it. See docs/SPEC.md 6.1 and D13.
- [x] Build phases 0 to 5 of the original spec, as of commit 8b3fe44.
- [x] Agent standards v2.0.0, CLAUDE.md split into docs/SPEC.md and a short
  rule list, hooks smoke-tested, repo-wide style cleanup.
- [x] 154 tests, `pyproject.toml`, ruff and mypy clean. `make agent-check` runs
  in the stop gate.
- [x] Repo skills: train-stage2, refresh-results, hitl-round, refresh-data,
  prompt-holdout, round2.

## Open questions and known issues

- Duplicate `word_count` and `char_count` columns in the feature files mean the
  models use the preprocessing word count, not the feature function's. Results
  are consistent. It was kept on purpose so round 1 and round 2 differ only in
  data. See docs/SPEC.md 6.5.
- The escalation thresholds are frozen on the pre-dedup val. Round 2
  recalibrates them.
- Stage 2 is 99.3% on escalated test rows. Near-duplicates and prompt
  memorization do not explain it. It is untested on unseen generators until
  round 2 (falcon_180b_v1 stays unseen for that). The prompt-held-out run is one
  draw of 3 prompts.
- Stage 1 and the combined pipeline degrade on unseen prompts, mostly through
  more false positives on human essays that Stage 1 auto-flags.
- The data, processed splits, adapters, experiments and review files are
  gitignored and exist only on this machine. There are no data checksums, so a
  result cannot be reproduced from the repo alone. `data/checksums.txt` is a
  protected path but does not exist yet.
- CI: the project-checks step in `.github/workflows/agent-checks.yml` is still
  commented out, because enabling it installs torch.
- `pre-commit` is configured but not installed as a git hook. `make setup`
  installs it.
- `.claude/style-ignore` skips the two generated tables, because their text
  comes from string literals in the code that writes them.
- The standards scripts in `scripts/agent/` are excluded from ruff and mypy.
- Mutation testing (`mutmut`) and coverage have not been run.
- Model code, the Streamlit app, and the evaluation entry points have no
  automated tests, because they need models, a browser, or write real result
  files.
