# Handoff

Read this first in a new session, then PROGRESS.md and CLAUDE.md. It says what
state the project is in, what is left, how to do it, and the quirks of this repo
that cost time to find.

## What this project is

A two-stage detector of AI-written student essays, built as a triage tool for
teachers. Stage 1 is an XGBoost model on 24 stylometric features. It clears or
flags the clear cases and escalates about 20% of essays to Stage 2, a
Qwen2.5-3B-Instruct model with a LoRA adapter. A teacher reviews the queue in a
Streamlit app. The reviewed essays plus a new AI source feed round 2. A fairness
check on non-native English writing and drift monitoring run throughout. It is
decision support only. Every flag needs a human to confirm it, and no code or
copy may imply an automatic verdict (CLAUDE.md rule 1).

The README is the portfolio write-up. docs/SPEC.md is the design and findings.
docs/DECISIONS.md is the reasons (D1 to D27).

## State in one paragraph

Everything is built, tested and documented except the HITL pass and the round-2
run that depends on it. Round 1 is complete and reported. The README has round-2
cells marked TBD. `make agent-check` passes (ruff, mypy, 154 tests, style).

## Branches

Five branches are stacked, each on the one before. `main` has not moved from
8b3fe44. No pull requests are open. Merge in this order:

1. `chore/agent-standards` (agent tooling, style cleanup, tests, lint fixes)
2. `fix/review-gate-calibration` (review gate calibrated on val, not test)
3. `fix/dedup-splits` (near-duplicate removal, regenerated results)
4. `feat/prompt-holdout` (prompt-held-out experiment and its skill)
5. `feat/remaining-phases` (drift, round 2 pipeline, trend plots, README,
   PROGRESS.md, this file)

All five are pushed to origin. Ask the owner before opening a PR, merging or
pushing anything new.

## What is left, in order

### 1. The HITL pass (the owner does the reviewing)

The owner types `/hitl-round`. It is a repo skill that only the owner can
start, so Claude cannot invoke it. It does this:

1. Archives the unfinished `hitl/reviews.jsonl` (36 old reviews, 28 of them in
   the new queue) into `hitl/archive/<date>/`. Archive, never delete.
2. Rebuilds the queue with `python -m stage2.build_review_queue`. Expect a queue
   of 823 essays: 525 "AI-generated" verdicts plus 298 low-confidence
   "Human-written" verdicts, from the cleaned val split of 1,242 escalated
   essays. The gate rate is 0.4168 and is written to `results/review_gate.json`.
3. The owner runs `streamlit run hitl/review_app.py` and reviews. The app hides
   the ground-truth label. Reviews append to `hitl/reviews.jsonl`.
4. When the owner says the pass is done, check the count against the queue size,
   then run `python -m hitl.apply_corrections --source human`. That writes
   `data/processed/hitl_corrections_human.csv`, which round 2 requires.
5. Report how many reviewer labels differ from Stage 2's, and how many carry the
   tag "possible false flag: non-native writing style".

Decision already made (D27): the app pre-selects Stage 2's suggestion, and the
owner chose to keep that. If the agreement rate looks implausibly high, removing
the default is the fix. Never report LLM-reviewer output (`hitl.llm_review`) as
human review.

### 2. Round 2

The owner types `/round2`. About 7 hours of LoRA training plus about 2 hours of
inference on this machine, which is Apple silicon with 64 GB and uses MPS. The
steps are in `.claude/skills/round2/SKILL.md`:

1. `python -m evaluation.round2 prepare` (stops with an error if the corrections
   file is missing, so do not use `--no-hitl`). An old dry run is sitting in
   `experiments/round2/`. This overwrites it.
2. Retrain in the background with `python -m stage2.finetune_lora` and the
   round-2 `--data-dir` and `--adapter-dir`. It resumes from checkpoints if
   interrupted.
3. Inference for `val_r2`, `test_r1`, `test_new_source`, `unseen_source` and
   `esl_eval`, confidence scoring for `val_r2` and `test_r1`, and round-1
   baselines for the two new-source groups.
4. `python -m evaluation.round2 report`, then drift and plots.

Round 2 adds `llama_70b_v1` (split 70/15/15) and the reviewed essays, labeled
with the reviewer's decision, and keeps `falcon_180b_v1` unseen. Both stages are
retrained from scratch and the bands and the review gate are recalibrated on
round-2 val rows. See docs/SPEC.md 6.8 and D24 to D26.

### 3. After round 2

- Fill the TBD cells in README.md from `results/round2.md`. The four cells are
  the escalation rate, Stage-1+2 accuracy, the combined ESL gap and the cost per
  essay.
- Record the result in docs/SPEC.md and docs/DECISIONS.md, and report every
  change as measured, including anything that got worse.
- `python -m drift.monitor --tag round2 --models-dir experiments/round2`, then
  `python -m viz.trend_plots`, then `make numbers` and `make agent-check`.
- Move the item from Next to Done in PROGRESS.md and commit.

### 4. Leftover items, none blocking

- Decide whether to fix the duplicate `word_count` and `char_count` columns in
  the feature files. It was kept for round 2 so the rounds differ only in data.
  Fixing it means retraining. See docs/SPEC.md 6.5.
- Enable the project-checks step in `.github/workflows/agent-checks.yml`. It
  installs torch, so it is slow.
- Install the pre-commit hook with `make setup`.
- Create `data/checksums.txt` for the processed data. It is already a protected
  path.
- Mutation testing (`mutmut`) and a coverage run.
- Tests for the model code, the review app and the evaluation entry points.
- `results/_before/` and `experiments/` are local scratch. Delete them if the
  disk is wanted.

## Key results to sanity-check against

These are round 1, from `results/`. If a rerun disagrees, find out why before
reporting.

| Item | Value |
|---|---|
| Stage-1 XGBoost test AUC | 0.9921 |
| Escalation rate, test | 0.2172 |
| Stage-1-only accuracy, all test | 0.9611 |
| Stage-1+2 accuracy, all test | 0.9936 |
| Stage 1 / Stage 2 accuracy, escalated rows | 0.8438 / 0.9932 |
| ESL false-positive gap, Stage 1 at 0.5 / combined | +0.0199 / +0.0008 |
| Review gate rate, val queue | 0.4168, 823 of 1,242 |
| Cost per essay, illustrative units | 5.58 |
| Prompt-held-out Stage-2 accuracy, unseen / seen prompts | 0.9944 / 0.9931 |
| Drift score PSI, llama_70b_v1 / falcon_180b_v1 | 0.509 / 0.037 |

Test split: 6,099 essays after removing 297 near-duplicates. Val: 6,168. ESL set:
5,936.

## Rules that matter

They are in CLAUDE.md. The ones most likely to be broken by accident:

- No automatic verdicts anywhere, including copy and comments.
- The held-out sources never appear in round-1 train, val, test or calibration.
- Never tune any threshold or gate on the test split.
- Log every metric honestly. Never round, cherry-pick or hand-edit a result.
  The cost units (1, 5, 25) are illustrative assumptions, not money.
- Stage 2 trains on escalated rows only.
- Do not read, print or commit raw essays, `.env`, or Kaggle credentials.
- Ask before pushing, merging, tagging, changing dependencies, CI or hooks.

## Quirks of working here

- Protected paths (`.claude/protected-paths`) ask the owner before an edit. They
  include CLAUDE.md, the Makefile, `.claude/`, `evaluation/`, `fairness/`,
  `results/`, `data/preprocess.py`, `stage1/escalation_policy.py`,
  `stage2/build_review_queue.py` and the review files. Use the Edit or Write
  tool for them. Editing them through a shell script bypasses the prompt.
- The Bash guard blocks some compound commands, with a message such as "Do not
  print secrets or the environment". It fired on long heredocs. Write files with
  the Write tool and keep shell commands short.
- `sleep N` followed by another command is blocked. For a long job, start it
  with `run_in_background` and watch it with Monitor, which expires after 30
  minutes and must be re-armed. Use `LC_ALL=C` with `tr` and `grep -a` when
  parsing a training progress bar, or it fails on the bar characters.
- The Stop hook runs `make agent-check` and keeps Claude working until it
  passes. The style check covers changed lines against the merge base with
  `origin/main`, and bans em dashes, semicolons in prose, bold lead-ins and
  mentions of CLAUDE.md in comments. Generated tables are skipped through
  `.claude/style-ignore`.
- Modules import each other as packages, so run them from the repo root with
  `python -m`. Activate the venv first with `source ghost/bin/activate`.
- Repo skills with side effects are owner-only: `/hitl-round`, `/round2`,
  `/train-stage2`, `/refresh-results`, `/refresh-data`, `/prompt-holdout`.

## Local-only state, not in git

`data/raw/`, `data/processed/`, `stage1/models/*.joblib`, `stage2/adapters/`,
`stage2/data/`, `experiments/`, `hitl/reviews.jsonl` and `results/_before/`. A
fresh clone has none of them, and there are no checksums.
`data/download.py` fetches the datasets and needs Kaggle credentials.

## A prompt to paste into the new conversation

> Read HANDOFF.md, PROGRESS.md and CLAUDE.md in this repo. We are ready for the
> HITL pass. Tell me the current branch and whether the tree is clean, confirm
> `make agent-check` passes, and confirm the review queue has 823 essays. Then
> wait for me to run `/hitl-round`. When I say the review is done, run
> `hitl.apply_corrections`, report the counts, and ask before starting round 2.
