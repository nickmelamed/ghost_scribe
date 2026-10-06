# AI-Generated Text Detection Flywheel

A two-stage detector of AI-written essays for teacher triage. An XGBoost
stylometric filter auto-clears or auto-flags confident cases and escalates
about 20% of ambiguous ones to a LoRA-tuned small LLM. Teachers review
escalations in a Streamlit app, and their corrections plus a previously
unseen generator feed a second training round and drift monitoring. The
false-positive gap on non-native English writing is tracked every round. It
is a portfolio project and a decision-support tool, never an automatic
verdict.

The full design, including the original build spec with its phases, is in
docs/SPEC.md. Read the relevant section before changing anything it covers.
Current status and next steps are in PROGRESS.md. Design decisions and their
reasons are in docs/DECISIONS.md. "Phase N" in the spec means the build
phases, not the phases of any cleanup task.

## Non-negotiable rules

1. No automatic verdicts. Every flagged or escalated case needs human
   confirmation before any action. No code path, UI copy, comment, or README
   text may imply an automatic accusation.
2. The held-out sources `falcon_180b_v1` and `llama_70b_v1` never appear in
   round-1 train, val, test, or threshold calibration. Describe them as a
   stand-in for a new model appearing, never as real chronological drift.
3. Log every metric honestly, even when a phase underperforms. Never round,
   cherry-pick, or fabricate a result. Count fallbacks, do not hide them.
4. Fine-tune Stage 2 on Stage-1-escalated cases only, never the full dataset.
5. Reviewers never see ground-truth labels. Review labels, not Stage 2's
   verdict, become training labels.
6. The test split is for evaluation only. Do not tune any threshold or gate
   on it.
7. ELLIPSE results are not a validated benchmark for all non-native writers,
   and the 1/5/25 cost units are illustrative assumptions, not financial
   data. Say so wherever they are reported.
8. Group by generation source, not by date. Use only a small open model
   (Qwen2.5-3B-Instruct or similar) with LoRA.
9. Never weaken a test, lint rule, or check to make it pass.
10. Ask before changing anything in `.claude/protected-paths`, the evaluation
    design, dependencies, CI, or hooks.

## Commands

```bash
python3 -m venv ghost && source ghost/bin/activate
pip install -r requirements.txt
make agent-check  # the fast checks the Stop hook runs
make numbers      # reported numbers must match results/
```

Pipeline entry points, run from the repo root as modules, for example
`python -m stage1.escalation_policy` and `streamlit run hitl/review_app.py`.
The data, models, adapters, and review files are gitignored and live only on
this machine.

## Where things live

- `data/`, `stage1/`, `stage2/`, `fairness/`, `evaluation/`, `hitl/`,
  `drift/`, `viz/`: pipeline code by stage. `tests/`: synthetic fixtures only.
- `results/`: generated tables. Never hand-edit a number in them.
- `docs/SPEC.md`: design spec. `docs/DECISIONS.md`: decision log.
- `.claude/rules/`: path-scoped rules. `.claude/skills/`: procedures.

## How to work here

- Plan before multi-file changes and wait for approval when a task spans
  several modules or touches the spec.
- A task is done when the Stop hook's checks pass and you have shown the
  output. Show commands and results, not claims.
- Commit in small atomic Conventional Commits (`type(scope): subject`), with
  code and its tests together, after each phase passes its acceptance check.
- You may branch and commit locally. Ask before pushing, opening or merging
  pull requests, tagging, or changing dependencies.
- Do not read, print, or commit raw essays, `.env`, or Kaggle credentials.
- If you repeat a multi-step procedure, propose a skill for it and wait for
  approval.
