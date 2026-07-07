# AI-Generated Text Detection Flywheel — Build Spec for Claude Code

## Context & Goal
Portfolio project demonstrating a production-style ML flywheel applied to
academic-integrity triage: a classical stylometric model as a fast
first-pass filter, a fine-tuned small LLM as an escalation layer for
ambiguous essays, a teacher-facing human-in-the-loop review queue, and drift
monitoring as new AI-generation sources appear. A dedicated fairness check
tracks false-positive rates on non-native English writing — a real,
well-documented failure mode of AI-text detectors — across every round.

## Definition of Done
All phases below complete, with:
- `results/metrics_comparison.md` showing Stage-1-only vs Stage-1+2 (round 1)
  vs Stage-1+2 (round 2, post-HITL)
- A false-positive-rate-by-subgroup (native vs non-native English writer)
  trend across rounds
- A drift-monitoring report across generation-source batches
- A README that reads as a coherent portfolio piece to a technical reviewer
  with no prior context, including an explicit ethics/limitations section

## Ground Rules
- **Dataset:** Kaggle's "LLM - Detect AI Generated Text" (DAIGT) competition
  dataset — real human essays plus AI-generated essays from multiple LLMs.
  Requires a free Kaggle account; flag this in Phase 1 and pause for the
  human if a manual download/API key step is needed.
- **Group by generation source, not by date.** There's no natural timestamp
  axis here, so essays are grouped by which model generated them. Hold out
  1–2 entirely unseen generation sources from initial training — these
  stand in for "a new AI model appears later" and drive both the round-2
  retraining and drift-monitoring phases. Document this substitution
  explicitly in the README; do not present it as real chronological drift.
- **Fairness data:** for the false-positive-rate check, use a publicly
  available L2/non-native-English essay corpus if one is accessible for
  research use. If a clean ESL-labeled corpus isn't available, use a
  documented proxy (e.g., essays with certain syntactic-complexity or
  L2-typical error profiles) and state clearly in the README that this is
  an approximation, not a validated benchmark.
- **Framing:** this is a decision-support triage tool for teachers, not an
  automated accusation system. Every escalated/flagged case requires human
  confirmation before any action is taken. State this explicitly in the
  README's ethics section — do not let any code path or UI copy imply an
  automatic verdict.
- **Cost-per-decision is an illustrative proxy, not real data.** Document
  assumed relative costs (e.g. Stage-1-only = 1 unit, Stage-2 escalation = 5
  units, human review = 25 units) explicitly as assumptions.
- **Small model only.** Use a small open-source instruction-tuned model
  (e.g. Llama-3.2-3B-Instruct or Qwen2.5-3B-Instruct) with LoRA via `peft`,
  4-bit quantization via `bitsandbytes` if compute-constrained.
- Log every metric honestly, even if a phase underperforms. Do not round or
  cherry-pick results to look better.
- Commit to git after each phase passes its acceptance check.

## Tech Stack
- Python 3.10+
- scikit-learn, xgboost
- nltk or spacy (POS tagging, stylometric features), textstat (readability)
- transformers, peft, bitsandbytes (optional)
- streamlit
- pandas, numpy, scipy (PSI/KS drift tests)
- matplotlib or plotly

## Repo Structure
```
ai-text-detection-flywheel/
├── README.md
├── pyproject.toml (or requirements.txt)
├── data/
│   ├── download.py          # DAIGT fetch/instructions
│   └── preprocess.py        # cleaning, generation-source labeling, splits
├── stage1/
│   ├── features.py          # stylometric/statistical feature extraction
│   ├── baseline_logreg.py
│   ├── baseline_xgb.py
│   └── escalation_policy.py
├── stage2/
│   ├── prepare_finetune_data.py
│   ├── finetune_lora.py
│   └── infer.py
├── fairness/
│   ├── data_prep_esl.py     # builds/loads non-native-English eval subset
│   └── esl_eval.py          # false-positive rate by subgroup
├── evaluation/
│   ├── metrics.py
│   └── evaluate.py
├── hitl/
│   ├── review_app.py        # teacher-facing Streamlit reviewer
│   └── apply_corrections.py
├── drift/
│   └── monitor.py           # PSI/KS across generation-source batches
├── viz/
│   └── trend_plots.py
└── results/
    ├── metrics_comparison.md
    └── figures/
```

## Phase 0 — Environment & Scaffold
1. Init git repo, create the structure above.
2. Create `pyproject.toml`/`requirements.txt` pinning the libs above.
3. Stub `README.md` with a project description (fill in fully in Phase 8).

**Acceptance:** `pip install -r requirements.txt` runs clean; structure matches spec.

## Phase 1 — Data Acquisition & Preprocessing
1. `data/download.py`: fetch the DAIGT dataset. If a manual Kaggle
   download/API key step is required, stop and tell the human exactly what
   to do, then continue once files are in place.
2. Label each essay with its generation source (human, or which LLM
   generated it).
3. Clean and prepare text; extract basic metadata needed for stylometric
   features downstream.
4. Split: hold out 1–2 entire generation sources (unseen at training time)
   for later phases; use the remaining sources for train/val/test.

**Acceptance:** preprocessed dataset exists with documented schema;
generation-source counts logged, including which sources are held out.

## Phase 2 — Stage 1: Classical Stylometric Filter
1. `features.py`: extract stylometric/statistical features — perplexity or
   n-gram-based burstiness proxies, sentence-length variance, POS-tag
   distribution, type-token ratio, punctuation patterns, readability scores.
2. Logistic regression + XGBoost classifiers on these features, predicting
   human vs AI-generated.
3. Calibrate escalation bands on the validation set: confident-human
   (auto-clear), confident-AI (auto-flag), ambiguous middle band (escalate
   to Stage 2). Target an initial escalation rate around 15–25%.
4. Evaluate: AUC-ROC, precision/recall/F1, escalation rate.
5. `fairness/esl_eval.py`: compute false-positive rate on the non-native
   English eval subset vs. the general test set at this stage, and log it.

**Acceptance:** Stage-1-only metrics table exists, including escalation rate
and the baseline fairness gap (subgroup FPR difference).

## Phase 3 — Stage 2: Fine-tuned LLM Escalation Layer
1. `prepare_finetune_data.py`: for escalated (middle-band) training cases,
   build prompts from the essay text, with the true label as the fine-tuning
   target and a short rationale (which stylistic cues suggest AI vs human).
2. LoRA fine-tune the chosen small model on this escalated subset only.
3. `infer.py`: run the fine-tuned model on escalated test cases.

**Acceptance:** fine-tuned adapter saved; inference runs end-to-end on a
sample of escalated cases and produces label + rationale.

## Phase 4 — Combined Pipeline Evaluation
1. Evaluate the full Stage-1 + Stage-2 pipeline: auto-decided cases keep
   Stage-1's verdict, escalated cases use Stage-2's.
2. Compare against Stage-1-alone specifically on the escalated subset.
3. Re-run the fairness check on the combined pipeline — does Stage 2
   improve or worsen the false-positive gap for non-native English writing
   on ambiguous cases? Report honestly either way.
4. Log to comparison table.

**Acceptance:** table shows Stage-1-only vs Stage-1+2 (round 1), with both
the escalated-subset comparison and the fairness-gap comparison called out
explicitly.

## Phase 5 — HITL Review Queue
1. `hitl/review_app.py` (Streamlit): show each escalated essay (text,
   Stage 2's verdict + rationale); let the teacher accept, overturn, or tag a
   failure mode — including an explicit **"possible false flag: non-native
   writing style"** tag so the tool actively surfaces this fairness issue.
2. `apply_corrections.py`: persist reviewer decisions as labeled examples
   for the next round.

**Acceptance:** app runs via `streamlit run hitl/review_app.py`; one full
review pass completed with corrections saved.

## Phase 6 — Flywheel Round 2
1. Introduce one of the held-out (unseen) generation sources, simulating a
   new AI model appearing, plus the HITL-corrected examples.
2. Retrain Stage 1 and re-run LoRA fine-tuning for Stage 2 with the
   expanded data.
3. Re-evaluate on a held-out slice; recompute escalation rate and the
   fairness gap.
4. Update the comparison table and trend: round 1 vs round 2 for both
   accuracy metrics and the fairness gap.

**Acceptance:** escalation rate, accuracy, and fairness-gap logged for both
rounds — report the actual direction of change honestly.

## Phase 7 — Drift Monitoring
1. `drift/monitor.py`: compute PSI and/or KS tests on stylometric feature
   distributions and Stage-1 score distributions across generation-source
   batches, treating each source as a stand-in for a sequential time step.
2. Flag sources exceeding conventional thresholds (PSI > 0.1 moderate,
   > 0.2 major), with particular attention to the previously-unseen source
   introduced in Phase 6.

**Acceptance:** drift report generated across all generation-source
batches, with the unseen source's drift result discussed explicitly.

## Phase 8 — Visualization & README
1. `viz/trend_plots.py`: plot escalation rate, fairness gap, and
   cost-per-decision (using the documented cost-unit assumptions) across
   rounds.
2. Write the full README:
   - Problem statement & motivation
   - Methodology (Stage 1 → Stage 2 → HITL → round 2 → drift monitoring)
   - Metrics table, fairness-gap trend, and drift report
   - Explicit statement of the cost-unit and generation-source-as-time
     assumptions as illustrative
   - **Ethics & limitations section**: this is a decision-support tool
     requiring human confirmation on every flagged case, not an automated
     accusation system; the ESL fairness data is a proxy, not a validated
     benchmark; false positives carry real consequences for students

**Acceptance:** README reads coherently to a reviewer with no prior context,
and the ethics section is substantive, not a token paragraph.

## Do NOT
- Do not fine-tune Stage 2 on the full dataset — escalated cases only.
- Do not build or imply any automated-accusation workflow — human
  confirmation is required at every flagged step, in code comments, UI copy,
  and the README.
- Do not present the ESL fairness results as a validated benchmark if a
  proxy dataset was used — state the limitation explicitly.
- Do not present the cost-unit numbers as real financial data.
- Do not skip logging a metric because a phase underperformed.
- Do not fabricate or round results to look better.