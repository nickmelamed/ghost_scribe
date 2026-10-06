# Spec: AI-Generated Text Detection Flywheel

Status: DRAFT for review. Items marked **[INFERRED]** come from reading the
code, not from the author. Items marked **[CONFIRMED]** were stated by the
author in the Phase 1 interview. Correct anything marked [INFERRED].

The original build spec (phases 0-8, ground rules, "Do NOT" list) is in
the "Build spec" section at the bottom of this file. Its headings are the
originals, one level deeper, so references to "Phase N" still work.

## 1. Purpose [CONFIRMED]

A two-stage AI-essay detector for teacher triage. An XGBoost stylometric
filter auto-clears or auto-flags confident cases and escalates about 20%
ambiguous ones to a LoRA-tuned small LLM. Teachers review escalations in a
Streamlit HITL app. Their corrections plus an unseen generator feed round
2 and drift monitoring. The false-positive gap on non-native (ESL) writing
is tracked every round. It is decision support only, never an automatic
verdict.

## 2. Rules that must not be broken [CONFIRMED unless noted]

1. **No automatic verdicts.** Every flagged or escalated case needs human
   confirmation. No code path or UI copy implies an automatic accusation.
2. **Held-out sources stay unseen.** `falcon_180b_v1` and `llama_70b_v1`
   never appear in round-1 train, val, test, or threshold calibration. They
   are described as a stand-in for a new model, not as real chronological
   drift.
3. **Honest metrics and proxies.** Every metric is logged even when bad, no
   rounding or cherry-picking. ELLIPSE ESL results are not a validated
   benchmark for all non-native writers. The 1/5/25 cost units are
   illustrative, not financial data.
4. **Stage 2 trains on escalated cases only** [INFERRED from CLAUDE.md "Do
   NOT". Not selected in the interview, so confirmed only by the marking.]
5. **Ground-truth labels never reach reviewers.** The review app and LLM
   reviewer drop the true label before display [INFERRED from code].

## 3. Pipeline as built [INFERRED]

### Data (`data/`)
- Source: `thedrcat/daigt-v2-train-dataset` (not the official competition
  file, which lacks per-model source labels). 44,868 raw rows.
- `preprocess.py` drops 3 AI rows with no recorded generator, sets
  `generation_source` ("human" or model name), normalizes line endings,
  strips whitespace, adds `char_count` and `word_count`.
- Holds out two sources (`falcon_180b_v1`, `llama_70b_v1`) into
  `held_out.csv`. Splits the rest 70/15/15, stratified by
  `generation_source`, seed 42.

### Stage 1 (`stage1/`)
- `features.py`: 24 features (length, sentence-length std, type-token
  ratio, bigram-burstiness proxy for perplexity, 8 POS ratios, 5
  punctuation rates, 3 readability scores). No language model is used, so
  there is no true perplexity.
- Logistic regression and XGBoost baselines. XGBoost is the decision
  model because its validation AUC is higher.
- `escalation_policy.py`: picks symmetric thresholds around 0.5 so that
  20% of validation rows fall in the middle band (`escalate`). Below the
  band is `auto_clear`, above is `auto_flag`. Thresholds are saved to
  `stage1/models/escalation_thresholds.json`.

### Fairness (`fairness/`)
- Eval set: ELLIPSE corpus (real English Language Learner essays, grades
  8-12, US schools), all human-written.
- Reports FPR at the 0.5 threshold and the auto-flag-only rate for ESL
  essays and for general-test human essays, and the difference (gap).
- Scope: ELL students in US schools only, not all non-native writers.

### Stage 2 (`stage2/`)
- Qwen2.5-3B-Instruct plus LoRA (r=16, alpha=32, attention projections),
  bf16 on Apple MPS, no 4-bit quantization on this machine.
- Training data: Stage-1-escalated train rows only. Target is the true
  label plus a templated one-sentence rationale built from the two most
  label-supportive features among the top 4 XGBoost features. The
  rationale is a heuristic, not a verified explanation.
- `infer.py` outputs label and rationale. If output cannot be parsed, the
  pipeline falls back to Stage 1's 0.5-threshold decision and counts it.
- `confidence.py`: two-way teacher-forced log-probability of
  "Label: AI-generated" vs "Label: Human-written".
- `build_review_queue.py`: every "AI-generated" verdict goes to the
  queue. Only the least-confident 20% of "Human-written" verdicts do.
  The 20% was calibrated using the escalated test split (see DECISIONS).

### Evaluation (`evaluation/`)
- `evaluate.py` (Stage 1 only) and `evaluate_combined.py` (Stage 1+2,
  round 1) write `results/metrics_comparison.md` and
  `results/phase{2,4}_metrics.json`.
- Combined rule: auto-decided rows keep Stage 1's verdict, escalated rows
  use Stage 2's.

### HITL (`hitl/`)
- `review_app.py`: Streamlit app over the review queue from the **val**
  split (test stays untouched). Reviewer accepts, overturns, or tags a
  failure mode, including "possible false flag: non-native writing
  style". Writes `hitl/reviews.jsonl` (gitignored).
- `llm_review.py`: a second reviewer using the base Qwen model with no
  adapter, writing `hitl/llm_reviews.jsonl`. It is an LLM opinion and must
  never be reported as human review.
- `apply_corrections.py`: turns reviews into labeled rows
  (`hitl_corrections_{human,llm}.csv`) using the reviewer's label, not
  Stage 2's. The two sources are kept separate.

## 4. Reported claims [CONFIRMED]

- Stage-1-only vs Stage-1+2 classification metrics, including the
  escalated-subset comparison (`results/metrics_comparison.md`,
  `results/phase*_metrics.json`).
- Drift (PSI/KS per generation source) and cost-per-decision across
  rounds. Not built yet.

Not selected in the interview, so not guarded by the number check unless
you add them: escalation rate and ESL fairness gap. They are in the same
results file today. **Please confirm whether they should be guarded.**

## 5. Not built yet [INFERRED]

Phase 6 (round 2 retrain with a held-out source plus HITL corrections),
Phase 7 (`drift/monitor.py`), Phase 8 (`viz/trend_plots.py`, full README),
and a completed human review pass (`hitl/reviews.jsonl` exists locally
but is unfinished and will be redone).

## 6. Findings and planned changes (from the Phase 1 review)

Confirmed by the author: markings in sections 2 and 4 are fine as drafted
(so rule 4 is guarded, and ESL gap and escalation rate stay unguarded).

1. **Test-set contamination: to be fixed.** `build_review_queue.py` sets
   the human-review gate rate (20%) from the escalated *test* split's
   errors. Planned fix: calibrate on val instead, then report the gate's
   behavior on test without tuning on it. This changes behavior and
   results, so it is a separate commit after the standards are installed,
   not part of the style cleanup. Numbers that depend on it must be
   regenerated and logged, with the before and after both recorded.
2. **99.3% on escalated test rows: checked, not explained by duplicates.**
   Read-only check on existing CSVs:
   - Exact text overlap: train/test 1, train/val 1, val/test 0. The
     preprocess docstring claims texts are unique, which is off by one.
   - Near-duplicates exist: 326 of 6,396 test rows (5.1%) share their
     first 200 normalized characters with a train row, and 37 of the 1,374
     escalated test rows share at least 80% of their 8-word shingles with
     one train essay.
   - Stage-2 accuracy on those 37 is 1.0000, but on the rest it is
     0.9933, so duplicates do not account for the high number.
   - All 15 prompts appear in train and test, so Stage 2 can learn
     prompt-specific wording. The dataset is not split by prompt.
   - 853 of 1,374 escalated test rows are human, with 99.77% accuracy.
     The weakest sources are mistral7binstruct_v2 (94.3%, n=105) and
     cohere-command (96.6%, n=29).
   - Not ruled out: Stage 2 reads the full text while Stage 1 sees 24
     summary features, so a large gap may be real. A prompt-held-out split
     would test whether it survives. Until then, report 99.3% with these
     caveats, and de-duplicate the splits before round 2.
3. **HITL pass to be redone entirely.** The existing `hitl/reviews.jsonl`
   is unfinished and will be superseded once the changes above and the
   round-2 data are in. Archive it (outside git, gitignored) rather than
   delete it, and review a fresh queue.

## 7. Decisions on scope (author, Phase 1)

1. **Remove near-duplicates across splits.** Recommended by Claude and
   accepted: near-duplicate train/test pairs inflate every metric, Stage 1
   included. Plan: drop test and val rows that share at least 80% of their
   8-word shingles with a train essay (threshold is a proposal, to be
   confirmed when implemented), and drop the 2 exact-text overlaps. Log
   how many rows were removed per source and split, and regenerate all
   downstream results. This changes the numbers in `results/`, so record
   before and after.
2. **Add a prompt-held-out evaluation.** Hold out whole essay prompts from
   Stage 1 and Stage 2 training and evaluate on them, to test whether the
   Stage-2 gain survives on unseen topics. Report it next to the main
   result, not instead of it. Which prompts to hold out is to be decided
   at implementation time and documented.

Both are behavior changes in `data/preprocess.py` and the evaluation code,
so they come after the standards are installed and tests exist
(characterization tests first, so the before-state is pinned). They also
feed the redo of the HITL pass.

## 8. Open questions

None outstanding from Phase 1.

---

## Build spec (original, from the first CLAUDE.md)

### Context & Goal
Portfolio project demonstrating a production-style ML flywheel applied to
academic-integrity triage: a classical stylometric model as a fast
first-pass filter, a fine-tuned small LLM as an escalation layer for
ambiguous essays, a teacher-facing human-in-the-loop review queue, and drift
monitoring as new AI-generation sources appear. A dedicated fairness check
tracks false-positive rates on non-native English writing — a real,
well-documented failure mode of AI-text detectors — across every round.

### Definition of Done
All phases below complete, with:
- `results/metrics_comparison.md` showing Stage-1-only vs Stage-1+2 (round 1)
  vs Stage-1+2 (round 2, post-HITL)
- A false-positive-rate-by-subgroup (native vs non-native English writer)
  trend across rounds
- A drift-monitoring report across generation-source batches
- A README that reads as a coherent portfolio piece to a technical reviewer
  with no prior context, including an explicit ethics/limitations section

### Ground Rules
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

### Tech Stack
- Python 3.10+
- scikit-learn, xgboost
- nltk or spacy (POS tagging, stylometric features), textstat (readability)
- transformers, peft, bitsandbytes (optional)
- streamlit
- pandas, numpy, scipy (PSI/KS drift tests)
- matplotlib or plotly

### Repo Structure
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

### Phase 0 — Environment & Scaffold
1. Init git repo, create the structure above.
2. Create `pyproject.toml`/`requirements.txt` pinning the libs above.
3. Stub `README.md` with a project description (fill in fully in Phase 8).

**Acceptance:** `pip install -r requirements.txt` runs clean; structure matches spec.

### Phase 1 — Data Acquisition & Preprocessing
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

### Phase 2 — Stage 1: Classical Stylometric Filter
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

### Phase 3 — Stage 2: Fine-tuned LLM Escalation Layer
1. `prepare_finetune_data.py`: for escalated (middle-band) training cases,
   build prompts from the essay text, with the true label as the fine-tuning
   target and a short rationale (which stylistic cues suggest AI vs human).
2. LoRA fine-tune the chosen small model on this escalated subset only.
3. `infer.py`: run the fine-tuned model on escalated test cases.

**Acceptance:** fine-tuned adapter saved; inference runs end-to-end on a
sample of escalated cases and produces label + rationale.

### Phase 4 — Combined Pipeline Evaluation
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

### Phase 5 — HITL Review Queue
1. `hitl/review_app.py` (Streamlit): show each escalated essay (text,
   Stage 2's verdict + rationale); let the teacher accept, overturn, or tag a
   failure mode — including an explicit **"possible false flag: non-native
   writing style"** tag so the tool actively surfaces this fairness issue.
2. `apply_corrections.py`: persist reviewer decisions as labeled examples
   for the next round.

**Acceptance:** app runs via `streamlit run hitl/review_app.py`; one full
review pass completed with corrections saved.

### Phase 6 — Flywheel Round 2
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

### Phase 7 — Drift Monitoring
1. `drift/monitor.py`: compute PSI and/or KS tests on stylometric feature
   distributions and Stage-1 score distributions across generation-source
   batches, treating each source as a stand-in for a sequential time step.
2. Flag sources exceeding conventional thresholds (PSI > 0.1 moderate,
   > 0.2 major), with particular attention to the previously-unseen source
   introduced in Phase 6.

**Acceptance:** drift report generated across all generation-source
batches, with the unseen source's drift result discussed explicitly.

### Phase 8 — Visualization & README
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

### Do NOT
- Do not fine-tune Stage 2 on the full dataset — escalated cases only.
- Do not build or imply any automated-accusation workflow — human
  confirmation is required at every flagged step, in code comments, UI copy,
  and the README.
- Do not present the ESL fairness results as a validated benchmark if a
  proxy dataset was used — state the limitation explicitly.
- Do not present the cost-unit numbers as real financial data.
- Do not skip logging a metric because a phase underperformed.
- Do not fabricate or round results to look better.
