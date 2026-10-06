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
4. **Stage 2 trains on escalated cases only** [INFERRED from the "Do NOT" list in
   the original build spec.]
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
- `dedup.py` runs after preprocessing and before features. It drops val,
  test and ESL rows with at least 80% of their 8-word shingles in one train
  essay, and test rows that do the same against the kept val. It removes the
  same essays from the derived files by text, and writes counts to
  `results/dedup_report.json` (see 6.4 and D20).

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
  queue. Only the least-confident fraction of "Human-written" verdicts do.
  The fraction is calibrated on the escalated val split (the smallest rate
  that queues every val error where Stage 2 said human and the essay was
  AI, plus a 5-point margin). The test split only gets a read-only check.
  Both are recorded in `results/review_gate.json` (see D13).

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

1. **Test-set contamination: fixed.** `build_review_queue.py` used to set
   the human-review gate rate (20%) from the escalated *test* split's
   errors. The rate is now calibrated on val, and test only gets a check
   that is never tuned on.
   - Before (20%, set from test): the gate queued 7 of 7 dangerous errors
     on test, but only 8 of 9 on val. The val queue had 676 of 1,280
     escalated essays (525 AI verdicts plus 151 human verdicts).
   - After (val-calibrated): the needed rate on val is 36.6%, because one
     error had 0.994 confidence. With the 5-point margin the gate rate is
     41.6%. The val queue has 838 of 1,280 (525 plus 313), and all 9 val
     errors are queued. On test the same rate queues 7 of 7 errors with
     872 of 1,374 essays in the queue.
   - The test result is weak evidence. Every test error sat below the 17th
     percentile, so any rate above about 17% would have caught all 7. With
     9 and 7 errors in total, the rate and the margin are noisy estimates.
   - Cost: the review queue is about 24% larger, and confidence gating still
     cannot promise to catch every confident mistake.
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
     would test whether it survives. That test has since run, see 6.6.
     Near-duplicates were removed, see 6.4.
3. **HITL pass to be redone entirely.** The existing `hitl/reviews.jsonl`
   is unfinished and will be superseded once the changes above and the
   round-2 data are in. Archive it (outside git, gitignored) rather than
   delete it, and review a fresh queue.

4. **Near-duplicates: removed, with a larger finding about the ESL set.**
   Overlap is the share of an essay's 8-word shingles found in one train
   essay, and 0.8 sits in a sparse region of the overlap distribution.
   - Removed: val 228 of 6,396, test 297 of 6,396 (236 near train plus 61
     near the kept val), ESL 546 of 6,482. All were human essays. No AI
     essay and no held-out essay was a near-duplicate.
   - **ESL contamination.** 8.4% of the ELLIPSE fairness set was already in
     Stage 1's training set, labeled human, and 538 of those 546 overlapped
     at 0.95 or more. The baseline false-positive rate on ESL essays was
     therefore measured partly on essays the model trained on.
   - Effect on results, with thresholds, models and adapter frozen:
     Stage-1 XGBoost test AUC 0.9924 to 0.9921, Stage-1+2 test accuracy
     0.9939 to 0.9936, and Stage-2 accuracy on escalated test rows 0.9934 to
     0.9932. Duplicates do not explain the 99.3%.
   - The ESL false-positive rate at 0.5 rose from 0.0538 to 0.0569, and the
     baseline gap from +0.0176 to +0.0199. That is in the expected direction,
     but at these sample sizes the change is within noise.
   - Escalation rate with the frozen thresholds is 0.2014 on val and 0.2172
     on test, still inside 15-25%.
   - Review gate on the cleaned val: needed rate 36.7%, gate rate 41.7%,
     queue 823 of 1,242 (66.3%), 9 of 9 val errors queued and 7 of 7 on test.
   - Limits: light edits can stay under the cutoff, and 47 val, 64 test and
     17 ESL rows with an overlap from 0.5 to 0.8 were kept. The general
     human test set probably also contains English-learner writers, since
     ELLIPSE and Persuade overlap, so it is not a clean non-ESL group.
5. **Duplicate `word_count` and `char_count` columns in the feature files.**
   `stage1.features` concatenates the preprocessing columns with its own, so
   the CSVs hold both, and pandas reads the second pair as `word_count.1`
   and `char_count.1`. Models use the first pair, which is the whitespace
   split from preprocessing. It differs from the feature function's token
   count by 4.3 words on average (about 1-2%). `char_count` is identical.
   Train and evaluation are consistent, so the reported results stand, but
   `extract_features` alone disagrees with the trained model's `word_count`.
   Not fixed, because fixing it means retraining every model. Decide before
   round 2.

6. **Prompt-held-out evaluation: the Stage-2 result survives, Stage 1 does
   not.** Three of the 15 prompts were held out with a seeded draw (seed 42:
   Mandatory extracurricular activities, Car-free cities, "A Cowboy Who Rode
   the Waves"). Stage 1 was retrained on the other 12 prompts, its bands
   were recalibrated on their val rows, and Stage 2 was fine-tuned for 3
   epochs on the 4,387 escalated rows of those prompts, with the main run's
   hyperparameters. Full tables are in `results/prompt_holdout.md`.
   - Stage-2 accuracy on escalated test rows was 0.9944 on the held-out
     prompts (354 rows, 95% interval 0.9796 to 0.9984), 0.9931 on the seen
     prompts (1,010 rows) and 0.9932 in the main run. So the 99.3% is not
     explained by prompt memorization.
   - Stage 1 alone is worse on unseen prompts. Test accuracy was 0.9280 on
     held-out prompts and 0.9636 on seen prompts. The escalation rate was
     0.2930 against 0.2065, and the false-positive rate on human essays at
     the 0.5 threshold was 0.0477 against 0.0340.
   - The combined pipeline is also worse on unseen prompts, even though Stage
     2 keeps its accuracy on the rows it sees. Accuracy was 0.9884 against
     0.9947. The human false-positive rate was 0.0087 against 0.0032, which
     is about 2.7 times higher, but that is 6 of 692 human essays against 10
     of 3,117, so the ratio is a rough estimate. Stage 2 only sees the
     escalated rows, so Stage-1 auto-flags stay wrong. Five of the 6
     held-out-prompt false positives and 8 of the 10 seen-prompt ones were
     Stage-1 auto-flags. The rest came from Stage 2.
   - Dangerous errors (said human, was AI) among escalated rows were 1 on the
     held-out prompts and 5 on the seen prompts.
   - Limits: this is one draw of 3 prompts, so a different draw could
     differ. It does not test unseen generators, which are reserved for round
     2. Why a language model reading the whole essay does so much better than
     the 24 summary features on these rows is still not explained.

7. **Drift monitoring across generation sources (Phase 7).** `drift/monitor.py`
   computes PSI and KS on the 24 features and on the Stage-1 score for each
   source, with the round-1 models. Each source stands in for a step in time,
   which is a substitution and not real drift. The full table is in
   `results/drift_round1.md`.
   - **Feature-level flags do not separate new from old sources.** Against
     the pooled AI reference, nearly every source shows 17 to 23 features over
     0.1, including sources the models trained on. A single generator differs
     in style from the mixture. Stage-1 score drift and routing are the more
     useful signals.
   - **Seen sources.** Score PSI against the pooled AI reference is major
     for 6 of 14 seen sources, and for 5 of them it is above the noise level.
     That reflects a source differing from the pool, not a new source,
     because against their own train rows the score PSI is at most 0.295.
     Stage 1 clears (misses) 0.8% or fewer of each seen source with at
     least 150 essays. It clears 7.7% of cohere-command and 5.8% of
     palm-text-bison1, which have 52 essays each, so that is 4 and 3 essays.
   - **llama_70b_v1 (unseen) drifts.** Score PSI is 0.509 (major, against a
     noise level of 0.015), Stage 1 clears 4.7% of its 1,172 essays with no
     review, and it escalates 44.3% of them, against 8.7% to 43.8% for the
     seen sources with at least 150 essays. This is the clearest case of a new generator shifting
     Stage 1's behavior, and round 2 introduces it.
   - **falcon_180b_v1 (unseen) mostly does not.** Score PSI is 0.037 (below
     the 0.1 threshold, above its 0.016 noise level), 0.9% is cleared and
     24.6% is escalated. It behaves like the seen sources, so it stays unseen
     in round 2 as a generalization test.
   - Limits: the reference is pooled, the batches differ in size by a factor
     of 40, and PSI with ten quantile bins is coarse on the near-binary
     Stage-1 score.

## 7. Decisions on scope (author, Phase 1)

1. **Remove near-duplicates across splits.** Done, see 6.4. Near-duplicate
   train/val/test pairs were removed at an overlap of at least 80%, the
   exact-text overlaps are covered by that rule, and the removals are logged
   per split, source and label.
2. **Add a prompt-held-out evaluation.** Done, see 6.6. Whole essay prompts
   were held out from both stages and tested next to the main result, with
   the prompts chosen by a seeded draw.

Both are behavior changes in `data/preprocess.py` and the evaluation code,
so they come after the standards are installed and tests exist
(characterization tests first, so the before-state is pinned). They also
feed the redo of the HITL pass.

## 8. Open questions

None outstanding from Phase 1.

---

## Build spec (original)

### Context & Goal
Portfolio project demonstrating a production-style ML flywheel applied to
academic-integrity triage: a classical stylometric model as a fast
first-pass filter, a fine-tuned small LLM as an escalation layer for
ambiguous essays, a teacher-facing human-in-the-loop review queue, and drift
monitoring as new AI-generation sources appear. A dedicated fairness check
tracks false-positive rates on non-native English writing, a real,
well-documented failure mode of AI-text detectors, across every round.

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
- Dataset: Kaggle's "LLM - Detect AI Generated Text" (DAIGT) competition
  dataset with real human essays plus AI-generated essays from multiple LLMs.
  Requires a free Kaggle account. Flag this in Phase 1 and pause for the
  human if a manual download/API key step is needed.
- **Group by generation source, not by date.** There's no natural timestamp
  axis here, so essays are grouped by which model generated them. Hold out
  1–2 entirely unseen generation sources from initial training. These
  stand in for "a new AI model appears later" and drive both the round-2
  retraining and drift-monitoring phases. Document this substitution
  explicitly in the README, and do not present it as real chronological drift.
- Fairness data: for the false-positive-rate check, use a publicly
  available L2/non-native-English essay corpus if one is accessible for
  research use. If a clean ESL-labeled corpus isn't available, use a
  documented proxy (e.g., essays with certain syntactic-complexity or
  L2-typical error profiles) and state clearly in the README that this is
  an approximation, not a validated benchmark.
- Framing: this is a decision-support triage tool for teachers, not an
  automated accusation system. Every escalated/flagged case requires human
  confirmation before any action is taken. State this explicitly in the
  README's ethics section. Do not let any code path or UI copy imply an
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

### Phase 0: Environment & Scaffold
1. Init git repo, create the structure above.
2. Create `pyproject.toml`/`requirements.txt` pinning the libs above.
3. Stub `README.md` with a project description (fill in fully in Phase 8).

Acceptance: `pip install -r requirements.txt` runs clean, and the structure matches spec.

### Phase 1: Data Acquisition & Preprocessing
1. `data/download.py`: fetch the DAIGT dataset. If a manual Kaggle
   download/API key step is required, stop and tell the human exactly what
   to do, then continue once files are in place.
2. Label each essay with its generation source (human, or which LLM
   generated it).
3. Clean and prepare text, and extract basic metadata needed for stylometric
   features downstream.
4. Split: hold out 1–2 entire generation sources (unseen at training time)
   for later phases, and use the remaining sources for train/val/test.

Acceptance: preprocessed dataset exists with documented schema, and
generation-source counts logged, including which sources are held out.

### Phase 2: Stage 1, Classical Stylometric Filter
1. `features.py`: extract stylometric/statistical features such as perplexity or
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

Acceptance: Stage-1-only metrics table exists, including escalation rate
and the baseline fairness gap (subgroup FPR difference).

### Phase 3: Stage 2, Fine-tuned LLM Escalation Layer
1. `prepare_finetune_data.py`: for escalated (middle-band) training cases,
   build prompts from the essay text, with the true label as the fine-tuning
   target and a short rationale (which stylistic cues suggest AI vs human).
2. LoRA fine-tune the chosen small model on this escalated subset only.
3. `infer.py`: run the fine-tuned model on escalated test cases.

Acceptance: fine-tuned adapter saved, and inference runs end-to-end on a
sample of escalated cases and produces label + rationale.

### Phase 4: Combined Pipeline Evaluation
1. Evaluate the full Stage-1 + Stage-2 pipeline: auto-decided cases keep
   Stage-1's verdict, escalated cases use Stage-2's.
2. Compare against Stage-1-alone specifically on the escalated subset.
3. Re-run the fairness check on the combined pipeline. Does Stage 2
   improve or worsen the false-positive gap for non-native English writing
   on ambiguous cases? Report honestly either way.
4. Log to comparison table.

Acceptance: table shows Stage-1-only vs Stage-1+2 (round 1), with both
the escalated-subset comparison and the fairness-gap comparison called out
explicitly.

### Phase 5: HITL Review Queue
1. `hitl/review_app.py` (Streamlit): show each escalated essay (text,
   Stage 2's verdict + rationale), and let the teacher accept, overturn, or tag a
   failure mode, including an explicit **"possible false flag: non-native
   writing style"** tag so the tool actively surfaces this fairness issue.
2. `apply_corrections.py`: persist reviewer decisions as labeled examples
   for the next round.

Acceptance: app runs via `streamlit run hitl/review_app.py`, and one full
review pass completed with corrections saved.

### Phase 6: Flywheel Round 2
1. Introduce one of the held-out (unseen) generation sources, simulating a
   new AI model appearing, plus the HITL-corrected examples.
2. Retrain Stage 1 and re-run LoRA fine-tuning for Stage 2 with the
   expanded data.
3. Re-evaluate on a held-out slice, then recompute escalation rate and the
   fairness gap.
4. Update the comparison table and trend: round 1 vs round 2 for both
   accuracy metrics and the fairness gap.

Acceptance: escalation rate, accuracy, and fairness-gap logged for both
rounds. Report the actual direction of change honestly.

### Phase 7: Drift Monitoring
1. `drift/monitor.py`: compute PSI and/or KS tests on stylometric feature
   distributions and Stage-1 score distributions across generation-source
   batches, treating each source as a stand-in for a sequential time step.
2. Flag sources exceeding conventional thresholds (PSI > 0.1 moderate,
   > 0.2 major), with particular attention to the previously-unseen source
   introduced in Phase 6.

Acceptance: drift report generated across all generation-source
batches, with the unseen source's drift result discussed explicitly.

### Phase 8: Visualization & README
1. `viz/trend_plots.py`: plot escalation rate, fairness gap, and
   cost-per-decision (using the documented cost-unit assumptions) across
   rounds.
2. Write the full README:
   - Problem statement & motivation
   - Methodology (Stage 1 → Stage 2 → HITL → round 2 → drift monitoring)
   - Metrics table, fairness-gap trend, and drift report
   - Explicit statement of the cost-unit and generation-source-as-time
     assumptions as illustrative
   - Ethics & limitations section: this is a decision-support tool
     requiring human confirmation on every flagged case, not an automated
     accusation system, the ESL fairness data is a proxy and not a validated
     benchmark, and false positives carry real consequences for students

Acceptance: README reads coherently to a reviewer with no prior context,
and the ethics section is substantive, not a token paragraph.

### Do NOT
- Do not fine-tune Stage 2 on the full dataset, escalated cases only.
- Do not build or imply any automated-accusation workflow. Human
  confirmation is required at every flagged step, in code comments, UI copy,
  and the README.
- Do not present the ESL fairness results as a validated benchmark if a
  proxy dataset was used. State the limitation explicitly.
- Do not present the cost-unit numbers as real financial data.
- Do not skip logging a metric because a phase underperformed.
- Do not fabricate or round results to look better.
