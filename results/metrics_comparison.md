# Metrics Comparison

All numbers below are computed directly from held-out val/test splits and the ELLIPSE ESL eval subset — nothing here is rounded or cherry-picked; underperformance is reported as-is (see CLAUDE.md ground rules).

## Phase 2 — Stage 1 only (classical stylometric filter)

### Classification metrics

| model | split | AUC-ROC | precision | recall | F1 |
|---|---|---|---|---|---|
| logreg | val | 0.9837 | 0.8958 | 0.9389 | 0.9168 |
| logreg | test | 0.9833 | 0.8866 | 0.9389 | 0.9120 |
| xgb | val | 0.9930 | 0.9423 | 0.9633 | 0.9527 |
| xgb | test | 0.9924 | 0.9364 | 0.9581 | 0.9471 |

XGBoost is used as the Stage-1 decision model for escalation banding (higher val AUC). Bands are calibrated on val to target a 15-25% escalation rate (see stage1/escalation_policy.py).

### Escalation bands (XGBoost)

| | val | test |
|---|---|---|
| t_low | 0.0438 | — |
| t_high | 0.9562 | — |
| escalation rate | 0.2001 | 0.2148 |

### Baseline fairness gap — false positive rate on non-native English writing

Non-native-English subset: ELLIPSE corpus (real English Language Learner essays, not a proxy for this population — see fairness/data_prep_esl.py for the scope limitation on who "non-native English writer" covers here). All rows are genuinely human-written, so any AI verdict is a false positive.

| subgroup | n | FPR (0.5 threshold) | FPR (auto-flag only) | escalate rate |
|---|---|---|---|---|
| general_test_human | 4106 | 0.0363 | 0.0039 | 0.2077 |
| esl_ellipse | 6482 | 0.0538 | 0.0043 | 0.2487 |

**Fairness gap (0.5 threshold): +0.0176** (ESL FPR minus general-test FPR)

**Fairness gap (auto-flag only): +0.0004**

## Phase 4 — Combined Stage-1 + Stage-2 pipeline (round 1)

Stage 2 fallback: 0 of 1374 escalated test rows and 0 of 1612 escalated ESL rows had unparseable Stage-2 output and fell back to Stage-1's raw threshold decision (logged, not silently dropped).

### Escalated subset only — Stage-1-alone vs Stage-1+2

n = 1374 escalated test rows

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Stage-1-alone (raw threshold on ambiguous cases) | 0.8435 | 0.7675 | 0.8426 | 0.8033 |
| Stage-1+2 (fine-tuned Stage-2 verdict) | 0.9934 | 0.9961 | 0.9866 | 0.9913 |

### Full test set — Stage-1-only vs Stage-1+2 (round 1)

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Stage-1-only | 0.9617 | 0.9364 | 0.9581 | 0.9471 |
| Stage-1+2 (round 1) | 0.9939 | 0.9921 | 0.9908 | 0.9915 |

### Fairness gap re-check — combined pipeline

| subgroup | n | FPR (combined-pipeline final verdict) |
|---|---|---|
| general_test_human | 4106 | 0.0044 |
| esl_ellipse | 6482 | 0.0054 |

**Fairness gap, combined pipeline: +0.0010** (ESL FPR minus general-test FPR)

**Phase 2 baseline (Stage-1 threshold-based) gap was +0.0176.** Stage 2 narrows the gap on the escalated/ambiguous cases it touches — reported as measured, not assumed.

