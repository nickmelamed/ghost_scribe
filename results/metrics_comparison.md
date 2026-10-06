# Metrics Comparison

All numbers below are computed directly from held-out val/test splits and the ELLIPSE ESL eval subset — nothing here is rounded or cherry-picked; underperformance is reported as-is (see CLAUDE.md ground rules).

## Phase 2 — Stage 1 only (classical stylometric filter)

### Classification metrics

| model | split | AUC-ROC | precision | recall | F1 |
|---|---|---|---|---|---|
| logreg | val | 0.9837 | 0.9007 | 0.9389 | 0.9194 |
| logreg | test | 0.9828 | 0.8925 | 0.9389 | 0.9151 |
| xgb | val | 0.9929 | 0.9443 | 0.9633 | 0.9537 |
| xgb | test | 0.9921 | 0.9396 | 0.9581 | 0.9488 |

XGBoost is used as the Stage-1 decision model for escalation banding (higher val AUC). Bands are calibrated on val to target a 15-25% escalation rate (see stage1/escalation_policy.py).

### Escalation bands (XGBoost)

| | val | test |
|---|---|---|
| t_low | 0.0438 | — |
| t_high | 0.9562 | — |
| escalation rate | 0.2014 | 0.2172 |

Rates are measured on the current val and test splits with the saved thresholds. The thresholds were calibrated to an escalation rate of 0.2001 on val at calibration time.

### Baseline fairness gap — false positive rate on non-native English writing

Non-native-English subset: ELLIPSE corpus (real English Language Learner essays, not a proxy for this population — see fairness/data_prep_esl.py for the scope limitation on who "non-native English writer" covers here). All rows are genuinely human-written, so any AI verdict is a false positive.

| subgroup | n | FPR (0.5 threshold) | FPR (auto-flag only) | escalate rate |
|---|---|---|---|---|
| general_test_human | 3809 | 0.0370 | 0.0042 | 0.2111 |
| esl_ellipse | 5936 | 0.0569 | 0.0047 | 0.2559 |

**Fairness gap (0.5 threshold): +0.0199** (ESL FPR minus general-test FPR)

**Fairness gap (auto-flag only): +0.0005**

## Phase 4 — Combined Stage-1 + Stage-2 pipeline (round 1)

Stage 2 fallback: 0 of 1325 escalated test rows and 0 of 1519 escalated ESL rows had unparseable Stage-2 output and fell back to Stage-1's raw threshold decision (logged, not silently dropped).

### Escalated subset only — Stage-1-alone vs Stage-1+2

n = 1325 escalated test rows

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Stage-1-alone (raw threshold on ambiguous cases) | 0.8438 | 0.7784 | 0.8426 | 0.8092 |
| Stage-1+2 (fine-tuned Stage-2 verdict) | 0.9932 | 0.9961 | 0.9866 | 0.9913 |

### Full test set — Stage-1-only vs Stage-1+2 (round 1)

| | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| Stage-1-only | 0.9611 | 0.9396 | 0.9581 | 0.9488 |
| Stage-1+2 (round 1) | 0.9936 | 0.9921 | 0.9908 | 0.9915 |

### Fairness gap re-check — combined pipeline

| subgroup | n | FPR (combined-pipeline final verdict) |
|---|---|---|
| general_test_human | 3809 | 0.0047 |
| esl_ellipse | 5936 | 0.0056 |

**Fairness gap, combined pipeline: +0.0008** (ESL FPR minus general-test FPR)

**Phase 2 baseline (Stage-1 threshold-based) gap was +0.0199.** Stage 2 narrows the gap on the escalated/ambiguous cases it touches — reported as measured, not assumed.

