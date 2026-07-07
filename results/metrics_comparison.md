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

