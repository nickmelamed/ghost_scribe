"""False-positive-rate check on non-native-English (ESL) writing.

Compares Stage 1's false-positive rate on the ELLIPSE ESL eval subset
(fairness/data_prep_esl.py — real English Language Learner essays, all
genuinely human-written) against the general test set's human essays.
Two FPR variants are reported:

  - fpr_threshold: P(predicted AI | true human) using the standard 0.5
    probability threshold. Comparable to FPR as usually reported in the
    detector-fairness literature.
  - fpr_auto_flag: P(auto-flagged for accusation | true human) using the
    calibrated escalation bands. This is the decision-relevant number for
    this tool specifically: it excludes essays that would be escalated to
    human review (not an accusation) and counts only essays that would be
    auto-flagged without any human in the loop.

Scope limitation: "ESL" here means the ELLIPSE corpus's specific
population (8th-12th grade English Language Learners in a US school
system), not the full space of non-native English writers this fairness
concern is generally about (e.g. adult L2 writers, writers educated
outside the US). Real, not-proxy data for the population it covers — see
fairness/data_prep_esl.py and the README ethics section.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from evaluation.metrics import auto_flag_rate, false_positive_rate
from stage1.escalation_policy import assign_band
from stage1.features import FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "stage1" / "models"


def compute_fairness_metrics() -> dict:
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    thresholds = json.loads((MODELS_DIR / "escalation_thresholds.json").read_text())
    t_low, t_high = thresholds["t_low"], thresholds["t_high"]

    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    esl = pd.read_csv(PROCESSED_DIR / "esl_eval_features.csv")

    test_human = test[test["label"] == 0]

    results = {}
    for name, df in [("general_test_human", test_human), ("esl_ellipse", esl)]:
        proba = xgb.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        bands = assign_band(proba, t_low, t_high)
        results[name] = {
            "n": int(len(df)),
            "fpr_threshold": false_positive_rate(df["label"].to_numpy(), y_pred),
            "fpr_auto_flag": auto_flag_rate(bands, df["label"].to_numpy(), positive_true_label=0),
            "escalate_rate": float((bands == "escalate").mean()),
        }

    results["fairness_gap_threshold"] = (
        results["esl_ellipse"]["fpr_threshold"] - results["general_test_human"]["fpr_threshold"]
    )
    results["fairness_gap_auto_flag"] = (
        results["esl_ellipse"]["fpr_auto_flag"] - results["general_test_human"]["fpr_auto_flag"]
    )
    return results


def main() -> int:
    results = compute_fairness_metrics()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
