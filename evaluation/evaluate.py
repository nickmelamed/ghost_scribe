"""Stage-1-only evaluation: classification metrics, escalation rate,
and the baseline ESL fairness gap. Writes results/metrics_comparison.md.

Run after stage1/baseline_logreg.py, stage1/baseline_xgb.py, and
stage1/escalation_policy.py have produced their saved artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluation.metrics import classification_metrics, escalation_rate
from fairness.esl_eval import compute_fairness_metrics
from stage1.escalation_policy import assign_band
from stage1.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "stage1" / "models"
RESULTS_PATH = ROOT / "results" / "metrics_comparison.md"


def evaluate_model(model, split_df: pd.DataFrame) -> dict:
    proba = model.predict_proba(split_df[FEATURE_COLUMNS])[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    return classification_metrics(split_df["label"].to_numpy(), y_pred, proba)


def main() -> int:
    val = pd.read_csv(PROCESSED_DIR / "val_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")

    logreg = joblib.load(MODELS_DIR / "logreg.joblib")
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    thresholds = json.loads((MODELS_DIR / "escalation_thresholds.json").read_text())
    t_low, t_high = thresholds["t_low"], thresholds["t_high"]

    model_metrics = {
        "logreg": {"val": evaluate_model(logreg, val), "test": evaluate_model(logreg, test)},
        "xgb": {"val": evaluate_model(xgb, val), "test": evaluate_model(xgb, test)},
    }

    xgb_test_proba = xgb.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_bands = assign_band(xgb_test_proba, t_low, t_high)
    test_escalation_rate = escalation_rate(test_bands)

    fairness = compute_fairness_metrics()

    lines = []
    lines.append("# Metrics Comparison\n")
    lines.append(
        "All numbers below are computed directly from held-out val/test "
        "splits and the ELLIPSE ESL eval subset — nothing here is rounded "
        "or cherry-picked; underperformance is reported as-is (see "
        "CLAUDE.md ground rules).\n"
    )

    lines.append("## Phase 2 — Stage 1 only (classical stylometric filter)\n")
    lines.append("### Classification metrics\n")
    lines.append("| model | split | AUC-ROC | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for model_name, splits in model_metrics.items():
        for split_name, m in splits.items():
            lines.append(
                f"| {model_name} | {split_name} | {m['auc_roc']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
            )
    lines.append("")

    lines.append(
        "XGBoost is used as the Stage-1 decision model for escalation "
        "banding (higher val AUC). Bands are calibrated on val to target "
        "a 15-25% escalation rate (see stage1/escalation_policy.py).\n"
    )
    lines.append("### Escalation bands (XGBoost)\n")
    lines.append("| | val | test |")
    lines.append("|---|---|---|")
    lines.append(f"| t_low | {t_low:.4f} | — |")
    lines.append(f"| t_high | {t_high:.4f} | — |")
    lines.append(
        f"| escalation rate | {thresholds['achieved_escalation_rate_val']:.4f} | "
        f"{test_escalation_rate:.4f} |"
    )
    lines.append("")

    lines.append("### Baseline fairness gap — false positive rate on non-native English writing\n")
    lines.append(
        "Non-native-English subset: ELLIPSE corpus (real English Language "
        "Learner essays, not a proxy for this population — see "
        "fairness/data_prep_esl.py for the scope limitation on who "
        "\"non-native English writer\" covers here). All rows are "
        "genuinely human-written, so any AI verdict is a false positive.\n"
    )
    lines.append("| subgroup | n | FPR (0.5 threshold) | FPR (auto-flag only) | escalate rate |")
    lines.append("|---|---|---|---|---|")
    for name in ["general_test_human", "esl_ellipse"]:
        r = fairness[name]
        lines.append(
            f"| {name} | {r['n']} | {r['fpr_threshold']:.4f} | "
            f"{r['fpr_auto_flag']:.4f} | {r['escalate_rate']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Fairness gap (0.5 threshold): {fairness['fairness_gap_threshold']:+.4f}** "
        f"(ESL FPR minus general-test FPR)\n"
    )
    lines.append(
        f"**Fairness gap (auto-flag only): {fairness['fairness_gap_auto_flag']:+.4f}**\n"
    )

    report = "\n".join(lines) + "\n"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report)

    print(report)
    print(f"Wrote {RESULTS_PATH}")

    out = {
        "model_metrics": model_metrics,
        "escalation": {**thresholds, "test_escalation_rate": test_escalation_rate},
        "fairness": fairness,
    }
    (ROOT / "results" / "phase2_metrics.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
