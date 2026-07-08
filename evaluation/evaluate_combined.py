"""Phase 4: evaluate the combined Stage-1 + Stage-2 pipeline.

Combines Stage 1's auto-decided verdicts (auto_clear -> human, auto_flag
-> AI) with Stage 2's verdict on escalated (ambiguous) rows, and:
  1. Compares Stage-1-alone vs Stage-1+2 specifically on the escalated
     subset (where Stage 2 actually changes the decision).
  2. Compares full-pipeline metrics: Stage-1-only (raw 0.5-threshold
     decision on every row) vs Stage-1+2 (combined decision).
  3. Re-runs the ESL fairness check using the combined-pipeline decision,
     and compares the fairness gap against the Phase 2 baseline.

Run after stage2/infer.py has produced predictions for the full escalated
test and esl_eval splits (not the --n_sample demo runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluation.metrics import classification_metrics
from stage1.escalation_policy import assign_band
from stage1.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "stage1" / "models"
RESULTS_PATH = ROOT / "results" / "metrics_comparison.md"


def score_and_band(df: pd.DataFrame, xgb, t_low: float, t_high: float) -> pd.DataFrame:
    proba = xgb.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    bands = assign_band(proba, t_low, t_high)
    df = df.copy()
    df["stage1_proba"] = proba
    df["stage1_pred"] = (proba >= 0.5).astype(int)
    df["stage1_band"] = bands
    return df


def attach_stage2(df: pd.DataFrame, stage2_path: Path) -> pd.DataFrame:
    """Join Stage-2 predictions onto df by essay text; fill missing/unparsed
    Stage-2 outputs with the Stage-1 raw threshold decision as a documented
    fallback (and report how often this happens — never silently)."""
    s2 = pd.read_csv(stage2_path)[["text", "stage2_pred_label"]]
    merged = df.merge(s2, on="text", how="left")

    escalated = merged["stage1_band"] == "escalate"
    missing_s2 = escalated & merged["stage2_pred_label"].isna()
    n_missing = int(missing_s2.sum())
    n_escalated = int(escalated.sum())

    merged["final_pred"] = merged["stage1_pred"]
    merged.loc[escalated, "final_pred"] = merged.loc[escalated, "stage2_pred_label"]
    merged.loc[missing_s2, "final_pred"] = merged.loc[missing_s2, "stage1_pred"]
    merged["final_pred"] = merged["final_pred"].astype(int)

    return merged, n_missing, n_escalated


def escalated_subset_comparison(df: pd.DataFrame) -> dict:
    esc = df[df["stage1_band"] == "escalate"]
    y_true = esc["label"].to_numpy()

    stage1_metrics = classification_metrics(y_true, esc["stage1_pred"].to_numpy(), esc["stage1_proba"].to_numpy())
    # Stage-2 has no probability score (generated text, not a classifier
    # score) so AUC-ROC isn't defined for it; report it as NaN rather than
    # a fabricated number, and use the hard label for precision/recall/F1.
    s2_pred = esc["final_pred"].to_numpy()
    stage2_metrics = {
        "auc_roc": float("nan"),
        "precision": float((s2_pred[y_true == 1] == 1).sum() / max((s2_pred == 1).sum(), 1)),
        "recall": float((s2_pred[y_true == 1] == 1).sum() / max((y_true == 1).sum(), 1)),
    }
    from sklearn.metrics import f1_score

    stage2_metrics["f1"] = float(f1_score(y_true, s2_pred, zero_division=0))
    stage2_metrics["accuracy"] = float((s2_pred == y_true).mean())
    stage1_metrics["accuracy"] = float((esc["stage1_pred"].to_numpy() == y_true).mean())

    return {"n": len(esc), "stage1_alone": stage1_metrics, "stage1_plus_2": stage2_metrics}


def full_pipeline_comparison(df: pd.DataFrame) -> dict:
    y_true = df["label"].to_numpy()
    stage1_only = classification_metrics(y_true, df["stage1_pred"].to_numpy(), df["stage1_proba"].to_numpy())
    stage1_only["accuracy"] = float((df["stage1_pred"].to_numpy() == y_true).mean())

    from sklearn.metrics import f1_score, precision_score, recall_score

    combined = {
        "auc_roc": float("nan"),  # combined decision is a hard label, not a score
        "precision": float(precision_score(y_true, df["final_pred"], zero_division=0)),
        "recall": float(recall_score(y_true, df["final_pred"], zero_division=0)),
        "f1": float(f1_score(y_true, df["final_pred"], zero_division=0)),
        "accuracy": float((df["final_pred"].to_numpy() == y_true).mean()),
    }
    return {"stage1_only": stage1_only, "stage1_plus_2": combined}


def combined_fpr(df_human: pd.DataFrame) -> float:
    return float(df_human["final_pred"].mean())


def main() -> int:
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    thresholds = json.loads((MODELS_DIR / "escalation_thresholds.json").read_text())
    t_low, t_high = thresholds["t_low"], thresholds["t_high"]

    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    esl = pd.read_csv(PROCESSED_DIR / "esl_eval_features.csv")

    test = score_and_band(test, xgb, t_low, t_high)
    esl = score_and_band(esl, xgb, t_low, t_high)

    test, n_missing_test, n_esc_test = attach_stage2(test, PROCESSED_DIR / "stage2_predictions_test.csv")
    esl, n_missing_esl, n_esc_esl = attach_stage2(esl, PROCESSED_DIR / "stage2_predictions_esl_eval.csv")

    print(f"test: {n_missing_test}/{n_esc_test} escalated rows fell back to Stage-1 (unparsed Stage-2 output)")
    print(f"esl_eval: {n_missing_esl}/{n_esc_esl} escalated rows fell back to Stage-1 (unparsed Stage-2 output)")

    esc_comparison = escalated_subset_comparison(test)
    full_comparison = full_pipeline_comparison(test)

    test_human = test[test["label"] == 0]
    general_fpr_combined = combined_fpr(test_human)
    esl_fpr_combined = combined_fpr(esl)
    fairness_gap_combined = esl_fpr_combined - general_fpr_combined

    # Phase 2 baseline for comparison
    phase2 = json.loads((ROOT / "results" / "phase2_metrics.json").read_text())
    baseline_gap_threshold = phase2["fairness"]["fairness_gap_threshold"]

    lines = []
    lines.append("## Phase 4 — Combined Stage-1 + Stage-2 pipeline (round 1)\n")

    lines.append(
        f"Stage 2 fallback: {n_missing_test} of {n_esc_test} escalated test rows and "
        f"{n_missing_esl} of {n_esc_esl} escalated ESL rows had unparseable Stage-2 output "
        "and fell back to Stage-1's raw threshold decision (logged, not silently dropped).\n"
    )

    lines.append("### Escalated subset only — Stage-1-alone vs Stage-1+2\n")
    lines.append(f"n = {esc_comparison['n']} escalated test rows\n")
    lines.append("| | accuracy | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|")
    s1 = esc_comparison["stage1_alone"]
    s2 = esc_comparison["stage1_plus_2"]
    lines.append(f"| Stage-1-alone (raw threshold on ambiguous cases) | {s1['accuracy']:.4f} | {s1['precision']:.4f} | {s1['recall']:.4f} | {s1['f1']:.4f} |")
    lines.append(f"| Stage-1+2 (fine-tuned Stage-2 verdict) | {s2['accuracy']:.4f} | {s2['precision']:.4f} | {s2['recall']:.4f} | {s2['f1']:.4f} |")
    lines.append("")

    lines.append("### Full test set — Stage-1-only vs Stage-1+2 (round 1)\n")
    lines.append("| | accuracy | precision | recall | F1 |")
    lines.append("|---|---|---|---|---|")
    f1_ = full_comparison["stage1_only"]
    f2_ = full_comparison["stage1_plus_2"]
    lines.append(f"| Stage-1-only | {f1_['accuracy']:.4f} | {f1_['precision']:.4f} | {f1_['recall']:.4f} | {f1_['f1']:.4f} |")
    lines.append(f"| Stage-1+2 (round 1) | {f2_['accuracy']:.4f} | {f2_['precision']:.4f} | {f2_['recall']:.4f} | {f2_['f1']:.4f} |")
    lines.append("")

    lines.append("### Fairness gap re-check — combined pipeline\n")
    lines.append("| subgroup | n | FPR (combined-pipeline final verdict) |")
    lines.append("|---|---|---|")
    lines.append(f"| general_test_human | {len(test_human)} | {general_fpr_combined:.4f} |")
    lines.append(f"| esl_ellipse | {len(esl)} | {esl_fpr_combined:.4f} |")
    lines.append("")
    lines.append(f"**Fairness gap, combined pipeline: {fairness_gap_combined:+.4f}** (ESL FPR minus general-test FPR)\n")
    lines.append(
        f"**Phase 2 baseline (Stage-1 threshold-based) gap was {baseline_gap_threshold:+.4f}.** "
        f"Stage 2 {'narrows' if abs(fairness_gap_combined) < abs(baseline_gap_threshold) else 'widens'} "
        "the gap on the escalated/ambiguous cases it touches — reported as measured, not assumed.\n"
    )

    report = "\n".join(lines) + "\n"
    with RESULTS_PATH.open("a") as f:
        f.write(report)

    print(report)
    print(f"Appended to {RESULTS_PATH}")

    out = {
        "escalated_subset_comparison": esc_comparison,
        "full_pipeline_comparison": full_comparison,
        "fairness_combined": {
            "general_test_human_fpr": general_fpr_combined,
            "esl_fpr": esl_fpr_combined,
            "gap": fairness_gap_combined,
            "baseline_gap_threshold": baseline_gap_threshold,
        },
        "stage2_fallback_counts": {
            "test": {"missing": n_missing_test, "escalated": n_esc_test},
            "esl_eval": {"missing": n_missing_esl, "escalated": n_esc_esl},
        },
    }
    (ROOT / "results" / "phase4_metrics.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
