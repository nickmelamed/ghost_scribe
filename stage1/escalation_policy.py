"""Calibrate and apply Stage-1 escalation bands.

XGBoost is used as the Stage-1 decision model (it beat logistic regression
on validation AUC — see results/metrics_comparison.md), scored against
three bands:
  - confident-human: auto-clear, no further action
  - confident-AI:    auto-flag for teacher review (NOT an automated
                      accusation — see CLAUDE.md; a human must confirm)
  - ambiguous:        escalate to Stage 2

Bands are calibrated on the validation set by taking the region of
predicted probability closest to the decision boundary (|p - 0.5|
smallest) that covers a target fraction of validation examples, then
fixing those probability thresholds for use at inference time. This
directly controls the escalation rate rather than hoping a fixed
probability cutoff like [0.4, 0.6] happens to land in range.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stage1.features import FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent / "models"
THRESHOLDS_PATH = MODELS_DIR / "escalation_thresholds.json"

TARGET_ESCALATION_RATE = 0.20  # midpoint of the 15-25% target band


def calibrate_bands(y_proba_val: np.ndarray, target_rate: float = TARGET_ESCALATION_RATE) -> tuple[float, float]:
    abs_dev = np.abs(y_proba_val - 0.5)
    d = float(np.percentile(abs_dev, target_rate * 100))
    t_low, t_high = 0.5 - d, 0.5 + d
    return t_low, t_high


def assign_band(y_proba: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    bands = np.where(
        y_proba < t_low,
        "auto_clear",
        np.where(y_proba > t_high, "auto_flag", "escalate"),
    )
    return bands


def main() -> int:
    val = pd.read_csv(PROCESSED_DIR / "val_features.csv")
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")

    val_proba = xgb.predict_proba(val[FEATURE_COLUMNS])[:, 1]
    t_low, t_high = calibrate_bands(val_proba)

    val_bands = assign_band(val_proba, t_low, t_high)
    achieved_rate = float((val_bands == "escalate").mean())

    thresholds = {
        "model": "xgb",
        "t_low": t_low,
        "t_high": t_high,
        "target_escalation_rate": TARGET_ESCALATION_RATE,
        "achieved_escalation_rate_val": achieved_rate,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))

    print(json.dumps(thresholds, indent=2))
    print(f"Wrote thresholds to {THRESHOLDS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
