"""XGBoost baseline on Stage-1 stylometric features."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from stage1.features import FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent / "models"
RANDOM_STATE = 42


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}_features.csv")


def train() -> XGBClassifier:
    train_df = load_split("train")
    n_pos = int((train_df["label"] == 1).sum())
    n_neg = int((train_df["label"] == 0).sum())
    scale_pos_weight = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return clf


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf = train()
    out_path = MODELS_DIR / "xgb.joblib"
    joblib.dump(clf, out_path)
    print(f"Trained XGBoost baseline, saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
