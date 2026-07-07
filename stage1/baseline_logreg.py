"""Logistic regression baseline on Stage-1 stylometric features."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stage1.features import FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent / "models"
RANDOM_STATE = 42


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}_features.csv")


def train() -> Pipeline:
    train_df = load_split("train")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return pipeline


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = train()
    out_path = MODELS_DIR / "logreg.joblib"
    joblib.dump(pipeline, out_path)
    print(f"Trained logistic regression baseline, saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
