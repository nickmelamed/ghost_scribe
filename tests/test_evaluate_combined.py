import math

import numpy as np
import pandas as pd

from evaluation import evaluate_combined as ec
from stage1.features import FEATURE_COLUMNS


class StubXGB:
    """Scores each row by its word_count divided by 100."""

    def predict_proba(self, X):
        p = X["word_count"].to_numpy() / 100.0
        return np.column_stack([1 - p, p])


def features_frame(word_counts, labels):
    df = pd.DataFrame({c: 0.0 for c in FEATURE_COLUMNS}, index=range(len(word_counts)))
    df["word_count"] = word_counts
    df["label"] = labels
    df["text"] = [f"essay {i}" for i in range(len(word_counts))]
    return df


def test_score_and_band_adds_proba_pred_and_band_without_touching_input():
    df = features_frame([10, 50, 90], [0, 1, 1])
    out = ec.score_and_band(df, StubXGB(), 0.3, 0.7)
    assert out["stage1_proba"].tolist() == [0.1, 0.5, 0.9]
    assert out["stage1_pred"].tolist() == [0, 1, 1]
    assert out["stage1_band"].tolist() == ["auto_clear", "escalate", "auto_flag"]
    assert "stage1_band" not in df.columns


def test_pred_threshold_is_inclusive_at_half():
    out = ec.score_and_band(features_frame([50], [1]), StubXGB(), 0.3, 0.7)
    assert out["stage1_pred"].tolist() == [1]


def test_attach_stage2_uses_stage2_on_escalated_rows_only(tmp_path):
    scored = ec.score_and_band(features_frame([10, 50, 55, 90], [0, 0, 1, 1]), StubXGB(), 0.3, 0.7)
    path = tmp_path / "s2.csv"
    pd.DataFrame(
        {"text": ["essay 0", "essay 1", "essay 2", "essay 3"], "stage2_pred_label": [1, 0, 1, 0]}
    ).to_csv(path, index=False)
    merged, n_missing, n_escalated = ec.attach_stage2(scored, path)
    assert (n_missing, n_escalated) == (0, 2)
    assert merged["final_pred"].tolist() == [0, 0, 1, 1]


def test_attach_stage2_falls_back_to_stage1_when_stage2_missing(tmp_path):
    scored = ec.score_and_band(features_frame([10, 50, 60], [0, 0, 1]), StubXGB(), 0.3, 0.7)
    path = tmp_path / "s2.csv"
    pd.DataFrame({"text": ["essay 1", "essay 2"], "stage2_pred_label": [0, float("nan")]}).to_csv(path, index=False)
    merged, n_missing, n_escalated = ec.attach_stage2(scored, path)
    assert (n_missing, n_escalated) == (1, 2)
    assert merged["final_pred"].tolist() == [0, 0, 1]
    assert merged["final_pred"].dtype == int


def test_escalated_subset_comparison_hand_worked():
    df = pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 1],
            "stage1_band": ["escalate", "escalate", "escalate", "escalate", "auto_flag"],
            "stage1_pred": [1, 0, 1, 0, 1],
            "stage1_proba": [0.6, 0.4, 0.55, 0.3, 0.99],
            "final_pred": [1, 1, 0, 0, 1],
        }
    )
    out = ec.escalated_subset_comparison(df)
    assert out["n"] == 4
    assert out["stage1_alone"]["accuracy"] == 0.5
    assert out["stage1_plus_2"]["accuracy"] == 1.0
    assert out["stage1_plus_2"]["precision"] == 1.0
    assert out["stage1_plus_2"]["recall"] == 1.0
    assert out["stage1_plus_2"]["f1"] == 1.0
    assert math.isnan(out["stage1_plus_2"]["auc_roc"])


def test_full_pipeline_comparison_reports_nan_auc_for_combined_decision():
    df = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "stage1_pred": [1, 0, 1, 0],
            "stage1_proba": [0.9, 0.4, 0.6, 0.1],
            "final_pred": [1, 1, 0, 0],
        }
    )
    out = ec.full_pipeline_comparison(df)
    assert out["stage1_only"]["accuracy"] == 0.5
    assert out["stage1_plus_2"]["accuracy"] == 1.0
    assert math.isnan(out["stage1_plus_2"]["auc_roc"])
    assert 0.0 <= out["stage1_only"]["auc_roc"] <= 1.0


def test_combined_fpr_is_mean_final_prediction_on_humans():
    assert ec.combined_fpr(pd.DataFrame({"final_pred": [1, 0, 0, 0]})) == 0.25
