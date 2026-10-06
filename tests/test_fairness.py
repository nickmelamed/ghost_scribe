import json

import pandas as pd

from fairness import esl_eval
from stage1.features import FEATURE_COLUMNS
from tests.helpers import StubXGB


def frame(word_counts, labels):
    df = pd.DataFrame({c: 0.0 for c in FEATURE_COLUMNS}, index=range(len(word_counts)))
    df["word_count"] = word_counts
    df["label"] = labels
    return df


def setup(monkeypatch, tmp_path, test_df, esl_df):
    test_df.to_csv(tmp_path / "test_features.csv", index=False)
    esl_df.to_csv(tmp_path / "esl_eval_features.csv", index=False)
    models = tmp_path / "models"
    models.mkdir()
    (models / "escalation_thresholds.json").write_text(json.dumps({"t_low": 0.3, "t_high": 0.7}))
    monkeypatch.setattr(esl_eval, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(esl_eval, "MODELS_DIR", models)
    monkeypatch.setattr(esl_eval.joblib, "load", lambda path: StubXGB())


def test_gap_is_esl_rate_minus_general_human_rate(monkeypatch, tmp_path):
    test_df = frame([10, 20, 80, 90, 95], [0, 0, 0, 0, 1])
    esl_df = frame([10, 60, 80, 90], [0, 0, 0, 0])
    setup(monkeypatch, tmp_path, test_df, esl_df)
    r = esl_eval.compute_fairness_metrics()
    assert r["general_test_human"]["n"] == 4
    assert r["general_test_human"]["fpr_threshold"] == 0.5
    assert r["general_test_human"]["fpr_auto_flag"] == 0.5
    assert r["general_test_human"]["escalate_rate"] == 0.0
    assert r["esl_ellipse"]["n"] == 4
    assert r["esl_ellipse"]["fpr_threshold"] == 0.75
    assert r["esl_ellipse"]["fpr_auto_flag"] == 0.5
    assert r["esl_ellipse"]["escalate_rate"] == 0.25
    assert r["fairness_gap_threshold"] == 0.25
    assert r["fairness_gap_auto_flag"] == 0.0


def test_general_set_excludes_ai_rows(monkeypatch, tmp_path):
    test_df = frame([95, 95, 10], [1, 1, 0])
    esl_df = frame([10], [0])
    setup(monkeypatch, tmp_path, test_df, esl_df)
    r = esl_eval.compute_fairness_metrics()
    assert r["general_test_human"]["n"] == 1
    assert r["general_test_human"]["fpr_threshold"] == 0.0


def test_negative_gap_is_reported_as_negative(monkeypatch, tmp_path):
    test_df = frame([90, 95], [0, 0])
    esl_df = frame([10, 10], [0, 0])
    setup(monkeypatch, tmp_path, test_df, esl_df)
    r = esl_eval.compute_fairness_metrics()
    assert r["fairness_gap_threshold"] == -1.0
