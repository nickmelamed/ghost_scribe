import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from evaluation import round2 as r2
from stage1.features import FEATURE_COLUMNS
from tests.helpers import StubXGB


def test_new_source_split_is_70_15_15_disjoint_and_seeded():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(100)]})
    train, val, test = r2.split_new_source(df)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert not set(train["text"]) & set(val["text"]) | set(train["text"]) & set(test["text"]) | set(val["text"]) & set(
        test["text"]
    )
    again = r2.split_new_source(df)
    assert train["text"].tolist() == again[0]["text"].tolist()


def test_source_choices_follow_the_agreed_design():
    assert (r2.NEW_SOURCE, r2.UNSEEN_SOURCE) == ("llama_70b_v1", "falcon_180b_v1")
    assert (r2.STAGE1_UNITS, r2.STAGE2_UNITS, r2.REVIEW_UNITS) == (1, 5, 25)


def test_hitl_rows_take_the_reviewer_label_and_keep_the_truth():
    val = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 1]})
    corrections = pd.DataFrame({"text": ["a", "b", "a"], "label": [1, 1, 0]})
    reviewed = r2.hitl_training_rows(val, corrections)
    assert reviewed["text"].tolist() == ["a", "b"]
    assert reviewed["label"].tolist() == [0, 1]
    assert reviewed["true_label"].tolist() == [0, 1]


def test_hitl_label_differs_from_truth_when_the_reviewer_overturns():
    val = pd.DataFrame({"text": ["a"], "label": [0]})
    reviewed = r2.hitl_training_rows(val, pd.DataFrame({"text": ["a"], "label": [1]}))
    assert reviewed["label"].tolist() == [1]
    assert reviewed["true_label"].tolist() == [0]


def test_hitl_rows_must_come_from_val():
    with pytest.raises(ValueError):
        r2.hitl_training_rows(pd.DataFrame({"text": ["a"], "label": [0]}), pd.DataFrame({"text": ["zzz"], "label": [0]}))


def test_drop_near_duplicates_removes_copies_and_counts_them():
    reference = [" ".join(f"w{i}" for i in range(20))]
    copy = reference[0]
    frames = {"g1": pd.DataFrame({"text": [copy, "something else entirely different here today now"]}), "g2": pd.DataFrame({"text": []})}
    kept, removed = r2.drop_near_duplicates(frames, reference)
    assert removed == {"g1": 1, "g2": 0}
    assert len(kept["g1"]) == 1
    assert len(kept["g2"]) == 0


def test_cost_per_decision_hand_worked():
    cost = r2.cost_per_decision(0.2, 0.1)
    assert cost["stage1"] == 1
    assert cost["stage2"] == pytest.approx(1.0)
    assert cost["review"] == pytest.approx(2.5)
    assert cost["total"] == pytest.approx(4.5)
    assert cost["stage1_only_total"] == 1


def scored_group(labels, bands, preds):
    return pd.DataFrame(
        {
            "text": [f"e{i}" for i in range(len(labels))],
            "label": labels,
            "stage1_band": bands,
            "stage1_pred": preds,
            "stage1_proba": [0.5] * len(labels),
            "prompt_name": "p",
        }
    )


def test_ai_only_metrics_hand_worked(tmp_path):
    scored = scored_group([1, 1, 1, 1], ["auto_clear", "escalate", "escalate", "auto_flag"], [0, 1, 0, 1])
    path = tmp_path / "s2.csv"
    pd.DataFrame({"text": ["e1", "e2"], "stage2_pred_label": [1, 0]}).to_csv(path, index=False)
    m = r2.ai_only_metrics(scored, path)
    assert m["n_rows"] == 4
    assert m["band_share"] == {"auto_clear": 0.25, "escalate": 0.5, "auto_flag": 0.25}
    assert m["stage1_recall"] == 0.5
    assert m["combined_recall"] == 0.5
    assert m["missed_after_stage2"] == 2
    assert m["cleared_by_stage1"] == 1
    assert m["stage2_fallbacks"] == 0


def test_human_only_metrics_hand_worked(tmp_path):
    scored = scored_group([0, 0, 0, 0], ["auto_clear", "escalate", "escalate", "auto_flag"], [0, 1, 0, 1])
    path = tmp_path / "s2.csv"
    pd.DataFrame({"text": ["e1", "e2"], "stage2_pred_label": [0, 0]}).to_csv(path, index=False)
    m = r2.human_only_metrics(scored, path)
    assert m["fpr_threshold"] == 0.5
    assert m["fpr_auto_flag"] == 0.25
    assert m["fpr_combined"] == 0.25
    assert m["escalate_rate"] == 0.5


def write_round1_results(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "phase2_metrics.json").write_text(
        json.dumps(
            {
                "escalation": {"test_escalation_rate": 0.2},
                "fairness": {"fairness_gap_threshold": 0.02, "fairness_gap_auto_flag": 0.001},
            }
        )
    )
    (directory / "phase4_metrics.json").write_text(
        json.dumps(
            {
                "escalated_subset_comparison": {"n": 200, "stage1_plus_2": {"accuracy": 0.99}},
                "full_pipeline_comparison": {
                    "stage1_only": {"accuracy": 0.96},
                    "stage1_plus_2": {"accuracy": 0.995},
                },
                "fairness_combined": {"gap": 0.001},
            }
        )
    )
    (directory / "review_gate.json").write_text(json.dumps({"test_check": {"queue_size": 150}}))


def test_round1_summary_reads_the_saved_results(tmp_path):
    write_round1_results(tmp_path)
    s = r2.round1_summary(tmp_path)
    assert s["escalation_rate"] == 0.2
    assert s["review_rate"] == pytest.approx(150 / 1000)
    assert s["gap_stage1"] == 0.02
    assert s["gap_combined"] == 0.001
    assert s["stage1_only_accuracy"] == 0.96
    assert s["escalated_stage2_accuracy"] == 0.99
    assert s["cost"]["total"] == pytest.approx(1 + 5 * 0.2 + 25 * 0.15)


def test_markdown_reports_both_rounds_without_dashes_or_semicolons(tmp_path):
    write_round1_results(tmp_path)
    round2 = {
        "escalation_rate": 0.18,
        "review_rate": 0.1,
        "gap_stage1": 0.015,
        "gap_combined": 0.0,
        "stage1_only_accuracy": 0.965,
        "stage1_plus_2_accuracy": 0.996,
        "escalated_stage2_accuracy": 0.992,
        "gate_rate": 0.4,
        "gate_val": {"needed_rate": 0.35},
        "gate_test_check": {"n_dangerous_errors_queued": 3, "n_dangerous_errors": 3},
        "cost": r2.cost_per_decision(0.18, 0.1),
    }
    group = {
        "n_rows": 10,
        "band_share": {"auto_clear": 0.1, "escalate": 0.4, "auto_flag": 0.5},
        "missed_after_stage2": 1,
    }
    results = {
        "meta": {"new_source": "llama_70b_v1", "unseen_source": "falcon_180b_v1", "hitl_reviewed_essays": 7},
        "round1": r2.round1_summary(tmp_path),
        "round2": round2,
        "test_new_source": group,
        "unseen_source": group,
    }
    text = r2.render_markdown(results)
    assert "| Escalation rate, test | 0.2000 | 0.1800 |" in text
    assert "Cost per essay, full flywheel" in text
    assert "test_new_source, round 1 | n/a" in text
    assert "—" not in text
    assert ";" not in text


class StubR2Model:
    """Score rows as a sigmoid of char_count."""

    feature_importances_ = np.arange(len(FEATURE_COLUMNS), dtype=float)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X["char_count"].to_numpy()))
        return np.column_stack([1 - p, p])


def synthetic_split(prefix, n, source_labels, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        source, label = source_labels[i % len(source_labels)]
        row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
        row["char_count"] += 1.5 * label
        row.update(
            {
                "text": f"{prefix} {source} {i}",
                "label": label,
                "generation_source": source,
                "prompt_name": "p",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_synthetic_project(tmp_path):
    processed, models = tmp_path / "processed", tmp_path / "models"
    processed.mkdir()
    models.mkdir()
    seen = [("human", 0), ("gpt", 1), ("human", 0)]
    synthetic_split("train", 240, seen, 1).to_csv(processed / "train_features.csv", index=False)
    val = synthetic_split("val", 90, seen, 2)
    val.to_csv(processed / "val_features.csv", index=False)
    synthetic_split("test", 90, seen, 3).to_csv(processed / "test_features.csv", index=False)
    held = pd.concat(
        [synthetic_split("ho", 100, [("llama_70b_v1", 1)], 4), synthetic_split("ho2", 50, [("falcon_180b_v1", 1)], 5)]
    )
    held.to_csv(processed / "held_out_features.csv", index=False)
    synthetic_split("esl", 40, [("human_esl", 0)], 6).to_csv(processed / "esl_eval_features.csv", index=False)
    joblib.dump(StubXGB(), models / "xgb.joblib")
    (models / "escalation_thresholds.json").write_text(json.dumps({"t_low": 0.3, "t_high": 0.7}))
    reviewed = val.iloc[:10][["text", "label", "generation_source", "prompt_name"]].copy()
    reviewed["label"] = [1 - x if i < 2 else x for i, x in enumerate(reviewed["label"])]
    reviewed.to_csv(processed / "hitl_corrections_human.csv", index=False)
    return processed, models


def test_prepare_needs_the_hitl_corrections_unless_told_otherwise(tmp_path, monkeypatch):
    monkeypatch.setattr(r2, "fit", lambda df: StubR2Model())
    processed, models = build_synthetic_project(tmp_path)
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        r2.prepare(tmp_path / "exp", processed, models, missing)
    meta = r2.prepare(tmp_path / "exp2", processed, models, missing, allow_missing_hitl=True)
    assert meta["hitl_reviewed_essays"] == 0


def test_prepare_builds_round_two_data_without_leaking_eval_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(r2, "fit", lambda df: StubR2Model())
    processed, models = build_synthetic_project(tmp_path)
    exp = tmp_path / "exp"
    meta = r2.prepare(exp, processed, models, processed / "hitl_corrections_human.csv")
    assert meta["hitl_reviewed_essays"] == 10
    assert meta["hitl_label_differs_from_truth"] == 2
    assert meta["rows"]["new_source_train"] == 70
    assert meta["rows"]["val_r2"] == 80 + 15
    train_texts = {json.loads(line)["text"] for line in (exp / "stage2_data" / "train.jsonl").open()}
    unseen = pd.read_csv(exp / "groups" / "unseen_source.csv")
    new_test = pd.read_csv(exp / "groups" / "test_new_source.csv")
    val_r2 = pd.read_csv(exp / "groups" / "val_r2.csv")
    assert len(unseen) == 50
    assert not train_texts & set(unseen["text"])
    assert not train_texts & set(new_test["text"])
    assert not train_texts & set(val_r2["text"])
    reviewed = set(pd.read_csv(processed / "hitl_corrections_human.csv")["text"])
    assert not reviewed & set(val_r2["text"])
    for name in r2.EVAL_GROUPS + ("val_r2",):
        assert (exp / "stage2_data" / f"{name}.jsonl").exists()
        assert (exp / "groups" / f"{name}.csv").exists()
    for name in ("test_new_source", "unseen_source"):
        assert (exp / "r1_stage2_data" / f"{name}.jsonl").exists()
    assert (exp / "xgb.joblib").exists()
    thresholds = json.loads((exp / "escalation_thresholds.json").read_text())
    assert (thresholds["t_low"], thresholds["t_high"]) == (meta["t_low"], meta["t_high"])
    saved = json.loads((exp / "meta.json").read_text())
    assert saved["new_source"] == "llama_70b_v1"
    assert not math.isnan(saved["t_low"])


def test_prepare_drops_esl_rows_that_copy_reviewed_essays(tmp_path, monkeypatch):
    monkeypatch.setattr(r2, "fit", lambda df: StubR2Model())
    processed, models = build_synthetic_project(tmp_path)
    reviewed = pd.read_csv(processed / "hitl_corrections_human.csv")
    esl = pd.read_csv(processed / "esl_eval_features.csv")
    esl.loc[0, "text"] = reviewed.loc[0, "text"]
    esl.to_csv(processed / "esl_eval_features.csv", index=False)
    meta = r2.prepare(tmp_path / "exp", processed, models, processed / "hitl_corrections_human.csv")
    assert meta["groups"]["esl_eval"]["removed_as_near_duplicates"] == 1
    assert meta["groups"]["esl_eval"]["rows"] == 39
