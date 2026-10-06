import json

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stage2 import build_review_queue as brq


def write_split(directory, split, verdicts, confidences, true_labels=None):
    n = len(verdicts)
    preds = pd.DataFrame(
        {
            "text": [f"{split} essay {i}" for i in range(n)],
            "true_label": true_labels if true_labels is not None else [0] * n,
            "stage2_pred_label": verdicts,
            "stage2_rationale": ["r"] * n,
        }
    )
    conf = pd.DataFrame({"text": preds["text"], "confidence": confidences})
    preds.to_csv(directory / f"stage2_predictions_{split}.csv", index=False)
    conf.to_csv(directory / f"stage2_confidence_{split}.csv", index=False)


def run_queue(monkeypatch, tmp_path, verdicts, confidences, true_labels=None):
    write_split(tmp_path, "val", verdicts, confidences, true_labels)
    monkeypatch.setattr(brq, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(brq, "GATE_REPORT_PATH", tmp_path / "review_gate.json")
    assert brq.main() == 0
    return pd.read_csv(tmp_path / "stage2_review_queue_val.csv")


def labeled_frame(verdicts, confidences, true_labels):
    return pd.DataFrame(
        {
            "text": [f"essay {i}" for i in range(len(verdicts))],
            "stage2_pred_label": verdicts,
            "confidence": confidences,
            "true_label": true_labels,
        }
    )


def test_gate_margin_is_five_points():
    assert brq.GATE_MARGIN == 0.05


def test_calibrate_gate_rate_hand_worked():
    # Ten human verdicts. The AI-labeled one is third least confident, so the
    # needed rate is 3/10. The AI verdict row is ignored.
    confidences = [0.99, 0.60, 0.98, 0.70, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]
    true_labels = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    df = labeled_frame([0] * 10, confidences, true_labels)
    ai_row = labeled_frame([1], [0.5], [1])
    needed, rate = brq.calibrate_gate_rate(pd.concat([df, ai_row], ignore_index=True))
    assert needed == 0.2
    assert rate == pytest.approx(0.25)


def test_calibrate_gate_rate_without_dangerous_errors_is_just_the_margin():
    df = labeled_frame([0] * 5, [0.9, 0.8, 0.7, 0.6, 0.5], [0] * 5)
    assert brq.calibrate_gate_rate(df) == (0.0, brq.GATE_MARGIN)


def test_calibrate_gate_rate_is_capped_at_one():
    df = labeled_frame([0] * 4, [0.5, 0.6, 0.7, 0.99], [0, 0, 0, 1])
    assert brq.calibrate_gate_rate(df) == (1.0, 1.0)


def test_a_confident_wrong_verdict_forces_a_large_rate():
    confidences = [0.5 + 0.01 * i for i in range(20)]
    true_labels = [0] * 19 + [1]
    needed, _ = brq.calibrate_gate_rate(labeled_frame([0] * 20, confidences, true_labels))
    assert needed == 1.0


def test_gate_report_counts_queue_and_missed_dangerous_errors():
    confidences = [0.99, 0.60, 0.98, 0.70, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92]
    true_labels = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    df = pd.concat([labeled_frame([0] * 10, confidences, true_labels), labeled_frame([1, 1], [0.9, 0.9], [1, 1])])
    report = brq.gate_report(df, 0.3)
    assert report == {
        "n_escalated": 12,
        "n_human_verdicts": 10,
        "n_gated_human_verdicts": 3,
        "queue_size": 5,
        "n_dangerous_errors": 2,
        "n_dangerous_errors_queued": 1,
    }


def test_every_ai_verdict_is_queued_regardless_of_confidence(monkeypatch, tmp_path):
    verdicts = [1, 1, 1, 0, 0, 0, 0, 0]
    confidences = [0.999, 0.99, 0.51, 0.9, 0.9, 0.9, 0.9, 0.9]
    queue = run_queue(monkeypatch, tmp_path, verdicts, confidences)
    assert {"val essay 0", "val essay 1", "val essay 2"} <= set(queue["text"])


def test_gated_human_verdicts_are_the_least_confident_up_to_the_calibrated_rate(monkeypatch, tmp_path):
    # Twenty human verdicts. The AI-labeled one is third least confident, so the
    # needed rate is 0.15 and the gate rate is 0.20, which queues four.
    confidences = [0.50 + 0.01 * i for i in range(20)]
    true_labels = [0] * 20
    true_labels[2] = 1
    queue = run_queue(monkeypatch, tmp_path, [0] * 20, confidences, true_labels)
    assert set(queue["text"]) == {"val essay 0", "val essay 1", "val essay 2", "val essay 3"}


def test_gate_rounds_down_so_a_small_human_pool_queues_none(monkeypatch, tmp_path):
    queue = run_queue(monkeypatch, tmp_path, [0, 0, 0, 0], [0.5, 0.6, 0.7, 0.8])
    assert len(queue) == 0


def test_queue_drops_the_confidence_column_and_keeps_the_prediction_schema(monkeypatch, tmp_path):
    queue = run_queue(monkeypatch, tmp_path, [1, 0], [0.9, 0.9])
    assert "confidence" not in queue.columns
    assert queue.columns.tolist() == ["text", "true_label", "stage2_pred_label", "stage2_rationale"]


def test_rate_is_set_on_val_and_test_only_gets_a_check(monkeypatch, tmp_path):
    val_conf = [0.50 + 0.01 * i for i in range(20)]
    val_true = [0] * 20
    val_true[2] = 1
    # On test the only dangerous error is the most confident verdict, so a
    # gate tuned on test would need a rate of 1.0.
    write_split(tmp_path, "test", [0] * 10, [0.5 + 0.01 * i for i in range(10)], [0] * 9 + [1])
    run_queue(monkeypatch, tmp_path, [0] * 20, val_conf, val_true)
    report = json.loads((tmp_path / "review_gate.json").read_text())
    assert report["calibrated_on"] == "val"
    assert report["needed_rate"] == 0.15
    assert report["gate_rate"] == pytest.approx(0.20)
    assert report["val"]["n_dangerous_errors_queued"] == 1
    assert report["test_check"]["n_human_verdicts"] == 10
    assert report["test_check"]["n_gated_human_verdicts"] == 2
    assert report["test_check"]["n_dangerous_errors"] == 1
    assert report["test_check"]["n_dangerous_errors_queued"] == 0


def test_report_has_no_test_check_when_test_files_are_missing(monkeypatch, tmp_path):
    run_queue(monkeypatch, tmp_path, [0, 1], [0.9, 0.9])
    assert json.loads((tmp_path / "review_gate.json").read_text())["test_check"] is None


@settings(deadline=None, max_examples=40)
@given(
    st.lists(
        st.tuples(
            st.integers(0, 1),
            st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
            st.integers(0, 1),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_no_ai_verdict_and_no_calibration_error_is_ever_skipped(tmp_path_factory, rows):
    tmp = tmp_path_factory.mktemp("queue")
    verdicts = [v for v, _, _ in rows]
    confidences = [c for _, c, _ in rows]
    true_labels = [t for _, _, t in rows]
    with pytest.MonkeyPatch.context() as mp:
        queue = run_queue(mp, tmp, verdicts, confidences, true_labels)
        report = json.loads((tmp / "review_gate.json").read_text())
    queued = set(queue["text"])
    for i, v in enumerate(verdicts):
        if v == 1:
            assert f"val essay {i}" in queued
    assert report["val"]["n_dangerous_errors_queued"] == report["val"]["n_dangerous_errors"]
    assert len(queue) == report["val"]["queue_size"]
