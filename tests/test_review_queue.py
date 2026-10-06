import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from stage2 import build_review_queue as brq


def run_queue(monkeypatch, tmp_path, verdicts, confidences):
    n = len(verdicts)
    preds = pd.DataFrame(
        {
            "text": [f"essay {i}" for i in range(n)],
            "true_label": [0] * n,
            "stage2_pred_label": verdicts,
            "stage2_rationale": ["r"] * n,
        }
    )
    conf = pd.DataFrame({"text": preds["text"], "confidence": confidences})
    preds.to_csv(tmp_path / "stage2_predictions_val.csv", index=False)
    conf.to_csv(tmp_path / "stage2_confidence_val.csv", index=False)
    monkeypatch.setattr(brq, "PROCESSED_DIR", tmp_path)
    assert brq.main() == 0
    return pd.read_csv(tmp_path / "stage2_review_queue_val.csv")


def test_gate_rate_is_twenty_percent():
    assert brq.TARGET_HUMAN_GATE_RATE == 0.20


def test_every_ai_verdict_is_queued_regardless_of_confidence(monkeypatch, tmp_path):
    verdicts = [1, 1, 1, 0, 0, 0, 0, 0]
    confidences = [0.999, 0.99, 0.51, 0.9, 0.9, 0.9, 0.9, 0.9]
    queue = run_queue(monkeypatch, tmp_path, verdicts, confidences)
    ai_texts = {"essay 0", "essay 1", "essay 2"}
    assert ai_texts <= set(queue["text"])


def test_only_the_least_confident_human_verdicts_are_queued(monkeypatch, tmp_path):
    verdicts = [0] * 10
    confidences = [0.99, 0.60, 0.98, 0.97, 0.55, 0.96, 0.95, 0.94, 0.93, 0.92]
    queue = run_queue(monkeypatch, tmp_path, verdicts, confidences)
    assert set(queue["text"]) == {"essay 1", "essay 4"}


def test_gate_rounds_down_so_a_small_human_pool_queues_none(monkeypatch, tmp_path):
    queue = run_queue(monkeypatch, tmp_path, [0, 0, 0, 0], [0.5, 0.6, 0.7, 0.8])
    assert len(queue) == 0


def test_queue_drops_the_confidence_column_and_keeps_the_prediction_schema(monkeypatch, tmp_path):
    queue = run_queue(monkeypatch, tmp_path, [1, 0], [0.9, 0.9])
    assert "confidence" not in queue.columns
    assert queue.columns.tolist() == ["text", "true_label", "stage2_pred_label", "stage2_rationale"]


@settings(deadline=None, max_examples=40)
@given(
    st.lists(
        st.tuples(st.integers(0, 1), st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        min_size=1,
        max_size=40,
    )
)
def test_no_ai_verdict_is_ever_skipped(tmp_path_factory, rows):
    tmp = tmp_path_factory.mktemp("queue")
    verdicts = [v for v, _ in rows]
    confidences = [c for _, c in rows]
    with pytest.MonkeyPatch.context() as mp:
        queue = run_queue(mp, tmp, verdicts, confidences)
    queued = set(queue["text"])
    for i, v in enumerate(verdicts):
        if v == 1:
            assert f"essay {i}" in queued
    n_human = verdicts.count(0)
    n_ai = verdicts.count(1)
    assert len(queue) == n_ai + int(n_human * brq.TARGET_HUMAN_GATE_RATE)
