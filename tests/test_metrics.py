import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from evaluation.metrics import auto_flag_rate, classification_metrics, escalation_rate, false_positive_rate

binary = st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=60)


def test_false_positive_rate_counts_predicted_ai_among_humans():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 0, 1, 0])
    assert false_positive_rate(y_true, y_pred) == 0.25


def test_false_positive_rate_is_nan_without_humans():
    assert math.isnan(false_positive_rate(np.array([1, 1]), np.array([1, 0])))


@given(binary, binary)
def test_false_positive_rate_is_a_fraction(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    yt, yp = np.array(y_true[:n]), np.array(y_pred[:n])
    rate = false_positive_rate(yt, yp)
    if (yt == 0).any():
        assert 0.0 <= rate <= 1.0
        assert rate == yp[yt == 0].mean()
    else:
        assert math.isnan(rate)


def test_escalation_rate_counts_only_escalate():
    bands = np.array(["auto_clear", "escalate", "auto_flag", "escalate"])
    assert escalation_rate(bands) == 0.5


@given(st.lists(st.sampled_from(["auto_clear", "auto_flag", "escalate"]), min_size=1, max_size=50))
def test_escalation_rate_is_a_fraction(bands):
    assert 0.0 <= escalation_rate(np.array(bands)) <= 1.0


def test_auto_flag_rate_defaults_to_true_humans():
    bands = np.array(["auto_flag", "auto_clear", "escalate", "auto_flag"])
    y_true = np.array([0, 0, 0, 1])
    assert auto_flag_rate(bands, y_true) == 1 / 3


def test_auto_flag_rate_can_target_the_ai_class():
    bands = np.array(["auto_flag", "auto_clear", "auto_flag"])
    y_true = np.array([1, 1, 0])
    assert auto_flag_rate(bands, y_true, positive_true_label=1) == 0.5


def test_auto_flag_rate_is_nan_when_class_absent():
    assert math.isnan(auto_flag_rate(np.array(["auto_flag"]), np.array([1])))


@settings(deadline=None)
@given(st.lists(st.sampled_from(["auto_clear", "auto_flag", "escalate"]), min_size=1, max_size=40), st.data())
def test_auto_flag_rate_never_exceeds_one(bands, data):
    y = np.array(data.draw(st.lists(st.integers(0, 1), min_size=len(bands), max_size=len(bands))))
    rate = auto_flag_rate(np.array(bands), y)
    assert math.isnan(rate) or 0.0 <= rate <= 1.0


def test_classification_metrics_perfect_scores():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == {"auc_roc": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_classification_metrics_hand_worked_example():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.4, 0.6, 0.3, 0.2])
    m = classification_metrics(y_true, y_pred, proba)
    assert m["precision"] == 2 / 3
    assert m["recall"] == 2 / 3
    assert math.isclose(m["f1"], 2 / 3)
    assert math.isclose(m["auc_roc"], 8 / 9)


def test_classification_metrics_zero_division_gives_zero():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])
    m = classification_metrics(y_true, y_pred, np.array([0.1, 0.2, 0.3]))
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
