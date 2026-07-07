"""Shared metric helpers used across evaluation phases."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def escalation_rate(bands: np.ndarray) -> float:
    return float((bands == "escalate").mean())


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """P(predicted positive | true negative). Assumes label 0 = human, 1 = AI."""
    negatives = y_true == 0
    if negatives.sum() == 0:
        return float("nan")
    return float(y_pred[negatives].mean())


def auto_flag_rate(bands: np.ndarray, y_true: np.ndarray, positive_true_label: int = 0) -> float:
    """Fraction of true-label==positive_true_label rows landing in auto_flag.

    For the fairness check this is the decision-relevant "false accusation
    without any human review" rate, as distinct from the raw 0.5-threshold
    false positive rate (which includes rows that would actually be
    escalated to human review rather than auto-flagged).
    """
    mask = y_true == positive_true_label
    if mask.sum() == 0:
        return float("nan")
    return float((bands[mask] == "auto_flag").mean())
