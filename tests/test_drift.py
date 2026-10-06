import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from drift import monitor
from stage1.features import FEATURE_COLUMNS

arrays = st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=20, max_size=200)


def test_thresholds_are_the_conventional_ones():
    assert (monitor.PSI_MODERATE, monitor.PSI_MAJOR) == (0.1, 0.2)
    assert monitor.HELD_OUT_SOURCES == ("falcon_180b_v1", "llama_70b_v1")


def test_level_boundaries():
    assert [monitor.level(x) for x in (0.0, 0.1, 0.1001, 0.2, 0.2001)] == [
        "none",
        "none",
        "moderate",
        "moderate",
        "major",
    ]


def test_psi_of_a_sample_with_the_reference_distribution_is_zero():
    ref = np.arange(100, dtype=float)
    assert monitor.psi(ref, ref) == pytest.approx(0.0, abs=1e-12)


def test_psi_hand_worked_two_bins():
    # Quantile edges split the reference at its median into two equal bins.
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    sample = np.array([0.0, 3.0, 3.0, 3.0])
    expected = (0.25 - 0.5) * math.log(0.25 / 0.5) + (0.75 - 0.5) * math.log(0.75 / 0.5)
    assert monitor.psi(ref, sample, n_bins=2) == pytest.approx(expected)


def test_psi_grows_as_the_sample_shifts_away():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=2000)
    values = [monitor.psi(ref, rng.normal(loc=shift, size=2000)) for shift in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values)
    assert values[0] < monitor.PSI_MODERATE
    assert values[-1] > monitor.PSI_MAJOR


def test_psi_of_a_constant_reference_is_zero():
    assert monitor.psi(np.zeros(50), np.arange(50, dtype=float)) == 0.0


@settings(deadline=None, max_examples=40)
@given(arrays, arrays)
def test_psi_is_never_negative_or_nan(reference, sample):
    value = monitor.psi(np.array(reference), np.array(sample))
    assert value >= 0.0
    assert not math.isnan(value)


def test_null_psi_shrinks_as_the_batch_grows_and_is_reproducible():
    ref = np.random.default_rng(1).normal(size=3000)
    small, large = monitor.null_psi_p95(ref, 30, reps=60), monitor.null_psi_p95(ref, 1500, reps=60)
    assert small > large
    assert monitor.null_psi_p95(ref, 30, reps=60) == small


class StubModel:
    """Score rows as a sigmoid of char_count."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X["char_count"].to_numpy()))
        return np.column_stack([1 - p, p])


def frame(n, shift, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS})
    df["char_count"] += shift
    return df


def test_batch_report_flags_a_shifted_batch_and_not_a_matching_one():
    thresholds = {"t_low": 0.2, "t_high": 0.8}
    reference = frame(3000, 0.0, 1)
    same = monitor.batch_report(reference, frame(600, 0.0, 2), StubModel(), thresholds)
    shifted = monitor.batch_report(reference, frame(600, 2.0, 3), StubModel(), thresholds)
    assert same["score_level"] == "none"
    assert same["features_major"] == 0
    assert shifted["score_level"] == "major"
    assert shifted["features_major"] == 1
    assert shifted["top_features"][0] == "char_count"
    assert shifted["score_psi_above_noise"] is True
    assert sum(shifted["band_share"].values()) == pytest.approx(1.0)
    assert shifted["band_share"]["auto_flag"] > same["band_share"]["auto_flag"]


def test_markdown_labels_the_time_substitution_and_has_no_dashes():
    batch = {
        "kind": "unseen AI source",
        "n": 10,
        "score_psi": 0.3,
        "score_level": "major",
        "score_psi_null_p95": 0.05,
        "score_ks": 0.4,
        "features_moderate_or_worse": 2,
        "features_major": 1,
        "top_features": ["char_count"],
        "features": {"char_count": {"psi": 0.5, "ks": 0.3}},
        "band_share": {"auto_clear": 0.1, "escalate": 0.2, "auto_flag": 0.7},
    }
    text = monitor.render_markdown({"batches": {"falcon_180b_v1": batch}})
    assert "not real chronological drift" in text
    assert "| falcon_180b_v1 | unseen AI source | 10 | 0.300 | major |" in text
    assert "| n/a |" in text
    assert "—" not in text
    assert ";" not in text
