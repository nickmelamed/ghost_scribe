import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from stage1.escalation_policy import TARGET_ESCALATION_RATE, assign_band, calibrate_bands

probas = st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=5, max_size=200)


def test_target_rate_is_twenty_percent():
    assert TARGET_ESCALATION_RATE == 0.20


def test_assign_band_boundaries_escalate():
    bands = assign_band(np.array([0.1, 0.2, 0.5, 0.8, 0.9]), 0.2, 0.8)
    assert bands.tolist() == ["auto_clear", "escalate", "escalate", "escalate", "auto_flag"]


def test_calibrate_bands_on_uniform_scores_hits_target():
    proba = np.linspace(0.0, 1.0, 1001)
    t_low, t_high = calibrate_bands(proba, 0.2)
    assert np.isclose(t_low, 0.4)
    assert np.isclose(t_high, 0.6)
    rate = (assign_band(proba, t_low, t_high) == "escalate").mean()
    assert abs(rate - 0.2) < 0.01


@given(probas)
def test_calibrated_bands_are_symmetric_around_half(values):
    t_low, t_high = calibrate_bands(np.array(values))
    assert np.isclose(t_low + t_high, 1.0)
    assert t_low <= 0.5 <= t_high


@given(probas)
def test_every_row_lands_in_exactly_one_known_band(values):
    arr = np.array(values)
    t_low, t_high = calibrate_bands(arr)
    bands = assign_band(arr, t_low, t_high)
    assert set(bands.tolist()) <= {"auto_clear", "auto_flag", "escalate"}
    assert (bands == "auto_clear").sum() + (bands == "auto_flag").sum() + (bands == "escalate").sum() == len(arr)


@given(probas, st.floats(min_value=0.05, max_value=0.5))
def test_wider_target_never_shrinks_the_band(values, target):
    arr = np.array(values)
    narrow = calibrate_bands(arr, target / 2)
    wide = calibrate_bands(arr, target)
    assert wide[0] <= narrow[0] + 1e-12
    assert wide[1] >= narrow[1] - 1e-12


def test_bands_ordering_follows_score():
    bands = assign_band(np.array([0.0, 0.3, 0.5, 0.7, 1.0]), 0.4, 0.6)
    assert bands.tolist() == ["auto_clear", "auto_clear", "escalate", "auto_flag", "auto_flag"]
