import pandas as pd

from stage1.features import FEATURE_COLUMNS
from stage2.prepare_finetune_data import (
    FEATURE_DISPLAY_NAMES,
    N_CUES_IN_RATIONALE,
    _feature_direction_for_ai,
    build_rationale,
)


def train_frame():
    base = {c: [1.0, 1.0, 3.0, 3.0] for c in FEATURE_COLUMNS}
    base["label"] = [0, 0, 1, 1]
    base["word_count"] = [10.0, 10.0, 5.0, 5.0]
    return pd.DataFrame(base)


def test_every_feature_has_a_display_name():
    assert set(FEATURE_DISPLAY_NAMES) == set(FEATURE_COLUMNS)


def test_direction_is_plus_one_when_ai_mean_is_higher():
    d = _feature_direction_for_ai(train_frame())
    assert d["char_count"] == 1
    assert d["word_count"] == -1


def test_direction_ties_count_as_minus_one():
    df = train_frame()
    df["comma_rate"] = 2.0
    assert _feature_direction_for_ai(df)["comma_rate"] == -1


def test_rationale_picks_the_two_most_supportive_cues_for_the_true_label():
    df = train_frame()
    direction = _feature_direction_for_ai(df)
    mean, std = df[FEATURE_COLUMNS].mean(), df[FEATURE_COLUMNS].std()
    row = mean.copy()
    row["char_count"] = mean["char_count"] + 5 * std["char_count"]
    row["word_count"] = mean["word_count"] - 5 * std["word_count"]
    row["noun_ratio"] = mean["noun_ratio"] + 0.1 * std["noun_ratio"]
    top = ["char_count", "word_count", "noun_ratio", "verb_ratio"]
    text = build_rationale(row, 1, top, direction, mean, std)
    assert N_CUES_IN_RATIONALE == 2
    assert text == (
        "essay length (characters) is higher than typical and "
        "essay length (words) is lower than typical, "
        "a pattern the training data associates more with AI-generated essays."
    )


def test_rationale_for_human_label_prefers_human_leaning_cues():
    df = train_frame()
    direction = _feature_direction_for_ai(df)
    mean, std = df[FEATURE_COLUMNS].mean(), df[FEATURE_COLUMNS].std()
    row = mean.copy()
    row["char_count"] = mean["char_count"] - 5 * std["char_count"]
    text = build_rationale(row, 0, ["char_count", "word_count"], direction, mean, std)
    assert text.startswith("essay length (characters) is lower than typical")
    assert text.endswith("a pattern the training data associates more with human-written essays.")


def test_rationale_with_zero_variance_feature_does_not_divide_by_zero():
    df = train_frame()
    direction = _feature_direction_for_ai(df)
    mean = df[FEATURE_COLUMNS].mean()
    std = df[FEATURE_COLUMNS].std()
    std["char_count"] = 0.0
    row = mean.copy()
    row["char_count"] = mean["char_count"] + 2
    text = build_rationale(row, 1, ["char_count"], direction, mean, std)
    assert "essay length (characters) is higher than typical" in text
