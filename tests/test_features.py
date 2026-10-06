import math

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stage1.features import FEATURE_COLUMNS, _bigram_burstiness, extract_features, extract_features_df

SAMPLE = "The cat sat. It was happy, wasn't it?"


def test_feature_columns_are_the_documented_24_in_order():
    assert len(FEATURE_COLUMNS) == 24
    assert FEATURE_COLUMNS[:3] == ["char_count", "word_count", "sentence_count"]
    assert FEATURE_COLUMNS[-3:] == ["flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog"]


def test_extract_features_returns_exactly_the_feature_columns():
    assert set(extract_features(SAMPLE)) == set(FEATURE_COLUMNS)


def test_extract_features_pins_counts_on_a_small_text():
    f = extract_features(SAMPLE)
    assert f["char_count"] == 37
    assert f["word_count"] == 9
    assert f["sentence_count"] == 2
    assert f["avg_word_length"] == 3.0
    assert f["avg_sentence_length"] == 6.0
    assert f["sentence_length_std"] == 2.0
    assert math.isclose(f["type_token_ratio"], 7 / 9)
    assert math.isclose(f["comma_rate"], 100 / 9)
    assert math.isclose(f["question_rate"], 100 / 9)
    assert f["exclamation_rate"] == 0.0


def test_empty_whitespace_and_non_string_text_give_zero_word_counts():
    for bad in ["", "   ", None, float("nan")]:
        f = extract_features(bad)
        assert f["word_count"] == 0
        assert f["sentence_count"] == 0
        assert f["type_token_ratio"] == 0.0
        assert f["noun_ratio"] == 0.0


def test_char_count_keeps_whitespace_but_non_strings_count_zero():
    assert extract_features("   ")["char_count"] == 3
    assert extract_features(None)["char_count"] == 0


def test_bigram_burstiness_short_text_is_zero():
    assert _bigram_burstiness(["a", "b"]) == 0.0


def test_bigram_burstiness_uniform_bigrams_is_zero():
    assert _bigram_burstiness(["a", "b", "c", "d"]) == 0.0


def test_bigram_burstiness_hand_worked():
    # bigram counts are [2, 1], mean 1.5, population std 0.5
    assert math.isclose(_bigram_burstiness(["a", "b", "a", "b"]), 0.5 / 1.5)


def test_extract_features_df_keeps_index_and_column_order():
    texts = pd.Series([SAMPLE, "Hello world."], index=[10, 20])
    df = extract_features_df(texts)
    assert df.index.tolist() == [10, 20]
    assert df.columns.tolist() == FEATURE_COLUMNS


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=300))
def test_features_are_finite_and_ratios_bounded(text):
    f = extract_features(text)
    for name in FEATURE_COLUMNS:
        assert math.isfinite(f[name]), name
    for name in ["noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio", "det_ratio",
                 "adp_ratio", "conj_ratio", "type_token_ratio"]:
        assert 0.0 <= f[name] <= 1.0, name
    assert f["char_count"] == len(text)
    assert f["word_count"] >= 0


@pytest.mark.parametrize("mark,col", [("!", "exclamation_rate"), (";", "semicolon_rate"), (".", "period_rate")])
def test_punctuation_rates_scale_with_word_count(mark, col):
    f = extract_features(f"one two three four five{mark}")
    assert math.isclose(f[col], 100 / 5)
