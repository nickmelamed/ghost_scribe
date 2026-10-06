import json
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from evaluation import prompt_holdout as ph
from stage1.features import FEATURE_COLUMNS

PROMPTS = [f"prompt {i:02d}" for i in range(15)]


def test_holdout_draw_is_three_seeded_distinct_prompts():
    assert (ph.HOLDOUT_SEED, ph.N_HOLDOUT_PROMPTS) == (42, 3)
    first = ph.choose_holdout_prompts(PROMPTS)
    assert first == ph.choose_holdout_prompts(list(reversed(PROMPTS)))
    assert len(set(first)) == 3
    assert set(first) <= set(PROMPTS)


def test_split_by_prompt_partitions_rows():
    df = pd.DataFrame({"prompt_name": ["a", "b", "c", "a"], "x": [1, 2, 3, 4]})
    seen, held = ph.split_by_prompt(df, ["a"])
    assert held["x"].tolist() == [1, 4]
    assert seen["x"].tolist() == [2, 3]


def test_wilson_interval_hand_worked():
    low, high = ph.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_keeps_perfect_scores_below_certainty():
    low, high = ph.wilson_interval(100, 100)
    assert high == pytest.approx(1.0)
    assert low < 1.0


def test_wilson_interval_with_no_rows_is_nan():
    assert all(math.isnan(x) for x in ph.wilson_interval(0, 0))


@given(st.integers(min_value=1, max_value=500), st.data())
def test_wilson_interval_brackets_the_proportion(n, data):
    k = data.draw(st.integers(min_value=0, max_value=n))
    low, high = ph.wilson_interval(k, n)
    assert 0.0 <= low <= k / n <= high <= 1.0


def synthetic_features(n_per_prompt, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for prompt in PROMPTS:
        for i in range(n_per_prompt):
            label = int(i % 3 == 0)
            row = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            row["char_count"] += 1.5 * label
            row.update(
                {
                    "text": f"{prompt} essay {i} {seed}",
                    "label": label,
                    "generation_source": "gpt" if label else "human",
                    "prompt_name": prompt,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


class StubModel:
    """Score rows by a sigmoid of char_count, so some rows are ambiguous."""

    feature_importances_ = np.arange(len(FEATURE_COLUMNS), dtype=float)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X["char_count"].to_numpy()))
        return np.column_stack([1 - p, p])


@pytest.fixture
def stub_fit(monkeypatch):
    monkeypatch.setattr(ph, "fit", lambda df: StubModel())


def write_features(directory, train, val, test):
    directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(directory / "train_features.csv", index=False)
    val.to_csv(directory / "val_features.csv", index=False)
    test.to_csv(directory / "test_features.csv", index=False)


def test_prepare_keeps_held_out_prompts_out_of_training_and_calibration(tmp_path, stub_fit):
    processed, exp = tmp_path / "processed", tmp_path / "exp"
    write_features(processed, synthetic_features(30, 1), synthetic_features(12, 2), synthetic_features(12, 3))
    meta = ph.prepare(exp, processed)
    held = set(meta["held_out_prompts"])
    assert held == set(ph.choose_holdout_prompts(PROMPTS))
    train_records = [json.loads(line) for line in (exp / "stage2_data" / "train.jsonl").open()]
    assert train_records
    assert all("target" in r for r in train_records)
    train_texts = {r["text"] for r in train_records}
    assert not any(any(p in t for p in held) for t in train_texts)
    assert meta["rows"]["train_seen"] == 12 * 30
    assert meta["rows"]["train_held_out"] == 3 * 30
    for name in ph.GROUPS:
        records = [json.loads(line) for line in (exp / "stage2_data" / f"{name}.jsonl").open()]
        assert all("target" not in r for r in records)
        assert len(records) == meta["test_groups"][name]["escalated"]


def test_thresholds_depend_only_on_val_rows_of_seen_prompts(tmp_path, stub_fit):
    train, test = synthetic_features(30, 1), synthetic_features(12, 3)
    val = synthetic_features(12, 2)
    held = ph.choose_holdout_prompts(PROMPTS)
    changed = val.copy()
    mask = changed["prompt_name"].isin(held)
    changed.loc[mask, "char_count"] += 50.0
    write_features(tmp_path / "a", train, val, test)
    write_features(tmp_path / "b", train, changed, test)
    a = ph.prepare(tmp_path / "exp_a", tmp_path / "a")
    b = ph.prepare(tmp_path / "exp_b", tmp_path / "b")
    assert (a["t_low"], a["t_high"]) == (b["t_low"], b["t_high"])


def scored_group():
    return pd.DataFrame(
        {
            "text": ["a", "b", "c", "d", "e"],
            "label": [1, 0, 1, 0, 0],
            "prompt_name": ["p1", "p1", "p2", "p2", "p2"],
            "stage1_proba": [0.99, 0.5, 0.6, 0.4, 0.01],
            "stage1_pred": [1, 1, 1, 0, 0],
            "stage1_band": ["auto_flag", "escalate", "escalate", "escalate", "auto_clear"],
        }
    )


def test_group_metrics_hand_worked(tmp_path):
    path = tmp_path / "s2.csv"
    pd.DataFrame({"text": ["b", "c", "d"], "stage2_pred_label": [0, 1, 1]}).to_csv(path, index=False)
    m = ph.group_metrics(scored_group(), path)
    assert m["n_rows"] == 5
    assert m["escalation_rate"] == 0.6
    assert m["stage2_fallbacks"] == 0
    esc = m["escalated_subset"]
    assert esc["n"] == 3
    assert esc["stage1_alone"]["accuracy"] == pytest.approx(2 / 3)
    assert esc["stage1_plus_2"]["accuracy"] == pytest.approx(2 / 3)
    assert esc["stage2_accuracy_ci95"] == ph.wilson_interval(2, 3)
    assert m["human_fpr_stage1_only"] == pytest.approx(1 / 3)
    assert m["human_fpr_combined"] == pytest.approx(1 / 3)
    assert m["dangerous_errors"] == 0
    assert m["escalated_by_prompt"]["p2"]["n_escalated"] == 2


def test_group_metrics_counts_a_dangerous_error(tmp_path):
    path = tmp_path / "s2.csv"
    pd.DataFrame({"text": ["b", "c", "d"], "stage2_pred_label": [0, 0, 0]}).to_csv(path, index=False)
    assert ph.group_metrics(scored_group(), path)["dangerous_errors"] == 1


def test_markdown_names_the_prompts_and_avoids_conclusions(tmp_path):
    path = tmp_path / "s2.csv"
    pd.DataFrame({"text": ["b", "c", "d"], "stage2_pred_label": [0, 1, 1]}).to_csv(path, index=False)
    group = ph.group_metrics(scored_group(), path)
    meta = {
        "holdout_seed": 42,
        "held_out_prompts": ["p1"],
        "rows": {"train_seen": 10, "train_held_out": 2},
        "stage2_train_escalated": 4,
    }
    text = ph.render_markdown({"meta": meta, "test_heldout_prompts": group, "test_seen_prompts": group})
    assert "p1" in text
    assert "Held-out prompts (seed 42): p1." in text
    assert "—" not in text
    assert ";" not in text
