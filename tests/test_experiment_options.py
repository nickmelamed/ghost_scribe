from pathlib import Path

import numpy as np
import pandas as pd

from stage1.baseline_xgb import fit
from stage1.features import FEATURE_COLUMNS
from stage2 import finetune_lora, infer
from stage2.prepare_finetune_data import SYSTEM_PROMPT, build_records


def test_infer_options_default_to_the_main_run_paths():
    args = infer.parse_args(["--split", "test"])
    assert args.data_dir == infer.DATA_DIR
    assert args.adapter_dir == infer.ADAPTER_DIR
    assert args.out_dir == infer.OUT_DIR
    assert args.split == "test"


def test_infer_options_can_point_at_another_experiment():
    args = infer.parse_args(
        ["--split", "custom_split", "--data-dir", "d", "--adapter-dir", "a", "--out-dir", "o", "--n_sample", "5"]
    )
    assert (args.data_dir, args.adapter_dir, args.out_dir) == (Path("d"), Path("a"), Path("o"))
    assert args.split == "custom_split"
    assert args.n_sample == 5


def test_finetune_options_default_to_the_main_run_paths():
    args = finetune_lora.parse_args([])
    assert args.data_dir == finetune_lora.DATA_DIR
    assert args.adapter_dir == finetune_lora.ADAPTER_DIR


def test_finetune_options_can_point_at_another_experiment():
    args = finetune_lora.parse_args(["--data-dir", "d", "--adapter-dir", "x/lora"])
    assert (args.data_dir, args.adapter_dir) == (Path("d"), Path("x/lora"))


def small_training_frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=60) for c in FEATURE_COLUMNS})
    df["label"] = [0, 1] * 30
    df["char_count"] += 3 * df["label"]
    return df


def test_fit_returns_a_classifier_that_scores_the_signal_feature():
    df = small_training_frame()
    clf = fit(df)
    proba = clf.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    assert proba[df["label"] == 1].mean() > proba[df["label"] == 0].mean()
    assert clf.get_params()["n_estimators"] == 300
    assert clf.get_params()["random_state"] == 42


def test_build_records_adds_a_target_only_when_asked():
    df = small_training_frame().iloc[:2].copy()
    df["text"] = ["essay one", "essay two"]
    df["generation_source"] = ["human", "gpt"]
    mean, std = df[FEATURE_COLUMNS].mean(), df[FEATURE_COLUMNS].std()
    direction = dict.fromkeys(FEATURE_COLUMNS, 1)
    top = ["char_count", "word_count"]
    plain = build_records(df, False, top, direction, mean, std)
    assert [r["text"] for r in plain] == ["essay one", "essay two"]
    assert all("target" not in r and r["system"] == SYSTEM_PROMPT for r in plain)
    assert plain[0]["user"] == "Essay:\nessay one"
    with_target = build_records(df, True, top, direction, mean, std)
    assert with_target[0]["target"].startswith("Label: Human-written\nRationale: ")
    assert with_target[1]["target"].startswith("Label: AI-generated\nRationale: ")
