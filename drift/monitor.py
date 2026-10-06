"""Drift monitoring across generation-source batches.

This dataset has no timestamps, so each generation source stands in for a
step in time, the way a new AI model would show up later. That is a
substitution, not real chronological drift.

Each batch is compared with the round-1 training data of the same kind:

  - AI sources are compared with the AI essays in train.
  - The human control batch is compared with the human essays in train.

Seen sources use their test-split rows. The held-out sources use all their
rows. They are the sources that were unseen when the models were trained.

A single source differs in style from the pooled reference even when the
models trained on it, so drift against the pool alone does not separate a new
source from an old one. Seen sources are therefore also compared with their
own train rows, which gives a stability baseline. Unseen sources have no such
rows.

For each batch the report gives the Population Stability Index (PSI) and the
Kolmogorov-Smirnov statistic on the 24 stylometric features and on the
Stage-1 score, plus how Stage 1 routes the batch. PSI above 0.1 is moderate
and above 0.2 is major, the usual rules of thumb. A small batch can pass those
thresholds by chance, so the report also gives the 95th percentile of the
score PSI that a same-size bootstrap sample of the reference reaches.

Run from the repo root:
    python -m drift.monitor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from stage1.escalation_policy import assign_band
from stage1.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "stage1" / "models"

PSI_MODERATE = 0.1
PSI_MAJOR = 0.2
N_BINS = 10
EPSILON = 1e-4
NULL_REPS = 200
SEED = 42
HELD_OUT_SOURCES = ("falcon_180b_v1", "llama_70b_v1")


def psi(reference: np.ndarray, sample: np.ndarray, n_bins: int = N_BINS) -> float:
    """Return the Population Stability Index of ``sample`` against ``reference``.

    Bin edges are quantiles of the reference, so each bin holds about the same
    share of it. A reference with too few distinct values to make two bins has
    no measurable PSI and gives 0.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    smp_share = np.histogram(sample, bins=edges)[0] / len(sample)
    ref_share = np.clip(ref_share, EPSILON, None)
    smp_share = np.clip(smp_share, EPSILON, None)
    return float(np.sum((smp_share - ref_share) * np.log(smp_share / ref_share)))


def level(value: float) -> str:
    if value > PSI_MAJOR:
        return "major"
    if value > PSI_MODERATE:
        return "moderate"
    return "none"


def null_psi_p95(reference: np.ndarray, n: int, reps: int = NULL_REPS, seed: int = SEED) -> float:
    """Return the 95th percentile of PSI for ``n`` rows resampled from the reference."""
    rng = np.random.default_rng(seed)
    values = [psi(reference, rng.choice(reference, size=n, replace=True)) for _ in range(reps)]
    return float(np.percentile(values, 95))


def batch_report(reference: pd.DataFrame, batch: pd.DataFrame, model: Any, thresholds: dict) -> dict:
    """Compare one batch with a reference frame of features and labels."""
    ref_score = model.predict_proba(reference[FEATURE_COLUMNS])[:, 1]
    score = model.predict_proba(batch[FEATURE_COLUMNS])[:, 1]
    features = {}
    for column in FEATURE_COLUMNS:
        ref_values, values = reference[column].to_numpy(), batch[column].to_numpy()
        features[column] = {"psi": psi(ref_values, values), "ks": float(stats.ks_2samp(ref_values, values).statistic)}
    psis = {name: f["psi"] for name, f in features.items()}
    bands = assign_band(score, thresholds["t_low"], thresholds["t_high"])
    score_psi = psi(ref_score, score)
    null_p95 = null_psi_p95(ref_score, len(batch))
    return {
        "n": len(batch),
        "score_psi": score_psi,
        "score_level": level(score_psi),
        "score_psi_null_p95": null_p95,
        "score_psi_above_noise": bool(score_psi > null_p95),
        "score_ks": float(stats.ks_2samp(ref_score, score).statistic),
        "features_moderate_or_worse": sum(v > PSI_MODERATE for v in psis.values()),
        "features_major": sum(v > PSI_MAJOR for v in psis.values()),
        "top_features": sorted(psis, key=lambda k: psis[k], reverse=True)[:3],
        "features": features,
        "band_share": {name: float((bands == name).mean()) for name in ("auto_clear", "escalate", "auto_flag")},
    }


def run(processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR) -> dict:
    train = pd.read_csv(processed_dir / "train_features.csv")
    test = pd.read_csv(processed_dir / "test_features.csv")
    held_out = pd.read_csv(processed_dir / "held_out_features.csv")
    model = joblib.load(models_dir / "xgb.joblib")
    thresholds = json.loads((models_dir / "escalation_thresholds.json").read_text())

    ref_ai, ref_human = train[train["label"] == 1], train[train["label"] == 0]
    batches: dict[str, dict] = {}
    batches["human (control)"] = {
        "kind": "human control",
        **batch_report(ref_human, test[test["label"] == 0], model, thresholds),
    }
    for source in sorted(test.loc[test["label"] == 1, "generation_source"].unique()):
        rows = test[(test["label"] == 1) & (test["generation_source"] == source)]
        own = batch_report(train[train["generation_source"] == source], rows, model, thresholds)
        batches[source] = {
            "kind": "seen AI source",
            **batch_report(ref_ai, rows, model, thresholds),
            "vs_own_train": {k: own[k] for k in ("score_psi", "score_level", "features_moderate_or_worse")},
        }
    for source in HELD_OUT_SOURCES:
        rows = held_out[held_out["generation_source"] == source]
        batches[source] = {"kind": "unseen AI source", **batch_report(ref_ai, rows, model, thresholds)}
    return {
        "psi_thresholds": {"moderate": PSI_MODERATE, "major": PSI_MAJOR},
        "n_bins": N_BINS,
        "reference_rows": {"train_ai": len(ref_ai), "train_human": len(ref_human)},
        "batches": batches,
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_markdown(report: dict) -> str:
    lines = ["# Drift report across generation-source batches", ""]
    lines.append(
        "Each generation source stands in for a step in time. This is a substitution, not real chronological "
        "drift. AI batches are compared with the AI essays in train and the human control with the human essays "
        "in train, using the round-1 models. PSI above 0.1 is moderate and above 0.2 is major. The noise column "
        "is the 95th percentile of the score PSI for a same-size bootstrap sample of the reference."
    )
    lines.append("")
    lines.append(
        "| Batch | Kind | Rows | Score PSI | Level | Noise p95 | Score KS | Features over 0.1 | Features over 0.2 "
        "| Score PSI vs own train rows | Auto-clear | Escalate | Auto-flag |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, b in report["batches"].items():
        own = f"{b['vs_own_train']['score_psi']:.3f}" if "vs_own_train" in b else "n/a"
        lines.append(
            f"| {name} | {b['kind']} | {b['n']} | {b['score_psi']:.3f} | {b['score_level']} | "
            f"{b['score_psi_null_p95']:.3f} | {b['score_ks']:.3f} | {b['features_moderate_or_worse']} | "
            f"{b['features_major']} | {own} | {pct(b['band_share']['auto_clear'])} | {pct(b['band_share']['escalate'])} | "
            f"{pct(b['band_share']['auto_flag'])} |"
        )
    lines.append("")
    lines.append(
        "For an AI batch the auto-clear share is the share Stage 1 would clear with no review, so it is the miss "
        "rate. For the human control the auto-flag share is the false-flag rate."
    )
    lines.append("")
    lines.append("## Top drifting features per batch")
    lines.append("")
    lines.append("| Batch | Top three features by PSI |")
    lines.append("|---|---|")
    for name, b in report["batches"].items():
        top = ", ".join(f"{f} ({b['features'][f]['psi']:.2f})" for f in b["top_features"])
        lines.append(f"| {name} | {top} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="round1", help="suffix for the output files")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)
    report = run(models_dir=args.models_dir)
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"drift_{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report)
    (out_dir / f"drift_{args.tag}.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
