"""Prompt-held-out evaluation of the Stage-1 plus Stage-2 pipeline.

The main run splits essays at random, so every essay prompt appears in train,
val and test. Stage 2 can then lean on prompt-specific wording. This
experiment holds out whole prompts from both stages and tests on them.

Steps, run from the repo root. Step 2 takes hours on this machine.

  1. python -m evaluation.prompt_holdout prepare
  2. python -m stage2.finetune_lora --data-dir experiments/prompt_holdout/stage2_data \\
         --adapter-dir experiments/prompt_holdout/adapters/lora
  3. python -m stage2.infer --split test_heldout_prompts --data-dir ... --adapter-dir ... --out-dir ...
     python -m stage2.infer --split test_seen_prompts --data-dir ... --adapter-dir ... --out-dir ...
  4. python -m evaluation.prompt_holdout report

Prepare retrains the Stage-1 XGBoost model on the seen prompts only,
recalibrates the escalation bands on the val rows of the seen prompts, and
builds Stage-2 training data from the escalated seen-prompt train rows only.
The held-out prompts never reach training or calibration. Report compares
the same retrained models on test rows from the held-out prompts and from the
seen prompts, next to the main run. Intermediate files go to the gitignored
experiments/ folder.

The held-out prompts come from one seeded draw, so the result is a single
split and can vary with the draw.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from evaluation.evaluate_combined import (
    attach_stage2,
    escalated_subset_comparison,
    full_pipeline_comparison,
    score_and_band,
)
from stage1.baseline_xgb import fit
from stage1.escalation_policy import TARGET_ESCALATION_RATE, calibrate_bands
from stage1.features import FEATURE_COLUMNS
from stage2.prepare_finetune_data import (
    N_IMPORTANT_FEATURES,
    _feature_direction_for_ai,
    build_records,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
EXP_DIR = ROOT / "experiments" / "prompt_holdout"
RESULTS_JSON = ROOT / "results" / "prompt_holdout.json"
RESULTS_MD = ROOT / "results" / "prompt_holdout.md"
MAIN_RUN_METRICS = ROOT / "results" / "phase4_metrics.json"

HOLDOUT_SEED = 42
N_HOLDOUT_PROMPTS = 3
GROUPS = ("test_heldout_prompts", "test_seen_prompts")


def choose_holdout_prompts(prompts: list[str], n: int = N_HOLDOUT_PROMPTS, seed: int = HOLDOUT_SEED) -> list[str]:
    """Draw ``n`` distinct prompts with a seeded generator, from the sorted names."""
    return random.Random(seed).sample(sorted(prompts), n)


def split_by_prompt(df: pd.DataFrame, held_out: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the rows on seen prompts and the rows on held-out prompts."""
    mask = df["prompt_name"].isin(held_out)
    return df[~mask], df[mask]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval for a proportion, clamped to [0, 1], or NaNs when n is 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    low = 0.0 if successes == 0 else max(0.0, centre - half)
    high = 1.0 if successes == n else min(1.0, centre + half)
    return low, high


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare(exp_dir: Path = EXP_DIR, processed_dir: Path = PROCESSED_DIR) -> dict:
    """Retrain Stage 1 on the seen prompts and write the Stage-2 data."""
    train = pd.read_csv(processed_dir / "train_features.csv")
    val = pd.read_csv(processed_dir / "val_features.csv")
    test = pd.read_csv(processed_dir / "test_features.csv")

    held_out = choose_holdout_prompts(train["prompt_name"].unique().tolist())
    train_seen, train_held = split_by_prompt(train, held_out)
    val_seen, val_held = split_by_prompt(val, held_out)
    test_seen, test_held = split_by_prompt(test, held_out)

    xgb = fit(train_seen)
    val_proba = xgb.predict_proba(val_seen[FEATURE_COLUMNS])[:, 1]
    t_low, t_high = calibrate_bands(val_proba)

    escalated_train = score_and_band(train_seen, xgb, t_low, t_high)
    escalated_train = escalated_train[escalated_train["stage1_band"] == "escalate"]
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS)
    top_features = importances.sort_values(ascending=False).head(N_IMPORTANT_FEATURES).index.tolist()
    direction = _feature_direction_for_ai(train_seen)
    mean, std = train_seen[FEATURE_COLUMNS].mean(), train_seen[FEATURE_COLUMNS].std()

    stage2_dir = exp_dir / "stage2_data"
    train_records = build_records(escalated_train, True, top_features, direction, mean, std)
    write_jsonl(stage2_dir / "train.jsonl", train_records)
    counts = {}
    for name, frame in (("test_heldout_prompts", test_held), ("test_seen_prompts", test_seen)):
        scored = score_and_band(frame, xgb, t_low, t_high)
        escalated = scored[scored["stage1_band"] == "escalate"]
        write_jsonl(stage2_dir / f"{name}.jsonl", build_records(escalated, False, top_features, direction, mean, std))
        counts[name] = {"rows": len(frame), "escalated": len(escalated)}

    held_texts = set(pd.concat([train_held, val_held, test_held])["text"])
    assert not held_texts & {r["text"] for r in train_records}, "held-out essay reached Stage-2 training"
    assert not set(train_seen["prompt_name"]) & set(held_out), "held-out prompt reached Stage-1 training"

    exp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, exp_dir / "xgb.joblib")
    meta = {
        "holdout_seed": HOLDOUT_SEED,
        "held_out_prompts": held_out,
        "target_escalation_rate": TARGET_ESCALATION_RATE,
        "t_low": float(t_low),
        "t_high": float(t_high),
        "rows": {
            "train_seen": len(train_seen),
            "train_held_out": len(train_held),
            "val_seen": len(val_seen),
            "val_held_out": len(val_held),
        },
        "stage2_train_escalated": len(train_records),
        "test_groups": counts,
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def group_metrics(df: pd.DataFrame, stage2_path: Path) -> dict:
    """Return Stage-1 and combined metrics for one group of scored test rows."""
    merged, n_missing, n_escalated = attach_stage2(df, stage2_path)
    escalated = merged[merged["stage1_band"] == "escalate"]
    n_esc = len(escalated)
    k1 = int((escalated["stage1_pred"] == escalated["label"]).sum())
    k2 = int((escalated["final_pred"] == escalated["label"]).sum())
    humans = merged[merged["label"] == 0]
    by_prompt = {
        str(prompt): {
            "n_escalated": len(g),
            "stage1_accuracy": float((g["stage1_pred"] == g["label"]).mean()),
            "stage2_accuracy": float((g["final_pred"] == g["label"]).mean()),
        }
        for prompt, g in escalated.groupby("prompt_name")
    }
    return {
        "n_rows": len(merged),
        "escalation_rate": n_esc / len(merged),
        "stage2_fallbacks": n_missing,
        "escalated_subset": {
            **escalated_subset_comparison(merged),
            "stage1_accuracy_ci95": wilson_interval(k1, n_esc),
            "stage2_accuracy_ci95": wilson_interval(k2, n_esc),
        },
        "full_pipeline": full_pipeline_comparison(merged),
        "human_fpr_stage1_only": float(humans["stage1_pred"].mean()),
        "human_fpr_combined": float(humans["final_pred"].mean()),
        "dangerous_errors": int(((escalated["final_pred"] == 0) & (escalated["label"] == 1)).sum()),
        "escalated_by_prompt": by_prompt,
        "n_escalated_check": n_escalated,
    }


def fmt(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"{fmt(ci[0])} to {fmt(ci[1])}"


def render_markdown(results: dict[str, Any]) -> str:
    meta = results["meta"]
    groups = {name: results[name] for name in GROUPS}
    main_run = results.get("main_run")
    columns = ["Held-out prompts", "Seen prompts"] + (["Main run, all prompts"] if main_run else [])

    def row(label: str, held: str, seen: str, main: str = "") -> str:
        cells = [label, held, seen] + ([main] if main_run else [])
        return "| " + " | ".join(cells) + " |"

    h, s = groups["test_heldout_prompts"], groups["test_seen_prompts"]
    lines = ["# Prompt-held-out evaluation", ""]
    lines.append(
        "Stage 1 and Stage 2 were retrained without the held-out prompts, using the same hyperparameters as the "
        "main run. The escalation bands were recalibrated on val rows of the seen prompts. The test rows below "
        "come from the cleaned test split."
    )
    lines.append("")
    lines.append("Held-out prompts (seed %d): %s." % (meta["holdout_seed"], ", ".join(meta["held_out_prompts"])))
    lines.append(
        f"Train rows kept {meta['rows']['train_seen']} of "
        f"{meta['rows']['train_seen'] + meta['rows']['train_held_out']}. "
        f"Stage 2 trained on {meta['stage2_train_escalated']} escalated rows."
    )
    lines.append("")
    lines.append("| Metric | " + " | ".join(columns) + " |")
    lines.append("|---|" + "---|" * len(columns))
    mr = main_run or {}
    lines.append(row("Test rows", str(h["n_rows"]), str(s["n_rows"])))
    lines.append(row("Escalation rate", fmt(h["escalation_rate"]), fmt(s["escalation_rate"]), fmt(mr.get("escalation_rate"))))
    lines.append(
        row(
            "Stage-1-only accuracy",
            fmt(h["full_pipeline"]["stage1_only"]["accuracy"]),
            fmt(s["full_pipeline"]["stage1_only"]["accuracy"]),
            fmt(mr.get("stage1_only_accuracy")),
        )
    )
    lines.append(
        row(
            "Stage-1+2 accuracy",
            fmt(h["full_pipeline"]["stage1_plus_2"]["accuracy"]),
            fmt(s["full_pipeline"]["stage1_plus_2"]["accuracy"]),
            fmt(mr.get("stage1_plus_2_accuracy")),
        )
    )
    lines.append(
        row("Escalated rows", str(h["escalated_subset"]["n"]), str(s["escalated_subset"]["n"]), str(mr.get("escalated_n", "n/a")))
    )
    lines.append(
        row(
            "Escalated, Stage-1 accuracy",
            fmt(h["escalated_subset"]["stage1_alone"]["accuracy"]),
            fmt(s["escalated_subset"]["stage1_alone"]["accuracy"]),
            fmt(mr.get("escalated_stage1_accuracy")),
        )
    )
    lines.append(
        row(
            "Escalated, Stage-2 accuracy",
            fmt(h["escalated_subset"]["stage1_plus_2"]["accuracy"]),
            fmt(s["escalated_subset"]["stage1_plus_2"]["accuracy"]),
            fmt(mr.get("escalated_stage2_accuracy")),
        )
    )
    lines.append(
        row(
            "Escalated, Stage-2 95% interval",
            fmt_ci(h["escalated_subset"]["stage2_accuracy_ci95"]),
            fmt_ci(s["escalated_subset"]["stage2_accuracy_ci95"]),
        )
    )
    lines.append(row("Human FPR, Stage-1 only", fmt(h["human_fpr_stage1_only"]), fmt(s["human_fpr_stage1_only"])))
    lines.append(row("Human FPR, combined", fmt(h["human_fpr_combined"]), fmt(s["human_fpr_combined"])))
    lines.append(row("Dangerous errors (said human, was AI)", str(h["dangerous_errors"]), str(s["dangerous_errors"])))
    lines.append(row("Stage-2 fallbacks", str(h["stage2_fallbacks"]), str(s["stage2_fallbacks"])))
    lines.append("")
    lines.append("## Escalated rows by prompt")
    lines.append("")
    lines.append("| Group | Prompt | Escalated rows | Stage-1 accuracy | Stage-2 accuracy |")
    lines.append("|---|---|---|---|---|")
    for name, label in (("test_heldout_prompts", "held out"), ("test_seen_prompts", "seen")):
        for prompt, p in sorted(groups[name]["escalated_by_prompt"].items()):
            lines.append(
                f"| {label} | {prompt} | {p['n_escalated']} | {fmt(p['stage1_accuracy'])} | {fmt(p['stage2_accuracy'])} |"
            )
    lines.append("")
    lines.append("## Reading this")
    lines.append("")
    lines.append(
        "This is one split of 3 of 15 prompts, so a different draw could give a different result. The intervals "
        "are Wilson 95% intervals on the escalated rows only. The main run column uses models trained on all "
        "prompts and so is not a like-for-like comparison. The seen-prompts column uses the same retrained models "
        "as the held-out column, so the difference between them reflects prompt shift."
    )
    return "\n".join(lines) + "\n"


def report(exp_dir: Path = EXP_DIR, processed_dir: Path = PROCESSED_DIR) -> dict:
    meta = json.loads((exp_dir / "meta.json").read_text())
    xgb = joblib.load(exp_dir / "xgb.joblib")
    test = pd.read_csv(processed_dir / "test_features.csv")
    seen, held = split_by_prompt(test, meta["held_out_prompts"])
    frames = {"test_heldout_prompts": held, "test_seen_prompts": seen}

    results: dict[str, Any] = {"meta": meta}
    for name, frame in frames.items():
        scored = score_and_band(frame, xgb, meta["t_low"], meta["t_high"])
        results[name] = group_metrics(scored, exp_dir / "predictions" / f"stage2_predictions_{name}.csv")

    if MAIN_RUN_METRICS.exists():
        main = json.loads(MAIN_RUN_METRICS.read_text())
        esc, full = main["escalated_subset_comparison"], main["full_pipeline_comparison"]
        results["main_run"] = {
            "escalation_rate": json.loads((ROOT / "results" / "phase2_metrics.json").read_text())["escalation"][
                "test_escalation_rate"
            ],
            "stage1_only_accuracy": full["stage1_only"]["accuracy"],
            "stage1_plus_2_accuracy": full["stage1_plus_2"]["accuracy"],
            "escalated_n": esc["n"],
            "escalated_stage1_accuracy": esc["stage1_alone"]["accuracy"],
            "escalated_stage2_accuracy": esc["stage1_plus_2"]["accuracy"],
        }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=float) + "\n")
    RESULTS_MD.write_text(render_markdown(results))
    return results


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in ("prepare", "report"):
        print("usage: python -m evaluation.prompt_holdout prepare|report")
        return 2
    if args[0] == "prepare":
        meta = prepare()
        print(json.dumps(meta, indent=2))
        print(f"Wrote Stage-1 model and Stage-2 data under {EXP_DIR}")
    else:
        report()
        print(RESULTS_MD.read_text())
        print(f"Wrote {RESULTS_JSON} and {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
