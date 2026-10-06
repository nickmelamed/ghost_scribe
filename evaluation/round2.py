"""Flywheel round 2: a new generation source plus human-reviewed corrections.

Round 2 adds one of the sources that were held out of round 1
(NEW_SOURCE, split 70/15/15 like the others) and the essays a teacher reviewed
in the HITL pass, labeled with the reviewer's decision. Both stages are
retrained from scratch and the escalation bands and the review gate are
recalibrated on the round-2 val rows. The other held-out source
(UNSEEN_SOURCE) stays unseen, as a test of generalization.

Reviewed essays come from the val split. They move into training and leave
the val rows used for calibration. Evaluation always uses the true labels.
Test rows, ESL rows and new-source eval rows that near-duplicate anything now
in training are dropped, the same rule as data.dedup.

Steps, run from the repo root. Training and inference take hours.

  1. python -m evaluation.round2 prepare
  2. python -m stage2.finetune_lora --data-dir experiments/round2/stage2_data \\
         --adapter-dir experiments/round2/adapters/lora
  3. For each group named in experiments/round2/meta.json, run stage2.infer
     with --data-dir, --adapter-dir and --out-dir pointed at the round-2
     folders, and stage2.confidence for val_r2 and test_r1. The round-1
     baselines for the new-source groups use r1_stage2_data and the round-1
     adapter.
  4. python -m evaluation.round2 report

Cost per decision uses illustrative relative units, not real money: 1 for the
Stage-1 pass on every essay, 5 for each Stage-2 escalation and 25 for each
essay a human reviews.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.dedup import NEAR_DUPLICATE_THRESHOLD, ShingleIndex, near_duplicates
from evaluation.evaluate_combined import attach_stage2, score_and_band
from evaluation.metrics import auto_flag_rate, false_positive_rate
from evaluation.prompt_holdout import group_metrics, write_jsonl
from stage1.baseline_xgb import fit
from stage1.escalation_policy import TARGET_ESCALATION_RATE, calibrate_bands
from stage1.features import FEATURE_COLUMNS
from stage2.build_review_queue import calibrate_gate_rate, gate_report
from stage2.prepare_finetune_data import N_IMPORTANT_FEATURES, _feature_direction_for_ai, build_records

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "stage1" / "models"
EXP_DIR = ROOT / "experiments" / "round2"
HITL_PATH = PROCESSED_DIR / "hitl_corrections_human.csv"
RESULTS_JSON = ROOT / "results" / "round2.json"
RESULTS_MD = ROOT / "results" / "round2.md"

NEW_SOURCE = "llama_70b_v1"
UNSEEN_SOURCE = "falcon_180b_v1"
SEED = 42
# Illustrative relative costs per decision, not real financial data.
STAGE1_UNITS, STAGE2_UNITS, REVIEW_UNITS = 1, 5, 25
EVAL_GROUPS = ("test_r1", "test_new_source", "unseen_source", "esl_eval")
CONFIDENCE_GROUPS = ("val_r2", "test_r1")


def split_new_source(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the new source's rows 70/15/15 into train, val and test."""
    train, rest = train_test_split(df, train_size=0.70, random_state=seed)
    val, test = train_test_split(rest, train_size=0.5, random_state=seed)
    return train, val, test


def hitl_training_rows(val: pd.DataFrame, corrections: pd.DataFrame) -> pd.DataFrame:
    """Return the reviewed val rows, labeled with the reviewer's decision.

    The true label is kept as ``true_label`` for analysis only. Training reads
    ``label``.
    """
    reviewed = val[val["text"].isin(corrections["text"])].copy()
    missing = set(corrections["text"]) - set(reviewed["text"])
    if missing:
        raise ValueError(f"{len(missing)} reviewed essays are not in the val split")
    decisions = corrections.drop_duplicates("text", keep="last").set_index("text")["label"]
    reviewed["true_label"] = reviewed["label"]
    reviewed["label"] = reviewed["text"].map(decisions).astype(int)
    return reviewed


def drop_near_duplicates(frames: dict[str, pd.DataFrame], reference_texts: list[str]) -> tuple[dict, dict]:
    """Drop rows that near-duplicate the reference. Return the frames and removal counts."""
    index = ShingleIndex(reference_texts)
    kept, removed = {}, {}
    for name, frame in frames.items():
        mask = near_duplicates(frame, [index]) if len(frame) else pd.Series([], dtype=bool)
        kept[name] = frame[~mask]
        removed[name] = int(mask.sum())
    return kept, removed


def prepare(
    exp_dir: Path = EXP_DIR,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    hitl_path: Path = HITL_PATH,
    allow_missing_hitl: bool = False,
) -> dict:
    """Assemble round-2 data, retrain Stage 1, and write the Stage-2 data."""
    if hitl_path.exists():
        corrections = pd.read_csv(hitl_path)
    elif allow_missing_hitl:
        corrections = pd.DataFrame({"text": [], "label": []})
    else:
        raise FileNotFoundError(f"{hitl_path} not found. Finish the HITL pass and run hitl.apply_corrections first.")

    train = pd.read_csv(processed_dir / "train_features.csv")
    val = pd.read_csv(processed_dir / "val_features.csv")
    test = pd.read_csv(processed_dir / "test_features.csv")
    held_out = pd.read_csv(processed_dir / "held_out_features.csv")
    esl = pd.read_csv(processed_dir / "esl_eval_features.csv")

    new_train, new_val, new_test = split_new_source(held_out[held_out["generation_source"] == NEW_SOURCE])
    unseen = held_out[held_out["generation_source"] == UNSEEN_SOURCE]
    reviewed = hitl_training_rows(val, corrections)

    train_r2 = pd.concat([train, new_train, reviewed], ignore_index=True)
    val_r2 = pd.concat([val[~val["text"].isin(reviewed["text"])], new_val], ignore_index=True)
    groups, removed = drop_near_duplicates(
        {"val_r2": val_r2, "test_r1": test, "test_new_source": new_test, "unseen_source": unseen, "esl_eval": esl},
        train_r2["text"].tolist(),
    )
    val_r2 = groups["val_r2"]

    xgb = fit(train_r2)
    t_low, t_high = calibrate_bands(xgb.predict_proba(val_r2[FEATURE_COLUMNS])[:, 1])
    escalated_train = score_and_band(train_r2, xgb, t_low, t_high)
    escalated_train = escalated_train[escalated_train["stage1_band"] == "escalate"]
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS)
    top_features = importances.sort_values(ascending=False).head(N_IMPORTANT_FEATURES).index.tolist()
    direction = _feature_direction_for_ai(train_r2)
    mean, std = train_r2[FEATURE_COLUMNS].mean(), train_r2[FEATURE_COLUMNS].std()

    stage2_dir = exp_dir / "stage2_data"
    train_records = build_records(escalated_train, True, top_features, direction, mean, std)
    write_jsonl(stage2_dir / "train.jsonl", train_records)
    counts: dict[str, Any] = {}
    for name, frame in groups.items():
        scored = score_and_band(frame, xgb, t_low, t_high)
        escalated = scored[scored["stage1_band"] == "escalate"]
        write_jsonl(stage2_dir / f"{name}.jsonl", build_records(escalated, False, top_features, direction, mean, std))
        (exp_dir / "groups").mkdir(parents=True, exist_ok=True)
        frame.to_csv(exp_dir / "groups" / f"{name}.csv", index=False)
        counts[name] = {"rows": len(frame), "escalated": len(escalated), "removed_as_near_duplicates": removed[name]}

    # Round-1 baselines for the groups round 1 never scored with Stage 2.
    xgb1 = joblib.load(models_dir / "xgb.joblib")
    th1 = json.loads((models_dir / "escalation_thresholds.json").read_text())
    for name in ("test_new_source", "unseen_source"):
        scored1 = score_and_band(groups[name], xgb1, th1["t_low"], th1["t_high"])
        esc1 = scored1[scored1["stage1_band"] == "escalate"]
        write_jsonl(
            exp_dir / "r1_stage2_data" / f"{name}.jsonl",
            build_records(esc1, False, top_features, direction, mean, std),
        )
        counts[name]["escalated_round1_bands"] = len(esc1)

    exp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb, exp_dir / "xgb.joblib")
    (exp_dir / "escalation_thresholds.json").write_text(json.dumps({"t_low": float(t_low), "t_high": float(t_high)}))
    meta = {
        "new_source": NEW_SOURCE,
        "unseen_source": UNSEEN_SOURCE,
        "seed": SEED,
        "hitl_reviewed_essays": len(reviewed),
        "hitl_label_differs_from_truth": int((reviewed["label"] != reviewed["true_label"]).sum()),
        "target_escalation_rate": TARGET_ESCALATION_RATE,
        "t_low": float(t_low),
        "t_high": float(t_high),
        "rows": {"train_r2": len(train_r2), "val_r2": len(val_r2), "new_source_train": len(new_train)},
        "stage2_train_escalated": len(train_records),
        "groups": counts,
        "confidence_groups": list(CONFIDENCE_GROUPS),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def ai_only_metrics(scored: pd.DataFrame, predictions: Path) -> dict:
    """Return routing and miss counts for a group where every essay is AI-generated."""
    merged, n_missing, _ = attach_stage2(scored, predictions)
    n = len(merged)
    shares = {name: float((merged["stage1_band"] == name).mean()) for name in ("auto_clear", "escalate", "auto_flag")}
    return {
        "n_rows": n,
        "band_share": shares,
        "stage1_recall": float((merged["stage1_pred"] == 1).mean()),
        "combined_recall": float((merged["final_pred"] == 1).mean()),
        "missed_after_stage2": int((merged["final_pred"] == 0).sum()),
        "cleared_by_stage1": int((merged["stage1_band"] == "auto_clear").sum()),
        "stage2_fallbacks": n_missing,
    }


def human_only_metrics(scored: pd.DataFrame, predictions: Path) -> dict:
    """Return false-positive rates for a group where every essay is human-written."""
    merged, n_missing, _ = attach_stage2(scored, predictions)
    labels = merged["label"].to_numpy()
    return {
        "n_rows": len(merged),
        "fpr_threshold": false_positive_rate(labels, merged["stage1_pred"].to_numpy()),
        "fpr_auto_flag": auto_flag_rate(merged["stage1_band"].to_numpy(), labels, positive_true_label=0),
        "fpr_combined": false_positive_rate(labels, merged["final_pred"].to_numpy()),
        "escalate_rate": float((merged["stage1_band"] == "escalate").mean()),
        "stage2_fallbacks": n_missing,
    }


def cost_per_decision(escalation_rate: float, review_rate: float) -> dict:
    """Return the per-essay cost in illustrative units and its parts."""
    parts = {
        "stage1": STAGE1_UNITS,
        "stage2": STAGE2_UNITS * escalation_rate,
        "review": REVIEW_UNITS * review_rate,
    }
    return {**parts, "total": float(sum(parts.values())), "stage1_only_total": STAGE1_UNITS}


def round1_summary(results_dir: Path = ROOT / "results") -> dict:
    """Collect the round-1 trend inputs from the saved result files."""
    phase2 = json.loads((results_dir / "phase2_metrics.json").read_text())
    phase4 = json.loads((results_dir / "phase4_metrics.json").read_text())
    gate = json.loads((results_dir / "review_gate.json").read_text())
    esc_rate = phase2["escalation"]["test_escalation_rate"]
    n_escalated = phase4["escalated_subset_comparison"]["n"]
    n_rows = round(n_escalated / esc_rate)
    review_rate = gate["test_check"]["queue_size"] / n_rows
    full = phase4["full_pipeline_comparison"]
    return {
        "escalation_rate": esc_rate,
        "review_rate": review_rate,
        "gap_stage1": phase2["fairness"]["fairness_gap_threshold"],
        "gap_auto_flag": phase2["fairness"]["fairness_gap_auto_flag"],
        "gap_combined": phase4["fairness_combined"]["gap"],
        "stage1_only_accuracy": full["stage1_only"]["accuracy"],
        "stage1_plus_2_accuracy": full["stage1_plus_2"]["accuracy"],
        "escalated_stage2_accuracy": phase4["escalated_subset_comparison"]["stage1_plus_2"]["accuracy"],
        "cost": cost_per_decision(esc_rate, review_rate),
    }


def round2_summary(groups: dict, gate_rate: float, gate_val: dict, gate_test: dict) -> dict:
    """Collect the round-2 trend inputs from the group metrics."""
    test = groups["test_r1"]
    esl = groups["esl_eval"]
    humans = test["human_fpr_stage1_only"], test["human_fpr_combined"]
    escalation_rate = test["escalation_rate"]
    review_rate = gate_test["queue_size"] / test["n_rows"]
    return {
        "escalation_rate": escalation_rate,
        "review_rate": review_rate,
        "gap_stage1": esl["fpr_threshold"] - humans[0],
        "gap_combined": esl["fpr_combined"] - humans[1],
        "stage1_only_accuracy": test["full_pipeline"]["stage1_only"]["accuracy"],
        "stage1_plus_2_accuracy": test["full_pipeline"]["stage1_plus_2"]["accuracy"],
        "escalated_stage2_accuracy": test["escalated_subset"]["stage1_plus_2"]["accuracy"],
        "gate_rate": gate_rate,
        "gate_val": gate_val,
        "gate_test_check": gate_test,
        "cost": cost_per_decision(escalation_rate, review_rate),
    }


def merged_with_confidence(predictions: Path, confidence: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions)
    conf = pd.read_csv(confidence)[["text", "confidence"]]
    return preds.merge(conf, on="text", how="left")


def report(exp_dir: Path = EXP_DIR) -> dict:
    meta = json.loads((exp_dir / "meta.json").read_text())
    xgb = joblib.load(exp_dir / "xgb.joblib")
    thresholds = {"t_low": meta["t_low"], "t_high": meta["t_high"]}
    scored = {
        name: score_and_band(pd.read_csv(exp_dir / "groups" / f"{name}.csv"), xgb, **thresholds)
        for name in EVAL_GROUPS
    }
    preds = exp_dir / "predictions"
    results: dict[str, Any] = {"meta": meta}
    results["test_r1"] = group_metrics(scored["test_r1"], preds / "stage2_predictions_test_r1.csv")
    for name in ("test_new_source", "unseen_source"):
        results[name] = ai_only_metrics(scored[name], preds / f"stage2_predictions_{name}.csv")
    results["esl_eval"] = human_only_metrics(scored["esl_eval"], preds / "stage2_predictions_esl_eval.csv")

    results["round1_models_on_new_groups"] = {}
    th1 = json.loads((MODELS_DIR / "escalation_thresholds.json").read_text())
    xgb1 = joblib.load(MODELS_DIR / "xgb.joblib")
    for name in ("test_new_source", "unseen_source"):
        path = exp_dir / "r1_predictions" / f"stage2_predictions_{name}.csv"
        if path.exists():
            scored1 = score_and_band(pd.read_csv(exp_dir / "groups" / f"{name}.csv"), xgb1, th1["t_low"], th1["t_high"])
            results["round1_models_on_new_groups"][name] = ai_only_metrics(scored1, path)

    conf = exp_dir / "confidence"
    val = merged_with_confidence(preds / "stage2_predictions_val_r2.csv", conf / "stage2_confidence_val_r2.csv")
    test = merged_with_confidence(preds / "stage2_predictions_test_r1.csv", conf / "stage2_confidence_test_r1.csv")
    needed, gate_rate = calibrate_gate_rate(val)
    results["round2"] = round2_summary(
        {"test_r1": results["test_r1"], "esl_eval": results["esl_eval"]},
        gate_rate,
        {"needed_rate": needed, **gate_report(val, gate_rate)},
        gate_report(test, gate_rate),
    )
    results["round1"] = round1_summary()
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=float) + "\n")
    RESULTS_MD.write_text(render_markdown(results))
    return results


def fmt(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def render_markdown(results: dict) -> str:
    r1, r2, meta = results["round1"], results["round2"], results["meta"]
    lines = ["# Flywheel round 2", ""]
    lines.append(
        f"Round 2 adds {meta['new_source']} and {meta['hitl_reviewed_essays']} essays reviewed by a teacher, "
        f"labeled with the reviewer's decision. {meta['unseen_source']} stays unseen. Both stages were retrained "
        "from scratch and the bands and review gate were recalibrated on round-2 val rows. Test rows that "
        "near-duplicate round-2 training data were dropped, so the round-1 test group is slightly smaller than "
        "in round 1. Costs are illustrative relative units, not real money."
    )
    lines.append("")
    lines.append("| Metric | Round 1 | Round 2 |")
    lines.append("|---|---|---|")
    for key, label in (
        ("escalation_rate", "Escalation rate, test"),
        ("review_rate", "Share of essays sent to human review"),
        ("stage1_only_accuracy", "Stage-1-only accuracy, test"),
        ("stage1_plus_2_accuracy", "Stage-1+2 accuracy, test"),
        ("escalated_stage2_accuracy", "Stage-2 accuracy on escalated rows"),
        ("gap_stage1", "ESL false-positive gap, Stage 1 at 0.5"),
        ("gap_combined", "ESL false-positive gap, combined"),
    ):
        lines.append(f"| {label} | {fmt(r1.get(key))} | {fmt(r2.get(key))} |")
    lines.append(f"| Cost per essay, Stage-1 only | {r1['cost']['stage1_only_total']} | {r2['cost']['stage1_only_total']} |")
    lines.append(f"| Cost per essay, full flywheel | {fmt(r1['cost']['total'])} | {fmt(r2['cost']['total'])} |")
    lines.append("")
    lines.append("## New and unseen sources")
    lines.append("")
    lines.append("| Group | Rows | Cleared by Stage 1 | Escalated | Auto-flagged | Missed after Stage 2 |")
    lines.append("|---|---|---|---|---|---|")
    for name in ("test_new_source", "unseen_source"):
        for label, g in (("round 1", results.get("round1_models_on_new_groups", {}).get(name)), ("round 2", results[name])):
            if g is None:
                lines.append(f"| {name}, {label} | n/a | n/a | n/a | n/a | n/a |")
                continue
            s = g["band_share"]
            lines.append(
                f"| {name}, {label} | {g['n_rows']} | {fmt(s['auto_clear'])} | {fmt(s['escalate'])} | "
                f"{fmt(s['auto_flag'])} | {g['missed_after_stage2']} |"
            )
    lines.append("")
    gate = r2["gate_val"]
    lines.append(
        f"Review gate: needed rate {fmt(gate['needed_rate'])} and gate rate {fmt(r2['gate_rate'])} on round-2 val. "
        f"Test check, not tuned on: {r2['gate_test_check']['n_dangerous_errors_queued']} of "
        f"{r2['gate_test_check']['n_dangerous_errors']} errors where Stage 2 said human and the essay was AI "
        "were queued."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in ("prepare", "report") or any(a not in ("--no-hitl",) for a in args[1:]):
        print("usage: python -m evaluation.round2 prepare [--no-hitl] | report")
        return 2
    if args[0] == "prepare":
        meta = prepare(allow_missing_hitl="--no-hitl" in args)
        print(json.dumps(meta, indent=2))
    else:
        report()
        print(RESULTS_MD.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
