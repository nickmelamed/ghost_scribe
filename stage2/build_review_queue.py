"""Build the actual human review queue from Stage 2's escalated verdicts,
using confidence-gated (asymmetric) sub-escalation.

Reviewing all ~1,280 Stage-1-escalated essays is unrealistic, and mostly
unnecessary: Stage 2 gets 99.2-99.5% of them right. But every escalated
case requires human confirmation before any action is taken, and this
project forbids any automated-accusation workflow. So confidence can
only be used to skip review in the direction that takes no action
against a student:

  - "AI-generated" verdicts: ALWAYS included, regardless of confidence.
    This is the accusatory direction, and skipping human review here
    would be exactly the automated-accusation pattern the project rules out.
  - "Human-written" verdicts: gated by confidence. Auto-clearing takes
    no action, so it's safe to skip review when Stage 2 is confident.
    Only the least-confident fraction is included.

The size of that fraction is calibrated on the escalated VAL split, where
ground truth is known only because the dataset is simulated. The needed
rate is the smallest one that queues every dangerous error on val
(predicted human, actually AI), and GATE_MARGIN is added on top. The test
split is never used to set it. It gets a read-only check of the chosen
rate, reported next to the calibration in results/review_gate.json.

Confidence gating cannot promise to catch every confident mistake. On val
one dangerous error had 0.994 confidence, so catching it needs a large
fraction of the human verdicts in the queue.

Output: data/processed/stage2_review_queue_val.csv, same schema as
data/processed/stage2_predictions_val.csv (what hitl/review_app.py
reads), just filtered down to the actual review queue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
GATE_REPORT_PATH = ROOT / "results" / "review_gate.json"

# Added to the smallest rate that catches every dangerous error on val. A
# fixed amount, not tuned, since there are only a handful of such errors.
GATE_MARGIN = 0.05


def load_split(split: str) -> pd.DataFrame:
    predictions = pd.read_csv(PROCESSED_DIR / f"stage2_predictions_{split}.csv")
    confidence = pd.read_csv(PROCESSED_DIR / f"stage2_confidence_{split}.csv")[["text", "confidence"]]
    return predictions.merge(confidence, on="text", how="left")


def human_verdicts_by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Return the "Human-written" verdicts, least confident first."""
    human = df[df["stage2_pred_label"] == 0]
    return human.sort_values("confidence", kind="stable")


def calibrate_gate_rate(df: pd.DataFrame, margin: float = GATE_MARGIN) -> tuple[float, float]:
    """Return the needed rate and the gate rate to use, from labeled rows.

    The needed rate is the smallest fraction of "Human-written" verdicts, taken
    least confident first, that includes every row whose true label is AI. The
    gate rate adds ``margin`` and is capped at 1.
    """
    human = human_verdicts_by_confidence(df)
    dangerous = np.flatnonzero(human["true_label"].to_numpy() == 1)
    needed = float((dangerous.max() + 1) / len(human)) if len(dangerous) else 0.0
    return needed, min(1.0, needed + margin)


def gate_report(df: pd.DataFrame, rate: float) -> dict:
    """Describe what a gate of ``rate`` queues and misses on labeled rows."""
    human = human_verdicts_by_confidence(df)
    n_gate = int(len(human) * rate)
    dangerous = human["true_label"] == 1
    n_ai = int((df["stage2_pred_label"] == 1).sum())
    return {
        "n_escalated": len(df),
        "n_human_verdicts": len(human),
        "n_gated_human_verdicts": n_gate,
        "queue_size": n_ai + n_gate,
        "n_dangerous_errors": int(dangerous.sum()),
        "n_dangerous_errors_queued": int(dangerous.iloc[:n_gate].sum()),
    }


def main() -> int:
    df = load_split("val")

    if df["confidence"].isnull().any():
        n_missing = int(df["confidence"].isnull().sum())
        print(f"WARNING: {n_missing} rows have no confidence score (missing from stage2_confidence_val.csv)")

    needed_rate, gate_rate = calibrate_gate_rate(df)

    ai_flagged = df[df["stage2_pred_label"] == 1]

    human_cleared = human_verdicts_by_confidence(df)
    n_gate = int(len(human_cleared) * gate_rate)
    human_gated = human_cleared.head(n_gate)  # least confident fraction

    queue = pd.concat([ai_flagged, human_gated]).sample(frac=1, random_state=42).reset_index(drop=True)
    queue = queue.drop(columns=["confidence"])

    out_path = PROCESSED_DIR / "stage2_review_queue_val.csv"
    queue.to_csv(out_path, index=False)

    report: dict[str, Any] = {
        "calibrated_on": "val",
        "margin": GATE_MARGIN,
        "needed_rate": needed_rate,
        "gate_rate": gate_rate,
        "val": gate_report(df, gate_rate),
        "test_check": None,
    }
    test_ready = all((PROCESSED_DIR / f"stage2_{kind}_test.csv").exists() for kind in ("predictions", "confidence"))
    if test_ready:
        report["test_check"] = gate_report(load_split("test"), gate_rate)
    GATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")

    n_human_auto_cleared = len(human_cleared) - len(human_gated)
    print(f"Full escalated pool: {len(df)}")
    print(f"  AI-generated verdicts (all sent to review): {len(ai_flagged)}")
    print(f"  Needed gate rate on val: {needed_rate:.1%}, margin {GATE_MARGIN:.0%}, gate rate {gate_rate:.1%}")
    print(f"  Human-written verdicts, gated to review (least confident {gate_rate:.1%}): {len(human_gated)}")
    print(f"  Human-written verdicts auto-cleared (no review, no action taken): {n_human_auto_cleared}")
    print(f"Review queue size: {len(queue)} ({len(queue)/len(df):.1%} of full pool)")
    check = report["test_check"]
    if check:
        print(
            f"Test check (not tuned on): {check['n_dangerous_errors_queued']}/{check['n_dangerous_errors']} "
            f"dangerous errors queued, queue size {check['queue_size']}/{check['n_escalated']}"
        )
    print(f"Wrote {out_path} and {GATE_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
