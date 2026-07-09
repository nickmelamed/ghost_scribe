"""Build the actual human review queue from Stage 2's escalated verdicts,
using confidence-gated (asymmetric) sub-escalation.

Reviewing all ~1,280 Stage-1-escalated essays is unrealistic, and mostly
unnecessary: Stage 2 gets 99.2-99.5% of them right (Phase 4 /
stage2/confidence.py validation on the test split). But CLAUDE.md is
explicit that every escalated case requires human confirmation before
any action is taken, and forbids any automated-accusation workflow — so
confidence can only be used to skip review in the direction that takes
no action against a student:

  - "AI-generated" verdicts: ALWAYS included, regardless of confidence.
    This is the accusatory direction; skipping human review here would
    be exactly the automated-accusation pattern CLAUDE.md rules out.
  - "Human-written" verdicts: gated by confidence. Auto-clearing takes
    no action, so it's safe to skip review when Stage 2 is confident.
    Only the least-confident fraction is included.

The confidence threshold for the "Human-written" gate is calibrated
against the escalated TEST set (known ground truth): of Stage 2's
errors there in the dangerous direction (predicted human, actually AI —
the case this gate exists to catch), the lowest-confidence one sat at
the 16th percentile among all human-written predictions. TARGET_RATE=0.20
below adds margin above that.

Output: data/processed/stage2_review_queue_val.csv, same schema as
data/processed/stage2_predictions_val.csv (what hitl/review_app.py
reads) — just filtered down to the actual review queue.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# Fraction of "Human-written"-predicted escalated cases to still send to
# human review (the least confident ones). Calibrated on the test split
# — see module docstring; includes margin above the empirically-needed
# ~17%.
TARGET_HUMAN_GATE_RATE = 0.20


def main() -> int:
    predictions = pd.read_csv(PROCESSED_DIR / "stage2_predictions_val.csv")
    confidence = pd.read_csv(PROCESSED_DIR / "stage2_confidence_val.csv")[["text", "confidence"]]
    df = predictions.merge(confidence, on="text", how="left")

    if df["confidence"].isnull().any():
        n_missing = int(df["confidence"].isnull().sum())
        print(f"WARNING: {n_missing} rows have no confidence score (missing from stage2_confidence_val.csv)")

    ai_flagged = df[df["stage2_pred_label"] == 1]

    human_cleared = df[df["stage2_pred_label"] == 0].sort_values("confidence")
    n_gate = int(len(human_cleared) * TARGET_HUMAN_GATE_RATE)
    human_gated = human_cleared.head(n_gate)  # least confident fraction

    queue = pd.concat([ai_flagged, human_gated]).sample(frac=1, random_state=42).reset_index(drop=True)
    queue = queue.drop(columns=["confidence"])

    out_path = PROCESSED_DIR / "stage2_review_queue_val.csv"
    queue.to_csv(out_path, index=False)

    n_human_auto_cleared = len(human_cleared) - len(human_gated)
    print(f"Full escalated pool: {len(df)}")
    print(f"  AI-generated verdicts (all sent to review): {len(ai_flagged)}")
    print(f"  Human-written verdicts, gated to review (least confident {TARGET_HUMAN_GATE_RATE:.0%}): {len(human_gated)}")
    print(f"  Human-written verdicts auto-cleared (no review, no action taken): {n_human_auto_cleared}")
    print(f"Review queue size: {len(queue)} ({len(queue)/len(df):.1%} of full pool)")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
