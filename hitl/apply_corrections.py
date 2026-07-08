"""Turn raw HITL review decisions (hitl/reviews.jsonl) into labeled
training examples for Phase 6 (flywheel round 2).

The teacher's final determination (reviewer_label) becomes the label —
never Stage 2's suggestion and never this project's internal ground
truth (which the review app never even sees; see hitl/review_app.py).
This is the actual point of human-in-the-loop: the human's call is what
gets learned from, whether or not they agreed with the model.

Output: data/processed/hitl_corrections.csv, schema-matched to
data/processed/train.csv (text, label, generation_source, prompt_name,
char_count, word_count) so Phase 6 can concatenate it directly onto the
training set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REVIEWS_PATH = ROOT / "hitl" / "reviews.jsonl"
VAL_FEATURES_PATH = ROOT / "data" / "processed" / "val_features.csv"
OUT_PATH = ROOT / "data" / "processed" / "hitl_corrections.csv"


def main() -> int:
    if not REVIEWS_PATH.exists():
        print(f"No reviews found at {REVIEWS_PATH}. Run `streamlit run hitl/review_app.py` first.")
        return 1

    reviews = pd.DataFrame([json.loads(l) for l in REVIEWS_PATH.open()])
    if reviews.empty:
        print("Reviews file exists but is empty — nothing to apply.")
        return 1

    # Keep only the latest decision per essay, in case something was re-reviewed.
    reviews = reviews.drop_duplicates(subset="text", keep="last")

    val_meta = pd.read_csv(VAL_FEATURES_PATH)[["text", "prompt_name", "char_count", "word_count"]]
    merged = reviews.merge(val_meta, on="text", how="left")

    out = pd.DataFrame(
        {
            "text": merged["text"],
            "label": merged["reviewer_label"],
            "generation_source": merged["generation_source"],
            "prompt_name": merged["prompt_name"],
            "char_count": merged["char_count"],
            "word_count": merged["word_count"],
        }
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n = len(out)
    n_overturned = int((~merged["reviewer_agreed"]).sum())
    n_esl_flag = int((merged["failure_tag"] == "possible false flag: non-native writing style").sum())

    print(f"Applied {n} reviewed corrections -> {OUT_PATH}")
    print(f"  overturned Stage 2's suggestion: {n_overturned}/{n} ({n_overturned/n:.1%})")
    print(f"  tagged 'possible false flag: non-native writing style': {n_esl_flag}/{n} ({n_esl_flag/n:.1%})")
    print("  failure tag breakdown:")
    for tag, count in merged["failure_tag"].value_counts().items():
        print(f"    {tag}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
