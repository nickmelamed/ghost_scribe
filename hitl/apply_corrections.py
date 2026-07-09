"""Turn raw HITL review decisions into labeled training examples for
Phase 6 (flywheel round 2).

The reviewer's final determination (reviewer_label) becomes the label —
never Stage 2's suggestion and never this project's internal ground
truth (which neither the human review app nor the LLM reviewer ever see;
see hitl/review_app.py and hitl/llm_review.py). This is the actual point
of human-in-the-loop: the reviewer's call is what gets learned from,
whether or not they agreed with the model.

Supports two review sources, kept separate rather than merged, so Phase
6+ can compare a round retrained on human corrections against one
retrained on LLM-reviewer corrections:

    python -m hitl.apply_corrections --source human   # hitl/reviews.jsonl
    python -m hitl.apply_corrections --source llm      # hitl/llm_reviews.jsonl

Output: data/processed/hitl_corrections_{source}.csv, schema-matched to
data/processed/train.csv (text, label, generation_source, prompt_name,
char_count, word_count) so Phase 6 can concatenate it directly onto the
training set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL_FEATURES_PATH = ROOT / "data" / "processed" / "val_features.csv"

SOURCES = {
    "human": ROOT / "hitl" / "reviews.jsonl",
    "llm": ROOT / "hitl" / "llm_reviews.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), default="human")
    args = parser.parse_args()

    reviews_path = SOURCES[args.source]
    out_path = ROOT / "data" / "processed" / f"hitl_corrections_{args.source}.csv"

    if not reviews_path.exists():
        launch_hint = (
            "Run `streamlit run hitl/review_app.py` first."
            if args.source == "human"
            else "Run `python -m hitl.llm_review` first."
        )
        print(f"No reviews found at {reviews_path}. {launch_hint}")
        return 1

    reviews = pd.DataFrame([json.loads(l) for l in reviews_path.open()])
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    n = len(out)
    n_overturned = int((~merged["reviewer_agreed"]).sum())
    n_esl_flag = int((merged["failure_tag"] == "possible false flag: non-native writing style").sum())

    print(f"[{args.source}] Applied {n} reviewed corrections -> {out_path}")
    print(f"  overturned Stage 2's suggestion: {n_overturned}/{n} ({n_overturned/n:.1%})")
    print(f"  tagged 'possible false flag: non-native writing style': {n_esl_flag}/{n} ({n_esl_flag/n:.1%})")
    print("  failure tag breakdown:")
    for tag, count in merged["failure_tag"].value_counts().items():
        print(f"    {tag}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
