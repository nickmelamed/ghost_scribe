"""Teacher-facing HITL review queue (Streamlit).

Shows each essay in the confidence-gated review queue (from the val
split, kept separate from the test split so that stays untouched for
evaluation) along with Stage 2's suggested label and rationale, and lets
the teacher record the final determination. The queue is not the full
set of Stage-1-escalated essays: see stage2/build_review_queue.py. Every
Stage-2 "AI-generated" verdict is included regardless of confidence,
since a human must confirm before any accusatory action. Confident
"Human-written" verdicts are auto-cleared without review, because no
action is taken on those either way.

This is a decision-support tool, not an automated accusation system: no
verdict shown here is final until a human confirms it below, and nothing
in this app takes any action on a student's behalf. The ground-truth
label this project happens to know (since the dataset is simulated) is
deliberately dropped before display and never used anywhere in this file.
A real deployment wouldn't have it, and showing it here would defeat
the point of a blind review.

Run with:
    streamlit run hitl/review_app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
# stage2_review_queue_val.csv is the confidence-gated queue (see
# stage2/build_review_queue.py): every Stage-2 "AI-generated" verdict,
# plus only the least-confident "Human-written" verdicts. Confident
# auto-clears never enter this file at all, since no review is required
# for them (no action is taken either way). This is not the full
# escalated pool.
PREDICTIONS_PATH = ROOT / "data" / "processed" / "stage2_review_queue_val.csv"
REVIEWS_PATH = ROOT / "hitl" / "reviews.jsonl"

FAILURE_TAGS = [
    "None",
    "possible false flag: non-native writing style",
    "Stage 2 rationale doesn't match the essay",
    "Other / unsure",
]

st.set_page_config(page_title="AI-Text Detection — Teacher Review Queue", layout="centered")


@st.cache_data
def load_queue() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    # Drop the ground-truth label so it can never reach the screen.
    df = df.drop(columns=["true_label"])
    return df.reset_index(drop=True)


def load_reviewed_texts() -> set[str]:
    if not REVIEWS_PATH.exists():
        return set()
    reviewed = set()
    with REVIEWS_PATH.open() as f:
        for line in f:
            reviewed.add(json.loads(line)["text"])
    return reviewed


def append_review(record: dict) -> None:
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    st.title("AI-Text Detection — Teacher Review Queue")
    st.caption(
        "Decision-support only. Every essay below either got an \"AI-generated\" "
        "suggestion (always reviewed, regardless of confidence) or a "
        "low-confidence \"Human-written\" suggestion — nothing here is an "
        "automated accusation, and no action is taken until you confirm a "
        "determination."
    )

    queue = load_queue()
    reviewed_texts = load_reviewed_texts()
    in_queue_mask = queue["text"].isin(reviewed_texts)
    pending = queue[~in_queue_mask].reset_index(drop=True)
    n_reviewed_in_queue = int(in_queue_mask.sum())

    st.sidebar.metric("Reviewed", f"{n_reviewed_in_queue} / {len(queue)}")

    if len(pending) == 0:
        st.success("Queue complete — every essay in the review queue has been reviewed.")
        return

    row = pending.iloc[0]

    st.progress(n_reviewed_in_queue / len(queue))
    st.markdown(f"**Essay** ({int(row['word_count']) if 'word_count' in row else len(row['text'].split())} words)"
                if "word_count" in row else "**Essay**")
    st.text_area("Essay text", row["text"], height=350, disabled=True, label_visibility="collapsed")

    stage2_label = "AI-generated" if row["stage2_pred_label"] == 1 else "Human-written"
    st.info(f"**Stage 2 suggests: {stage2_label}**\n\nRationale: {row['stage2_rationale']}")
    st.caption("This is a model-generated suggestion based on stylometric cues — use your own judgment.")

    with st.form(key=f"review_form_{row.name}"):
        determination = st.radio(
            "Your final determination:",
            ["Human-written", "AI-generated"],
            index=0 if row["stage2_pred_label"] == 0 else 1,
        )
        failure_tag = st.selectbox("Flag a failure mode (optional):", FAILURE_TAGS)
        notes = st.text_input("Additional notes (optional):")
        submitted = st.form_submit_button("Submit & Next")

        if submitted:
            reviewer_label = 1 if determination == "AI-generated" else 0
            record = {
                "text": row["text"],
                "generation_source": row["generation_source"],
                "stage2_pred_label": int(row["stage2_pred_label"]),
                "stage2_rationale": row["stage2_rationale"],
                "reviewer_label": reviewer_label,
                "reviewer_agreed": reviewer_label == int(row["stage2_pred_label"]),
                "failure_tag": failure_tag,
                "notes": notes,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            append_review(record)
            st.rerun()


if __name__ == "__main__":
    main()
