"""Teacher-facing HITL review queue (Streamlit).

Shows each Stage-1-escalated essay (from the val split — kept separate
from the test split, which stays untouched for evaluation) along with
Stage 2's suggested label and rationale, and lets the teacher record the
final determination.

This is a decision-support tool, not an automated accusation system: no
verdict shown here is final until a human confirms it below, and nothing
in this app takes any action on a student's behalf. The ground-truth
label this project happens to know (since the dataset is simulated) is
deliberately dropped before display and never used anywhere in this file
— a real deployment wouldn't have it, and showing it here would defeat
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
PREDICTIONS_PATH = ROOT / "data" / "processed" / "stage2_predictions_val.csv"
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
    # Ground truth exists only because this project's data is simulated.
    # Drop it immediately so it can never end up on screen or in a
    # decision — a real teacher reviewing real essays wouldn't have it.
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
        "Decision-support only. Every essay below was flagged as ambiguous by the "
        "automated pipeline and requires your review — nothing here is an automated "
        "accusation, and no action is taken until you confirm a determination."
    )

    queue = load_queue()
    reviewed_texts = load_reviewed_texts()
    pending = queue[~queue["text"].isin(reviewed_texts)].reset_index(drop=True)

    st.sidebar.metric("Reviewed", f"{len(reviewed_texts)} / {len(queue)}")

    if len(pending) == 0:
        st.success("Queue complete — every escalated essay has been reviewed.")
        return

    row = pending.iloc[0]

    st.progress(len(reviewed_texts) / len(queue))
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
