"""Clean, label, and split the raw DAIGT dataset.

Input:  data/raw/train_v2_drcat_02.csv (thedrcat/daigt-v2-train-dataset)
Output: data/processed/{train,val,test,held_out}.csv
        data/processed/dataset_summary.md  (schema + generation-source counts)

Schema of the raw file (44,868 rows, no nulls, no duplicate texts):
    text            str   the essay
    label           int   0 = human, 1 = AI-generated
    prompt_name     str   which essay prompt/topic was answered
    source          str   dataset provenance; for label==1 this is the
                           generating model, for label==0 it is either
                           "persuade_corpus" or "train_essays" (both human)
    RDizzl3_seven   bool  a community-added quality flag from the source
                           dataset, unrelated to our labeling; unused here

What this script does:
1. Drops 3 rows where source=="train_essays" and label==1 — these are
   AI-generated but the specific generating model is not recorded, so
   they cannot be assigned a generation source. (Everything else is
   unambiguous.)
2. Builds `generation_source`: "human" for all label==0 rows (collapsing
   the two human-provenance source values), else the model name from
   `source`.
3. Cleans text: strips leading/trailing whitespace, collapses internal
   \\r\\n to \\n. Adds basic metadata (char_count, word_count) that
   downstream stylometric feature extraction (stage1/features.py) will
   build on.
4. Holds out two entire generation sources (see HELD_OUT_SOURCES below)
   as unseen data, standing in for "a new AI model appears later" per
   CLAUDE.md — NOT real chronological drift, since this dataset has no
   timestamp axis. These rows are written to held_out.csv and excluded
   from train/val/test.
5. Splits the remaining ("seen") pool into train/val/test (70/15/15),
   stratified by generation_source so every remaining source — including
   the small ones — appears in all three splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "train_v2_drcat_02.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Two distinct, sizeable model families held out entirely from initial
# training/val/test. Reintroduced in Phase 6 (flywheel round 2) and used
# as the focal "new source" in Phase 7 (drift monitoring).
HELD_OUT_SOURCES = ["falcon_180b_v1", "llama_70b_v1"]

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    ambiguous = (df["source"] == "train_essays") & (df["label"] == 1)
    n_dropped = int(ambiguous.sum())
    df = df[~ambiguous].copy()

    df["generation_source"] = df["source"]
    df.loc[df["label"] == 0, "generation_source"] = "human"

    df["text"] = df["text"].str.replace("\r\n", "\n", regex=False).str.strip()

    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    df = df[["text", "label", "generation_source", "prompt_name", "char_count", "word_count"]]
    return df, n_dropped


def split_seen_pool(seen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(
        seen,
        train_size=TRAIN_FRAC,
        stratify=seen["generation_source"],
        random_state=RANDOM_STATE,
    )
    val_frac_of_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val, test = train_test_split(
        temp,
        train_size=val_frac_of_temp,
        stratify=temp["generation_source"],
        random_state=RANDOM_STATE,
    )
    return train, val, test


def write_summary(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    held_out: pd.DataFrame,
    n_dropped: int,
) -> str:
    lines = []
    lines.append("# DAIGT Dataset — Preprocessing Summary\n")
    lines.append(
        "Source: [thedrcat/daigt-v2-train-dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset), "
        "the community-compiled DAIGT dataset (human essays from the Persuade "
        "corpus + AI essays from a range of LLMs), used because the official "
        "competition training set lacks multi-model generation-source labels.\n"
    )
    lines.append("## Schema (post-processing)\n")
    lines.append("| column | type | meaning |")
    lines.append("|---|---|---|")
    lines.append("| text | str | essay text |")
    lines.append("| label | int | 0 = human, 1 = AI-generated |")
    lines.append("| generation_source | str | \"human\", or the generating model's name |")
    lines.append("| prompt_name | str | essay prompt/topic |")
    lines.append("| char_count | int | character length of text |")
    lines.append("| word_count | int | whitespace-split word count |\n")

    lines.append(
        f"Dropped {n_dropped} row(s) with `source==\"train_essays\"` and `label==1`: "
        "AI-generated but the specific generating model is not recorded in the "
        "source data, so no `generation_source` could be assigned.\n"
    )

    lines.append(f"## Held-out generation sources (unseen at initial training)\n")
    lines.append(
        "Standing in for \"a new AI model appears later\" (this dataset has no "
        "real timestamp axis — see CLAUDE.md). Reintroduced in Phase 6, used as "
        "the focal case in Phase 7 drift monitoring.\n"
    )
    for src in HELD_OUT_SOURCES:
        n = int((df["generation_source"] == src).sum())
        lines.append(f"- `{src}`: {n} rows")
    lines.append("")

    lines.append("## Generation-source counts, full dataset\n")
    lines.append("| generation_source | label | count | held out |")
    lines.append("|---|---|---|---|")
    counts = df.groupby(["generation_source", "label"]).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=False)
    for _, row in counts.iterrows():
        held = "yes" if row["generation_source"] in HELD_OUT_SOURCES else ""
        lines.append(f"| {row['generation_source']} | {row['label']} | {row['count']} | {held} |")
    lines.append("")

    lines.append("## Split sizes (seen pool only, held-out sources excluded)\n")
    lines.append("| split | rows | human | AI |")
    lines.append("|---|---|---|---|")
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_human = int((split_df["label"] == 0).sum())
        n_ai = int((split_df["label"] == 1).sum())
        lines.append(f"| {name} | {len(split_df)} | {n_human} | {n_ai} |")
    lines.append(f"| held_out | {len(held_out)} | {int((held_out['label']==0).sum())} | {int((held_out['label']==1).sum())} |")
    lines.append("")

    lines.append("## Generation-source counts by split\n")
    lines.append("| generation_source | train | val | test | held_out |")
    lines.append("|---|---|---|---|---|")
    all_sources = sorted(df["generation_source"].unique())
    for src in all_sources:
        n_train = int((train["generation_source"] == src).sum())
        n_val = int((val["generation_source"] == src).sum())
        n_test = int((test["generation_source"] == src).sum())
        n_held = int((held_out["generation_source"] == src).sum())
        lines.append(f"| {src} | {n_train} | {n_val} | {n_test} | {n_held} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not RAW_PATH.exists():
        print(f"Raw data not found at {RAW_PATH}. Run data/download.py first.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, n_dropped = load_and_clean(RAW_PATH)

    held_out_mask = df["generation_source"].isin(HELD_OUT_SOURCES)
    held_out = df[held_out_mask].copy()
    seen = df[~held_out_mask].copy()

    train, val, test = split_seen_pool(seen)

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    held_out.to_csv(OUT_DIR / "held_out.csv", index=False)

    summary = write_summary(df, train, val, test, held_out, n_dropped)
    (OUT_DIR / "dataset_summary.md").write_text(summary)

    print(summary)
    print(f"Wrote train/val/test/held_out CSVs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
