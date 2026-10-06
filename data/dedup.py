"""Remove near-duplicate essays from the evaluation splits.

Some human essays appear in both train and val/test, and many ELLIPSE essays
(the ESL fairness set) are already in train, labeled human. That inflates every
metric and understates the ESL false-positive rate. This step drops them.

An essay is a near-duplicate when at least NEAR_DUPLICATE_THRESHOLD of its
8-word shingles also occur in a single reference essay. Exact repeats are
always dropped. The rules, with train as the reference:

  - val:      drop rows that near-duplicate a train essay
  - test:     drop rows that near-duplicate a train essay or a kept val essay
  - esl_eval: drop rows that near-duplicate a train essay

Train and the held-out generation sources are left alone, because train is
the reference and the held-out sources show no overlap with it.

Light edits can slip under the cutoff, since one changed word in the middle of
an essay removes up to 8 windows. The report counts the rows kept with an
overlap between BORDERLINE_THRESHOLD and the cutoff, so that limit is visible.

The same essays are removed from the derived files (features, Stage-2
predictions and confidences, Stage-2 JSONL), matched by text. Kept rows are
unchanged, which is valid only while the Stage-1 models, the escalation
thresholds, and the Stage-2 adapter are unchanged. Counts per split, source
and label go to results/dedup_report.json, with no essay text.

Run after data.preprocess and fairness.data_prep_esl, and before
stage1.features:
    python -m data.dedup
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
STAGE2_DATA_DIR = ROOT / "stage2" / "data"
REPORT_PATH = ROOT / "results" / "dedup_report.json"

SHINGLE_WORDS = 8
NEAR_DUPLICATE_THRESHOLD = 0.8
BORDERLINE_THRESHOLD = 0.5
# Derived files that exist per split, keyed by the split they belong to.
DERIVED_CSV = ("{split}_features.csv", "stage2_predictions_{split}.csv", "stage2_confidence_{split}.csv")


def shingles(text: str, n: int = SHINGLE_WORDS) -> set[int]:
    """Return hashes of the n-word windows of ``text``, lowercased, punctuation ignored."""
    words = re.sub(r"\W+", " ", str(text).lower()).split()
    return {hash(" ".join(words[i : i + n])) for i in range(max(len(words) - n + 1, 0))}


class ShingleIndex:
    """Map shingles to the reference essays that contain them."""

    def __init__(self, texts: list[str]):
        self.texts = set(texts)
        self.by_shingle: dict[int, list[int]] = defaultdict(list)
        for doc_id, text in enumerate(texts):
            for s in shingles(text):
                self.by_shingle[s].append(doc_id)

    def max_overlap(self, text: str) -> float:
        """Return the largest share of ``text``'s shingles found in one reference essay."""
        if text in self.texts:
            return 1.0
        query = shingles(text)
        if not query:
            return 0.0
        shared: dict[int, int] = defaultdict(int)
        for s in query:
            for doc_id in self.by_shingle.get(s, ()):
                shared[doc_id] += 1
        return max(shared.values()) / len(query) if shared else 0.0


def max_overlaps(df: pd.DataFrame, indexes: list[ShingleIndex]) -> pd.Series:
    """Return each row's largest overlap with any essay in any of the indexes."""
    overlaps = [df["text"].map(index.max_overlap) for index in indexes]
    return pd.concat(overlaps, axis=1).max(axis=1)


def near_duplicates(df: pd.DataFrame, indexes: list[ShingleIndex], threshold: float = NEAR_DUPLICATE_THRESHOLD) -> pd.Series:
    """Return a boolean mask of rows in ``df`` that near-duplicate any indexed reference."""
    return max_overlaps(df, indexes) >= threshold


def drop_texts_from_csv(path: Path, texts: set[str]) -> int:
    """Rewrite a CSV without the rows whose ``text`` is in ``texts``. Return how many were removed."""
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    keep = ~df["text"].isin(texts)
    if (~keep).any():
        df[keep].to_csv(path, index=False)
    return int((~keep).sum())


def drop_texts_from_jsonl(path: Path, texts: set[str]) -> int:
    """Rewrite a JSONL file without the records whose ``text`` is in ``texts``."""
    if not path.exists():
        return 0
    records = [json.loads(line) for line in path.open()]
    kept = [r for r in records if r["text"] not in texts]
    if len(kept) != len(records):
        path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return len(records) - len(kept)


def summarize(df: pd.DataFrame, overlap: pd.Series) -> dict:
    removed = overlap >= NEAR_DUPLICATE_THRESHOLD
    gone = df[removed]
    borderline = (overlap >= BORDERLINE_THRESHOLD) & ~removed
    return {
        "n_before": len(df),
        "n_removed": len(gone),
        "n_after": len(df) - len(gone),
        "n_borderline_kept": int(borderline.sum()),
        "removed_by_label": {str(k): int(v) for k, v in gone["label"].value_counts().items()},
        "removed_by_source": {str(k): int(v) for k, v in gone["generation_source"].value_counts().items()},
    }


def main() -> int:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    esl = pd.read_csv(PROCESSED_DIR / "esl_eval.csv")

    train_index = ShingleIndex(train["text"].tolist())

    val_overlap = max_overlaps(val, [train_index])
    val_kept = val[val_overlap < NEAR_DUPLICATE_THRESHOLD]
    test_overlap = max_overlaps(test, [train_index, ShingleIndex(val_kept["text"].tolist())])
    esl_overlap = max_overlaps(esl, [train_index])

    splits = {"val": (val, val_overlap), "test": (test, test_overlap), "esl_eval": (esl, esl_overlap)}
    report: dict[str, Any] = {
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "borderline_threshold": BORDERLINE_THRESHOLD,
        "shingle_words": SHINGLE_WORDS,
        "reference": "train",
        "splits": {name: summarize(df, overlap) for name, (df, overlap) in splits.items()},
    }

    n_total = sum(info["n_removed"] for info in report["splits"].values())
    if n_total == 0:
        print("No near-duplicates found. Nothing removed, report left as it was.")
        return 0

    for name, (df, overlap) in splits.items():
        texts = set(df.loc[overlap >= NEAR_DUPLICATE_THRESHOLD, "text"])
        drop_texts_from_csv(PROCESSED_DIR / f"{name}.csv", texts)
        for pattern in DERIVED_CSV:
            drop_texts_from_csv(PROCESSED_DIR / pattern.format(split=name), texts)
        drop_texts_from_jsonl(STAGE2_DATA_DIR / f"{name}.jsonl", texts)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for name, info in report["splits"].items():
        print(
            f"{name}: removed {info['n_removed']} of {info['n_before']}, kept {info['n_after']}, "
            f"{info['n_borderline_kept']} kept with borderline overlap"
        )
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
