"""Build Stage-2 fine-tuning/eval data from Stage-1's escalated (ambiguous
middle-band) cases only — never the full dataset (see CLAUDE.md Do NOT).

For each split (train/val/test), scores rows with the Stage-1 XGBoost
model, applies the calibrated escalation bands, and keeps only the
"escalate" rows. Escalated train rows get a supervised fine-tuning target
(true label + a short templated rationale); escalated val/test rows are
saved without a target completion, for use at inference time (Phase 3
infer.py) and combined-pipeline evaluation (Phase 4).

Rationale generation is deterministic and grounded in real Stage-1
statistics, NOT model-generated or fabricated: for each escalated row, we
pick the 2 stylometric features (among the 4 most important to the
XGBoost model) whose value most strongly points, in the direction the
training population associates with that row's true label, and phrase a
one-sentence cue from them. This is a templated heuristic explanation for
teacher context, not a verified causal or linguistic analysis — documented
as such in the README ethics section.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stage1.escalation_policy import assign_band
from stage1.features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "stage1" / "models"
OUT_DIR = ROOT / "stage2" / "data"

MAX_ESSAY_CHARS = 1500
N_IMPORTANT_FEATURES = 4
N_CUES_IN_RATIONALE = 2

FEATURE_DISPLAY_NAMES = {
    "char_count": "essay length (characters)",
    "word_count": "essay length (words)",
    "sentence_count": "number of sentences",
    "avg_word_length": "average word length",
    "avg_sentence_length": "average sentence length",
    "sentence_length_std": "sentence-length variability",
    "type_token_ratio": "vocabulary diversity (type-token ratio)",
    "bigram_burstiness": "word-pair repetition burstiness",
    "noun_ratio": "noun usage rate",
    "verb_ratio": "verb usage rate",
    "adj_ratio": "adjective usage rate",
    "adv_ratio": "adverb usage rate",
    "pronoun_ratio": "pronoun usage rate",
    "det_ratio": "determiner usage rate",
    "adp_ratio": "preposition usage rate",
    "conj_ratio": "conjunction usage rate",
    "comma_rate": "comma frequency",
    "period_rate": "period frequency",
    "exclamation_rate": "exclamation-mark frequency",
    "question_rate": "question-mark frequency",
    "semicolon_rate": "semicolon frequency",
    "flesch_reading_ease": "Flesch reading-ease score",
    "flesch_kincaid_grade": "Flesch-Kincaid grade level",
    "gunning_fog": "Gunning Fog readability index",
}

SYSTEM_PROMPT = (
    "You are an assistant supporting a teacher's academic-integrity review. "
    "You will be shown a student essay that a first-pass classifier could "
    "not confidently categorize. Decide whether the essay is more likely "
    "human-written or AI-generated, and briefly explain which stylistic "
    "cues suggest this. This is a decision-support suggestion only — a "
    "human teacher must review and confirm any final determination. "
    "Respond in exactly this format:\n"
    "Label: <Human-written or AI-generated>\n"
    "Rationale: <one sentence>"
)


def build_user_message(text: str) -> str:
    truncated = text[:MAX_ESSAY_CHARS]
    suffix = " [...]" if len(text) > MAX_ESSAY_CHARS else ""
    return f"Essay:\n{truncated}{suffix}"


def _feature_direction_for_ai(train_df: pd.DataFrame) -> dict[str, int]:
    """+1 if higher values of a feature are more typical of AI text in the
    training population, -1 if lower values are."""
    means_ai = train_df.loc[train_df["label"] == 1, FEATURE_COLUMNS].mean()
    means_human = train_df.loc[train_df["label"] == 0, FEATURE_COLUMNS].mean()
    return {f: (1 if means_ai[f] > means_human[f] else -1) for f in FEATURE_COLUMNS}


def build_rationale(
    row: pd.Series,
    label: int,
    top_features: list[str],
    direction_for_ai: dict[str, int],
    pop_mean: pd.Series,
    pop_std: pd.Series,
) -> str:
    label_direction = 1 if label == 1 else -1  # which direction supports this row's true label
    scored = []
    for f in top_features:
        z = (row[f] - pop_mean[f]) / (pop_std[f] if pop_std[f] > 0 else 1.0)
        support = z * direction_for_ai[f] * label_direction  # higher = stronger support for true label
        scored.append((support, f, z))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[:N_CUES_IN_RATIONALE]

    label_word = "AI-generated" if label == 1 else "human-written"
    clauses = []
    for _, f, z in chosen:
        direction_word = "higher" if z > 0 else "lower"
        clauses.append(f"{FEATURE_DISPLAY_NAMES[f]} is {direction_word} than typical")
    cue_text = " and ".join(clauses)
    return f"{cue_text}, a pattern the training data associates more with {label_word} essays."


def score_and_band(df: pd.DataFrame, xgb, t_low: float, t_high: float) -> pd.DataFrame:
    proba = xgb.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    bands = assign_band(proba, t_low, t_high)
    df = df.copy()
    df["stage1_proba"] = proba
    df["stage1_band"] = bands
    return df


def main() -> int:
    xgb = joblib.load(MODELS_DIR / "xgb.joblib")
    thresholds = json.loads((MODELS_DIR / "escalation_thresholds.json").read_text())
    t_low, t_high = thresholds["t_low"], thresholds["t_high"]

    train_df = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    direction_for_ai = _feature_direction_for_ai(train_df)
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS)
    top_features = importances.sort_values(ascending=False).head(N_IMPORTANT_FEATURES).index.tolist()
    pop_mean = train_df[FEATURE_COLUMNS].mean()
    pop_std = train_df[FEATURE_COLUMNS].std()

    print(f"Top-{N_IMPORTANT_FEATURES} XGBoost features used for rationale cues: {top_features}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test", "esl_eval"]:
        df = pd.read_csv(PROCESSED_DIR / f"{split}_features.csv")
        df = score_and_band(df, xgb, t_low, t_high)
        escalated = df[df["stage1_band"] == "escalate"].reset_index(drop=True)

        records = []
        for _, row in escalated.iterrows():
            user_msg = build_user_message(row["text"])
            record = {
                "text": row["text"],
                "label": int(row["label"]),
                "generation_source": row["generation_source"],
                "system": SYSTEM_PROMPT,
                "user": user_msg,
            }
            if split == "train":
                rationale = build_rationale(row, int(row["label"]), top_features, direction_for_ai, pop_mean, pop_std)
                label_word = "AI-generated" if row["label"] == 1 else "Human-written"
                record["target"] = f"Label: {label_word}\nRationale: {rationale}"
            records.append(record)

        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(records)} escalated rows -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
