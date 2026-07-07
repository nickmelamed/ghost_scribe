"""Build the non-native-English (ESL) fairness eval subset.

Source: the ELLIPSE Corpus (English Language Learner Insight, Proficiency
and Skills Evaluation), via the Kaggle mirror matthewjansen/ellipse-corpus
(CC-BY-NC-SA-4.0). This is a *real* labeled non-native-English corpus, not
a proxy: every essay was written by an 8th-12th grade English Language
Learner (ELL) in a US school as part of the ELLIPSE study, with human
proficiency ratings across six dimensions (cohesion, syntax, vocabulary,
phraseology, grammar, conventions). We only use the essay text and the
fact of ELL authorship — the proficiency scores themselves are not used
here.

All essays in this subset are genuinely human-written (label=0). Running
Stage 1 on them and checking how many get misclassified as AI-generated
gives a real false-positive-rate-on-non-native-English-writing measurement,
not an approximation of one — the approximation, if any, is that ELL
status is a proxy for "non-native English writer" in the broader sense
CLAUDE.md's fairness check is concerned with (ELL specifically means
enrolled in English-learner services in a US school; it does not cover
e.g. adult L2 English writers outside that system). This scope limitation
is documented in the README ethics section.

Output: data/processed/esl_eval.csv with columns matching the main
pipeline's schema (text, label, generation_source, prompt_name) so it can
be run through the same feature extraction and inference code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ellipse_corpus.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "esl_eval.csv"


def main() -> int:
    if not RAW_PATH.exists():
        print(
            f"Raw ELLIPSE corpus not found at {RAW_PATH}.\n"
            "Download it with:\n"
            "  kaggle datasets download -d matthewjansen/ellipse-corpus "
            f"-p {RAW_PATH.parent} --unzip\n"
            f"  mv {RAW_PATH.parent}/train.csv {RAW_PATH}"
        )
        return 1

    df = pd.read_csv(RAW_PATH)
    out = pd.DataFrame(
        {
            "text": df["full_text"].str.replace("\r\n", "\n", regex=False).str.strip(),
            "label": 0,
            "generation_source": "human_esl_ellipse",
            "prompt_name": df["prompt"],
        }
    )
    out["char_count"] = out["text"].str.len()
    out["word_count"] = out["text"].str.split().str.len()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} ESL eval rows to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
