"""Stylometric / statistical feature extraction for Stage 1.

Stage 1 is deliberately classical: no language model is used, so there is
no true perplexity. Instead we approximate the "how surprising/repetitive
is this text" signal with an n-gram burstiness proxy (coefficient of
variation of within-document bigram frequencies) — documented here as an
approximation, not a substitute for LM perplexity.

Feature groups (see FEATURE_COLUMNS for the exact list):
  - length:        char_count, word_count, sentence_count,
                    avg_word_length, avg_sentence_length
  - variance:       sentence_length_std   (sentence-length variance proxy)
  - lexical:        type_token_ratio
  - burstiness:      bigram_burstiness    (n-gram repetition proxy for
                                            perplexity; see module docstring)
  - POS ratios:      noun_ratio, verb_ratio, adj_ratio, adv_ratio,
                      pronoun_ratio, det_ratio, adp_ratio, conj_ratio
  - punctuation:      comma_rate, period_rate, exclamation_rate,
                      question_rate, semicolon_rate   (per 100 words)
  - readability:      flesch_reading_ease, flesch_kincaid_grade,
                      gunning_fog
"""

from __future__ import annotations

from collections import Counter

import nltk
import numpy as np
import pandas as pd
import textstat
from tqdm import tqdm

FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "avg_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "bigram_burstiness",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "pronoun_ratio",
    "det_ratio",
    "adp_ratio",
    "conj_ratio",
    "comma_rate",
    "period_rate",
    "exclamation_rate",
    "question_rate",
    "semicolon_rate",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
]

_POS_GROUPS = {
    "noun_ratio": ("NN", "NNS", "NNP", "NNPS"),
    "verb_ratio": ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
    "adj_ratio": ("JJ", "JJR", "JJS"),
    "adv_ratio": ("RB", "RBR", "RBS"),
    "pronoun_ratio": ("PRP", "PRP$", "WP", "WP$"),
    "det_ratio": ("DT", "PDT", "WDT"),
    "adp_ratio": ("IN",),
    "conj_ratio": ("CC",),
}


def _bigram_burstiness(words: list[str]) -> float:
    """Coefficient of variation of within-document bigram frequencies.

    Higher = word pairs repeat unevenly (some bigrams reused a lot,
    others never), lower = more uniform bigram usage. A proxy for
    n-gram-level "burstiness" in lieu of true LM perplexity.
    """
    if len(words) < 3:
        return 0.0
    bigrams = list(zip(words[:-1], words[1:]))
    counts = np.array(list(Counter(bigrams).values()), dtype=float)
    if counts.mean() == 0:
        return 0.0
    return float(counts.std() / counts.mean())


def extract_features(text: str) -> dict[str, float]:
    text = text if isinstance(text, str) else ""
    sentences = nltk.sent_tokenize(text) if text.strip() else []
    words = nltk.word_tokenize(text) if text.strip() else []
    word_tokens = [w for w in words if any(c.isalpha() for c in w)]
    words_lower = [w.lower() for w in word_tokens]

    char_count = len(text)
    word_count = len(word_tokens)
    sentence_count = len(sentences)

    avg_word_length = (
        float(np.mean([len(w) for w in word_tokens])) if word_tokens else 0.0
    )
    sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences] if sentences else []
    avg_sentence_length = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sentence_length_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    type_token_ratio = (
        len(set(words_lower)) / len(words_lower) if words_lower else 0.0
    )
    bigram_burstiness = _bigram_burstiness(words_lower)

    pos_tags = [tag for _, tag in nltk.pos_tag(word_tokens)] if word_tokens else []
    pos_total = len(pos_tags) if pos_tags else 1
    pos_ratios = {
        name: sum(1 for t in pos_tags if t in tags) / pos_total
        for name, tags in _POS_GROUPS.items()
    }

    denom_words = max(word_count, 1)
    punctuation = {
        "comma_rate": 100 * text.count(",") / denom_words,
        "period_rate": 100 * text.count(".") / denom_words,
        "exclamation_rate": 100 * text.count("!") / denom_words,
        "question_rate": 100 * text.count("?") / denom_words,
        "semicolon_rate": 100 * text.count(";") / denom_words,
    }

    try:
        readability = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text),
        }
    except Exception:
        readability = {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
        }

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_std": sentence_length_std,
        "type_token_ratio": type_token_ratio,
        "bigram_burstiness": bigram_burstiness,
        **pos_ratios,
        **punctuation,
        **readability,
    }


def extract_features_df(texts: pd.Series, desc: str = "features") -> pd.DataFrame:
    rows = [extract_features(t) for t in tqdm(texts, desc=desc)]
    return pd.DataFrame(rows, index=texts.index)[FEATURE_COLUMNS]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
    splits = sys.argv[1:] or ["train", "val", "test", "held_out"]
    for split in splits:
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            print(f"skip {split}: {path} not found")
            continue
        df = pd.read_csv(path)
        feats = extract_features_df(df["text"], desc=split)
        out = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
        out_path = PROCESSED_DIR / f"{split}_features.csv"
        out.to_csv(out_path, index=False)
        print(f"{split}: wrote {out_path} ({out.shape})")
