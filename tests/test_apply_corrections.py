import json
import sys

import pandas as pd

from hitl import apply_corrections as ac

ESL_TAG = "possible false flag: non-native writing style"


def review(text, reviewer_label, agreed, tag="None", source="human"):
    return {
        "text": text,
        "generation_source": source,
        "stage2_pred_label": 1 - reviewer_label if not agreed else reviewer_label,
        "reviewer_label": reviewer_label,
        "reviewer_agreed": agreed,
        "failure_tag": tag,
    }


def setup(monkeypatch, tmp_path, reviews, argv=("apply_corrections",)):
    reviews_path = tmp_path / "reviews.jsonl"
    if reviews is not None:
        reviews_path.write_text("".join(json.dumps(r) + "\n" for r in reviews))
    meta = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "prompt_name": ["p1", "p2", "p3"],
            "char_count": [1, 2, 3],
            "word_count": [1, 1, 1],
        }
    )
    meta.to_csv(tmp_path / "val_features.csv", index=False)
    monkeypatch.setattr(ac, "SOURCES", {"human": reviews_path, "llm": tmp_path / "llm.jsonl"})
    monkeypatch.setattr(ac, "VAL_FEATURES_PATH", tmp_path / "val_features.csv")
    monkeypatch.setattr(ac, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", list(argv))


def test_label_comes_from_the_reviewer_not_from_stage_two(monkeypatch, tmp_path, capsys):
    setup(monkeypatch, tmp_path, [review("a", 0, agreed=False), review("b", 1, agreed=True)])
    assert ac.main() == 0
    out = pd.read_csv(tmp_path / "data" / "processed" / "hitl_corrections_human.csv")
    assert out["label"].tolist() == [0, 1]
    assert out.columns.tolist() == ["text", "label", "generation_source", "prompt_name", "char_count", "word_count"]
    assert out["prompt_name"].tolist() == ["p1", "p2"]


def test_latest_decision_per_essay_wins(monkeypatch, tmp_path, capsys):
    setup(monkeypatch, tmp_path, [review("a", 1, agreed=True), review("a", 0, agreed=False)])
    assert ac.main() == 0
    out = pd.read_csv(tmp_path / "data" / "processed" / "hitl_corrections_human.csv")
    assert len(out) == 1
    assert out["label"].tolist() == [0]


def test_summary_counts_overturned_and_esl_tagged(monkeypatch, tmp_path, capsys):
    setup(
        monkeypatch,
        tmp_path,
        [review("a", 0, agreed=False, tag=ESL_TAG), review("b", 1, agreed=True), review("c", 0, agreed=False)],
    )
    assert ac.main() == 0
    text = capsys.readouterr().out
    assert "overturned Stage 2's suggestion: 2/3" in text
    assert "non-native writing style': 1/3" in text


def test_missing_reviews_file_returns_one_and_points_to_the_app(monkeypatch, tmp_path, capsys):
    setup(monkeypatch, tmp_path, None)
    assert ac.main() == 1
    assert "streamlit run hitl/review_app.py" in capsys.readouterr().out


def test_empty_reviews_file_returns_one(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path, [])
    assert ac.main() == 1
