import json

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from data import dedup
from data.dedup import NEAR_DUPLICATE_THRESHOLD, ShingleIndex, near_duplicates, shingles


def essay(prefix, n=30, change=None):
    words = [f"{prefix}{i}" for i in range(n)]
    for i in change or []:
        words[i] = f"edited{i}"
    return " ".join(words) + "."


def test_threshold_and_window_are_the_documented_values():
    assert NEAR_DUPLICATE_THRESHOLD == 0.8
    assert dedup.SHINGLE_WORDS == 8


def test_shingles_ignore_case_and_punctuation():
    assert shingles("One two, THREE four five six seven eight nine.") == shingles(
        "one two three four five six seven eight nine"
    )


def test_text_shorter_than_one_window_has_no_shingles():
    assert shingles("only seven words in this short text") == set()
    assert len(shingles("one two three four five six seven eight")) == 1


def test_exact_repeat_is_full_overlap_even_when_short():
    index = ShingleIndex(["tiny essay"])
    assert index.max_overlap("tiny essay") == 1.0


def test_unrelated_essay_has_zero_overlap():
    assert ShingleIndex([essay("a")]).max_overlap(essay("b")) == 0.0


def test_partial_overlap_hand_worked():
    # 20 words give 13 windows. Sharing the first 10 words keeps 3 of them.
    reference = " ".join(f"w{i}" for i in range(20))
    query = " ".join([f"w{i}" for i in range(10)] + [f"x{i}" for i in range(10, 20)])
    assert ShingleIndex([reference]).max_overlap(query) == pytest.approx(3 / 13)


def test_overlap_is_measured_against_the_single_best_reference():
    a = " ".join(f"w{i}" for i in range(20))
    b = " ".join(f"v{i}" for i in range(20))
    mixed = " ".join([f"w{i}" for i in range(10)] + [f"v{i}" for i in range(10, 20)])
    assert ShingleIndex([a, b]).max_overlap(mixed) == pytest.approx(3 / 13)


def test_a_light_edit_still_counts_as_a_near_duplicate():
    # One changed word in the middle of 30 destroys 8 of 23 windows, leaving
    # 15/23 = 0.65, so it is NOT a near-duplicate at 0.8. A change at the very
    # end destroys only one window.
    index = ShingleIndex([essay("p")])
    one_edit_middle = essay("p", change=[15])
    one_edit_end = essay("p", change=[29])
    assert index.max_overlap(one_edit_middle) == pytest.approx(15 / 23)
    assert index.max_overlap(one_edit_end) == pytest.approx(22 / 23)


def test_near_duplicates_mask_is_inclusive_at_the_threshold():
    reference = ShingleIndex([essay("p", n=12)])
    query_at = essay("p", n=12, change=[11])
    df = pd.DataFrame({"text": [query_at, essay("q")]})
    assert reference.max_overlap(query_at) == pytest.approx(4 / 5)
    assert near_duplicates(df, [reference]).tolist() == [True, False]


def test_near_duplicates_checks_every_reference_index():
    one, two = ShingleIndex([essay("a")]), ShingleIndex([essay("b")])
    df = pd.DataFrame({"text": [essay("a"), essay("b"), essay("c")]})
    assert near_duplicates(df, [one, two]).tolist() == [True, True, False]


@settings(deadline=None, max_examples=50)
@given(st.lists(st.text(alphabet="abcdefg ", min_size=0, max_size=80), min_size=1, max_size=5), st.text(max_size=80))
def test_overlap_is_a_fraction_and_members_are_full_overlap(references, query):
    index = ShingleIndex(references)
    assert 0.0 <= index.max_overlap(query) <= 1.0
    for ref in references:
        assert index.max_overlap(ref) == 1.0


def write_csv(path, rows):
    pd.DataFrame(rows, columns=["text", "label", "generation_source"]).to_csv(path, index=False)


def setup_data(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    stage2 = tmp_path / "stage2"
    processed.mkdir()
    stage2.mkdir()
    train = [(essay("t1"), 0, "human"), (essay("t2"), 1, "gpt")]
    val = [(essay("t1", change=[29]), 0, "human"), (essay("v1"), 0, "human"), (essay("v2"), 1, "gpt")]
    test = [
        (essay("t2"), 1, "gpt"),
        (essay("v1", change=[29]), 0, "human"),
        (essay("x1"), 0, "human"),
        (essay("x2"), 1, "mistral"),
    ]
    esl = [(essay("t1"), 0, "human_esl_ellipse"), (essay("e1"), 0, "human_esl_ellipse")]
    write_csv(processed / "train.csv", train)
    write_csv(processed / "val.csv", val)
    write_csv(processed / "test.csv", test)
    write_csv(processed / "esl_eval.csv", esl)
    for name, rows in [("val", val), ("test", test), ("esl_eval", esl)]:
        pd.DataFrame({"text": [r[0] for r in rows], "word_count": 1}).to_csv(processed / f"{name}_features.csv", index=False)
        pd.DataFrame({"text": [r[0] for r in rows], "stage2_pred_label": 0}).to_csv(
            processed / f"stage2_predictions_{name}.csv", index=False
        )
        (stage2 / f"{name}.jsonl").write_text("".join(json.dumps({"text": r[0]}) + "\n" for r in rows))
    monkeypatch.setattr(dedup, "PROCESSED_DIR", processed)
    monkeypatch.setattr(dedup, "STAGE2_DATA_DIR", stage2)
    monkeypatch.setattr(dedup, "REPORT_PATH", tmp_path / "dedup_report.json")
    return processed, stage2


def test_main_drops_near_duplicates_from_val_test_and_esl_only(monkeypatch, tmp_path):
    processed, _ = setup_data(monkeypatch, tmp_path)
    assert dedup.main() == 0
    train = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "val.csv")
    test = pd.read_csv(processed / "test.csv")
    esl = pd.read_csv(processed / "esl_eval.csv")
    assert len(train) == 2
    assert val["text"].tolist() == [essay("v1"), essay("v2")]
    # test drops a train repeat and a near-duplicate of a kept val essay
    assert test["text"].tolist() == [essay("x1"), essay("x2")]
    assert esl["text"].tolist() == [essay("e1")]


def test_main_removes_the_same_essays_from_derived_files(monkeypatch, tmp_path):
    processed, stage2 = setup_data(monkeypatch, tmp_path)
    dedup.main()
    for name, expected in [("val", 2), ("test", 2), ("esl_eval", 1)]:
        assert len(pd.read_csv(processed / f"{name}_features.csv")) == expected
        assert len(pd.read_csv(processed / f"stage2_predictions_{name}.csv")) == expected
        assert len((stage2 / f"{name}.jsonl").read_text().splitlines()) == expected


def test_report_has_counts_by_label_and_source_and_no_essay_text(monkeypatch, tmp_path):
    setup_data(monkeypatch, tmp_path)
    dedup.main()
    raw = (tmp_path / "dedup_report.json").read_text()
    report = json.loads(raw)
    assert report["threshold"] == 0.8
    assert report["borderline_threshold"] == 0.5
    assert report["splits"]["val"] == {
        "n_before": 3,
        "n_removed": 1,
        "n_after": 2,
        "n_borderline_kept": 0,
        "removed_by_label": {"0": 1},
        "removed_by_source": {"human": 1},
    }
    assert report["splits"]["test"]["removed_by_label"] == {"1": 1, "0": 1}
    assert report["splits"]["esl_eval"]["removed_by_source"] == {"human_esl_ellipse": 1}
    assert "t1" not in raw


def test_second_run_removes_nothing_and_keeps_the_original_report(monkeypatch, tmp_path):
    setup_data(monkeypatch, tmp_path)
    dedup.main()
    first = (tmp_path / "dedup_report.json").read_text()
    assert dedup.main() == 0
    assert (tmp_path / "dedup_report.json").read_text() == first


def test_missing_derived_files_are_skipped(monkeypatch, tmp_path):
    processed, stage2 = setup_data(monkeypatch, tmp_path)
    (processed / "val_features.csv").unlink()
    (stage2 / "val.jsonl").unlink()
    assert dedup.main() == 0
    assert len(pd.read_csv(processed / "val.csv")) == 2


def test_borderline_rows_are_kept_but_counted(monkeypatch, tmp_path):
    processed, _ = setup_data(monkeypatch, tmp_path)
    # A middle edit leaves an overlap of 15/23 = 0.65 with a train essay.
    val = pd.read_csv(processed / "val.csv")
    val.loc[len(val)] = [essay("t2", change=[15]), 1, "gpt"]
    val.to_csv(processed / "val.csv", index=False)
    dedup.main()
    report = json.loads((tmp_path / "dedup_report.json").read_text())
    assert report["splits"]["val"]["n_borderline_kept"] == 1
    assert report["splits"]["val"]["n_removed"] == 1
    assert essay("t2", change=[15]) in pd.read_csv(processed / "val.csv")["text"].tolist()
