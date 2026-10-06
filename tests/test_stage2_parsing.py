import pytest

from hitl import llm_review
from stage2 import infer, prepare_finetune_data


def test_infer_parse_output_reads_label_and_rationale():
    label, rationale = infer.parse_output("Label: AI-generated\nRationale: Sentences are uniform.")
    assert label == 1
    assert rationale == "Sentences are uniform."


def test_infer_parse_output_is_case_insensitive_and_maps_human_to_zero():
    label, rationale = infer.parse_output("label: human-written\nrationale: varied sentence length")
    assert label == 0
    assert rationale == "varied sentence length"


def test_infer_parse_output_unparseable_gives_none_and_empty_rationale():
    assert infer.parse_output("I cannot tell.") == (None, "")


def test_infer_parse_output_keeps_label_when_rationale_missing():
    assert infer.parse_output("Label: Human-written") == (0, "")


def test_infer_parse_output_takes_the_first_label_mentioned():
    label, _ = infer.parse_output("Label: Human-written\nRationale: not Label: AI-generated")
    assert label == 0


def test_llm_parse_output_reads_all_three_fields():
    raw = "Analysis: Repetitive phrasing.\nDetermination: AI-generated\nFailure tag: None"
    assert llm_review.parse_output(raw) == (1, "None", "Repetitive phrasing.")


def test_llm_parse_output_hedged_label_still_counts():
    label, _, _ = llm_review.parse_output("Determination: Possible AI-generated\nFailure tag: None")
    assert label == 1


def test_llm_parse_output_both_or_neither_label_is_none():
    assert llm_review.parse_output("Determination: AI-generated or Human-written")[0] is None
    assert llm_review.parse_output("Determination: unsure")[0] is None
    assert llm_review.parse_output("no structure at all") == (None, "None", "")


@pytest.mark.parametrize(
    "raw_tag,expected",
    [
        ("'possible false flag: non-native writing style'", "possible false flag: non-native writing style"),
        ("something invented", "Other / unsure"),
        ("", "None"),
    ],
)
def test_llm_parse_output_normalizes_failure_tags(raw_tag, expected):
    raw = f"Determination: Human-written\nFailure tag: {raw_tag}"
    assert llm_review.parse_output(raw)[1] == expected


def test_llm_analysis_stops_at_determination():
    raw = "Analysis: first line\nsecond line\nDetermination: Human-written"
    assert llm_review.parse_output(raw)[2] == "first line\nsecond line"


def test_llm_build_user_message_truncates_long_essays_only():
    short = llm_review.build_user_message("short essay", "AI-generated", "because")
    assert "[...]" not in short
    assert short.endswith("Other model's suggestion: AI-generated — because")
    long = llm_review.build_user_message("x" * 2000, "Human-written", "r")
    assert ("x" * llm_review.MAX_ESSAY_CHARS + " [...]") in long
    assert "x" * (llm_review.MAX_ESSAY_CHARS + 1) not in long


def test_finetune_build_user_message_truncates_at_limit():
    msg = prepare_finetune_data.build_user_message("y" * 1600)
    assert msg == "Essay:\n" + "y" * 1500 + " [...]"
    assert prepare_finetune_data.build_user_message("short") == "Essay:\nshort"
