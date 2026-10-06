import json

import pytest

from evaluation.round2 import cost_per_decision
from tests.test_round2 import write_round1_results
from viz import trend_plots as tp

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def round2_summary():
    return {
        "escalation_rate": 0.18,
        "review_rate": 0.1,
        "gap_stage1": 0.015,
        "gap_combined": 0.0,
        "cost": cost_per_decision(0.18, 0.1),
    }


def test_without_round_two_results_the_second_round_is_none(tmp_path):
    write_round1_results(tmp_path)
    round1, round2 = tp.load_rounds(tmp_path)
    assert round2 is None
    assert round1["escalation_rate"] == 0.2


def test_with_round_two_results_both_rounds_come_from_that_file(tmp_path):
    write_round1_results(tmp_path)
    saved = {"round1": {"escalation_rate": 0.3}, "round2": round2_summary()}
    (tmp_path / "round2.json").write_text(json.dumps(saved))
    round1, round2 = tp.load_rounds(tmp_path)
    assert round1 == {"escalation_rate": 0.3}
    assert round2["review_rate"] == 0.1


def test_series_leaves_a_pending_round_empty():
    assert tp.series({"x": 1.0}, None, lambda r: r["x"]) == [1.0, None]
    assert tp.series({"x": 1.0}, {"x": 2.0}, lambda r: r["x"]) == [1.0, 2.0]


@pytest.mark.parametrize("with_round_two", [False, True])
def test_main_writes_three_png_charts(tmp_path, with_round_two):
    write_round1_results(tmp_path)
    if with_round_two:
        from evaluation.round2 import round1_summary

        saved = {"round1": round1_summary(tmp_path), "round2": round2_summary()}
        (tmp_path / "round2.json").write_text(json.dumps(saved))
    out = tmp_path / "figures"
    paths = tp.main(tmp_path, out)
    assert [p.name for p in paths] == ["rates.png", "fairness_gap.png", "cost_per_decision.png"]
    for path in paths:
        data = path.read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert len(data) > 5000
