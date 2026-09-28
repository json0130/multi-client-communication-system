"""
tests/test_paper_live_stats.py
==============================
The paper's live figures come only from live, full-tour runs, and every other
kind of run is counted rather than silently dropped.
"""

from tools.paper_live_stats import split_by_kind


def test_only_live_full_tours_are_kept_and_everything_is_counted():
    runs = {"r1": {"mode": "live", "format": "full_tour"},
            "r2": {"mode": "live", "format": "solo"},
            "r3": {"mode": "gazebo", "format": "full_tour"}}
    rows = [{"run_id": "r1"}, {"run_id": "r1"}, {"run_id": "r2"},
            {"run_id": "r3"}, {"run_id": "gone"}]
    kept, counts = split_by_kind(rows, runs)
    assert kept == [{"run_id": "r1"}, {"run_id": "r1"}]
    assert counts == {("live", "full_tour"): 2, ("live", "solo"): 1,
                      ("gazebo", "full_tour"): 1, ("unknown", "unknown"): 1}
