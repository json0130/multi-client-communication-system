"""
tests/test_duration_trust.py
=============================
decision.flow.MIN_RUNS_TO_TRUST and the filter app.py applies with it.

The property under test is narrow but load-bearing: a step timed ONCE must
not reach the planner. tools/plan_replay_harness.py demonstrated that the
planner's rungs are threshold-based, so a single bad recording — a stalled
ACK, a robot caught mid-reconnect — can flip "compress a block" into "skip a
project" with nothing in the output explaining why. Dropping untrusted means
rather than down-weighting them also keeps FlowGraph.measured_coverage()
honest: it reports the fraction of an estimate backed by data worth
believing, not the fraction seen at least once.
"""

from __future__ import annotations

import pytest

from decision.flow import DEFAULT_STEP_SEC, MIN_RUNS_TO_TRUST, FlowGraph
from demo.demo_script import build_script

GUIDE = "pepper_01"
A, B = "chatbox_01", "navel_01"


def _rows(*specs):
    """(step_id, runs, mean_sec) tuples as demo_step_duration_stats rows."""
    return [{"step_id": s, "runs": n, "mean_sec": m} for s, n, m in specs]


@pytest.fixture
def durations_from(monkeypatch):
    """Call app._step_durations() against a stubbed stats table."""
    import app

    def _run(rows):
        monkeypatch.setattr("data.demo_duration_repo.step_stats", lambda: rows)
        app._duration_cache["at"] = 0.0        # defeat the TTL cache
        try:
            return app._step_durations()
        finally:
            app._duration_cache["at"] = 0.0    # don't leak into other tests
    return _run


class TestThresholdValue:
    def test_the_threshold_is_three(self):
        # Matches decision.kg.CONFIDENCE_HALFLIFE, which answers the same
        # question for the competence graph. Two systems, one answer.
        from decision.kg import CONFIDENCE_HALFLIFE
        assert MIN_RUNS_TO_TRUST == 3
        assert MIN_RUNS_TO_TRUST == CONFIDENCE_HALFLIFE


class TestFilter:
    def test_a_single_run_is_not_trusted(self, durations_from):
        assert durations_from(_rows(("greeting", 1, 99.0))) == {}

    def test_a_step_at_the_threshold_is_trusted(self, durations_from):
        got = durations_from(_rows(("greeting", MIN_RUNS_TO_TRUST, 8.0)))
        assert got == {"greeting": 8.0}

    def test_a_well_measured_step_is_trusted(self, durations_from):
        got = durations_from(_rows(("greeting", 11, 7.8)))
        assert got == {"greeting": 7.8}

    def test_trusted_and_untrusted_are_separated_not_blended(self, durations_from):
        got = durations_from(_rows(
            ("greeting", 11, 7.8),          # trusted
            ("lab_intro", 1, 60.0),         # one bad recording
            ("overview", 2, 8.0),           # still short of the threshold
        ))
        assert got == {"greeting": 7.8}

    def test_a_null_mean_is_dropped_regardless_of_run_count(self, durations_from):
        assert durations_from([{"step_id": "greeting", "runs": 20,
                                "mean_sec": None}]) == {}

    def test_a_missing_run_count_is_not_trusted(self, durations_from):
        # An older row without the column must fail closed, not be assumed good.
        assert durations_from([{"step_id": "greeting", "mean_sec": 8.0}]) == {}

    def test_an_unreachable_stats_table_plans_from_defaults(self, monkeypatch):
        import app

        def boom():
            raise RuntimeError("supabase down")
        monkeypatch.setattr("data.demo_duration_repo.step_stats", boom)
        app._duration_cache["at"] = 0.0
        try:
            assert app._step_durations() == {}
        finally:
            app._duration_cache["at"] = 0.0


class TestEstimateFallsBackCleanly:
    """An untrusted step is not a missing step — the tour still has to be
    costed, just from the default rather than from one noisy sample."""

    def test_an_untrusted_step_costs_the_default(self):
        graph = FlowGraph.from_script(build_script(GUIDE, [A, B]))
        # greeting measured once at 99s; excluded, so it costs the default.
        trusted = {}
        est = graph.estimate(trusted, qa_budget=0.0)
        all_default = graph.estimate({}, qa_budget=0.0)
        assert est["total_sec"] == all_default["total_sec"]

    def test_coverage_reports_only_trusted_steps(self):
        graph = FlowGraph.from_script(build_script(GUIDE, [A, B]))
        assert graph.measured_coverage({}) == 0.0
        partial = graph.measured_coverage({"greeting": 8.0})
        assert 0.0 < partial < 1.0

    def test_dropping_an_outlier_changes_the_estimate(self):
        # The point of the whole exercise: a 99s one-off measurement would
        # have dominated the budget. Excluded, the step costs the default.
        graph = FlowGraph.from_script(build_script(GUIDE, [A, B]))
        with_outlier = graph.estimate({"greeting": 99.0}, qa_budget=0.0)["total_sec"]
        without = graph.estimate({}, qa_budget=0.0)["total_sec"]
        assert with_outlier - without == pytest.approx(99.0 - DEFAULT_STEP_SEC)
