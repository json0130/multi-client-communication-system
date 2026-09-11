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


class TestRowsForOneStepAreCombined:
    """
    demo_step_duration_stats groups by (step_id, block_robot_id), so one
    step_id can appear on several rows — transition_to_silbot_01 belongs to
    whichever block precedes Silbot, which changes with the robot ordering.
    FlowGraph.estimate looks steps up by step_id alone, so those rows have to
    be combined.

    Before this, a dict comprehension over the rows kept whichever Supabase
    returned last: the planner used a different duration for the same step
    depending on row order, and a well-observed row could be silently
    replaced by a barely-observed one.
    """

    def test_rows_for_the_same_step_are_run_weighted(self, durations_from):
        got = durations_from(_rows(
            ("transition_to_silbot_01", 9, 10.0),
            ("transition_to_silbot_01", 1, 20.0),
        ))
        # (9*10 + 1*20) / 10 = 11.0 — not 20.0, and not 15.0 either.
        assert got["transition_to_silbot_01"] == pytest.approx(11.0)

    def test_the_threshold_applies_to_the_combined_count(self, durations_from):
        # Two rows of two runs each is four observations of the same step.
        got = durations_from(_rows(
            ("transition_to_silbot_01", 2, 10.0),
            ("transition_to_silbot_01", 2, 14.0),
        ))
        assert got["transition_to_silbot_01"] == pytest.approx(12.0)

    def test_combined_rows_below_the_threshold_are_still_dropped(self, durations_from):
        got = durations_from(_rows(
            ("transition_to_silbot_01", 1, 10.0),
            ("transition_to_silbot_01", 1, 14.0),
        ))
        assert got == {}

    def test_row_order_does_not_change_the_answer(self, durations_from):
        a = durations_from(_rows(("s", 9, 10.0), ("s", 1, 20.0)))
        b = durations_from(_rows(("s", 1, 20.0), ("s", 9, 10.0)))
        assert a == b


class TestTheBudgetIsNotLearnedFromItsOwnConstants:
    """
    A window that ran out its own allocation measures the allocation, not the
    visitor.

    Half the first live corpus was exactly that: 23 of 46 windows closed by
    timeout, 19 of them at exactly 5.0s (ALREADY_ENGAGED_QA_SEC) and 3 at
    exactly 60.0s. Feeding those back made the estimator a closed loop —
    shrink a window to five seconds, record five seconds, pull the median
    down, shrink more windows. The pooled median over everything was 5.0s:
    the system had learned its own constant and was about to present it as a
    measurement of how long visitors want to talk.
    """

    def _rows(self, monkeypatch, rows):
        from data import demo_duration_repo
        monkeypatch.setattr(demo_duration_repo, "qa_windows", lambda **k: rows)
        return demo_duration_repo

    def _w(self, seconds, closed_by="policy", block="chatbox_01"):
        return {"seconds": seconds, "closed_by": closed_by,
                "block_robot_id": block}

    def test_timed_out_windows_do_not_move_the_estimate(self, monkeypatch):
        repo = self._rows(monkeypatch,
                          [self._w(30.0) for _ in range(20)]
                          + [self._w(5.0, "timeout") for _ in range(40)])
        got = repo.suggested_qa_budget()
        assert got["budget_sec"] == 30.0, "the estimator learned its own shrink"
        assert got["self_terminated"] == 40

    def test_they_are_counted_rather_than_silently_dropped(self, monkeypatch):
        # Right-censored, not junk: the visitor may well have wanted longer,
        # and how many were cut off is itself worth seeing.
        repo = self._rows(monkeypatch, [self._w(5.0, "timeout")] * 3)
        assert repo.suggested_qa_budget()["self_terminated"] == 3

    def test_the_gate_counts_usable_windows_not_all_of_them(self, monkeypatch):
        # 46 windows passed a threshold of 10 while only 22 were informative.
        repo = self._rows(monkeypatch,
                          [self._w(30.0) for _ in range(5)]
                          + [self._w(5.0, "timeout") for _ in range(100)])
        got = repo.suggested_qa_budget()
        assert got["basis"] == "default"
        assert got["windows"] == 5

    def test_the_open_floor_is_not_a_project_qa_window(self, monkeypatch):
        # It is the end-of-demo floor, a different animal. On the live corpus
        # its single window WAS the returned budget.
        repo = self._rows(monkeypatch,
                          [self._w(30.0) for _ in range(20)]
                          + [self._w(999.0, "policy", None)])
        got = repo.suggested_qa_budget()
        assert got["windows"] == 20
        assert got["budget_sec"] == 30.0

    def test_it_medians_windows_not_per_step_means(self, monkeypatch):
        # One busy step and one quiet one. Medianing the two STEP means gives
        # the midpoint of 10 and 100; medianing the windows gives what a
        # window actually looks like.
        repo = self._rows(monkeypatch,
                          [self._w(10.0) for _ in range(19)]
                          + [self._w(100.0, "policy", "navel_01")])
        assert repo.suggested_qa_budget()["budget_sec"] == 10.0

    def test_a_long_tail_does_not_drag_the_centre(self, monkeypatch):
        repo = self._rows(monkeypatch,
                          [self._w(20.0) for _ in range(20)] + [self._w(1000.0)])
        assert repo.suggested_qa_budget()["budget_sec"] == 20.0

    def test_too_little_evidence_says_so(self, monkeypatch):
        repo = self._rows(monkeypatch, [self._w(30.0) for _ in range(3)])
        got = repo.suggested_qa_budget(default_sec=90.0)
        assert (got["budget_sec"], got["basis"]) == (90.0, "default")

    def test_the_threshold_reflects_where_more_data_stops_helping(self):
        # Bootstrapped on the real corpus: the median's 95% interval is
        # 4.3-52.7s at n=10 and 9.7-42.6s at n=20, and barely moves after.
        from data.demo_duration_repo import MIN_WINDOWS_TO_TRUST
        assert MIN_WINDOWS_TO_TRUST >= 20

    def test_an_unreadable_row_does_not_take_the_estimate_with_it(self, monkeypatch):
        repo = self._rows(monkeypatch,
                          [self._w(30.0) for _ in range(20)]
                          + [{"seconds": None, "closed_by": "policy",
                              "block_robot_id": "chatbox_01"}])
        assert repo.suggested_qa_budget()["budget_sec"] == 30.0

    def test_per_block_medians_apply_the_same_exclusion(self, monkeypatch):
        # The loop guard has to hold everywhere a window length feeds a
        # figure that SETS window lengths, and the moment two estimators
        # apply it separately one of them drifts. _windows_by_block is the
        # one place it is written down.
        from data import demo_duration_repo
        rows = ([self._w(40.0, "policy", "silbot_01") for _ in range(6)]
                + [self._w(5.0, "timeout", "silbot_01") for _ in range(50)])
        monkeypatch.setattr(demo_duration_repo, "qa_windows", lambda **k: rows)
        assert demo_duration_repo.qa_median_by_block()["silbot_01"] == 40.0

    def test_a_thinly_observed_block_is_absent_rather_than_guessed(self, monkeypatch):
        # The planner can then tell "this block runs long" from "nothing is
        # known about this block", and default the second to the flat share.
        from data import demo_duration_repo
        rows = ([self._w(40.0, "policy", "silbot_01") for _ in range(6)]
                + [self._w(90.0, "policy", "navel_01")])
        monkeypatch.setattr(demo_duration_repo, "qa_windows", lambda **k: rows)
        got = demo_duration_repo.qa_median_by_block()
        assert "silbot_01" in got and "navel_01" not in got

    def test_the_open_floor_is_excluded_here_too(self, monkeypatch):
        from data import demo_duration_repo
        rows = [self._w(40.0, "policy", None) for _ in range(20)]
        monkeypatch.setattr(demo_duration_repo, "qa_windows", lambda **k: rows)
        assert demo_duration_repo.qa_median_by_block() == {}


class TestLimitHitWindowsCountAsAtLeastThatLong:
    """
    Dropping the windows that hit their time limit was safe while few windows
    had one. With a limit on every window from the start of the tour, the
    windows that hit it are the long ones, for the projects visitors want
    more of — so dropping them pushed exactly those projects' medians down,
    which shrank their next allocation, which made them hit the limit sooner.
    On the first allocated live runs, Navel came out at 39s dropped against
    57s counted as censored.
    """

    def _m(self, obs):
        from data.demo_duration_repo import censored_median
        return censored_median(obs)

    def test_without_censoring_it_is_the_ordinary_median(self):
        assert self._m([(10, True), (20, True), (30, True)]) == (20, False)
        assert self._m([(10, True), (20, True), (30, True), (40, True)]) == (25, False)

    def test_the_live_navel_case(self):
        obs = [(4, True), (18, True), (39, True), (57, True), (61, True),
               (60, False), (104, False)]
        median, lower = self._m(obs)
        assert median == 57 and lower is False

    def test_censoring_never_lowers_the_estimate_below_dropping_it(self):
        import random
        import statistics
        rng = random.Random(3)
        for _ in range(200):
            ended = [rng.uniform(5, 120) for _ in range(rng.randint(3, 12))]
            cut = [rng.uniform(5, 150) for _ in range(rng.randint(0, 8))]
            m, _ = self._m([(t, True) for t in ended] + [(t, False) for t in cut])
            # The ordinary median of the ended windows alone, lower-middle
            # convention to match Kaplan-Meier on an exact half.
            dropped = sorted(ended)[(len(ended) - 1) // 2]
            assert m >= dropped - 1e-9

    def test_all_the_long_windows_cut_off_gives_a_lower_bound(self):
        obs = [(10, True), (100, False), (120, False), (140, False)]
        median, lower = self._m(obs)
        assert lower is True and median == 140

    def test_short_cut_off_windows_say_almost_nothing(self):
        # Windows cut after 5s say only "longer than five seconds" — the old
        # corpus had 19 of them, and they must not drag the estimate.
        obs = [(30, True)] * 20 + [(5, False)] * 40
        assert self._m(obs) == (30, False)

    def test_nothing_to_estimate(self):
        assert self._m([]) == (None, False)

    def test_the_per_block_median_uses_the_cut_off_windows(self, monkeypatch):
        from data import demo_duration_repo
        rows = ([self._w(s) for s in (4, 18, 39, 57, 61)]
                + [self._w(s, "timeout") for s in (60, 104)])
        monkeypatch.setattr(demo_duration_repo, "qa_windows", lambda **k: rows)
        assert demo_duration_repo.qa_median_by_block()["navel_01"] == 57.0

    def test_cut_off_windows_do_not_count_toward_the_trust_threshold(self, monkeypatch):
        # Four visitor-ended windows plus any number cut off is still four.
        from data import demo_duration_repo
        rows = ([self._w(40) for _ in range(4)]
                + [self._w(90, "timeout") for _ in range(10)])
        monkeypatch.setattr(demo_duration_repo, "qa_windows", lambda **k: rows)
        assert "navel_01" not in demo_duration_repo.qa_median_by_block()

    def _w(self, seconds, closed_by="policy"):
        return {"seconds": seconds, "closed_by": closed_by,
                "block_robot_id": "navel_01"}

