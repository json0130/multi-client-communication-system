"""
tests/test_duration_basis.py
============================
Which evidence the planner costs a tour with (app._duration_basis / _qa_basis):
a live run uses live observations once there are MIN_RUNS_TO_TRUST of them,
else the newest sim-to-real prior, else the default; a simulated run uses
simulated observations only. Every estimate names its source.
"""

from __future__ import annotations

import pytest

from decision.models import Action, PlanOp, PlanOpKind

STEPS = ("navel_001_greeting", "lab_intro", "navel_001_project_problem")


class _Step:
    def __init__(self, step_id):
        self.step_id = step_id


def _stat(step, mode, runs, mean, fmt="full_tour"):
    return {"step_id": step, "mode": mode, "format": fmt, "runs": runs, "mean_sec": mean}


@pytest.fixture
def basis(monkeypatch):
    import app

    def _run(stats, priors, live, steps=STEPS):
        monkeypatch.setattr("data.demo_duration_repo.step_stats_by_mode", lambda: stats)
        monkeypatch.setattr("data.demo_duration_repo.current_priors", lambda: priors)
        app._duration_cache.clear()
        app._duration_cache["at"] = 0.0
        try:
            return app._duration_basis(live, [_Step(s) for s in steps])
        finally:
            app._duration_cache.clear()
            app._duration_cache["at"] = 0.0
    return _run


PRIORS = [{"robot_id": "navel_001", "step_id": "navel_001_greeting", "kind": "step",
           "median_sec": 6.0, "n": 42, "version": 1},
          {"robot_id": "pepper_001", "step_id": "lab_intro", "kind": "step",
           "median_sec": 15.0, "n": 93, "version": 1}]


def test_a_live_run_with_no_live_data_uses_the_prior_and_says_so(basis):
    durations, sources = basis([], PRIORS, live=True)
    assert durations["navel_001_greeting"] == 6.0
    assert sources == {"navel_001_greeting": "prior#1", "lab_intro": "prior#1",
                       "navel_001_project_problem": "default"}


def test_enough_live_runs_replace_the_prior(basis):
    stats = [_stat("navel_001_greeting", "live", 3, 8.0)]
    durations, sources = basis(stats, PRIORS, live=True)
    assert durations["navel_001_greeting"] == 8.0 and sources["navel_001_greeting"] == "live"


def test_too_few_live_runs_keep_the_prior(basis):
    stats = [_stat("navel_001_greeting", "live", 2, 99.0)]
    durations, sources = basis(stats, PRIORS, live=True)
    assert durations["navel_001_greeting"] == 6.0 and sources["navel_001_greeting"] == "prior#1"


def test_a_simulated_run_never_uses_live_data_or_priors(basis):
    stats = [_stat("lab_intro", "gazebo", 5, 14.0), _stat("lab_intro", "live", 9, 99.0)]
    durations, sources = basis(stats, PRIORS, live=False)
    assert durations == {"lab_intro": 14.0}
    assert sources["lab_intro"] == "sim" and sources["navel_001_greeting"] == "default"


def test_simulated_evidence_never_reaches_a_live_run(basis):
    stats = [_stat("navel_001_project_problem", "mock", 50, 9.0)]
    _durations, sources = basis(stats, [], live=True)
    assert sources["navel_001_project_problem"] == "default"


def test_solo_and_unknown_runs_are_left_out(basis):
    stats = [_stat("lab_intro", "gazebo", 5, 14.0, fmt="solo"),
             _stat("lab_intro", "sim_unknown", 5, 14.0)]
    durations, _ = basis(stats, [], live=False)
    assert durations == {}


def test_the_qa_basis_falls_back_to_the_prior_only_for_live_runs(monkeypatch):
    import app
    monkeypatch.setattr("data.demo_duration_repo.qa_windows", lambda: [])
    monkeypatch.setattr("data.demo_run_repo.run_modes", lambda: {})
    monkeypatch.setattr("data.demo_duration_repo.current_priors", lambda: [
        {"robot_id": "navel_001", "step_id": "qa_invite_navel_001", "kind": "qa",
         "median_sec": 48.4, "n": 28, "version": 1}])
    measured, sources = app._qa_basis(True, ["navel_001", "pepper_001"])
    assert measured == {"navel_001": 48.4}
    assert sources == {"navel_001": "prior#1", "pepper_001": "default"}
    measured, sources = app._qa_basis(False, ["navel_001"])
    assert measured == {} and sources == {"navel_001": "default"}


def test_a_plans_basis_is_logged_with_it():
    ops = [PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id="navel_001", seconds=40)]
    basis = {"run_mode": "live", "steps": {"prior": 2}}
    assert Action.revise(ops, basis=basis).payload()["basis"] == basis
    assert "basis" not in Action.revise(ops).payload()
    assert Action.revise(ops, basis=basis) == Action.revise(ops)   # not part of identity
