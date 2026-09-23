"""
tests/test_plan_replay_harness.py
==================================
tools/plan_replay_harness.py — the tool built to diagnose "PLAN_REVISE
sometimes skips a project, sometimes just shortens it" without a live demo.

Two things worth pinning here: that the determinism check actually catches a
real mismatch rather than always reporting "deterministic" (a check that
can't fail is not a check), and that the drift experiment reproduces the
reported symptom — the same scenario, same budget, same play-head position,
landing on a different rung purely because duration data moved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.plan_replay_harness import (
    check_determinism,
    diff_under_duration_drift,
    run_scenario,
)

GUIDE = "pepper_01"
PROJECTS = ["chatbox_01", "navel_01", "silbot_01"]

# Chosen so the tour comfortably overruns a 300s budget and lands mid-ladder
# (tighten Q&A, then compress) — not "already fits", not "impossible".
TIGHT_BUDGET = 300.0

# Chosen so the tour is close enough to DRIFT_BOUNDARY_BUDGET that a modest
# duration drift tips it from "tighten + compress" into "also skip one" —
# the exact ladder-rung transition the reported bug described.
def _block(robot):
    """Per-block durations for the current script shape: one merged
    intro+handoff, greeting, prompt, and the project talk split across the
    PROJECT_CHECKLIST points."""
    return {
        f"introduce_{robot}": 18,
        f"{robot}_greeting": 12,
        f"{robot}_prompt": 6,
        f"{robot}_project_problem": 20,
        f"{robot}_project_approach": 20,
        f"{robot}_project_impact": 15,
    }


DRIFT_BOUNDARY_BUDGET = 485.0
"""A budget sitting just above the skip rung. Undrifted, the tour fits by
tightening Q&A and compressing; at +15% it no longer does and a project has
to go. Re-derived when the script gained the merged intro+handoff and the
per-checklist-point project steps — the old 420s sat comfortably inside the
compress band under the new arithmetic and demonstrated nothing. Re-derived
again (460 -> 485) when build_script() grew two more navigation steps
(approach_visitors before the greeting, and the guide's own nav-in for
chatbox_01) neither of which has an entry in BOUNDARY_DURATIONS below, so
each falls back to DEFAULT_STEP_SEC and — being absent from the dict —
does NOT scale under diff_under_duration_drift's drift_pct multiply,
diluting the drift's effect and pushing the boundary up by more than the
steps' own combined ~24s."""

BOUNDARY_DURATIONS = {
    "greeting": 15, "lab_intro": 15, "overview": 10,
    **_block("chatbox_01"), **_block("navel_01"), **_block("silbot_01"),
    "transition_to_navel_01": 8, "transition_to_silbot_01": 8,
    "wrap_up": 15, "open_floor": 10,
}


def _boundary_scenario(**overrides) -> dict:
    base = dict(
        guide_id=GUIDE, project_ids=PROJECTS,
        budget_sec=DRIFT_BOUNDARY_BUDGET, elapsed_sec=0.0, at_robot=None, at_role="",
        durations=dict(BOUNDARY_DURATIONS), kg_edges=[], kg_links=[], topics=[],
    )
    base.update(overrides)
    return base


class TestRunScenario:
    def test_a_generous_budget_needs_no_ops(self):
        r = run_scenario(guide_id=GUIDE, project_ids=PROJECTS, budget_sec=10_000.0)
        assert r["ops"] == []
        assert r["fits_already"] is True

    def test_a_tight_budget_produces_ops(self):
        r = run_scenario(guide_id=GUIDE, project_ids=PROJECTS, budget_sec=TIGHT_BUDGET)
        assert r["ops"]
        assert r["estimate"]["total_sec"] <= TIGHT_BUDGET or not r["feasible"]

    def test_play_head_position_changes_the_remaining_graph(self):
        from_start = run_scenario(guide_id=GUIDE, project_ids=PROJECTS,
                                   budget_sec=TIGHT_BUDGET, at_robot=None, at_role="")
        from_last_qa = run_scenario(guide_id=GUIDE, project_ids=PROJECTS,
                                     budget_sec=TIGHT_BUDGET, at_robot="silbot_01", at_role="qa")
        # Parked at the last block's own Q&A, only its own steps remain —
        # nothing left to compress or skip ahead of it.
        assert from_last_qa["estimate"]["total_sec"] < from_start["estimate"]["total_sec"]

    def test_no_remaining_blocks_returns_no_ops_cleanly(self):
        r = run_scenario(guide_id=GUIDE, project_ids=[], budget_sec=1.0)
        assert r["ops"] == []


class TestDeterminismCheck:
    def test_the_pure_path_is_deterministic_by_construction(self):
        # plan_for_budget has no I/O and no sampling — this must always pass.
        # A failure here would mean a real bug in the pure planning path.
        det = check_determinism(_boundary_scenario(), n=8)
        assert det["deterministic"] is True
        assert det["mismatched_runs"] == []

    def test_a_genuinely_different_scenario_is_not_silently_reported_same(self):
        # Sanity check on the checker itself: two DIFFERENT scenarios must
        # not compare equal. If this failed, check_determinism's comparison
        # would be too coarse to catch a real mismatch.
        tight = check_determinism(_boundary_scenario(budget_sec=200.0), n=1)
        loose = check_determinism(_boundary_scenario(budget_sec=10_000.0), n=1)
        assert tight["ops"] != loose["ops"]


class TestDurationDriftExperiment:
    def test_drift_can_flip_the_rung_at_a_boundary_scenario(self):
        # Reproduces the reported symptom: identical visitor profile, budget,
        # and play-head position, but the rung fired differs purely because
        # duration data moved — exactly what accumulating more measured
        # samples between two real demo runs would do.
        drift = diff_under_duration_drift(_boundary_scenario(), drift_pct=0.15)
        assert drift["rung_changed"] is True
        after_kinds = [op["kind"] for op in drift["after"]["ops"]]
        assert "skip" in after_kinds

    def test_zero_drift_never_changes_the_rung(self):
        drift = diff_under_duration_drift(_boundary_scenario(), drift_pct=0.0)
        assert drift["rung_changed"] is False

    def test_a_scenario_nowhere_near_a_boundary_is_stable_under_drift(self):
        # A budget so generous that no plausible drift would matter — the
        # drift explanation should not overclaim everywhere, only near a
        # rung transition.
        roomy = _boundary_scenario(budget_sec=10_000.0)
        drift = diff_under_duration_drift(roomy, drift_pct=0.15)
        assert drift["rung_changed"] is False
