"""
tests/test_movement_pricing.py
==============================
Navigation steps cost time only for a guide that can actually move.

Real robots ACK a navigation step at once; the Gazebo guide declares the
"navigation" module and really drives. Priced at DEFAULT_STEP_SEC regardless,
a real lab tour was estimated ~190s too long and the planner cut projects.
"""

from __future__ import annotations

import pytest

import app
from decision.flow import FlowGraph
from demo.demo_script import build_script

GUIDE = "pepper_01"
PROJECTS = ["chatbox_01", "navel_01", "silbot_01"]


class _Robot:
    def __init__(self, modules):
        self.enabled_modules = set(modules)


class _Registry:
    def __init__(self, guide_modules):
        self._guide = _Robot(guide_modules)

    def get(self, robot_id):
        return self._guide if robot_id == GUIDE else None


@pytest.fixture
def steps():
    return build_script(GUIDE, PROJECTS)


def _nav_ids(steps):
    return {s.step_id for s in steps if s.step_kind == "navigation"}


def test_a_guide_that_cannot_move_spends_no_time_moving(monkeypatch, steps):
    monkeypatch.setitem(app._registry_ref, "r", _Registry(["gpt", "rag"]))
    # A Gazebo-measured 40s must not leak into a real run either.
    measured = {"navigate_to_silbot_01": 40.0, "greeting": 15.0}
    priced = app._price_movement(measured, steps)
    assert all(priced[sid] == 0.0 for sid in _nav_ids(steps))
    assert priced["greeting"] == 15.0

    graph = FlowGraph.from_script(steps)
    before = graph.estimate({"greeting": 15.0})["total_sec"]
    assert before - graph.estimate(priced)["total_sec"] == 12.0 * len(_nav_ids(steps))


def test_a_guide_that_can_move_keeps_its_timings(monkeypatch, steps):
    monkeypatch.setitem(app._registry_ref, "r", _Registry(["gpt", "navigation"]))
    measured = {"navigate_to_silbot_01": 40.0}
    assert app._price_movement(measured, steps) == measured


def test_no_registry_yet_is_treated_as_cannot_move(monkeypatch, steps):
    monkeypatch.setitem(app._registry_ref, "r", None)
    priced = app._price_movement({}, steps)
    assert set(priced) == _nav_ids(steps)


def test_the_planners_own_step_view_carries_the_step_kind(monkeypatch, steps):
    """In production the planner prices obs.remaining_steps — StepRefs built
    from /demo/status dicts, not DemoSteps. Without step_kind on that path
    every budgeted plan raised AttributeError and fell back to the ladder."""
    from decision.observation import DemoRunTracker, build_observation
    status = {"state": "waiting_ack", "step_idx": 0, "step_id": steps[0].step_id,
              "steps": [{"step_id": s.step_id, "robot_id": s.robot_id, "role": s.role,
                         "qa_window": s.qa_window, "block_robot_id": s.block_robot_id,
                         "step_kind": s.step_kind} for s in steps]}
    obs = build_observation(status=status, registry=None, tracker=DemoRunTracker())
    monkeypatch.setitem(app._registry_ref, "r", _Registry(["gpt"]))
    priced = app._price_movement({}, obs.remaining_steps)
    assert set(priced) == _nav_ids(steps[1:]) and priced
