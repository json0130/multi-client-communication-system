"""
Q&A time is shared out when the tour starts, and every decision says which
visitor the run was for.

The allocation used to run only when the planner did, which was only under
time pressure — so in four live runs no window in a relaxed tour ever got a
budget, and a stated interest changed nothing about Q&A. And the decision
log did not record the visitor profile, so the four runs could not be told
apart afterwards.
"""

from __future__ import annotations

import time

from decision import DecisionRecorder, MemoryDecisionSink
from decision.models import PlanOp, PlanOpKind
from decision.observation import DemoRunTracker, build_observation
from decision.visitor_profile import VisitorProfile
from demo.demo_orchestrator import DemoOrchestrator, DemoState
from demo.demo_script import build_script
from gateway.websocket_gateway import WebSocketGateway

GUIDE, A, B = "pepper_01", "chatbox_01", "navel_01"


class _Registry:
    def get(self, cid):
        return None

    def get_all(self, exclude_id=None):
        return []


def _wired(planner):
    sink = MemoryDecisionSink()
    gw = WebSocketGateway(_Registry(), recorder=DecisionRecorder(sink),
                          flow_planner=planner)
    gw.send_to_robot = lambda *a, **k: None
    orch = DemoOrchestrator(gw, recorder=DecisionRecorder(sink))
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)
    orch._state, orch._idx = DemoState.RUNNING, 0
    orch._started_at, orch._time_budget_sec = time.time(), 900.0
    orch._run_id, orch._revisions = "run-test", []
    orch._visitor_profile = VisitorProfile(interest_text="the emotion work",
                                           style="business", topics=("topic:emotion",))
    return gw, orch, sink


def _qa_timeouts(orch):
    return {s.block_robot_id: s.qa_timeout for s in orch._script
            if s.qa_window and s.block_robot_id}


class TestAllocationAtStart:
    def test_the_budgets_are_applied_before_the_first_window(self):
        plan = {"ops": [PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=A, seconds=40),
                        PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B, seconds=120)]}
        gw, orch, _ = _wired(lambda obs: plan)
        gw.plan_at_start()
        assert _qa_timeouts(orch) == {A: 40, B: 120}

    def test_only_qa_budgets_are_applied_nothing_is_cut(self):
        # Compressing and skipping are for a tour actually running late.
        plan = {"ops": [PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=A, seconds=40),
                        PlanOp(PlanOpKind.SKIP, robot_id=B)]}
        gw, orch, _ = _wired(lambda obs: plan)
        before = [s.step_id for s in orch._script]
        gw.plan_at_start()
        assert [s.step_id for s in orch._script] == before

    def test_the_allocation_is_logged_as_a_decision(self):
        plan = {"ops": [PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=A, seconds=40)]}
        gw, _orch, sink = _wired(lambda obs: plan)
        gw.plan_at_start()
        assert sink.decisions_by_mechanism().get("flow_planner_start") == 1

    def test_no_plan_means_no_change(self):
        # build_flow_plan returns None without a time budget.
        gw, orch, _ = _wired(lambda obs: None)
        before = _qa_timeouts(orch)
        gw.plan_at_start()
        assert _qa_timeouts(orch) == before

    def test_a_failing_planner_never_stops_the_tour(self):
        def boom(obs):
            raise RuntimeError("database down")
        gw, _orch, _ = _wired(boom)
        gw.plan_at_start()      # must not raise

    def test_starting_a_tour_runs_it(self):
        calls = []
        gw, orch, _ = _wired(lambda obs: calls.append(obs) or None)
        orch._state = DemoState.IDLE
        orch._run_loop = lambda: None           # don't actually run the tour
        orch.start(time_budget_sec=900)
        assert len(calls) == 1


class TestTheRunIsIdentifiable:
    def test_every_observation_carries_the_visitor_profile(self):
        _gw, orch, _ = _wired(lambda obs: None)
        obs = build_observation(status=orch.get_status(), registry=None,
                                tracker=DemoRunTracker())
        d = obs.as_dict()
        assert d["run_id"] == "run-test"
        assert d["visitor_style"] == "business"
        assert d["visitor_interest"] == "the emotion work"
        assert d["visitor_topics"] == ["topic:emotion"]

    def test_no_profile_is_recorded_as_empty_not_missing(self):
        _gw, orch, _ = _wired(lambda obs: None)
        orch._visitor_profile = None
        d = build_observation(status=orch.get_status(), registry=None,
                              tracker=DemoRunTracker()).as_dict()
        assert d["visitor_interest"] == "" and d["visitor_topics"] == []


class TestEachTourStartsFresh:
    """A server that had run one tour cut every Q&A window in the next to 5s:
    the engagement counts were never reset, so every project looked as if
    the visitor had already asked about it."""

    def test_questions_from_the_last_tour_do_not_carry_over(self):
        gw, orch, _ = _wired(lambda obs: None)
        gw.tracker.note_visitor_turn(A, "how does it work?")
        assert gw.tracker.engagement_for(A)["turns"] == 1
        orch._state = DemoState.IDLE
        orch._run_loop = lambda: None
        orch.start(time_budget_sec=900)
        assert gw.tracker.engagement_for(A)["turns"] == 0

    def test_so_the_next_tours_qa_is_not_shortened(self):
        gw, orch, _ = _wired(lambda obs: None)
        gw.tracker.note_visitor_turn(A, "how does it work?")
        orch._state = DemoState.IDLE
        orch._run_loop = lambda: None
        orch.start(time_budget_sec=900)
        step = next(s for s in orch._script if s.qa_window and s.block_robot_id == A)
        assert orch._shrink_if_already_engaged(step).qa_timeout == step.qa_timeout
