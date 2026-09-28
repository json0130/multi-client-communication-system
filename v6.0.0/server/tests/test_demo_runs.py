"""
tests/test_demo_runs.py
=======================
Schema v2 phase 2: every tour gets a demo_runs row, and the log rows written
during it name the run and say who made each correction.

The mode matters most: only 'live' runs may feed the planner's priors, so a
simulated or scripted tour that recorded itself as 'live' would quietly skew
the next real one.
"""

from __future__ import annotations

import pytest
from flask import Flask

import demo.demo_orchestrator as orch_mod
from decision import DecisionRecorder, MemoryDecisionSink
from decision.models import (
    Action, DecisionPoint, Observation, build_correction, build_decision,
)
from demo.demo_orchestrator import DemoOrchestrator, DemoState
from demo.demo_script import build_script
from gateway.demo_gateway import create_demo_gateway

GUIDE, A = "pepper_01", "chatbox_01"


class StubGateway:
    def send_to_robot(self, client_id, data):
        pass

    def generate_demo_step(self, robot_id, instruction):
        return instruction


@pytest.fixture
def runs(monkeypatch):
    monkeypatch.setattr(orch_mod, "_reset_gazebo_positions", lambda: None)
    rows, sink = [], MemoryDecisionSink()
    o = DemoOrchestrator(
        StubGateway(), recorder=DecisionRecorder(sink),
        session_context=lambda: {"scenario_id": "lab_demo", "session_id": "s1"},
        run_sink=rows.append,
        run_mode=lambda robots: "gazebo" if robots[0] == "sim_guide" else "mock",
    )
    o._run_loop = lambda: None      # don't actually run the tour
    return o, rows, sink


# ── Rows ──────────────────────────────────────────────────────────────────────

class TestEventRows:
    def test_a_decision_in_a_run_names_it_as_recorded(self):
        row = build_decision(DecisionPoint.QA_ADVANCE, Action.advance(), "m",
                             Observation(run_id="run-1")).as_row()
        assert row["run_id"] == "run-1" and row["run_id_source"] == "recorded"

    def test_a_decision_outside_a_run_names_none(self):
        row = build_decision(DecisionPoint.QA_ADVANCE, Action.advance(), "m",
                             Observation()).as_row()
        assert row["run_id"] is None and row["run_id_source"] is None

    def test_a_correction_is_human_unless_said_otherwise(self):
        row = build_correction(DecisionPoint.QA_ADVANCE, Action.advance(),
                               "operator").as_row()
        assert row["origin"] == "human" and row["run_id"] is None

    def test_a_correction_carries_run_and_origin(self):
        row = build_correction(DecisionPoint.QA_ADVANCE, Action.advance(), "operator",
                               run_id="run-1", origin="scripted").as_row()
        assert (row["run_id"], row["run_id_source"], row["origin"]) == \
            ("run-1", "recorded", "scripted")


# ── The run's lifecycle ───────────────────────────────────────────────────────

class TestRunRow:
    def test_start_writes_the_run(self, runs):
        o, rows, _ = runs
        o.start(robot_ids=[GUIDE, A], time_budget_sec=600, supervisor_id="jay")
        (row,) = rows
        assert row["id"] == o._run_id and row["id"].startswith("run-")
        assert row["mode"] == "mock" and row["scenario_id"] == "lab_demo"
        assert row["time_budget_sec"] == 600 and row["supervisor_id"] == "jay"
        assert "ended_at" not in row and row["started_at"]

    def test_an_explicit_mode_wins(self, runs):
        o, rows, _ = runs
        o.start(robot_ids=[GUIDE, A], mode="scripted_visitor")
        assert rows[0]["mode"] == "scripted_visitor"

    def test_the_default_mode_comes_from_the_runs_robots(self, runs):
        o, rows, _ = runs
        o.start(robot_ids=["sim_guide", A])
        assert rows[0]["mode"] == "gazebo"
        assert rows[0]["format"] == "full_tour"

    def test_a_failing_mode_lookup_is_sim_unknown_never_live(self, runs):
        o, rows, _ = runs
        o._run_mode = lambda robots: 1 / 0
        o.start(robot_ids=[GUIDE, A])
        assert rows[0]["mode"] == "sim_unknown"

    def test_stop_ends_the_run_once_with_the_whole_row(self, runs):
        o, rows, _ = runs
        o.start(robot_ids=[GUIDE, A])
        o.stop()
        o.stop()
        assert len(rows) == 2
        assert rows[1]["ended_at"] and rows[1]["mode"] == rows[0]["mode"]

    def test_a_run_left_in_error_is_ended_by_the_next_start(self, runs):
        o, rows, _ = runs
        o.start(robot_ids=[GUIDE, A])
        first = o._run_id
        o._state = DemoState.ERROR
        o._run_id = first
        o.start(robot_ids=[GUIDE, A])
        ended = [r for r in rows if r.get("ended_at")]
        assert [r["id"] for r in ended] == [first]

    def test_a_failing_sink_never_stops_the_tour(self, runs):
        o, _, _ = runs
        o._run_sink = lambda row: 1 / 0
        o.start(robot_ids=[GUIDE, A])
        assert o.get_status()["state"] == "running"

    def test_the_run_is_active_only_until_it_ends(self, runs):
        o, _, _ = runs
        o.start(robot_ids=[GUIDE, A])
        assert o.active_run_id() == o._run_id == o.get_status()["run_id"]
        o.stop()
        assert o.active_run_id() is None and o.get_status()["run_id"] is None


class TestCorrectionsNameTheRun:
    def test_origin_and_run_reach_the_correction(self, runs):
        o, _, sink = runs
        o.start(robot_ids=[GUIDE, A])
        o.manual_next(source="operator", origin="scripted")
        (c,) = sink.corrections
        assert c.run_id == o._run_id and c.origin == "scripted"

    def test_the_default_origin_is_human(self, runs):
        o, _, sink = runs
        o.start(robot_ids=[GUIDE, A])
        o.qa_end(source="operator")
        assert sink.corrections[0].origin == "human"


# ── HTTP ──────────────────────────────────────────────────────────────────────

class FakeOrchestrator:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*a, **k):
            self.calls.append((name, k))
            return {"applied": [], "ignored": [], "total": 0}
        return record

    def get_status(self):
        return {}


@pytest.fixture
def client():
    o = FakeOrchestrator()
    app = Flask(__name__)
    app.register_blueprint(create_demo_gateway(o))
    c = app.test_client()
    c.orch = o
    return c


class TestGateway:
    def test_a_named_mode_is_passed_through(self, client):
        assert client.post("/demo/start", json={"mode": "gazebo"}).status_code == 200
        assert client.orch.calls[0][1]["mode"] == "gazebo"

    def test_no_mode_leaves_it_to_the_orchestrator(self, client):
        client.post("/demo/start", json={})
        assert client.orch.calls[0][1]["mode"] is None

    def test_an_unknown_mode_is_refused(self, client):
        assert client.post("/demo/start", json={"mode": "sim"}).status_code == 400
        assert client.orch.calls == []

    @pytest.mark.parametrize("route", ["/demo/next", "/demo/qa", "/demo/qa_end"])
    def test_origin_reaches_the_orchestrator(self, client, route):
        assert client.post(route, json={"origin": "scripted"}).status_code == 200
        assert client.orch.calls[0][1]["origin"] == "scripted"

    def test_origin_defaults_to_human(self, client):
        client.post("/demo/next", json={})
        assert client.orch.calls[0][1]["origin"] == "human"

    def test_an_unknown_origin_is_refused(self, client):
        assert client.post("/demo/next", json={"origin": "robot"}).status_code == 400
        assert client.orch.calls == []

    def test_revise_carries_origin(self, client):
        client.post("/demo/revise", json={"ops": [{"kind": "drop_remaining"}],
                                          "origin": "test"})
        assert client.orch.calls[0][1]["origin"] == "test"


# ── Repo + app wiring ─────────────────────────────────────────────────────────

class TestRepo:
    def test_an_unknown_foreign_key_retries_without_it(self, monkeypatch):
        from data import demo_run_repo
        sent = []

        class Table:
            def upsert(self, row, **_k):
                sent.append(row)
                return self

            def execute(self):
                if "scenario_id" in sent[-1]:
                    raise RuntimeError("23503 foreign key violation")

        monkeypatch.setattr(demo_run_repo, "get_client",
                            lambda: type("C", (), {"table": lambda s, n: Table()})())
        assert demo_run_repo.upsert_run(
            {"id": "run-1", "mode": "live", "scenario_id": "nope", "supervisor_id": None})
        assert sent[-1] == {"id": "run-1", "mode": "live"}

    def test_a_refused_write_is_false_not_raised(self, monkeypatch):
        from data import demo_run_repo
        monkeypatch.setattr(demo_run_repo, "get_client", lambda: 1 / 0)
        assert demo_run_repo.upsert_run({"id": "run-1", "scenario_id": "x"}) is False


class TestDefaultMode:
    def test_a_guide_that_can_move_is_gazebo(self, monkeypatch):
        import app

        class Inst:
            enabled_modules = {"navigation"}
        monkeypatch.setitem(app._registry_ref, "r",
                            type("R", (), {"get": lambda s, cid: Inst() if cid == "g" else None})())
        monkeypatch.setitem(app._profiles_ref, "p", None)
        assert app.default_run_mode(["g", "navel_01"]) == "gazebo"
        assert app.default_run_mode(["pepper_01", "navel_01"]) == "mock"

    def test_live_only_when_every_robot_is_physical(self, monkeypatch):
        import app

        class Profiles:
            def is_physical(self, rid):
                return rid.endswith("_001")
        monkeypatch.setitem(app._registry_ref, "r", None)
        monkeypatch.setitem(app._profiles_ref, "p", Profiles())
        assert app.default_run_mode(["pepper_001", "navel_001"]) == "live"
        assert app.default_run_mode(["pepper_001", "navel_01"]) == "mock"   # one twin: not live
        assert app.default_run_mode([]) == "mock"


# ── Supervisor ────────────────────────────────────────────────────────────────

class TestSupervisor:
    def test_a_known_supervisor_is_passed_to_the_run(self, client, monkeypatch):
        from data import supervisor_repo
        monkeypatch.setattr(supervisor_repo, "supervisor_exists", lambda sid: sid == "jay")
        assert client.post("/demo/start", json={"supervisor_id": "jay"}).status_code == 200
        assert client.orch.calls[0][1]["supervisor_id"] == "jay"

    def test_an_unknown_supervisor_is_refused(self, client, monkeypatch):
        from data import supervisor_repo
        monkeypatch.setattr(supervisor_repo, "supervisor_exists", lambda sid: False)
        assert client.post("/demo/start", json={"supervisor_id": "who"}).status_code == 400
        assert client.orch.calls == []

    def test_an_unreadable_table_does_not_block_the_tour(self, client, monkeypatch):
        from data import supervisor_repo
        monkeypatch.setattr(supervisor_repo, "supervisor_exists", lambda sid: None)
        assert client.post("/demo/start", json={"supervisor_id": "jay"}).status_code == 200

    def test_none_named_is_none(self, client):
        client.post("/demo/start", json={})
        assert client.orch.calls[0][1]["supervisor_id"] is None

    def test_the_list_route(self, client, monkeypatch):
        from data import supervisor_repo
        monkeypatch.setattr(supervisor_repo, "list_supervisors",
                            lambda: [{"id": "jay", "name": "Jay"}])
        assert client.get("/supervisors").get_json() == \
            {"supervisors": [{"id": "jay", "name": "Jay"}]}

    def test_the_repo_degrades_to_empty(self, monkeypatch):
        from data import supervisor_repo
        monkeypatch.setattr(supervisor_repo, "get_client", lambda: 1 / 0)
        assert supervisor_repo.list_supervisors() == []
        assert supervisor_repo.supervisor_exists("jay") is None

    def test_every_correction_in_the_run_names_the_supervisor(self, runs):
        from gateway.websocket_gateway import WebSocketGateway
        o, _, sink = runs

        class NoRobots:
            def get(self, cid):
                return None

            def get_all(self, exclude_id=None):
                return []

        gw = WebSocketGateway(NoRobots(), recorder=o._recorder)
        gw.set_demo_orchestrator(o)
        o._session_context = gw.session_context
        o.start(robot_ids=[GUIDE, A], supervisor_id="jay")
        assert gw.session_context() == {"supervisor_id": "jay"}
        o.manual_next(source="operator")
        assert sink.corrections[0].supervisor_id == "jay"
        o.stop()
        assert gw.session_context() == {}
