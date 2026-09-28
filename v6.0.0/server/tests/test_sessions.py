"""
tests/test_sessions.py
======================
Schema v2 phase 2: every robot connection has a sessions row, keyed by the
same session_id its log rows carry, linked to the users row still written for
it (dual write), and closed when the connection goes.
"""

from __future__ import annotations

import pytest

from data import robot_repo, session_repo, user_repo
from data.robot_repo import RobotRecord
from robot.robot_registry import RobotRegistry


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(robot_repo, "get_robot", lambda cid: RobotRecord(
        client_id=cid, robot_name="Navel", robot_role="", is_active=False,
        allowed_tags=[], modules=[], scenario_id="lab_demo"))
    monkeypatch.setattr(robot_repo, "set_active", lambda cid, a: True)
    monkeypatch.setattr(user_repo, "create_user", lambda name=None: 77)
    monkeypatch.setattr(session_repo, "start_session",
                        lambda *a, **k: log.append(("start", a, k)) or True)
    monkeypatch.setattr(session_repo, "end_session",
                        lambda sid: log.append(("end", sid)) or True)
    return log


class TestRegistry:
    def test_a_connection_opens_its_session(self, calls):
        inst = RobotRegistry().connect("navel_001")
        (kind, args, kw), = calls
        assert kind == "start"
        assert args == (inst.identity.session_id, "navel_001")
        assert kw["scenario_id"] == "lab_demo" and kw["legacy_user_id"] == 77
        assert inst.user_id == 77          # the users row is still written

    def test_disconnect_closes_it(self, calls):
        reg = RobotRegistry()
        sid = reg.connect("navel_001").identity.session_id
        reg.disconnect("navel_001")
        assert calls[-1] == ("end", sid)

    def test_disconnecting_an_unknown_robot_closes_nothing(self, calls):
        RobotRegistry().disconnect("nobody")
        assert calls == []

    def test_shutdown_closes_every_session(self, calls):
        reg = RobotRegistry()
        reg.connect("a")
        reg.connect("b")
        reg.shutdown()
        assert sum(1 for c in calls if c[0] == "end") == 2

    def test_idle_cleanup_closes_it(self, calls):
        reg = RobotRegistry()
        reg.connect("a").last_active = 0
        reg._cleanup_idle()
        assert calls[-1][0] == "end"


class TestRepo:
    def test_start_writes_the_row(self, fake_supabase):
        assert session_repo.start_session("navel_001:1", "navel_001", "lab_demo", 77, 0.0)
        (row,) = fake_supabase.writes["sessions"]
        assert row["id"] == "navel_001:1" and row["legacy_user_id"] == 77
        assert row["started_at"].startswith("1970-01-01")

    def test_a_failed_user_row_is_not_linked(self, fake_supabase):
        session_repo.start_session("x:1", "x", None, 0)
        assert fake_supabase.writes["sessions"][0]["legacy_user_id"] is None

    def test_an_unknown_scenario_retries_without_it(self, monkeypatch):
        sent = []

        class Table:
            def upsert(self, row, **_k):
                sent.append(row)
                return self

            def execute(self):
                if sent[-1]["scenario_id"]:
                    raise RuntimeError("23503")

        monkeypatch.setattr(session_repo, "get_client",
                            lambda: type("C", (), {"table": lambda s, n: Table()})())
        assert session_repo.start_session("x:1", "x", "nope")
        assert sent[-1]["scenario_id"] is None

    def test_failures_are_false_not_raised(self, monkeypatch):
        monkeypatch.setattr(session_repo, "get_client", lambda: 1 / 0)
        assert session_repo.start_session("x:1", "x") is False
        assert session_repo.end_session("x:1") is False

    def test_end_stamps_ended_at(self, fake_supabase):
        session_repo.end_session("x:1")
        assert "ended_at" in fake_supabase.writes["sessions"][0]["_update"]
