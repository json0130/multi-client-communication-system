"""
tests/test_robot_retire.py
==========================
The dashboard's "delete robot" retires it: retired_at is stamped and the row
stays (log rows, and later knowledge-graph rows, reference it). GET /robots
hides retired robots; registering one again brings it back. is_active is the
connection flag and is not what retiring sets.
"""

from __future__ import annotations

import pytest
from flask import Flask

from data import robot_repo
from gateway.http_gateway import create_http_gateway


class FakeWS:
    def __init__(self):
        self.disconnected = []

    def get_connected_ids(self):
        return ["a"]

    def disconnect_robot(self, cid):
        self.disconnected.append(cid)


@pytest.fixture
def client(monkeypatch):
    ws = FakeWS()
    app = Flask(__name__)
    app.register_blueprint(create_http_gateway(registry=None, ws_gateway=ws))
    c = app.test_client()
    c.ws = ws
    return c


class TestRoutes:
    def test_delete_disconnects_and_retires(self, client, monkeypatch):
        retired = []
        monkeypatch.setattr(robot_repo, "retire_robot", lambda cid: retired.append(cid) or True)
        assert client.delete("/robots/a").status_code == 200
        assert client.ws.disconnected == ["a"] and retired == ["a"]

    def test_a_failed_retire_is_500(self, client, monkeypatch):
        monkeypatch.setattr(robot_repo, "retire_robot", lambda cid: False)
        assert client.delete("/robots/a").status_code == 500

    @pytest.mark.parametrize("qs,expected", [("", False), ("?include_retired=1", True)])
    def test_the_list_hides_retired_unless_asked(self, client, monkeypatch, qs, expected):
        asked = []
        monkeypatch.setattr(robot_repo, "list_robot_rows",
                            lambda include_retired=False: asked.append(include_retired)
                            or [{"client_id": "a", "retired_at": None}])
        robots = client.get("/robots" + qs).get_json()["robots"]
        assert asked == [expected]
        assert robots[0]["ws_connected"] is True and "retired_at" in robots[0]


class TestRepo:
    def test_retire_stamps_retired_at_only(self, fake_supabase):
        assert robot_repo.retire_robot("a")
        (w,) = fake_supabase.writes["robots"]
        assert list(w["_update"]) == ["retired_at"]

    def test_registering_again_clears_it(self, fake_supabase):
        robot_repo.upsert_robot("a", "A", "", [], [])
        assert fake_supabase.writes["robots"][0]["retired_at"] is None

    def test_the_default_list_filters_retired(self, monkeypatch):
        calls = []

        class Q:
            def select(self, *a):
                return self

            def is_(self, *a):
                calls.append(a)
                return self

            def execute(self):
                return type("R", (), {"data": [{"client_id": "a"}]})()

        monkeypatch.setattr(robot_repo, "get_client",
                            lambda: type("C", (), {"table": lambda s, n: Q()})())
        assert robot_repo.list_robot_rows() == [{"client_id": "a"}]
        assert calls == [("retired_at", "null")]
        calls.clear()
        robot_repo.list_robot_rows(include_retired=True)
        assert calls == []

    def test_an_unreadable_table_lists_nothing(self, monkeypatch):
        monkeypatch.setattr(robot_repo, "get_client", lambda: 1 / 0)
        assert robot_repo.list_robot_rows() == []
