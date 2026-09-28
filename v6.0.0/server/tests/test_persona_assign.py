"""
tests/test_persona_assign.py
============================
Assigning a persona sets the robot's persona_id and role prompt, and never its
modules / allowed_tags: those describe hardware, and overwriting them could
take "navigation" away from a robot that can move.
"""

from __future__ import annotations

import pytest
from flask import Flask

from data import persona_repo, robot_repo
from data.persona_repo import PersonaRecord
from data.robot_repo import RobotRecord
from gateway.persona_gateway import create_persona_gateway

PERSONA = PersonaRecord(
    id="p1", name="Cheerful", description="", robot_role="You are cheerful.",
    allowed_tags=["[DEFAULT]"], modules=["gpt"], voice_config={"rate": 1},
    capabilities={}, personality={}, is_default=False, created_at="",
)


class FakeWS:
    def __init__(self):
        self.sent = []

    def get_connected_ids(self):
        return ["navel_001"]

    def send_to_robot(self, cid, data):
        self.sent.append((cid, data))


@pytest.fixture
def setup(monkeypatch):
    writes = []
    monkeypatch.setattr(robot_repo, "get_robot", lambda cid: RobotRecord(
        client_id=cid, robot_name="Navel", robot_role="", is_active=True,
        allowed_tags=["[WAVE]"], modules=["speech", "navigation"]))
    monkeypatch.setattr(persona_repo, "get_by_id", lambda pid: PERSONA)
    monkeypatch.setattr(robot_repo, "set_persona",
                        lambda *a: writes.append(a) or True)
    ws = FakeWS()
    app = Flask(__name__)
    app.register_blueprint(create_persona_gateway(ws))
    return app.test_client(), writes, ws


def test_only_persona_and_role_are_written(setup):
    c, writes, _ = setup
    assert c.post("/robots/navel_001/persona", json={"persona_id": "p1"}).status_code == 200
    assert writes == [("navel_001", "p1", "You are cheerful.")]


def test_the_live_update_carries_no_hardware_fields(setup):
    c, _, ws = setup
    c.post("/robots/navel_001/persona", json={"persona_id": "p1"})
    (_cid, event), = ws.sent
    assert event["event"] == "persona_update" and event["voice_config"] == {"rate": 1}
    assert "modules" not in event and "allowed_tags" not in event


def test_a_failed_write_is_500(setup, monkeypatch):
    c, _, ws = setup
    monkeypatch.setattr(robot_repo, "set_persona", lambda *a: False)
    assert c.post("/robots/navel_001/persona", json={"persona_id": "p1"}).status_code == 500
    assert ws.sent == []


def test_the_repo_updates_only_those_columns(fake_supabase):
    assert robot_repo.set_persona("navel_001", "p1", "role")
    assert fake_supabase.writes["robots"] == [
        {"_update": {"persona_id": "p1", "robot_role": "role"}}]
