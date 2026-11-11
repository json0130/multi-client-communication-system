"""
tests/test_roles_talk.py
========================
Two purposes for talking to a robot.

  * Roles-style (no demo running): the role a persona was assigned, nothing from the DB: no retrieved chat
    history, no stored project facts, and what is said is not written back.
  * The demo: the DB role (not the assigned one), plus the DB's chat history and facts.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.websocket_gateway import WebSocketGateway
from robot import robot_instance as ri
from robot.robot_instance import RobotInstance
from tests.test_qa_more_questions_race import FakeRegistry


class StreamLLM:
    def __init__(self):
        self.systems, self.histories = [], []

    def is_available(self):
        return True

    def stream_with_history(self, system, history, user_msg):
        self.systems.append(system)
        self.histories.append(list(history))
        yield "[DEFAULT] Hello there."


@pytest.fixture
def robot(monkeypatch):
    monkeypatch.setattr(ri.robot_repo, "get_robot", lambda cid: SimpleNamespace(
        robot_role="DB ROLE: a lab guide", allowed_tags=["[DEFAULT]"], access_level=None, scenario_id=None))
    r = RobotInstance("chatbox_01", "ChatBox", user_id=1, enabled_modules=set())
    r.llm = StreamLLM()
    r.calls = []
    r._get_rag_context = lambda message: r.calls.append("rag") or []
    r._persist = lambda message, text: r.calls.append("persist")
    r._get_active_peers = lambda: []
    return r


def test_demo_talk_reads_and_writes_the_db_history(robot):
    robot.process_chat_stream("hi", lambda *a: None)
    assert robot.calls == ["rag", "persist"] and len(robot._history) == 2


def test_roles_talk_uses_no_db_history_and_stores_nothing(robot):
    robot.process_chat_stream("hi", lambda *a: None, use_history=False)
    assert robot.calls == [] and robot._history == []


def test_roles_talk_keeps_its_own_conversation_apart_from_the_demos(robot):
    robot.process_chat_stream("demo question", lambda *a: None)
    robot.process_chat_stream("one", lambda *a: None, use_history=False)
    robot.process_chat_stream("two", lambda *a: None, use_history=False)
    assert robot.llm.histories[1] == []                         # none of the demo's turns
    assert len(robot.llm.histories[2]) == 2                     # the one roles turn before it
    assert len(robot._history) == 2                             # the demo's was left alone


def test_a_new_role_starts_a_new_roles_conversation(robot):
    robot.process_chat_stream("one", lambda *a: None, use_history=False)
    robot.assigned_role = "You are a shy tutor."
    robot.process_chat_stream("two", lambda *a: None, use_history=False)
    assert robot.llm.histories[1] == []


def test_the_assigned_role_is_for_roles_talk_only_the_demo_uses_the_db_role(robot):
    robot.assigned_role = "ASSIGNED ROLE: a shy tutor"
    robot.process_chat_stream("hi", lambda *a: None)
    robot.process_chat_stream("hi", lambda *a: None, use_history=False)
    demo, roles = robot.llm.systems
    assert "DB ROLE" in demo and "ASSIGNED ROLE" not in demo
    assert "ASSIGNED ROLE" in roles and "DB ROLE" not in roles


def _gw(state):
    gw = WebSocketGateway(FakeRegistry([]))
    gw.style_framing_for = lambda cid: "STYLE"
    gw.grounding_for = lambda cid, msg: ["a stored fact"]
    gw._demo_orchestrator = SimpleNamespace(get_status=lambda: {"state": state})
    return gw


@pytest.mark.parametrize("state", ["running", "waiting_ack", "qa_window", "paused"])
def test_a_demo_in_progress_keeps_the_facts_and_history(state):
    kw = _gw(state).answer_kwargs("chatbox_01", "how does it work")
    assert kw["grounded_facts"] == ["a stored fact"] and "use_history" not in kw


@pytest.mark.parametrize("state", ["idle", "completed", "error"])
def test_outside_the_demo_nothing_comes_from_the_db(state):
    kw = _gw(state).answer_kwargs("chatbox_01", "how does it work")
    assert kw == {"style_framing": "", "grounded_facts": (), "use_history": False}


def test_no_orchestrator_is_outside_the_demo():
    gw = _gw("idle")
    gw._demo_orchestrator = None
    assert gw.answer_kwargs("chatbox_01", "hi")["use_history"] is False


def test_assigning_a_persona_sets_the_role_the_server_talks_as():
    from gateway.roles_gateway import DEFAULT_ROBOTS, GatewayHub
    chatbox = SimpleNamespace(robot_name="ChatBox", assigned_role=None)
    registry = SimpleNamespace(get=lambda cid: chatbox if cid == "chatbox_01" else None)
    ws = SimpleNamespace(get_connected_ids=lambda: ["chatbox_01"], send_to_robot=lambda cid, d: None,
                         ack_listeners=[], demo_start_listeners=[])
    hub = GatewayHub(ws, DEFAULT_ROBOTS, registry=registry)
    hub.push_persona("Shy tutor", "You are a shy tutor.", {"O": 0.0})
    assert chatbox.assigned_role == "You are a shy tutor."
