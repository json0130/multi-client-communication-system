"""
ChatBox leads, and passes Navel's subject over to Navel.

The guide asks ChatBox for its intro and project. When they ask about Navel's
work, ChatBox hands over out loud and Navel answers in its own voice. The
routing is the lab tour's (KGRouter); what is new is the script (ChatBox only)
and that Navel counts as standing beside ChatBox despite the lab profile's
separate_stations.
"""

from __future__ import annotations

import pytest
from flask import Flask

from decision import DecisionRecorder, MemoryDecisionSink
from decision.kg import RobotTopicEdge
from decision.kg_policy import KGRouter
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
from demo.demo_script import build_solo_script
from gateway.http_gateway import create_http_gateway
from gateway.websocket_gateway import WebSocketGateway
from tests.test_stations_and_names import (A, B, EMOTION, RAG, TOPICS, _Registry,
                                     _Robot, _ask)

CHATBOX, NAVEL = A, B


@pytest.fixture(autouse=True)
def _instant(monkeypatch):
    monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds", lambda _t: 0.0)
    monkeypatch.setattr("conversation.turn._speaking_seconds", lambda _t: 0.0)
    monkeypatch.setattr("demo.demo_orchestrator._reset_gazebo_positions", lambda: None)


def _pair(colocated=True):
    robots = [_Robot(CHATBOX, "ChatBox"), _Robot(NAVEL, "Navel")]
    registry = _Registry(robots)
    edges = [RobotTopicEdge(robot_id=NAVEL, topic_id=EMOTION, specialised=True),
             RobotTopicEdge(robot_id=CHATBOX, topic_id=RAG, specialised=True)]
    recorder = DecisionRecorder(MemoryDecisionSink())
    gw = WebSocketGateway(registry, recorder=recorder,
                          kg_router_factory=lambda: KGRouter(edges, [], TOPICS, explore=False))
    gw.presence.set_separate_stations(True)     # what profiles/lab_demo.yaml says
    gw.grounding_for = lambda rid, msg: [f"fact of {rid}"]
    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_solo_script(CHATBOX, partner_id=NAVEL))
    if colocated:
        orch._colocated = [CHATBOX, NAVEL]
        orch._set_colocated(orch._colocated)
    gw.set_demo_orchestrator(orch)
    orch._state = DemoState.QA_WINDOW
    orch._idx = next(i for i, s in enumerate(orch._script) if s.role == StepRole.QA)
    gw.send_to_robot = lambda cid, data: None
    app = Flask(__name__)
    app.register_blueprint(create_http_gateway(registry, gw))
    c = app.test_client()
    c.robots, c.orch = {r.client_id: r for r in robots}, orch
    return c


def test_script_is_the_leads_alone_and_mentions_the_partner():
    steps = build_solo_script(CHATBOX, partner_id=NAVEL)
    assert {s.robot_id for s in steps} == {CHATBOX}
    assert NAVEL in next(s for s in steps if s.role == StepRole.QA).text
    assert NAVEL not in build_solo_script(CHATBOX)[-2].text


def test_two_robots_without_pepper_start_the_pair_and_end_clears_it():
    gw = type("G", (), {"presence": WebSocketGateway(_Registry([]), recorder=None).presence,
                        "send_to_robot": lambda *a: None,
                        "generate_demo_step": lambda *a: "x"})()
    orch = DemoOrchestrator(gw, transition_delay=0.0)
    orch.start(robot_ids=[CHATBOX, NAVEL])
    try:
        assert {s.robot_id for s in orch._script} == {CHATBOX}
        assert orch.colocated == [CHATBOX, NAVEL]
        assert gw.presence.has_override(NAVEL)
    finally:
        orch.stop()
        orch._end_run()
    assert not gw.presence.has_override(NAVEL)


def test_a_question_about_navels_subject_goes_to_navel():
    c = _pair()
    _ask(c, "can i ask more about the emotion recognition?", to=CHATBOX)
    assert c.robots[NAVEL].calls, "ChatBox should have passed it to Navel"
    assert c.robots[NAVEL].calls[-1].get("standing_in_for") is None


def test_naming_navel_passes_it_over():
    c = _pair()
    _ask(c, "navel, can you tell us about your project?", to=CHATBOX)
    assert c.robots[NAVEL].calls


def test_chatboxs_own_subject_stays_with_chatbox():
    c = _pair()
    _ask(c, "can i ask more about retrieval augmented generation?", to=CHATBOX)
    assert c.robots[CHATBOX].calls and not c.robots[NAVEL].calls


def test_without_the_co_location_navel_would_be_treated_as_away():
    c = _pair(colocated=False)
    _ask(c, "can i ask more about the emotion recognition?", to=CHATBOX)
    assert not c.robots[NAVEL].calls
