"""
tests/test_qa_more_questions_race.py
=====================================
Regression for a real bug: a visitor said "lets move on" during ChatBox's
Q&A window. The advance was decided and logged correctly (mechanism
ADVANCE_PHRASE), but ChatBox got sent ANOTHER "any other questions?" prompt
right afterward anyway — the visitor had to say it a second time before the
demo actually moved to the next robot. The same phrase closed Navel's window
on the first try later in the same run.

Root cause, found by reading gateway/http_gateway.py::robot_chat() against
demo/demo_orchestrator.py: qa_end() only sets an event —
    self._qa_end_event.set()
— it does not synchronously flip self._state. The flip happens later, in the
_run_loop background thread blocked in _open_qa_window(), AFTER it records
the window's duration (a call that can hit real network I/O — this exact run
also logged "[demo_kg_repo] apply_observation failed: Server disconnected").

Meanwhile the Flask thread handling /chat calls qa_end() and then immediately
fires its OWN LLM call (process_chat_stream, for the robot's in-character
reply) and checks get_status()["state"] == "qa_window" right after. Whichever
finishes first decides whether a stale "still in qa_window" read causes a
spurious extra prompt. It is a race, not a per-robot difference — ChatBox
losing it and Navel winning it later in the same run is exactly what a race
looks like.

This test proves it deterministically by never starting the orchestrator's
background thread at all — the most extreme (and, under load, common) case
of "the state hasn't flipped yet". Before the fix, this reproduces every
time, not occasionally.
"""

from __future__ import annotations

import pytest
from flask import Flask

from core.rbac import AccessLevel, RobotIdentity
from decision import (DecisionPoint, DecisionRecorder, Mechanism,
                      MemoryDecisionSink)
from decision.kg_policy import RoutingDecision
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
from demo.demo_script import build_script
from gateway.http_gateway import create_http_gateway
from gateway.websocket_gateway import WebSocketGateway

GUIDE = "pepper_01"
A = "chatbox_jetson_001"
B = "navel_001"
SCENARIO = "lab_demo"


class _ChatResult:
    def __init__(self, text):
        self.response = text
        self.emotion_tag = "DEFAULT"
        self.clean_text = text
        self.is_delegation = False
        self.delegation_target = None


class FakeInstance:
    """Mirrors tests/test_decision_wiring.py's fake, plus a stubbed chat call —
    the one extra thing robot_chat() needs that the WS-layer tests don't."""

    def __init__(self, client_id, name, role, level=AccessLevel.LOCAL, intent="continue"):
        self.client_id = client_id
        self.robot_name = name
        self.access_level = level
        self._role = role
        self._intent = intent

    @property
    def identity(self):
        return RobotIdentity(
            robot_id=self.client_id, scenario_id=SCENARIO,
            session_id=f"sess-{self.client_id}",
            access_level=self.access_level, role=self._role,
        )

    def classify_qa_intent(self, message):
        return self._intent

    def process_chat_stream(self, message, on_sentence):
        # A real robot's own conversational reply — exactly the call that
        # used to race the orchestrator's background state flip. Tagged with
        # client_id so a reroute test can tell WHICH robot actually answered.
        text = f"[{self.client_id}] Great, let's move on!"
        on_sentence(text, "DEFAULT")
        return _ChatResult(text)


class FakeRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}

    def get(self, client_id):
        return self._by_id.get(client_id)

    def get_all(self, exclude_id=None):
        return [i for i in self._by_id.values() if i.client_id != exclude_id]


@pytest.fixture
def client():
    """A live Flask test client wired exactly as app.create_app() wires it,
    parked mid-tour in ChatBox's open Q&A window — with NO background
    run-loop thread, so the orchestrator's state can never advance on its
    own. That is what makes the race reproduce every time instead of
    sometimes."""
    registry = FakeRegistry([
        FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
        FakeInstance(A, "ChatBox", "RAG research"),
        FakeInstance(B, "Navel", "Emotion research"),
    ])
    sink = MemoryDecisionSink()
    recorder = DecisionRecorder(sink)
    gw = WebSocketGateway(registry, recorder=recorder)

    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)

    orch._state = DemoState.QA_WINDOW
    orch._idx = next(
        i for i, s in enumerate(orch._script)
        if s.block_robot_id == A and s.role == StepRole.QA
    )

    sent = []
    gw.send_to_robot = lambda cid, data: sent.append((cid, data))

    app = Flask(__name__)
    app.register_blueprint(create_http_gateway(registry, gw))
    test_client = app.test_client()
    test_client.sent = sent
    return test_client


class TestNoDoublePromptOnAdvance:
    def test_advance_phrase_does_not_resend_more_questions(self, client):
        resp = client.post(f"/robots/{A}/chat", json={"message": "lets move on"})
        assert resp.status_code == 200

        step_ids = [d.get("step_id") for _, d in client.sent if d.get("event") == "demo_step"]
        assert "_qa_more_questions" not in step_ids

    def test_bare_affirmation_does_not_resend_more_questions(self, client):
        resp = client.post(f"/robots/{A}/chat", json={"message": "yes"})
        assert resp.status_code == 200

        step_ids = [d.get("step_id") for _, d in client.sent if d.get("event") == "demo_step"]
        assert "_qa_more_questions" not in step_ids

    def test_a_real_question_still_gets_the_more_questions_prompt(self, client):
        # The resend exists for a reason — a genuine follow-up must still get
        # it. Only an ADVANCE this same turn should suppress it.
        resp = client.post(f"/robots/{A}/chat", json={"message": "how accurate is it"})
        assert resp.status_code == 200

        step_ids = [d.get("step_id") for _, d in client.sent if d.get("event") == "demo_step"]
        assert "_qa_more_questions" in step_ids


class _FakeKGRouter:
    """Always names `target_id` as the answer — stands in for a competence
    graph confident enough to override the receiver.

    **kwargs absorbs the presence-related arguments the real KGRouter.decide
    now takes (remaining_block_ids / guide_robot_id / absent_robot_ids); this
    fake is about the reroute-execution path, not about presence."""

    def __init__(self, target_id, topic_id="topic:llm"):
        self._target_id = target_id
        self._topic_id = topic_id

    def decide(self, utterance, robot_ids, **kwargs):
        # target_id None models a router with no opinion — an utterance whose
        # topic never resolved.
        if self._target_id is None:
            return None
        return RoutingDecision(robot_id=self._target_id, topic_id=self._topic_id,
                               topic_label="LLMs", reason="argmax", score=0.9,
                               candidates_considered=2)


@pytest.fixture
def client_routing_to_navel():
    """Same wiring as `client`, except the competence graph is confident
    ChatBox's mic-holder (Pepper, in this scenario) is not who should answer —
    Navel should. Regression for a real report: a visitor's technical
    follow-up about ChatBox's own research was answered BY Pepper, in
    Pepper's voice, because Pepper's mic picked up the question — the QA_ROUTE
    decision was computed and logged but nothing ever acted on it."""
    registry = FakeRegistry([
        FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
        FakeInstance(A, "ChatBox", "RAG research"),
        FakeInstance(B, "Navel", "Emotion research"),
    ])
    sink = MemoryDecisionSink()
    recorder = DecisionRecorder(sink)
    gw = WebSocketGateway(registry, recorder=recorder,
                          kg_router_factory=lambda: _FakeKGRouter(B))

    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)

    orch._state = DemoState.QA_WINDOW
    orch._idx = next(
        i for i, s in enumerate(orch._script)
        if s.block_robot_id == A and s.role == StepRole.QA
    )

    sent = []
    gw.send_to_robot = lambda cid, data: sent.append((cid, data))

    app = Flask(__name__)
    app.register_blueprint(create_http_gateway(registry, gw))
    test_client = app.test_client()
    test_client.sent = sent
    return test_client


class TestQARouteExecution:
    def test_the_named_robot_speaks_the_answer_not_the_receiver(self, client_routing_to_navel):
        client_routing_to_navel.post(f"/robots/{GUIDE}/chat",
                                     json={"message": "how does the memory work exactly"})
        answers = [d.get("text") for cid, d in client_routing_to_navel.sent
                  if cid == B and d.get("event") == "chat_sentence"]
        assert any(f"[{B}]" in text for text in answers)

    def test_the_receiver_never_speaks_the_answer_itself(self, client_routing_to_navel):
        client_routing_to_navel.post(f"/robots/{GUIDE}/chat",
                                     json={"message": "how does the memory work exactly"})
        pepper_answers = [d.get("text") for cid, d in client_routing_to_navel.sent
                          if cid == GUIDE and d.get("event") == "chat_sentence"
                          and f"[{GUIDE}]" in (d.get("text") or "")]
        assert pepper_answers == []

    def test_the_receiver_gets_a_handoff_line_naming_the_target(self, client_routing_to_navel):
        client_routing_to_navel.post(f"/robots/{GUIDE}/chat",
                                     json={"message": "how does the memory work exactly"})
        pepper_lines = [d.get("text") for cid, d in client_routing_to_navel.sent
                        if cid == GUIDE and d.get("event") == "chat_sentence"]
        assert any("Navel" in (t or "") for t in pepper_lines)

    def test_the_more_questions_prompt_follows_the_target_not_the_receiver(self, client_routing_to_navel):
        client_routing_to_navel.post(f"/robots/{GUIDE}/chat",
                                     json={"message": "how does the memory work exactly"})
        targets = [cid for cid, d in client_routing_to_navel.sent
                  if d.get("event") == "demo_step" and d.get("step_id") == "_qa_more_questions"]
        assert targets == [B]

    def test_no_kg_router_means_the_receiver_still_answers(self, client):
        # The baseline (client fixture has no kg_router_factory): unchanged
        # behaviour, no handoff, no reroute.
        client.post(f"/robots/{A}/chat", json={"message": "how accurate is it"})
        answers = [d.get("text") for cid, d in client.sent
                  if cid == A and d.get("event") == "chat_sentence"]
        assert any(f"[{A}]" in (t or "") for t in answers)

    def test_the_graph_naming_the_same_robot_that_received_it_is_not_a_reroute(self):
        # If the graph agrees with the receiver, there is nothing to hand off —
        # must behave exactly like the no-router baseline (no extra chat_sentence).
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG research"),
            FakeInstance(B, "Navel", "Emotion research"),
        ])
        recorder = DecisionRecorder(MemoryDecisionSink())
        gw = WebSocketGateway(registry, recorder=recorder,
                              kg_router_factory=lambda: _FakeKGRouter(A))
        instance = registry.get(A)
        target_instance, target_id, handoff = gw.route_question(instance, "how accurate is it")
        assert target_id == A
        assert handoff is None

    def test_a_router_naming_an_unconnected_robot_falls_back_to_the_receiver(self):
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG research"),
        ])
        recorder = DecisionRecorder(MemoryDecisionSink())
        gw = WebSocketGateway(registry, recorder=recorder,
                              kg_router_factory=lambda: _FakeKGRouter("ghost_robot"))
        instance = registry.get(A)
        target_instance, target_id, handoff = gw.route_question(instance, "how accurate is it")
        assert target_id == A
        assert handoff is None


class TestUnresolvedQuestionsGoToThePresenter:
    """
    Regression: mid-way through Silbot's block a visitor asked "so which
    techniqe do you use". Nothing in the vocabulary matches "technique", so
    the topic did not resolve, routing had no opinion, and the question was
    answered by whoever's microphone caught it — the guide. Pepper then
    invented an answer on Silbot's behalf: "Silbot specializes in using
    machine learning techniques."

    Inside a robot's own block an unresolvable question is almost certainly
    about that robot, so it answers. The guide speaking for a specialist is
    the failure worth removing.
    """

    def _wire(self, sink_list):
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG research"),
            FakeInstance(B, "Navel", "Emotion research"),
        ])
        recorder = DecisionRecorder(MemoryDecisionSink())
        # A router that resolves nothing, which is what an unmatched word does.
        gw = WebSocketGateway(registry, recorder=recorder,
                              kg_router_factory=lambda: _FakeKGRouter(None),
                              kg_observer=sink_list.extend)
        orch = DemoOrchestrator(gw, recorder=recorder,
                                session_context=gw.session_context)
        orch.load_script(build_script(GUIDE, [A, B]))
        gw.set_demo_orchestrator(orch)
        orch._state = DemoState.QA_WINDOW
        orch._idx = next(i for i, s in enumerate(orch._script)
                         if s.block_robot_id == A and s.role == StepRole.QA)
        gw.on_qa_window_open()
        gw.send_to_robot = lambda cid, d: None
        return gw, registry

    def test_the_presenting_robot_answers_not_the_guide(self):
        observed = []
        gw, registry = self._wire(observed)
        result = gw._decide(DecisionPoint.QA_ROUTE, registry.get(GUIDE),
                            "so which techniqe do you use")
        assert result.action.robot_id == A
        assert result.mechanism == "presenter_fallback"

    def test_it_writes_no_observation(self):
        # Nothing resolved, so there is no edge this turn belongs to.
        observed = []
        gw, registry = self._wire(observed)
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(GUIDE),
                   "so which techniqe do you use")
        gw.on_qa_window_close()
        assert observed == []

    def test_the_presenter_receiving_it_directly_is_not_rerouted(self):
        observed = []
        gw, registry = self._wire(observed)
        result = gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                            "so which techniqe do you use")
        assert result.mechanism == Mechanism.RECEIVER
