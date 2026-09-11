"""
Robots stand at separate stations; visitors name robots.

Two live-run findings. In the final open-floor Q&A every question went to
whichever robot presented last. And routing assumed every robot could hear
every question — in the real lab ChatBox, Navel and Silbot stand at three
different stations, so a question about a robot the group is not standing
with can only be deferred to its station or answered by Pepper, who walks
with the group.
"""

from __future__ import annotations

import pytest
from flask import Flask

from core.rbac import AccessLevel, RobotIdentity
from decision import DecisionRecorder, MemoryDecisionSink
from decision.kg import RobotTopicEdge
from decision.kg_policy import KGRouter
from decision.policy import robots_named
from decision.presence import PresenceTracker
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
from demo.demo_script import build_script
from gateway.http_gateway import create_http_gateway
from gateway.websocket_gateway import WebSocketGateway

GUIDE, A, B = "pepper_01", "chatbox_01", "navel_01"
EMOTION, RAG = "topic:emotion-recognition", "topic:retrieval-augmented-generation"
TOPICS = [{"id": EMOTION, "label": "emotion recognition"},
          {"id": RAG, "label": "retrieval augmented generation"}]
PEERS = [{"client_id": GUIDE, "robot_name": "Pepper"},
         {"client_id": A, "robot_name": "ChatBox"},
         {"client_id": B, "robot_name": "Navel"}]


@pytest.fixture(autouse=True)
def _instant_speech(monkeypatch):
    """No real speaking-time pauses; these tests are about who answers."""
    monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds", lambda _t: 0.0)
    monkeypatch.setattr("gateway.http_gateway._speaking_seconds", lambda _t: 0.0,
                        raising=False)


# ── Names ────────────────────────────────────────────────────────────────────

class TestRobotsNamed:
    def test_a_robot_name_is_found(self):
        assert robots_named("okay can i ask you about chatbox", PEERS) == [A]

    def test_case_and_punctuation_do_not_matter(self):
        assert robots_named("Navel, how does it work?", PEERS) == [B]

    def test_the_client_id_counts_too(self):
        assert robots_named("what does navel_01 do", PEERS) == [B]

    def test_only_whole_words(self):
        assert robots_named("the chatboxes in my house", PEERS) == []

    def test_two_names_are_both_reported(self):
        # The caller only acts on exactly one — naming two is not a choice.
        assert set(robots_named("is navel better than chatbox?", PEERS)) == {A, B}

    def test_no_name_is_no_match(self):
        assert robots_named("how does emotion recognition work?", PEERS) == []


# ── Stations ─────────────────────────────────────────────────────────────────

class TestSeparateStations:
    def test_off_by_default_so_nothing_changes_without_the_setting(self):
        t = PresenceTracker()
        assert t.absent([A, B], A) == set()

    def test_only_the_presenting_robot_is_within_reach(self):
        t = PresenceTracker()
        t.set_separate_stations(True)
        assert t.absent([A, B], A) == {B}

    def test_in_the_closing_no_project_robot_is_within_reach(self):
        t = PresenceTracker()
        t.set_separate_stations(True)
        assert t.absent([A, B], None) == {A, B}

    def test_an_override_still_wins(self):
        # "Navel has come over to the group" can still be said explicitly.
        t = PresenceTracker()
        t.set_separate_stations(True)
        t.set_in_conversation(B, True, source="operator")
        assert t.absent([A, B], A) == set()

    def test_the_lab_profile_declares_separate_stations(self):
        from core.profiles.registry import ProfileRegistry
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reg = ProfileRegistry.from_directory(os.path.join(here, "profiles"))
        assert reg.separate_stations() is True


# ── End to end, through the HTTP route ───────────────────────────────────────

class _Result:
    def __init__(self, text):
        self.response = text
        self.clean_text = text
        self.emotion_tag = "DEFAULT"
        self.is_delegation = False
        self.delegation_target = None


class _Robot:
    def __init__(self, cid, name, level=AccessLevel.LOCAL):
        self.client_id, self.robot_name = cid, name
        self.access_level, self.scenario_id = level, "lab_demo"
        self.robot_role = name
        self.calls = []

    @property
    def identity(self):
        return RobotIdentity(robot_id=self.client_id, scenario_id=self.scenario_id,
                             session_id=f"sess-{self.client_id}",
                             access_level=self.access_level, role=self.robot_role)

    def classify_qa_intent(self, message):
        return "continue"

    def process_chat_stream(self, message, on_sentence, **kw):
        self.calls.append(kw)
        text = f"[{self.client_id}] answer."
        on_sentence(text, "DEFAULT")
        return _Result(text)


class _Registry:
    def __init__(self, robots):
        self._by = {r.client_id: r for r in robots}

    def get(self, cid):
        return self._by.get(cid)

    def get_all(self, exclude_id=None):
        return [r for r in self._by.values() if r.client_id != exclude_id]


def _make(stations=True, at="open_floor"):
    robots = [_Robot(GUIDE, "Pepper", AccessLevel.GLOBAL),
              _Robot(A, "ChatBox"), _Robot(B, "Navel")]
    registry = _Registry(robots)
    edges = [RobotTopicEdge(robot_id=B, topic_id=EMOTION, specialised=True),
             RobotTopicEdge(robot_id=A, topic_id=RAG, specialised=True)]
    recorder = DecisionRecorder(MemoryDecisionSink())
    gw = WebSocketGateway(registry, recorder=recorder,
                          kg_router_factory=lambda: KGRouter(edges, [], TOPICS, explore=False))
    gw.presence.set_separate_stations(stations)
    gw.grounding_for = lambda rid, msg: [f"fact of {rid}"]

    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)
    orch._state = DemoState.QA_WINDOW
    if at == "open_floor":
        orch._idx = next(i for i, s in enumerate(orch._script) if s.step_id == "open_floor")
    else:   # ChatBox's own Q&A, with Navel's block still ahead
        orch._idx = next(i for i, s in enumerate(orch._script)
                         if s.block_robot_id == A and s.role == StepRole.QA)

    sent = []
    gw.send_to_robot = lambda cid, data: sent.append((cid, data))
    app = Flask(__name__)
    app.register_blueprint(create_http_gateway(registry, gw))
    c = app.test_client()
    c.sent, c.robots = sent, {r.client_id: r for r in robots}
    return c


def _ask(c, text, to=GUIDE):
    c.post(f"/robots/{to}/chat", json={"message": text})


class TestTheFinalQandA:
    def test_pepper_answers_for_a_robot_at_another_station(self):
        c = _make(stations=True)
        _ask(c, "can i ask more about the emotion recognition?")
        pepper = c.robots[GUIDE].calls
        assert pepper, "Pepper did not answer"
        assert pepper[-1].get("standing_in_for") == "Navel"
        assert pepper[-1]["grounded_facts"] == [f"fact of {B}"]
        assert not c.robots[B].calls

    def test_a_named_robot_at_another_station_is_answered_by_pepper(self):
        c = _make(stations=True)
        _ask(c, "okay can i ask you about chatbox")
        assert c.robots[GUIDE].calls[-1].get("standing_in_for") == "ChatBox"
        assert not c.robots[A].calls

    def test_without_stations_the_owner_answers(self):
        c = _make(stations=False)
        _ask(c, "can i ask more about the emotion recognition?")
        assert c.robots[B].calls, "the owner should answer when everyone is present"


class TestMidTour:
    def test_a_named_robot_still_ahead_is_deferred_to_its_station(self):
        c = _make(stations=True, at="chatbox_qa")
        _ask(c, "what can navel tell me about this?")
        assert not c.robots[B].calls
        spoken = " ".join(d.get("text", "") for _cid, d in c.sent)
        assert "Navel" in spoken and "station" in spoken

    def test_a_named_robot_the_group_is_standing_with_answers(self):
        c = _make(stations=True, at="chatbox_qa")
        _ask(c, "chatbox, how does that work?")
        assert c.robots[A].calls
        assert "standing_in_for" not in c.robots[A].calls[-1]


class TestTheStandInPrompt:
    def test_the_prompt_tells_the_guide_to_answer_rather_than_hand_over(self):
        from robot.robot_instance import RobotInstance

        class _LLM:
            system = None

            def is_available(self):
                return True

            def stream_with_history(self, system, history, user):
                _LLM.system = system
                yield "[DEFAULT] Navel's project reads facial cues."

        inst = RobotInstance.__new__(RobotInstance)
        inst.llm = _LLM()
        inst.robot_name, inst._robot_role, inst._allowed_tags = "Pepper", "Guide", ["DEFAULT"]
        inst._history, inst._max_history = [], 10
        inst.last_active = 0
        inst._refresh_role_from_db = lambda: None
        inst._get_rag_context = lambda m: []
        inst._get_active_peers = lambda: []
        inst._persist = lambda *a: None
        inst._current_user_emotion = lambda: None
        inst._parse_delegation = lambda t: (False, None)
        inst.process_chat_stream("how does navel read emotions?", lambda *a: None,
                                 standing_in_for="Navel")
        assert "standing in for Navel" in _LLM.system
        assert "Do not hand it over" in _LLM.system

    def test_the_hand_over_section_is_left_out_when_standing_in(self):
        # With it present, Pepper followed its "RobotX, can you take it?"
        # template and called over a robot at another station.
        from robot.prompt_builder import build_delegation_prompt
        peers = [{"client_id": "navel_01", "robot_name": "Navel",
                  "robot_role": "Emotion research", "declared_topics": ["emotion recognition"]}]
        normal, _ = build_delegation_prompt("Pepper", "Guide", ["[DEFAULT]"], "q", peers, [])
        standin, _ = build_delegation_prompt("Pepper", "Guide", ["[DEFAULT]"], "q", peers, [],
                                             standing_in_for="Navel")
        assert "TEAMMATES & DELEGATION" in normal
        assert "TEAMMATES & DELEGATION" not in standin
        assert "target_robot_id" not in standin

    def test_the_prompt_has_no_real_method_to_copy(self):
        # A worked example naming FAISS made Pepper, standing in for Navel,
        # say emotion recognition used "a FAISS index" in 3 of 10 answers.
        from robot.prompt_builder import build_delegation_prompt
        system, _ = build_delegation_prompt("Pepper", "Guide", ["[DEFAULT]"], "q", [], [],
                                            grounded_facts=["method: something"])
        assert "FAISS" not in system
