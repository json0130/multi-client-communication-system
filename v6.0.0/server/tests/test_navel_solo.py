"""
tests/test_navel_solo.py
=========================
Navel presenting alone, with a human as the guide.

The human introduces the lab out loud, then hands over ("Navel, over to you").
Until then Navel must say nothing — its mic hears the whole intro. After the
hand-off it greets, presents, takes questions and hands back, and the Q&A
logic must behave as it does in the full tour even though there is no guide
robot for it to lean on.
"""

from __future__ import annotations

import json
import os
import re
import time
from types import SimpleNamespace

import pytest

from core.rbac import AccessLevel, RobotIdentity
from decision import DecisionRecorder, MemoryDecisionSink, is_handoff_cue
from decision.models import Action
from decision.policy import Mechanism, PolicyResult
from demo.demo_orchestrator import DemoOrchestrator, DemoState, DemoStep, StepRole
from demo.demo_script import build_script, build_solo_script
from gateway.websocket_gateway import WebSocketGateway

NAVEL = "navel_001"
SCENARIO = "lab_demo"


# ── Fakes ────────────────────────────────────────────────────────────────────

class FakeInstance:
    def __init__(self, client_id, heard=""):
        self.client_id, self.robot_name = client_id, client_id
        self.access_level = AccessLevel.LOCAL
        self.speech_calls = []
        self._heard = heard
        self.says_handoff = False       # what the LLM fallback answers
        self.asked = []                 # what the LLM fallback was asked about

    def classify_handoff(self, heard):
        self.asked.append(heard)
        return self.says_handoff

    @property
    def identity(self):
        return RobotIdentity(robot_id=self.client_id, scenario_id=SCENARIO,
                             session_id=f"sess-{self.client_id}",
                             access_level=self.access_level, role="researcher")

    def transcribe(self, audio_b64):
        self.speech_calls.append(audio_b64)
        return SimpleNamespace(transcription=self._heard, confidence=1.0)


class FakeRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}

    def get(self, client_id):
        return self._by_id.get(client_id)

    def get_all(self, exclude_id=None):
        return [i for i in self._by_id.values() if i.client_id != exclude_id]


class RecordingGateway:
    """Just enough gateway for the orchestrator's run loop."""

    def __init__(self):
        self.sent, self.instructions = [], []

    def send_to_robot(self, client_id, data):
        self.sent.append((client_id, data))

    def generate_demo_step(self, robot_id, instruction):
        self.instructions.append(instruction)
        return "[GREET] Hello everyone!"


def _wait_until(pred, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def running(monkeypatch):
    monkeypatch.setattr("demo.demo_orchestrator._reset_gazebo_positions", lambda: None)
    gw = RecordingGateway()
    orch = DemoOrchestrator(gw, transition_delay=0.0)
    orch.start(robot_ids=[NAVEL])
    assert _wait_until(lambda: orch.get_status()["state"] == "awaiting_cue")
    yield gw, orch
    orch.stop()


# ── The script ───────────────────────────────────────────────────────────────

class TestSoloScript:

    def test_every_step_is_the_robots_and_none_move_it(self):
        steps = build_solo_script(NAVEL)
        assert {s.robot_id for s in steps} == {NAVEL}
        assert not [s for s in steps if s.step_kind == "navigation"]

    def test_order_wait_greet_present_qa_hand_back(self):
        roles = [s.role for s in build_solo_script(NAVEL)]
        assert roles == [StepRole.OPENING, StepRole.HANDOFF,
                         StepRole.PROJECT, StepRole.PROJECT, StepRole.PROJECT,
                         StepRole.QA, StepRole.CLOSING]

    def test_it_waits_for_the_guide_first(self):
        assert build_solo_script(NAVEL)[0].step_kind == "cue"

    def test_qa_waits_for_move_on_and_the_end_does_not_reopen_the_floor(self):
        steps = build_solo_script(NAVEL)
        qa = next(s for s in steps if s.role == StepRole.QA)
        assert qa.qa_window and qa.qa_timeout == 0
        assert not steps[-1].qa_window

    def test_the_greeting_survives_compression(self):
        # It answers the human's hand-off; GREETING would be trimmed.
        greeting = build_solo_script(NAVEL)[1]
        assert greeting.role not in StepRole.COMPRESSIBLE


# ── Waiting for the hand-off ─────────────────────────────────────────────────

class TestHandOffPhrases:

    @pytest.mark.parametrize("said", [
        "Navel, over to you",
        "okay navel, can you explain about your project?",
        "Could you tell us about your research?",
        "Take it away, Navel!",
        "Navel, please introduce yourself",
        "Navel, what are you working on?",
    ])
    def test_a_hand_off_is_recognised(self, said):
        assert is_handoff_cue(said)

    @pytest.mark.parametrize("said", [
        "Welcome everyone to the CARES lab.",
        "In a moment Navel will explain its research.",
        "Navel studies emotion-aware interaction.",
    ])
    def test_the_guides_own_intro_is_not(self, said):
        assert not is_handoff_cue(said)


class TestTheRunWaits:

    def test_nothing_is_said_until_the_hand_off(self, running):
        gw, orch = running
        time.sleep(0.1)
        assert gw.sent == [] and gw.instructions == []

    def test_the_hand_off_starts_the_greeting_and_it_hears_what_was_said(self, running):
        gw, orch = running
        assert orch.cue_received("Navel, over to you")
        assert _wait_until(lambda: gw.instructions)
        assert "Navel, over to you" in gw.instructions[0]
        assert gw.sent[0][1]["step_id"] == f"{NAVEL}_greeting"

    def test_next_on_the_dashboard_also_hands_over_without_skipping_qa(self, running):
        gw, orch = running
        orch.manual_next(source="operator")
        assert _wait_until(lambda: gw.instructions)
        assert orch._skip_next_qa is False

    def test_a_cue_outside_the_wait_is_refused(self, running):
        gw, orch = running
        orch.cue_received("over to you")
        assert _wait_until(lambda: gw.instructions)
        assert orch.cue_received("over to you") is False


class TestTheGatewayStaysQuiet:

    @pytest.fixture
    def gw(self):
        navel = FakeInstance(NAVEL, heard="Navel, over to you")
        gw = WebSocketGateway(FakeRegistry([navel]),
                              recorder=DecisionRecorder(MemoryDecisionSink()))
        orch = DemoOrchestrator(gw)
        orch.load_script(build_solo_script(NAVEL))
        orch._state = DemoState.AWAITING_CUE
        gw.set_demo_orchestrator(orch)
        gw.sent = []
        gw.send_to_robot = lambda cid, data: gw.sent.append((cid, data))
        gw.cues = []
        orch.cue_received = lambda text: gw.cues.append(text) or True
        return gw

    def test_an_unlisted_hand_off_starts_it_through_the_llm(self, gw):
        navel = gw._registry.get(NAVEL)
        navel.says_handoff = True
        gw._on_message(NAVEL, {"type": "chat", "message": "Navel, go ahead"})
        assert navel.asked == ["Navel, go ahead"]
        assert gw.cues == ["Navel, go ahead"]
        assert gw.sent == []

    def test_a_listed_phrase_never_waits_for_the_llm(self, gw):
        gw._on_message(NAVEL, {"type": "chat", "message": "Navel, over to you"})
        assert gw._registry.get(NAVEL).asked == []
        assert gw.cues == ["Navel, over to you"]

    def test_the_guides_intro_gets_no_reply(self, gw):
        gw._on_message(NAVEL, {"type": "chat", "message": "Welcome to the CARES lab!"})
        assert gw.sent == [] and gw.cues == []

    def test_a_typed_or_transcribed_hand_off_is_passed_on(self, gw):
        gw._on_message(NAVEL, {"type": "chat", "message": "Navel, over to you"})
        assert gw.cues == ["Navel, over to you"]
        assert gw.sent == []

    def test_audio_is_transcribed_without_generating_a_reply(self, gw):
        gw._on_message(NAVEL, {"type": "speech", "audio": "UklGRg=="})
        assert gw._registry.get(NAVEL).speech_calls == ["UklGRg=="]   # transcribe() never generates
        assert gw.cues == ["Navel, over to you"]
        assert gw.sent == []


# ── Q&A with no separate guide ───────────────────────────────────────────────

def _moderator_gateway(script, responder_is):
    registry = FakeRegistry([FakeInstance(r) for r in {s.robot_id for s in script}])
    gw = WebSocketGateway(registry, recorder=DecisionRecorder(MemoryDecisionSink()))
    orch = DemoOrchestrator(gw)
    orch.load_script(script)
    gw.set_demo_orchestrator(orch)
    orch._state = DemoState.QA_WINDOW
    orch._idx = next(i for i, s in enumerate(script) if s.role == StepRole.QA)
    gw.send_to_robot = lambda cid, data: None

    def moderator_offers_to_wrap_up(point, decider, utterance=""):
        gw._scratch.wrap_up_text = "Shall we wrap up?"
        return PolicyResult(Action.guide_interject(responder_is), Mechanism.LLM_MODERATOR)
    gw._decide = moderator_offers_to_wrap_up
    return gw


class TestModeratorStillJudges:

    def test_a_solo_robots_answers_are_judged(self):
        gw = _moderator_gateway(build_solo_script(NAVEL), NAVEL)
        assert gw.check_qa_auto_close(NAVEL, "We read facial expressions.") == "wrap_up"
        gw.on_qa_window_close()

    def test_a_guide_robot_in_the_full_tour_is_still_guarded(self):
        gw = _moderator_gateway(build_script("pepper_01", ["navel_01"]), "pepper_01")
        assert gw.check_qa_auto_close("pepper_01", "Happy to help.") is None


# ── Project 56 knowledge ─────────────────────────────────────────────────────

SERVER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = os.path.dirname(SERVER)


class TestTheTalkIsGrounded:

    def _orch(self, lookup):
        gw = RecordingGateway()
        return gw, DemoOrchestrator(gw, facts_lookup=lookup)

    def test_project_points_carry_the_facts(self):
        gw, orch = self._orch(lambda rid: ["Method B plans one gesture per segment."])
        project = next(s for s in build_solo_script(NAVEL) if s.role == StepRole.PROJECT)
        orch._send_step(project)
        assert "Method B plans one gesture per segment." in gw.instructions[-1]

    def test_other_steps_are_worded_as_before(self):
        gw, orch = self._orch(lambda rid: ["Method B plans one gesture per segment."])
        orch._send_step(build_solo_script(NAVEL)[1])   # the greeting
        assert "Method B" not in gw.instructions[-1]

    def test_a_failed_lookup_never_stops_the_talk(self):
        def broken(rid):
            raise RuntimeError("database unreachable")
        gw, orch = self._orch(broken)
        project = DemoStep(step_id="p", robot_id=NAVEL, text="Explain.", generate=True,
                           role=StepRole.PROJECT)
        orch._send_step(project)
        assert gw.sent[-1][1]["step_id"] == "p"


class TestProject56Migration:
    """The table's CHECK constraints reject a bad row only when the migration
    is applied to the live database — too late to find a typo."""

    SQL = open(os.path.join(SERVER, "data", "migrations", "016_navel_project56.sql")).read()
    FACTS = re.findall(r"\('(topic:[a-z-]+)', '(\w+)',\s*\n\s*'((?:[^']|'')*)'\)", SQL)

    def test_there_are_facts_for_every_new_topic(self):
        topics = set(re.findall(r"\('(topic:[a-z-]+)',\s+'[a-z ]+',", self.SQL))
        assert topics and topics <= {t for t, _, _ in self.FACTS}

    def test_every_fact_passes_the_tables_checks(self):
        allowed = {"method", "model", "dataset", "metric", "hardware",
                   "limitation", "publication"}
        for _topic, kind, fact in self.FACTS:
            assert kind in allowed
            assert 3 <= len(fact.replace("''", "'")) <= 600

    def test_written_for_the_twin_only(self):
        # navel_001 gets these through a recorded sim-to-real transfer
        # (tools/model_transfer.py), never by being written twice.
        assert self.SQL.count("(VALUES ('navel_01')) AS r(robot_id)") >= 2
        assert "('navel_001')) AS r(robot_id)" not in self.SQL


class TestOnePersona:

    CONFIGS = ("navel_client/client_config.json",
               "gazebo_client/configs/navel_01_sim.json",
               "client/test_configs/navel_01.json")

    def test_every_navel_config_describes_project_56_the_same_way(self):
        roles = {json.load(open(os.path.join(V6, p)))["robot_role"] for p in self.CONFIGS}
        assert len(roles) == 1
        assert "Project 56" in roles.pop()


class TestNavelGestureTags:
    """Navel performs a fixed set of 11 gestures (DEFAULT, POINT, EMPHASISE,
    AGREE, LISTEN, UNCERTAIN, DENY, GREET, HAPPY, SURPRISE, SAD). A tag it
    has no gesture for — the script used to ask for [WAVE] — is just skipped."""

    def _tags(self, path):
        return json.load(open(os.path.join(V6, path)))["allowed_tags"]

    def test_every_navel_config_lists_the_same_gestures(self):
        tags = {tuple(self._tags(p)) for p in TestOnePersona.CONFIGS}
        assert len(tags) == 1 and len(tags.pop()) == 11

    def test_the_solo_script_only_asks_for_gestures_navel_has(self):
        allowed = set(self._tags("navel_client/client_config.json"))
        asked = {tag for s in build_solo_script(NAVEL)
                 for tag in re.findall(r"\[[A-Z]+\]", s.text)}
        assert asked and asked <= allowed, asked - allowed
