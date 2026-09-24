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

import time
from types import SimpleNamespace

import pytest

from core.rbac import AccessLevel, RobotIdentity
from decision import DecisionRecorder, MemoryDecisionSink, is_handoff_cue
from decision.models import Action
from decision.policy import Mechanism, PolicyResult
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
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

    @property
    def identity(self):
        return RobotIdentity(robot_id=self.client_id, scenario_id=SCENARIO,
                             session_id=f"sess-{self.client_id}",
                             access_level=self.access_level, role="researcher")

    def process_speech(self, audio_b64, chat=True):
        self.speech_calls.append(chat)
        return SimpleNamespace(transcription=self._heard, confidence=1.0, chat=None)


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
        return "[WAVE] Hello everyone!"


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

    def test_the_guides_intro_gets_no_reply(self, gw):
        gw._on_message(NAVEL, {"type": "chat", "message": "Welcome to the CARES lab!"})
        assert gw.sent == [] and gw.cues == []

    def test_a_typed_or_transcribed_hand_off_is_passed_on(self, gw):
        gw._on_message(NAVEL, {"type": "chat", "message": "Navel, over to you"})
        assert gw.cues == ["Navel, over to you"]
        assert gw.sent == []

    def test_audio_is_transcribed_without_generating_a_reply(self, gw):
        gw._on_message(NAVEL, {"type": "speech", "audio": "UklGRg=="})
        assert gw._registry.get(NAVEL).speech_calls == [False]
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
