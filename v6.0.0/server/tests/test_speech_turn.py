"""
tests/test_speech_turn.py
=========================
A spoken turn goes through the same pipeline as a typed one.

Every voice robot sends `speech`, and that path used to wait for the whole
LLM reply (process_chat, not streamed) and skip routing, grounding,
acknowledgements and the more-questions prompt. These pin it to
conversation/turn.py: transcribe, echo the transcription, then handle_turn.
"""

from __future__ import annotations

import pytest

from core.rbac import AccessLevel
from decision import DecisionRecorder, MemoryDecisionSink
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
from demo.demo_script import build_script
from gateway.websocket_gateway import WebSocketGateway
from tests.test_qa_more_questions_race import A, B, GUIDE, FakeInstance, FakeRegistry


class _STT:
    def __init__(self, text):
        self.transcription = text
        self.confidence = 90.0


class VoiceInstance(FakeInstance):
    heard = ""

    def transcribe(self, audio_b64):
        return _STT(self.heard)


@pytest.fixture
def gw():
    registry = FakeRegistry([
        VoiceInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
        VoiceInstance(A, "ChatBox", "RAG research"),
        VoiceInstance(B, "Navel", "Emotion research"),
    ])
    recorder = DecisionRecorder(MemoryDecisionSink())
    gw = WebSocketGateway(registry, recorder=recorder)
    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)
    orch._state = DemoState.QA_WINDOW
    orch._idx = next(i for i, s in enumerate(orch._script)
                     if s.block_robot_id == A and s.role == StepRole.QA)
    gw.sent = []
    gw.send_to_robot = lambda cid, data: gw.sent.append((cid, data))
    gw.say = lambda text: _speak(gw, registry, text)
    return gw


def _speak(gw, registry, text):
    registry.get(A).heard = text
    gw._on_message(A, {"type": "speech", "audio": "UklGRg=="})
    return [(cid, d) for cid, d in gw.sent]


def test_a_spoken_question_is_streamed_like_a_typed_one(gw):
    sent = gw.say("how does the retrieval work")
    events = [d["event"] for _, d in sent]
    assert events[0] == "speech_response"
    assert sent[0][1]["transcription"] == "how does the retrieval work"
    assert "response" not in sent[0][1]          # the answer is not bundled
    assert any(d["event"] == "chat_sentence" and "forty two" in d["text"]
               for _, d in sent)
    # Same follow-up as the typed path, which the old speech path never sent.
    assert any(d.get("step_id") == "_qa_more_questions" for _, d in sent)


def test_a_spoken_nod_is_not_sent_to_the_llm(gw):
    sent = gw.say("okay thank you")
    lines = [d["text"] for _, d in sent if d["event"] == "chat_sentence"]
    assert lines == ["Of course."]


def test_a_spoken_move_on_closes_the_window_without_generating(gw):
    # The old path answered with the first sentence of an LLM reply. Now,
    # like typed chat, the window closes and the scripted transition speaks.
    sent = gw.say("lets move on")
    assert gw._demo_orchestrator._qa_end_event.is_set()
    assert not any(d["event"] == "chat_sentence" for _, d in sent)


def test_silence_is_only_echoed(gw):
    sent = gw.say("   ")
    assert [d["event"] for _, d in sent] == ["speech_response"]
