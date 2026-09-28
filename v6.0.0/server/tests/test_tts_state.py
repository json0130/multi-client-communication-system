"""
tests/test_tts_state.py
=======================
A robot that reports its TTS state (tts_state) overrides the word-count
estimate of when it will fall silent, and quiet-aware Q&A timers act on it.

Clients that never report keep the estimate — see test_qa_auto_close.py.
"""

from __future__ import annotations

import threading
import time

from gateway.websocket_gateway import WebSocketGateway

A, B = "robot_a", "robot_b"


class _Registry:
    def get(self, cid):
        return object()

    def disconnect(self, cid):
        pass


def _gw():
    return WebSocketGateway(_Registry())


def test_an_idle_report_ends_that_robots_estimate():
    gw = _gw()
    gw.note_speech(A, {"event": "chat_sentence", "text": "word " * 40})
    assert gw.seconds_until_quiet() > 5
    gw.note_tts_state(A, False)
    assert gw.seconds_until_quiet() == 0.0


def test_an_idle_report_leaves_other_robots_estimates_alone():
    gw = _gw()
    gw.note_speech(A, {"event": "chat_sentence", "text": "word " * 40})
    gw.note_speech(B, {"event": "chat_sentence", "text": "word " * 20})
    gw.note_tts_state(A, False)
    assert 5 < gw.seconds_until_quiet() <= 20 / 2.4 + 0.1


def test_still_speaking_outlasts_a_short_estimate():
    gw = _gw()
    gw.note_tts_state(A, True)
    assert gw.seconds_until_quiet() >= gw.SPEAKING_FLOOR_SEC


def test_the_report_arrives_as_a_message():
    gw = _gw()
    gw._on_message(A, {"type": "tts_state", "speaking": True})
    assert gw.seconds_until_quiet() > 0
    gw._on_message(A, {"type": "tts_state", "speaking": False})
    assert gw.seconds_until_quiet() == 0.0


def test_a_quiet_aware_action_waits_for_the_robot_then_the_silence():
    gw = _gw()
    fired = threading.Event()
    gw.note_tts_state(A, True)
    gw._schedule_qa_action(0.05, fired.set, silence=0.05)
    assert not fired.wait(0.3)          # still speaking: re-armed, not fired
    gw.note_tts_state(A, False)
    assert fired.wait(0.5)


def test_finishing_early_pulls_the_action_forward():
    gw = _gw()
    fired = threading.Event()
    gw.note_speech(A, {"event": "chat_sentence", "text": "word " * 40})   # ~17s
    gw._schedule_qa_action(gw.seconds_until_quiet() + 0.05, fired.set, silence=0.05)
    gw.note_tts_state(A, False)
    started = time.time()
    assert fired.wait(1.0)
    assert time.time() - started < 1.0


def test_a_cancelled_action_is_not_revived_by_an_idle_report():
    gw = _gw()
    fired = threading.Event()
    gw.note_tts_state(A, True)
    gw._schedule_qa_action(5.0, fired.set, silence=0.05)
    gw.cancel_qa_auto_close("visitor spoke")
    gw.note_tts_state(A, False)
    assert not fired.wait(0.3)
