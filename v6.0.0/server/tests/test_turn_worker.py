"""
tests/test_turn_worker.py
=========================
A robot's connection keeps reading while one of its turns is answering, and a
new turn stops the old answer at its next sentence.

The reader thread used to run the whole turn (STT + LLM) itself, so nothing
else from that robot was read until the answer finished: not its acks, not
the visitor talking over it.
"""

from __future__ import annotations

import json
import threading

from conversation.turn import handle_turn
from gateway.websocket_gateway import RobotConnection
from tests.test_speech_turn import gw  # noqa: F401  (fixture)
from tests.test_qa_more_questions_race import A


def _conn(on_message):
    return RobotConnection("r1", "127.0.0.1", 1, on_message=on_message,
                           on_close=lambda _cid: None)


def test_an_ack_is_read_while_a_turn_is_running():
    started, release, seen = threading.Event(), threading.Event(), []

    def on_message(cid, data, cancelled=None):
        seen.append(data["type"])
        if data["type"] == "chat":
            started.set()
            release.wait(2)

    conn = _conn(on_message)
    conn._handle_message(None, json.dumps({"type": "chat", "message": "hi"}))
    assert started.wait(2)
    conn._handle_message(None, json.dumps({"type": "ack", "step_id": "s1"}))
    assert seen == ["chat", "ack"]          # the ack did not wait for the turn
    release.set()
    conn.disconnect()


def test_a_new_turn_cancels_the_one_still_answering():
    started, release, flags = threading.Event(), threading.Event(), []

    def on_message(cid, data, cancelled=None):
        flags.append(cancelled)
        if data["message"] == "first":
            started.set()
            release.wait(2)

    conn = _conn(on_message)
    conn._handle_message(None, json.dumps({"type": "chat", "message": "first"}))
    assert started.wait(2)
    conn._handle_message(None, json.dumps({"type": "chat", "message": "second"}))
    assert flags[0].is_set()
    release.set()
    conn.disconnect()


def test_a_cancelled_answer_stops_at_its_next_sentence(gw):  # noqa: F811
    cancel = threading.Event()
    inst = gw._registry.get(A)

    def three_sentences(message, on_sentence, **kw):
        on_sentence("One.", "DEFAULT")
        cancel.set()                        # the visitor speaks again here
        on_sentence("Two.", "")
        on_sentence("Three.", "")

    inst.process_chat_stream = three_sentences
    turn = handle_turn(gw, inst, A, "how does the retrieval work", cancelled=cancel)

    spoken = [d["text"] for _, d in gw.sent if d["event"] == "chat_sentence"]
    assert turn.outcome == "cancelled"
    assert spoken == ["One."]


def test_sentences_and_the_barge_in_carry_the_turn_id(gw):  # noqa: F811
    turn = handle_turn(gw, gw._registry.get(A), A, "how does the retrieval work")
    tagged = [d for _, d in gw.sent if d["event"] in ("tts_stop", "chat_sentence")]
    assert tagged and all(d["turn_id"] == turn.turn_id for d in tagged)
