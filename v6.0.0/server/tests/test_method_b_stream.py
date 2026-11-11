"""
tests/test_method_b_stream.py
=============================
Method B inside the server: a Navel reply goes out one sentence at a time, each with its planned gesture as
its emotion_tag. Anyone else's reply is untouched, and a planner that is slow or broken never holds speech.
"""
from __future__ import annotations

import json
import time

import pytest

from conversation.method_b import MethodB, _Strict
from conversation.turn import handle_turn
from modules.action_tag_generator import ChatLLM, navel_tagger
from tests.test_qa_more_questions_race import A, B
from tests.test_speech_turn import gw  # noqa: F401  (fixture)

PICK = {"Welcome": "GREET", "sorry": "SAD", "Over there": "POINT"}


class ScriptedLLM:
    """Answers the picker's {"tag","confidence"} request by keyword in the sentence being scored."""
    available = True

    def __init__(self, delay=0.0, boom=None):
        self.delay, self.boom = delay, boom

    def respond(self, system, user, history, max_tokens=140):
        if self.boom:
            raise self.boom
        time.sleep(self.delay)
        sentence = user.split("Generated robot dialogue:\n")[1].split("\n")[0]
        tag = next((t for k, t in PICK.items() if k in sentence), "DEFAULT")
        return json.dumps({"tag": tag, "confidence": 0.9})


def _mb(llm, timeout=2.0):
    return MethodB(navel_tagger(ChatLLM(_Strict(llm)), overrun_seconds=10), timeout)


def _say(mb, sentences, model_tag="DEFAULT"):
    sent = []
    s = mb.stream("navel_001", "hi", sent.append)
    for t in sentences:
        s.put({"event": "chat_sentence", "text": t, "emotion_tag": model_tag, "turn_id": 1})
    s.close()
    return sent


def test_each_sentence_goes_out_in_order_with_its_gesture():
    sent = _say(_mb(ScriptedLLM()), ["Welcome to the lab.", "I am sorry that was hard.", "Over there is the arm."])
    assert [p["text"] for p in sent] == ["Welcome to the lab.", "I am sorry that was hard.", "Over there is the arm."]
    assert [p["emotion_tag"] for p in sent] == ["GREET", "SAD", "POINT"]
    assert all(p["turn_id"] == 1 for p in sent)


def test_the_same_gesture_is_not_repeated_back_to_back():
    sent = _say(_mb(ScriptedLLM()), ["Welcome one.", "Welcome two.", "Over there is the arm."])
    assert [p["emotion_tag"] for p in sent] == ["GREET", "", "POINT"]       # the repeat is plain speech


@pytest.mark.parametrize("llm", [ScriptedLLM(boom=RuntimeError("ollama down")),
                                 type("Err", (), {"available": True,
                                                  "respond": lambda *a, **k: "[LLM error: refused]"})()])
def test_a_failing_planner_still_sends_the_plain_sentence(llm):
    sent = _say(_mb(llm), ["Welcome to the lab."], model_tag="HAPPY")
    assert sent == [{"event": "chat_sentence", "text": "Welcome to the lab.", "emotion_tag": "HAPPY", "turn_id": 1}]


def test_a_slow_planner_does_not_hold_speech():
    t0 = time.monotonic()
    sent = _say(_mb(ScriptedLLM(delay=1.5), timeout=0.2), ["Welcome to the lab."], model_tag="HAPPY")
    assert time.monotonic() - t0 < 1.0 and sent[0]["emotion_tag"] == "HAPPY"


def test_planning_overlaps_across_sentences():
    # Four sentences at 0.4 s each is 1.6 s if planned one after another; planned as they arrive, two at a time.
    t0 = time.monotonic()
    sent = _say(_mb(ScriptedLLM(delay=0.4), timeout=5), ["Welcome a.", "Over there b.", "I am sorry c.", "Welcome d."])
    assert len(sent) == 4 and time.monotonic() - t0 < 1.3


def _streams(inst, sentences):
    def process_chat_stream(message, on_sentence, **kw):
        for t in sentences:
            on_sentence(t, "DEFAULT")
        return type("R", (), {"clean_text": " ".join(sentences), "is_delegation": False,
                              "delegation_target": None, "emotion_tag": "DEFAULT"})()
    inst.process_chat_stream = process_chat_stream


def _answer(gw, robot_id):
    inst = gw._registry.get(robot_id)
    _streams(inst, ["Welcome to the lab.", "Over there is the arm."])
    gw.route_question = lambda instance, message, ask_first=False: (inst, robot_id, None)
    handle_turn(gw, inst, robot_id, "tell me about it")
    return [d for cid, d in gw.sent if cid == robot_id and d["event"] == "chat_sentence"
            and d["text"] in ("Welcome to the lab.", "Over there is the arm.")]


def test_a_navel_turn_is_tagged(gw):
    gw.method_b = _mb(ScriptedLLM())
    assert [d["emotion_tag"] for d in _answer(gw, B)] == ["GREET", "POINT"]


def test_another_robots_turn_is_untouched(gw):
    gw.method_b = _mb(ScriptedLLM())
    assert [d["emotion_tag"] for d in _answer(gw, A)] == ["DEFAULT", "DEFAULT"]


def test_method_b_off_leaves_navel_plain(gw):
    assert gw.method_b is None
    assert [d["emotion_tag"] for d in _answer(gw, B)] == ["DEFAULT", "DEFAULT"]


def test_a_configured_model_turns_method_b_on_and_off_turns_it_off():
    from types import SimpleNamespace as NS
    from conversation.method_b import build_method_b
    assert build_method_b(NS(ollama_model="qwen2.5:7b", ai_server="", timeout_sec=2.0), "http://x") is not None
    assert build_method_b(NS(ollama_model="off", ai_server="", timeout_sec=2.0), "http://x") is None


def test_a_later_sentence_may_wait_for_its_tag_while_the_earlier_ones_are_still_being_said():
    # On one Ollama a tag queues behind the reply still being written, so it arrives late. By then the first
    # sentence is being spoken, and the second cannot be said before it ends: waiting costs nothing.
    long_first = "Welcome " + "word " * 9 + "here."          # ~4 s to say
    t0 = time.monotonic()
    sent = _say(_mb(ScriptedLLM(delay=0.8), timeout=0.2), [long_first, "Over there is the arm."])
    assert [p["emotion_tag"] for p in sent] == ["DEFAULT", "POINT"]      # the first had no one to wait for
    assert time.monotonic() - t0 < 3.0
