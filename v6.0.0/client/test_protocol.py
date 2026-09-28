"""
test_protocol.py — how the shared robot client handles what the server pushes.

Run from v6.0.0/client:  python3 -m pytest test_protocol.py

Speech events are handled in arrival order on one lane, and sentences from a
turn the server has since stopped (tts_stop with a newer turn_id) are dropped.
"""

import json
import time

import pytest

from client import BasicClient, OutputModule


class _Recorder(OutputModule):
    def __init__(self):
        super().__init__("recorder", {})
        self.said = []

    def initialize(self): return True
    def start(self): return True
    def stop(self): pass

    def process_output(self, data):
        time.sleep(0.01)                  # enough to reorder racing threads
        self.said.append(data["text"])
        return True


@pytest.fixture
def robot(tmp_path):
    cfg = tmp_path / "client_config.json"
    cfg.write_text(json.dumps({"client_id": "t1", "ws_port": 1}))
    bot = BasicClient(str(cfg))
    out = _Recorder()
    bot.register_output_module(out)
    return bot, out


def _push(bot, **msg):
    bot.server_connection._dispatch(msg)


def _settle(out, n, timeout=2.0):
    deadline = time.time() + timeout
    while len(out.said) < n and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)


def test_streamed_sentences_are_spoken_in_order(robot):
    bot, out = robot
    words = [f"Sentence {i}." for i in range(20)]
    for w in words:
        _push(bot, event="chat_sentence", text=w, turn_id=1)
    _settle(out, len(words))
    assert out.said == words


def test_sentences_from_a_stopped_turn_are_dropped(robot):
    bot, out = robot
    _push(bot, event="tts_stop", turn_id=5)
    _push(bot, event="chat_sentence", text="Old answer.", turn_id=4)
    _push(bot, event="chat_sentence", text="New answer.", turn_id=5)
    _settle(out, 1)
    assert out.said == ["New answer."]


def test_untagged_sentences_are_always_spoken(robot):
    # Older servers send no turn_id.
    bot, out = robot
    _push(bot, event="tts_stop", turn_id=9)
    _push(bot, event="chat_sentence", text="Of course.")
    _settle(out, 1)
    assert out.said == ["Of course."]
