"""
test_robots.py — run.py finds the right robot, and the speech contract a new robot inherits holds.

Run from v6.0.0/client:  python3 -m pytest test_robots.py
"""

import glob
import importlib
import importlib.util
import json
import threading

import pytest

from client import BasicClient
from run import load


def test_a_config_without_a_robot_module_runs_the_generic_robot():
    from robots.generic import Robot
    assert load("irobi") is Robot and load("generic") is Robot


@pytest.mark.skipif(importlib.util.find_spec("navel") is not None, reason="the Navel SDK is installed here")
def test_a_robot_whose_sdk_is_missing_fails_rather_than_falling_back():
    with pytest.raises(ModuleNotFoundError) as e:
        load("navel")
    assert e.value.name == "navel"


@pytest.mark.parametrize("name", ["generic", "chatbox", "pepper", "_template"])
def test_every_robot_is_a_basic_client(name):
    assert issubclass(load(name), BasicClient)


def test_every_config_names_its_robot():
    paths = glob.glob("configs/*.json") + glob.glob("configs/*/*.json")
    assert len(paths) >= 13
    for path in paths:
        cfg = json.load(open(path))
        assert cfg["client_id"] and cfg["server_url"], path


def test_the_generic_robot_builds_from_a_lab_config_and_retunes_its_voice():
    robot = load("generic")("configs/lab/silbot_01.json")
    assert set(robot.output_modules) == {"console_output", "edge_tts_output"}
    assert not robot.input_modules                      # no "speech" module: no mic, no typed input
    robot.on_persona_update({"persona_name": "tutor", "voice_config": {"tts_voice": "en-GB-RyanNeural"}})
    assert robot.output_modules["edge_tts_output"]._voice == "en-GB-RyanNeural"


class _Client:
    def __init__(self):
        self.is_speaking, self.tts_started_event = threading.Event(), threading.Event()
        self.states = []

    def on_tts_state(self, speaking):
        self.states.append(speaking)


def test_a_robot_that_only_overrides_say_gets_acks_tts_state_and_barge_in():
    from robots._template import Speech

    said, release = [], threading.Event()

    class Robot(Speech):
        def say(self, sentence):
            said.append(sentence)
            release.wait(2)

    client = _Client()
    tts = Robot("speech", {})
    tts.set_client(client)
    tts.start()
    try:
        done = threading.Event()
        tts.speak_with_callback("[WAVE] First one. Second one. Third one.", callback=done.set)
        for _ in range(100):
            if said:
                break
            threading.Event().wait(0.02)
        assert said == ["First one."]                    # tag stripped, one sentence at a time
        assert tts.interrupt() == "Second one. Third one."  # what was never said, to resume
        release.set()
        assert done.wait(2)                               # the demo step still gets its ACK
        assert said == ["First one."]
        assert client.states == [True, False]
    finally:
        tts.stop()
