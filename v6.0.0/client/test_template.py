"""
test_template.py — what a new robot built from robots/_template.py inherits.

Run from the client folder:  python3 -m pytest test_template.py
Exported with the kit (export_kit.sh), so it needs none of this lab's robots.
"""

import json
import threading

from run import load


def test_a_robot_without_a_module_runs_the_generic_robot():
    from robots.generic import Robot
    assert load("some_new_robot") is Robot


def test_the_generic_robot_builds_from_a_config_and_retunes_its_voice(tmp_path):
    cfg = tmp_path / "r.json"
    cfg.write_text(json.dumps({"client_id": "r_001", "server_url": "http://127.0.0.1:5000", "modules": ["gpt"]}))
    robot = load("generic")(str(cfg))
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
