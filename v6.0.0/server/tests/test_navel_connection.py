"""
tests/test_navel_connection.py
===============================
The server and the shared robot client, talking over a real WebSocket.

Every other test fakes one side of the wire. This one runs the real
WebSocketGateway + DemoOrchestrator against the real BasicClient (v6.0.0/client,
which navel_client/run.py extends on the robot) on a localhost port, and walks
the Navel solo demo end to end: the guide's intro is ignored, the hand-off
starts it, every scripted step is spoken and acked, a question gets a streamed
answer, an interruption stops speech, and "no more questions" ends it.

Only what needs the outside world is stubbed: the robot's address (a DB read),
LLM text, and the robot instance's answer. No DB, no Ollama, no robot. What it
pins down is the wire contract a Navel client has to honour — see CONTRACT.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from core.rbac import AccessLevel, RobotIdentity
from decision import DecisionRecorder, MemoryDecisionSink
from demo.demo_orchestrator import DemoOrchestrator
from gateway.websocket_gateway import WebSocketGateway

# Loaded by file path, not by putting v6.0.0/client on sys.path: that folder
# has its own robot.py, and once it is on the path a later `import robot`
# (the server's package) can resolve to it. client.py imports nothing local.
_CLIENT_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "client", "client.py")
_spec = importlib.util.spec_from_file_location("shared_robot_client", _CLIENT_PY)
_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_client)
BasicClient, OutputModule = _client.BasicClient, _client.OutputModule

NAVEL = "navel_001"
ANSWER = "Method B scores each gesture tag against what the segment means."

CONTRACT = """
server -> robot : demo_step {step_id, text, require_ack}   robot replies ack {step_id}
                  chat_sentence {text, emotion_tag}         (streamed answer; no reply)
                  tts_stop {turn_id}                        robot stops speaking
robot -> server : chat {message}                            (what the mic heard)
"""


# ── Robot side ───────────────────────────────────────────────────────────────

class Voice(OutputModule):
    """Records what Navel would have said. Scripted steps finish at once
    unless `hold` is set, which leaves one mid-sentence for an interruption."""

    def __init__(self):
        super().__init__("voice", {})
        self.said, self.stopped = [], 0
        self.hold = threading.Event()

    def initialize(self): return True
    def start(self): return True
    def stop(self): pass

    def speak_with_callback(self, text, callback=None):
        self.said.append(("step", text))
        if callback and not self.hold.is_set():
            threading.Timer(0.01, callback).start()
        return True

    def process_output(self, data):
        self.said.append(("sentence", data["text"]))
        return True

    def interrupt(self):
        self.stopped += 1


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Server side ──────────────────────────────────────────────────────────────

class FakeInstance:
    client_id, robot_name, access_level = NAVEL, "Navel", AccessLevel.LOCAL

    @property
    def identity(self):
        return RobotIdentity(robot_id=NAVEL, scenario_id="lab_demo", session_id="s",
                             access_level=self.access_level, role="researcher")

    def classify_qa_intent(self, message):
        return "continue"

    def classify_handoff(self, heard):
        return False                    # only the listed phrase hands over here

    def process_chat_stream(self, message, on_sentence, **_kw):
        on_sentence(ANSWER, "DEFAULT")
        return SimpleNamespace(response=f"[DEFAULT] {ANSWER}", clean_text=ANSWER,
                               emotion_tag="DEFAULT", is_delegation=False,
                               delegation_target=None)


class FakeRegistry:
    def __init__(self):
        self.navel = FakeInstance()

    def get(self, cid):
        return self.navel if cid == NAVEL else None

    def get_all(self, exclude_id=None):
        return [] if exclude_id == NAVEL else [self.navel]

    def connect(self, cid):
        return self.navel

    def disconnect(self, cid):
        pass


def _wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def wired(tmp_path, monkeypatch):
    port = _free_port()
    cfg = tmp_path / "navel.json"
    cfg.write_text(json.dumps({"client_id": NAVEL, "robot_name": "Navel",
                               "ws_port": port, "server_url": "http://127.0.0.1:9"}))
    robot = BasicClient(str(cfg))
    voice = Voice()
    robot.register_output_module(voice)
    acked = []
    real_ack = robot.send_ack
    robot.send_ack = lambda step_id: (acked.append(step_id), real_ack(step_id))
    robot.server_connection.start_server()          # no HTTP registration

    # Timing that exists for real speakers only.
    monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds", lambda _t: 0.0)
    monkeypatch.setattr(DemoOrchestrator, "POST_PROJECT_GRACE_SEC", 0.0)
    monkeypatch.setattr("demo.demo_orchestrator._reset_gazebo_positions", lambda: None)
    monkeypatch.setattr("data.robot_repo.get_robot_address",
                        lambda cid: ("127.0.0.1", port) if cid == NAVEL else None)

    gw = WebSocketGateway(FakeRegistry(), recorder=DecisionRecorder(MemoryDecisionSink()))
    gw.generate_demo_step = lambda rid, instruction: "[DEFAULT] A generated line."
    orch = DemoOrchestrator(gw, transition_delay=0.0)
    gw.set_demo_orchestrator(orch)

    assert _wait_until(lambda: _port_open(port)), "robot WS server never came up"
    assert gw.connect_robot(NAVEL)
    assert _wait_until(robot.server_connection.is_connected), "server never connected"

    yield SimpleNamespace(gw=gw, orch=orch, robot=robot, voice=voice, acked=acked)

    orch.stop()
    gw.disconnect_robot(NAVEL)


def _port_open(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _state(w):
    return w.orch.get_status()["state"]


def _hear(w, text):
    """Navel's mic picks something up."""
    w.robot.send_to_server("chat", text)


def _handed_over(w):
    w.orch.start(robot_ids=[NAVEL])
    assert _wait_until(lambda: _state(w) == "awaiting_cue")
    _hear(w, "Navel, over to you")


# ── The demo, over the wire ──────────────────────────────────────────────────

class TestSoloDemoOverTheWire:

    def test_the_guides_intro_reaches_the_server_and_gets_no_reply(self, wired):
        w = wired
        w.orch.start(robot_ids=[NAVEL])
        assert _wait_until(lambda: _state(w) == "awaiting_cue")
        _hear(w, "Welcome everyone to the CARES lab.")
        time.sleep(0.3)
        assert w.voice.said == []
        assert _state(w) == "awaiting_cue"

    def test_the_whole_script_is_spoken_and_acked_then_questions_open(self, wired):
        w = wired
        _handed_over(w)
        assert _wait_until(lambda: _state(w) == "qa_window"), _state(w)
        assert w.acked == [f"{NAVEL}_greeting",
                           f"{NAVEL}_project_problem",
                           f"{NAVEL}_project_approach",
                           f"{NAVEL}_project_impact",
                           f"qa_invite_{NAVEL}"]
        assert all(kind == "step" for kind, _ in w.voice.said)

    def test_a_question_gets_a_streamed_answer(self, wired):
        w = wired
        _handed_over(w)
        assert _wait_until(lambda: _state(w) == "qa_window")
        _hear(w, "How does Method B choose gestures?")
        assert _wait_until(lambda: ("sentence", ANSWER) in w.voice.said)

    def test_no_more_questions_hands_back_and_completes(self, wired):
        w = wired
        _handed_over(w)
        assert _wait_until(lambda: _state(w) == "qa_window")
        _hear(w, "no more questions, thank you")
        assert _wait_until(lambda: _state(w) == "completed"), _state(w)
        assert w.acked[-1] == "wrap_up"

    def test_speaking_over_navel_stops_it_and_opens_questions(self, wired):
        w = wired
        w.voice.hold.set()                       # the greeting never finishes
        _handed_over(w)
        assert _wait_until(lambda: w.voice.said)  # Navel is mid-sentence
        _hear(w, "Sorry, what does co-speech mean?")
        assert _wait_until(lambda: w.voice.stopped >= 1)
        assert _wait_until(lambda: _state(w) == "qa_window"), _state(w)
