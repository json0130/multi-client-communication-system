"""
tests/test_roles_gateway.py
===========================
The Roles page on the server's own robot connections: GatewayHub (the persona stack's hub interface over
WebSocketGateway), the ack routing for its steps, and the page's HTTP API with a stub LLM.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from gateway.roles_gateway import DEFAULT_ROBOTS, GatewayHub, start_roles_service
from gateway.websocket_gateway import WebSocketGateway

OCEAN = {"O": 0.6, "C": 0.8, "E": -0.3, "A": 0.7, "N": -0.5}


NAMES = {"chatbox_jetson_001": "ChatBox", "navel_001": "Navel",        # the real robots
         "chatbox_01": "ChatBox", "navel_01": "Navel", "pepper_01": "Pepper"}      # the fake ones (client/test_configs)


class _Robot:
    def __init__(self, name):
        self.robot_name = name


class _Registry:
    """the server's robot registry: get(client_id).robot_name"""

    def __init__(self, ids=()):
        self.ids = set(ids)

    def get(self, cid):
        return _Robot(NAMES[cid]) if cid in self.ids else None


class FakeWS:
    """what GatewayHub needs of WebSocketGateway; `acker` plays a robot that acks every step it is told"""

    def __init__(self, up=("chatbox_jetson_001", "navel_001"), acker=False):
        self.ack_listeners, self.demo_start_listeners, self.up, self.sent, self.acker = [], [], list(up), [], acker
        self.registry = _Registry(up)

    def get_connected_ids(self):
        return list(self.up)

    def send_to_robot(self, cid, data):
        self.sent.append((cid, data))
        if self.acker and data.get("require_ack"):
            threading.Timer(0.05, lambda: [f(data["step_id"]) for f in self.ack_listeners]).start()


def test_persona_goes_to_each_robot_in_the_form_its_client_takes():
    ws, saved = FakeWS(), []
    hub = GatewayHub(ws, DEFAULT_ROBOTS, on_persona=lambda *a: saved.append(a), registry=ws.registry)
    out = hub.push_persona("Tutor", "teach", OCEAN, {"rate": "-9%", "pitch": "-1.0st"}, "chatbox")
    assert out == {"robots": 1, "sent": True}
    cid, p = ws.sent[0]
    assert cid == "chatbox_jetson_001" and p["event"] == "persona_update" and p["robot_role"] == "teach"
    assert p["personality"]["O"] == 0.8 and p["voice_config"]["rate"] == "-9%" and p["voice_config"]["language"] == "en"
    hub.push_persona("Tutor", "teach", OCEAN, {"rate": "-9%", "pitch": "-1.0st"}, "navel")
    cid, p = ws.sent[1]
    assert cid == "navel_001" and set(p) == {"event", "persona_name", "personality"}     # Navel keeps its own role/voice
    assert hub.voices["navel"]["pitch"] == "-1.0st"
    assert [(c, n) for c, n, _ in saved] == [("chatbox_jetson_001", "Tutor"), ("navel_001", "Tutor")]


def test_only_connected_robots_are_addressed():
    ws = FakeWS(up=["navel_001"])
    hub = GatewayHub(ws, DEFAULT_ROBOTS, registry=ws.registry)
    assert [r["robot_name"] for r in hub.robots()] == ["Navel"] and hub.robots("chatbox") == []
    assert hub.say("hi", "chatbox") == {"robots": 0, "sent": False}
    assert not ws.sent and "ChatBox" in hub.describe("chatbox")


@pytest.mark.parametrize("chatbox, navel", [("chatbox_jetson_001", "navel_001"),      # real robots
                                            ("chatbox_01", "navel_01")])              # fake ones
def test_robots_are_matched_by_name_whatever_their_client_ids(chatbox, navel):
    ws = FakeWS(up=[chatbox, navel, "pepper_01"])                # Pepper is connected too and is no Roles robot
    hub = GatewayHub(ws, DEFAULT_ROBOTS, registry=ws.registry)
    assert {r["robot_name"]: r["client_id"] for r in hub.robots()} == {"ChatBox": chatbox, "Navel": navel}
    hub.push_persona("Tutor", "teach", OCEAN, robot="navel")
    hub.say("hi", "chatbox")
    assert [(c, d["event"]) for c, d in ws.sent] == [(navel, "persona_update"), (chatbox, "demo_step")]


def test_a_pinned_client_id_wins_and_an_unnamed_connection_is_ignored():
    ws = FakeWS(up=["navel_01", "navel_001"])
    pinned = [dict(DEFAULT_ROBOTS[1], client_id="navel_001")]
    assert GatewayHub(ws, pinned, registry=ws.registry).robots()[0]["client_id"] == "navel_001"
    assert GatewayHub(ws, DEFAULT_ROBOTS[1:], registry=ws.registry).robots()[0]["client_id"] == "navel_01"
    assert GatewayHub(ws, DEFAULT_ROBOTS[1:]).robots() == []     # no registry: nothing to match names against


def test_saving_a_role_uses_the_connected_client_id():
    ws, saved = FakeWS(up=["chatbox_01", "navel_01"]), []
    hub = GatewayHub(ws, DEFAULT_ROBOTS, on_persona=lambda *a: saved.append(a), registry=ws.registry)
    hub.push_persona("Tutor", "teach", OCEAN, robot="navel")
    assert saved == [("navel_01", "Tutor", "teach")]


def test_say_steps_waits_for_each_ack_and_navel_gets_voice_tags():
    ws = FakeWS(acker=True)
    hub = GatewayHub(ws, DEFAULT_ROBOTS, registry=ws.registry)
    hub.set_voice("navel", {"rate": "+0%", "pitch": "-5.0st", "volume": "+0dB"})
    out = hub.say_steps(["[GREET] Hello!", "Nice to meet you."], robot="navel")
    assert out == {"robots": 1, "sent": True, "steps": 2}
    texts = [d["text"] for _, d in ws.sent]
    assert texts[0].startswith("<vol,") and texts[0].endswith("[GREET] Hello!") and texts[1].endswith("Nice to meet you.")
    assert all(d["step_id"].startswith("persona-") and d["require_ack"] for _, d in ws.sent)
    assert not any(d["text"] == "[DEFAULT]" for _, d in ws.sent)                         # the ChatBox's pose tag only
    ws.sent.clear()
    hub.say_steps(["Hi"], robot="chatbox")
    assert ws.sent[-1][1]["text"] == "[DEFAULT]"


def test_gateway_routes_persona_acks_to_the_roles_page_not_the_demo():
    gw = WebSocketGateway(_Registry(), recorder=None)
    heard, demo = [], []
    gw.ack_listeners.append(heard.append)
    gw._demo_orchestrator = type("O", (), {"receive_ack": lambda self, s: demo.append(s)})()
    gw._registry = type("R", (), {"get": lambda self, cid: object()})()
    gw._on_message("navel_001", {"type": "ack", "step_id": "persona-3"})
    gw._on_message("navel_001", {"type": "ack", "step_id": "step-7"})
    assert heard == ["persona-3"] and demo == ["step-7"]


class StubLLM:
    available = True
    label = "stub"

    def respond(self, system, user, history, max_tokens=140):
        return "[HAPPY] Hello there. Welcome to the lab."


@pytest.fixture
def page(tmp_path):
    ws = FakeWS(acker=True)
    server, hub = start_roles_service(ws, str(tmp_path), port=0, host="127.0.0.1", action_tags=False, llm=StubLLM(),
                                    server_registry=ws.registry)
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)
    yield call, ws, hub
    server.shutdown()


def test_a_relative_data_dir_still_saves_a_persona(tmp_path, monkeypatch):
    """the default ROLES_DATA_DIR is './roles_data'; assigning a role used to look for roles_data/./roles_data/"""
    monkeypatch.chdir(tmp_path)
    ws = FakeWS(acker=True)
    server, _hub = start_roles_service(ws, "./roles_data", port=0, host="127.0.0.1", action_tags=False,
                                       llm=StubLLM(), server_registry=ws.registry)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/persona",
                                     data=json.dumps({"robot": "navel", "role": "role:tutor", "ocean": OCEAN}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            assert json.load(r)["ok"] is True
    finally:
        server.shutdown()


def test_the_page_lists_the_servers_robots_and_the_merged_graph(page):
    call, ws, hub = page
    code, robots = call("/robots")
    assert code == 200 and robots["workspaces"] == ["chatbox", "navel"] and robots["manage"] is False
    assert {r["robot_name"] for r in robots["robots"]} == {"ChatBox", "Navel"}
    ids = {n["id"] for n in call("/graph.json")[1]["nodes"]}
    assert {"chatbox", "navel"} <= ids
    assert any(i.startswith("role:") for i in ids)                                      # the shared role library
    assert call("/registry/add", {"name": "Stuart", "kind": "connects-in"})[0] == 400   # robots are added on the server's tab


def test_assigning_a_role_from_the_page_reaches_that_robot_only(page):
    call, ws, hub = page
    code, a = call("/persona", {"robot": "navel", "role": "role:tutor", "ocean": OCEAN})
    assert code == 200 and a["robot"] == {"robots": 1, "sent": True}
    assert [c for c, d in ws.sent if d["event"] == "persona_update"] == ["navel_001"]
    assert call("/persona?robot=navel")[1]["assigned"]["navel"]["role"] == "role:tutor"
    assert call("/persona?robot=chatbox")[1]["assigned"].get("chatbox") is None


def test_a_chat_reply_is_spoken_by_the_robot_you_talk_to(page):
    call, ws, hub = page
    code, out = call("/chat", {"message": "hi", "robot": "navel"})
    assert code == 200 and out["robot"]["sent"]
    deadline = time.time() + 5
    while time.time() < deadline and not any(d["event"] == "demo_step" for _, d in ws.sent):
        time.sleep(0.05)
    steps = [(c, d) for c, d in ws.sent if d["event"] == "demo_step"]
    assert steps and all(c == "navel_001" for c, _ in steps) and "Welcome to the lab" in steps[0][1]["text"]


def test_action_buttons_per_robot(page):
    call, ws, hub = page
    assert call("/action", {"robot": "navel", "tag": "agree"})[0] == 200
    assert ws.sent[-1][0] == "navel_001" and ws.sent[-1][1]["text"] == "[AGREE]"
    assert call("/action", {"robot": "navel", "tag": "GREETING"})[0] == 400                 # a ChatBox tag


# ── Roles page next to a real demo run ────────────────────────────────────────────

CHATBOX, NAVEL = "chatbox_jetson_001", "navel_001"


class _Conn:
    """A robot connected to the real gateway: records what it is sent, and acks every step like the client does."""

    def __init__(self, gw, cid):
        self.gw, self.cid, self.got = gw, cid, []

    def is_connected(self):
        return True

    def send(self, data):
        self.got.append(data)
        if data.get("require_ack"):
            threading.Timer(0.05, self.gw._on_message, (self.cid, {"type": "ack", "step_id": data["step_id"]})).start()


def _wait(cond, t=5.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.03)
    return cond()


def test_roles_page_works_during_a_demo_run_and_after_it_is_stopped(tmp_path, monkeypatch):
    from demo.demo_orchestrator import DemoOrchestrator, DemoState
    from tests.test_stations_and_names import _Registry as Reg, _Robot
    monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds", lambda _t: 0.0)
    monkeypatch.setattr("conversation.turn._speaking_seconds", lambda _t: 0.0)
    monkeypatch.setattr("demo.demo_orchestrator._reset_gazebo_positions", lambda: None)

    gw = WebSocketGateway(Reg([_Robot(CHATBOX, "ChatBox"), _Robot(NAVEL, "Navel")]), recorder=None)
    conns = gw._connections = {CHATBOX: _Conn(gw, CHATBOX), NAVEL: _Conn(gw, NAVEL)}
    orch = DemoOrchestrator(gw, transition_delay=0.0)
    gw.set_demo_orchestrator(orch)
    server, hub = start_roles_service(gw, str(tmp_path), port=0, host="127.0.0.1", action_tags=False, llm=StubLLM(),
                                    server_registry=gw._registry)
    base = f"http://127.0.0.1:{server.server_port}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    try:
        orch.start(robot_ids=[CHATBOX, NAVEL])                       # Start on the dashboard: the ChatBox-led pair
        assert _wait(lambda: orch.get_status()["state"] == DemoState.AWAITING_CUE.value)
        # with the demo running (waiting for the guide's cue) the page still reaches both robots
        assert {r["robot_name"] for r in call("/robots")["robots"]} == {"ChatBox", "Navel"}
        call("/persona", {"robot": "navel", "role": "role:tutor", "ocean": OCEAN})
        assert any(d["event"] == "persona_update" for d in conns[NAVEL].got)
        before = len(conns[NAVEL].got)

        orch.stop()                                                  # Stop
        assert orch.get_status()["state"] == DemoState.IDLE.value
        assert set(gw.get_connected_ids()) == {CHATBOX, NAVEL}       # stopping does not drop the robots

        # after the run: assign roles and talk, on the same connections
        call("/persona", {"robot": "chatbox", "role": "role:storyteller", "ocean": OCEAN})
        assert any(d["event"] == "persona_update" and d.get("persona_name") for d in conns[CHATBOX].got)
        t0 = time.time()
        assert call("/chat", {"message": "hi", "robot": "navel"})["robot"]["sent"]
        assert _wait(lambda: len(conns[NAVEL].got) > before)
        assert _wait(lambda: not hub._acks)                          # every step was acked and routed to the page
        assert time.time() - t0 < 5
        assert all(d["step_id"].startswith("persona-") for d in conns[NAVEL].got[before:] if d["event"] == "demo_step")
        assert call("/persona?robot=navel")["assigned"]["navel"]["role"] == "role:tutor"
    finally:
        orch.stop()
        orch._end_run()
        server.shutdown()


def test_starting_the_demo_puts_the_base_voice_back_not_the_personas():
    ws = FakeWS()
    hub = GatewayHub(ws, DEFAULT_ROBOTS, registry=ws.registry)
    hub.push_persona("Tutor", "teach", OCEAN, {"rate": "-9%", "pitch": "-1.0st"})
    ws.sent.clear()
    for fn in ws.demo_start_listeners:
        fn()
    assert hub.voices == {}                                                    # Navel's persona tags are dropped
    assert [c for c, _ in ws.sent] == ["chatbox_jetson_001"]                   # Navel's client keeps its own voice
    v = ws.sent[0][1]["voice_config"]
    assert v["rate"] == "+0%" and v["pitch"] == "+0.0st" and v["language"] == "en"
    ws.sent.clear()
    for fn in ws.demo_start_listeners:
        fn()
    assert ws.sent == []                                                       # nothing assigned since: nothing to undo
