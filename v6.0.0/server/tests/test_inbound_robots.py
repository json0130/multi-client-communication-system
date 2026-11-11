"""
tests/test_inbound_robots.py
============================
A robot the server cannot reach (the Jetson ChatBox, behind NAT / on Tailscale) dials IN to the server
instead of waiting to be dialled. The server treats the socket like any other robot connection.
"""

from __future__ import annotations

import json
import socket
import time

import pytest
from websockets.sync.client import connect

from gateway.websocket_gateway import WebSocketGateway


class _Registry:
    def __init__(self, known=("chatbox_jetson_001",)):
        self.known, self.connected, self.disconnected = set(known), [], []

    def connect(self, cid):
        if cid not in self.known:
            return None
        self.connected.append(cid)
        return object()

    def disconnect(self, cid):
        self.disconnected.append(cid)

    def get(self, cid):
        return object() if cid in self.connected else None


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _wait(cond, t=5.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


@pytest.fixture
def server():
    registry, heard = _Registry(), []
    gw = WebSocketGateway(registry, recorder=None)
    gw._on_message = lambda cid, data, *a: heard.append((cid, data))
    port = _port()
    gw.start_inbound_server("127.0.0.1", port)
    assert _wait(lambda: _can_connect(port))
    return gw, registry, heard, port


def _can_connect(port):
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close(); return True
    except OSError:
        return False


def _hello(port, cid="chatbox_jetson_001", token=""):
    ws = connect(f"ws://127.0.0.1:{port}")
    ws.send(json.dumps({"type": "hello", "client_id": cid, "token": token}))
    return ws


def test_a_robot_that_dials_in_is_connected_and_can_be_sent_to_and_heard(server):
    gw, registry, heard, port = server
    ws = _hello(port)
    assert _wait(lambda: gw.get_connected_ids() == ["chatbox_jetson_001"])
    assert registry.connected == ["chatbox_jetson_001"]            # the registry made its instance, as for a dialled-out robot

    gw.send_to_robot("chatbox_jetson_001", {"event": "demo_step", "step_id": "s1", "text": "hello"})
    assert json.loads(ws.recv(timeout=5))["step_id"] == "s1"

    ws.send(json.dumps({"type": "ack", "step_id": "s1"}))
    assert _wait(lambda: ("chatbox_jetson_001", {"type": "ack", "step_id": "s1"}) in heard)
    ws.close()


def test_dropping_the_connection_marks_the_robot_gone(server):
    gw, registry, heard, port = server
    ws = _hello(port)
    assert _wait(lambda: gw.get_connected_ids())
    ws.close()
    assert _wait(lambda: registry.disconnected == ["chatbox_jetson_001"] and not gw.get_connected_ids())


def test_a_redial_replaces_the_old_connection_without_dropping_the_new_one(server):
    gw, registry, heard, port = server
    first = _hello(port)
    assert _wait(lambda: len(registry.connected) == 1)
    second = _hello(port)
    assert _wait(lambda: len(registry.connected) == 2)
    time.sleep(0.5)
    assert gw.get_connected_ids() == ["chatbox_jetson_001"]
    assert registry.disconnected == []                              # the stale one closing did not unregister the live one
    gw.send_to_robot("chatbox_jetson_001", {"event": "tts_stop"})
    assert json.loads(second.recv(timeout=5))["event"] == "tts_stop"
    first.close(); second.close()


def test_an_unregistered_robot_and_a_bad_token_are_refused():
    registry = _Registry()
    gw = WebSocketGateway(registry, recorder=None)
    port = _port()
    gw.start_inbound_server("127.0.0.1", port, token="secret")
    assert _wait(lambda: _can_connect(port))
    for kwargs in ({"token": "wrong"}, {"token": "secret", "cid": "nobody_001"}):
        ws = _hello(port, **kwargs)
        assert _wait(lambda: not gw.get_connected_ids())
        with pytest.raises(Exception):
            ws.recv(timeout=3)                                      # the server closed it
    assert registry.connected == []


def test_connect_on_a_robot_that_registered_as_dialling_in_does_not_dial_out():
    gw = WebSocketGateway(_Registry(), recorder=None)
    gw.mark_inbound("chatbox_jetson_001")
    assert gw.connect_robot("chatbox_jetson_001") is True          # nothing to initiate, and no database address needed
    assert gw.get_connected_ids() == []
