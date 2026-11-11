"""
tests/test_socketio_robots.py
=============================
The ChatBox client (unchanged) connects to `server_url` over Socket.IO and emits client_init. The server
takes it into the same pool as every other robot, on the same port as the REST API.
"""

from __future__ import annotations

import threading

import pytest
import socketio
from flask import Flask
from werkzeug.serving import make_server

from gateway.websocket_gateway import WebSocketGateway
from tests.test_inbound_robots import _Registry, _wait


@pytest.fixture
def server():
    registry, heard = _Registry(), []
    gw = WebSocketGateway(registry, recorder=None)
    gw._on_message = lambda cid, data, *a: heard.append((cid, data))
    api = Flask(__name__)
    api.add_url_rule("/health", "health", lambda: "ok")
    srv = make_server("127.0.0.1", 0, api, threaded=True)
    api.wsgi_app = gw.socketio_app(api.wsgi_app)                       # what app.py does
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield gw, registry, heard, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _robot(url, cid="chatbox_jetson_001"):
    sio, got, init = socketio.Client(), [], []
    sio.on("client_init_response", init.append)
    for ev in ("demo_step", "persona_update", "tts_stop"):
        sio.on(ev, lambda d, ev=ev: got.append((ev, d)))
    sio.connect(url, transports=["websocket", "polling"])
    sio.emit("client_init", {"robot_name": "ChatBox", "client_id": cid, "modules": ["gpt"]})
    return sio, got, init


def test_the_chatbox_socketio_client_is_connected_sent_to_and_heard(server):
    gw, registry, heard, url = server
    sio, got, init = _robot(url)
    assert _wait(lambda: gw.get_connected_ids() == ["chatbox_jetson_001"]) and _wait(lambda: init)
    assert init[0]["success"] and registry.connected == ["chatbox_jetson_001"]

    gw.send_to_robot("chatbox_jetson_001", {"event": "demo_step", "step_id": "s1", "text": "[HAPPY] hi"})
    assert _wait(lambda: got) and got[0][0] == "demo_step" and got[0][1]["step_id"] == "s1"

    sio.emit("ack", {"type": "ack", "step_id": "s1"})                  # what ChatBox client.send() emits
    assert _wait(lambda: ("chatbox_jetson_001", {"type": "ack", "step_id": "s1"}) in heard)
    sio.disconnect()
    assert _wait(lambda: registry.disconnected == ["chatbox_jetson_001"] and not gw.get_connected_ids())


def test_an_unregistered_robot_is_told_so_and_not_pooled(server):
    gw, registry, heard, url = server
    sio, got, init = _robot(url, cid="stranger")
    assert _wait(lambda: init) and not init[0]["success"] and "register" in init[0]["message"]
    assert gw.get_connected_ids() == []
    sio.disconnect()


def test_the_rest_api_still_answers_on_the_same_port(server):
    import urllib.request
    assert urllib.request.urlopen(server[3] + "/health", timeout=3).read() == b"ok"
