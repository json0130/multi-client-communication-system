"""
tests/test_dial_forever.py
==========================
The server dials every robot that registers, and keeps dialing until it answers.

It used to dial only when an operator pressed Connect, and gave up after 10 tries,
so a robot switched on after the server (or a server restarted under a running
robot) never showed up as connected.
"""

from __future__ import annotations

import threading
import time

import pytest
from flask import Flask

import gateway.websocket_gateway as wsg
from data import robot_repo
from gateway.http_gateway import create_http_gateway
from gateway.websocket_gateway import RobotConnection, WebSocketGateway
from tests.test_qa_more_questions_race import FakeRegistry


def test_a_refused_robot_is_retried_far_past_the_old_limit(monkeypatch):
    monkeypatch.setattr(wsg, "RECONNECT_DELAY", 0)
    monkeypatch.setattr(wsg, "RECONNECT_MAX_DELAY", 0)
    attempts, closed = [], []
    conn = RobotConnection("r1", "127.0.0.1", 1, on_message=lambda *a: None,
                           on_close=closed.append)

    def refused():      # what a dial to a powered-off robot does: error, then close
        attempts.append(1)
        threading.Thread(target=conn._handle_close, args=(None, None, None), daemon=True).start()
    monkeypatch.setattr(conn, "_start_thread", refused)

    conn._handle_close(None, None, None)
    deadline = time.time() + 3
    while len(attempts) < 25 and time.time() < deadline:
        time.sleep(0.01)
    conn.disconnect()
    assert len(attempts) >= 25            # the old cap was 10
    assert closed == []                   # none of them was ever a live connection to lose


def test_a_drop_after_connecting_is_reported_once_then_retried(monkeypatch):
    monkeypatch.setattr(wsg, "RECONNECT_DELAY", 0)
    monkeypatch.setattr(wsg, "RECONNECT_MAX_DELAY", 0)
    closed, opened, dialed = [], [], threading.Event()
    conn = RobotConnection("r1", "127.0.0.1", 1, on_message=lambda *a: None,
                           on_close=closed.append, on_open=opened.append)
    monkeypatch.setattr(conn, "_start_thread", dialed.set)
    conn._on_open(None)
    assert opened == ["r1"] and conn.is_connected()
    conn._handle_close(None, None, None)
    assert closed == ["r1"] and not conn.is_connected() and dialed.is_set()
    conn.disconnect()


def test_a_stopped_connection_is_not_redialed(monkeypatch):
    dialed = []
    conn = RobotConnection("r1", "127.0.0.1", 1, on_message=lambda *a: None,
                           on_close=lambda c: None)
    monkeypatch.setattr(conn, "_start_thread", lambda: dialed.append(1))
    conn.disconnect()
    conn._handle_close(None, None, None)
    assert dialed == []


def test_registering_again_cuts_the_backoff_short(monkeypatch):
    monkeypatch.setattr(wsg, "RECONNECT_DELAY", 30)
    dialed = threading.Event()
    conn = RobotConnection("r1", "127.0.0.1", 1, on_message=lambda *a: None,
                           on_close=lambda c: None)
    monkeypatch.setattr(conn, "_start_thread", dialed.set)
    threading.Thread(target=conn._handle_close, args=(None, None, None), daemon=True).start()
    time.sleep(0.1)
    assert not dialed.is_set()            # sleeping out a 30 s backoff
    conn.retry_now()
    assert dialed.wait(2)
    conn.disconnect()


class _FakeConn:
    made = []

    def __init__(self, client_id, ip, port, on_message, on_close, on_open=None):
        self.client_id, self.ip, self.port = client_id, ip, port
        self.up, self.disconnected, self.retried = False, False, 0
        _FakeConn.made.append(self)

    def connect(self):
        pass

    def disconnect(self):
        self.disconnected = True

    def is_connected(self):
        return self.up

    def retry_now(self):
        self.retried += 1


@pytest.fixture
def gw(monkeypatch):
    _FakeConn.made = []
    monkeypatch.setattr(wsg, "RobotConnection", _FakeConn)
    addr = {"navel_001": ("10.0.0.5", 8765)}
    monkeypatch.setattr(robot_repo, "get_robot_address", lambda cid: addr.get(cid))
    g = WebSocketGateway(FakeRegistry([]))
    g.addr = addr
    return g


def test_connect_robot_dials_once_and_nudges_a_robot_that_is_not_up(gw):
    assert gw.connect_robot("navel_001") and gw.connect_robot("navel_001")
    (conn,) = _FakeConn.made
    assert conn.retried == 1              # the second registration, while it was still down
    conn.up = True
    gw.connect_robot("navel_001")
    assert conn.retried == 1              # already up: left alone


def test_a_robot_that_moves_is_dialed_at_its_new_address(gw):
    gw.connect_robot("navel_001")
    gw.addr["navel_001"] = ("10.0.0.9", 8765)
    gw.connect_robot("navel_001")
    old, new = _FakeConn.made
    assert old.disconnected and new.ip == "10.0.0.9"


def test_only_robots_that_answered_count_as_connected(gw):
    gw.connect_robot("navel_001")
    assert gw.get_connected_ids() == []   # dialing is not connected
    _FakeConn.made[0].up = True
    assert gw.get_connected_ids() == ["navel_001"]


def test_unknown_address_is_not_dialed(gw):
    assert gw.connect_robot("nobody") is False and _FakeConn.made == []


class _WS:
    def __init__(self):
        self.dialed = []

    def connect_robot(self, cid):
        self.dialed.append(cid)


@pytest.fixture
def client(monkeypatch):
    class Row:
        client_id, robot_name = "navel_001", "Navel"
    monkeypatch.setattr(robot_repo, "upsert_robot", lambda **kw: Row())
    ws = _WS()
    app = Flask(__name__)
    app.register_blueprint(create_http_gateway(registry=None, ws_gateway=ws))
    c = app.test_client()
    c.ws = ws
    return c


def test_registering_dials_the_robot(client):
    r = client.post("/robots/register", json={"client_id": "navel_001", "robot_name": "Navel"})
    assert r.status_code == 200 and client.ws.dialed == ["navel_001"]


def test_a_robot_that_dials_in_itself_is_not_dialed(client):
    client.post("/robots/register", json={"client_id": "navel_001", "robot_name": "Navel",
                                          "connect_mode": "inbound"})
    assert client.ws.dialed == []
