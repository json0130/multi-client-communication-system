"""
test_dial_out.py — a client with `connect_to_server` dials the central server instead of listening.

Run from v6.0.0/client:  python3 -m pytest test_dial_out.py
"""

import json
import socket
import threading
import time

from websockets.sync.server import serve

from client import ServerConnection


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _wait(cond, t=8.0):
    end = time.time() + t
    while time.time() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_the_client_dials_in_says_hello_and_then_behaves_like_a_dialled_connection():
    port, hellos, received, sockets = _port(), [], [], []

    def handler(ws):
        hellos.append(json.loads(ws.recv()))
        sockets.append(ws)
        ws.send(json.dumps({"event": "demo_step", "step_id": "s1", "text": "hi"}))
        for raw in ws:
            received.append(json.loads(raw))

    server = serve(handler, "127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    got = []
    conn = ServerConnection(8765, f"http://127.0.0.1:5000",
                            {"client_id": "chatbox_jetson_001", "connect_to_server": True, "server_ws_port": port})
    conn.register_handler("demo_step", got.append)
    conn.start_server()
    try:
        assert _wait(lambda: hellos and got)
        assert hellos[0]["type"] == "hello" and hellos[0]["client_id"] == "chatbox_jetson_001"
        assert got[0]["step_id"] == "s1"                            # server -> robot, same handlers as ever
        assert conn.send({"type": "ack", "step_id": "s1"})          # robot -> server
        assert _wait(lambda: received == [{"type": "ack", "step_id": "s1"}])

        sockets[0].close()                                          # the server restarts: the client redials by itself
        assert _wait(lambda: len(hellos) == 2)
    finally:
        server.shutdown()


def test_without_the_setting_the_client_still_listens():
    conn = ServerConnection(_port(), "http://127.0.0.1:5000", {"client_id": "x"})
    assert not conn.client_config.get("connect_to_server")


def test_a_web_page_request_to_a_robots_websocket_port_is_not_logged_as_an_error():
    """A browser tab or bookmark on a robot's port is refused with 426; that is not worth a traceback."""
    import logging
    from client import _QuietWebProbes
    f = _QuietWebProbes()

    def rec(msg, exc=None):
        return logging.LogRecord("websockets.server", logging.ERROR, __file__, 1, msg, None,
                                 (type(exc), exc, None) if exc else None)

    class InvalidUpgrade(Exception):
        pass

    assert not f.filter(rec("connection rejected (426 Upgrade Required)"))
    assert not f.filter(rec("opening handshake failed", InvalidUpgrade("invalid Connection header: keep-alive")))
    assert f.filter(rec("opening handshake failed", ValueError("something real")))
    assert f.filter(rec("connection open"))
