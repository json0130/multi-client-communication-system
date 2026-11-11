"""
tests/test_shutdown.py
======================
Ctrl+C must be able to finish. shutdown() used to disconnect every robot while holding the connection lock;
closing a dial-in (Socket.IO) link runs its close handler on the same thread, and that handler takes the lock,
so the process hung and every further Ctrl+C re-entered the stuck handler.
"""
from __future__ import annotations

import threading

from gateway.websocket_gateway import WebSocketGateway


class _Registry:
    def disconnect(self, cid):
        pass

    def shutdown(self):
        pass

    def get(self, cid):
        return None


class _DialInLink:
    """Disconnecting it reports the close straight back, on the caller's thread, like sio.disconnect does."""

    def __init__(self, gw, cid):
        self.gw, self.cid = gw, cid

    def is_connected(self):
        return False

    def disconnect(self):
        self.gw._on_robot_close(self.cid)


def test_shutdown_does_not_deadlock_on_a_dial_in_link():
    gw = WebSocketGateway(_Registry())
    gw._connections["chatbox_jetson_001"] = _DialInLink(gw, "chatbox_jetson_001")
    t = threading.Thread(target=gw.shutdown, daemon=True)
    t.start()
    t.join(3)
    assert not t.is_alive() and gw._connections == {}
