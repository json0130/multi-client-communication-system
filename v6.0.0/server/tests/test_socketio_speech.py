"""
tests/test_socketio_speech.py
=============================
Speech from the ChatBox Socket.IO client: one base64 WAV per spoken turn, in a single message.
"""

from __future__ import annotations

import base64
import io
import wave

import pytest

from tests.test_inbound_robots import _wait
from tests.test_socketio_robots import _robot, server  # noqa: F401  (the fixture)


def _wav_b64(seconds, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x01\x02" * int(rate * seconds))
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("seconds", [3, 20, 30])
def test_a_spoken_turn_of_any_length_the_client_can_record_reaches_the_gateway(server, seconds):
    gw, registry, heard, url = server
    sio, got, init = _robot(url)
    assert _wait(lambda: init)
    audio = _wav_b64(seconds)
    sio.emit("speech", {"type": "speech", "audio": audio})
    assert _wait(lambda: any(d.get("type") == "speech" and d.get("audio") == audio for _, d in heard), t=8)
    sio.disconnect()
