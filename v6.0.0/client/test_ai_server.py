"""
test_ai_server.py — ai_server.py against a fake AI server speaking the same wire protocol.

Run from the client folder:  python3 -m pytest test_ai_server.py
"""

import base64
import io
import json
import socket
import threading
import wave

import pytest

from ai_server import AIServer, AIServerError


def _wav(frames=b"\x00\x01" * 160, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(frames)
    return buf.getvalue()


def _exact(conn, n):
    buf = b""
    while len(buf) < n:
        buf += conn.recv(n - len(buf))
    return buf


@pytest.fixture
def fake():
    """Answers each request with handle(request) -> (command, content, parameters); records requests."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen()
    seen, handle = [], {}

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                header = conn.recv(32); conn.sendall(header)
                req = json.loads(_exact(conn, int(header)))
                seen.append(req)
                command, content, params = handle["f"](req)
                body = json.dumps({"robot_id": req["robot_id"], "request_id": req["request_id"],
                                   "command": command, "content": content, "parameters": params}).encode()
                conn.sendall(str(len(body)).encode())
                assert conn.recv(32) == str(len(body)).encode()      # we wait for the client's echo
                conn.sendall(body)

    threading.Thread(target=serve, daemon=True).start()
    yield AIServer("127.0.0.1", srv.getsockname()[1], robot_id="stuart_001", timeout=5), seen, handle
    srv.close()


def test_tts_returns_the_wav_and_sends_the_voice(fake):
    ai, seen, handle = fake
    handle["f"] = lambda req: ([], base64.b64encode(_wav()).decode(), {"format": "wav"})
    assert ai.tts("Hello.", voice="af_heart") == _wav()
    assert seen[0]["command"] == ["TTS"] and seen[0]["parameters"] == {"voice": "af_heart"}
    assert seen[0]["content"] == "Hello." and seen[0]["robot_id"] == "stuart_001"


def test_stt_sends_raw_pcm_with_its_format(fake):
    ai, seen, handle = fake
    handle["f"] = lambda req: ([], "where are the space books", {})
    assert ai.stt(_wav()) == "where are the space books"
    p = seen[0]["parameters"]
    assert seen[0]["command"] == ["STT"] and p == {"format": "raw_pcm", "dtype": "int16", "sample_rate": 16000, "channels": 1}
    assert base64.b64decode(seen[0]["content"]) == b"\x00\x01" * 160


def test_llm_face_and_emotion(fake):
    ai, seen, handle = fake
    handle["f"] = lambda req: ([], "Hi!", {"detections": [{"name": "Jay"}]})
    assert ai.llm("Say hi") == "Hi!"
    assert ai.face(b"jpg", "enroll", name="Jay") == [{"name": "Jay"}]
    assert seen[1]["parameters"] == {"operation": "enroll", "format": "jpeg", "name": "Jay"}
    assert ai.emotion(b"jpg") == [{"name": "Jay"}] and seen[2]["command"] == ["EMOTION"]


def test_an_err_reply_raises(fake):
    ai, _, handle = fake
    handle["f"] = lambda req: (["ERR"], "Unknown local TTS voice", {})
    with pytest.raises(AIServerError, match="Unknown local TTS voice"):
        ai.tts("Hi", voice="nope")


def test_config_and_unreachable(monkeypatch):
    monkeypatch.delenv("AI_HOST", raising=False)
    ai = AIServer.from_config({"ai_server": "10.0.0.5:7000", "client_id": "r_001"})
    assert (ai.host, ai.port, ai.robot_id) == ("10.0.0.5", 7000, "r_001")
    with pytest.raises(AIServerError, match="no AI server"):
        AIServer.from_config({})
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    with pytest.raises(AIServerError):
        AIServer("127.0.0.1", port, timeout=2).llm("hi")
