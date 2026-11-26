"""
ai_server.py — the lab AI server's models (LLM, STT, TTS, FACE, EMOTION) for an adapter.
=========================================================================================
For a robot that has no voice, ears or camera models of its own:

    from ai_server import AIServer
    ai = AIServer.from_config(self.config)      # "ai_server": "HOST[:PORT]", or AI_HOST / AI_PORT
    wav = ai.tts("Hello there.")                # bytes of a WAV, ready to play
    text = ai.stt(open("q.wav", "rb").read())   # transcript
    faces = ai.face(jpeg_bytes, "recognize")    # [{"box": ..., "name": ...}, ...]

The CARES server already does the conversation (LLM) and transcribes the robot's
microphone itself; use these when the adapter needs a model on the robot's side.

Wire protocol (the AI server's, TCP, default port 7898): the sender writes the JSON
length as ASCII digits, waits for them to be echoed back, then writes the JSON; the
reply comes back the same way. One request per connection.
    request : {robot_id, request_id, command: ["TTS"], parameters, content}
    reply   : same ids, command [] on success, ["ERR"] with the message in content
Stdlib only. The server's twin of this is server/modules/ai_server_llm.py.
"""

import base64
import io
import json
import os
import socket
import uuid
import wave

DEFAULT_PORT = 7898
MAX_BYTES = 8 * 1024 * 1024


class AIServerError(RuntimeError):
    """Unreachable, a protocol fault, or an ERR reply."""


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise AIServerError("the AI server closed the connection mid-message")
        buf.extend(chunk)
    return bytes(buf)


def _send(sock, message: dict) -> None:
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    if not 0 < len(body) <= MAX_BYTES:
        raise AIServerError("request over the 8 MiB limit")
    header = str(len(body)).encode("ascii")
    sock.sendall(header)
    if _recv_exact(sock, len(header)) != header:
        raise AIServerError("the AI server did not echo the length")
    sock.sendall(body)


def _recv(sock) -> dict:
    # ponytail: one recv for the digits, as the server team's own client does; the sender
    # writes nothing more until we echo them, so they arrive alone.
    header = sock.recv(32)
    if not header.isdigit() or not 0 < int(header) <= MAX_BYTES:
        raise AIServerError("bad length header from the AI server")
    sock.sendall(header)
    return json.loads(_recv_exact(sock, int(header)))


class AIServer:
    def __init__(self, host: str, port: int = DEFAULT_PORT, robot_id: str = "robot", timeout: float = 60.0):
        if not host:
            raise AIServerError('no AI server: set "ai_server": "HOST[:PORT]" in the config, or AI_HOST')
        self.host, self.port, self.robot_id, self.timeout = host, int(port), robot_id, timeout

    @classmethod
    def from_config(cls, config: dict, **kw) -> "AIServer":
        spec = config.get("ai_server") or os.environ.get("AI_HOST", "")
        host, _, port = spec.partition(":")
        return cls(host, int(port or os.environ.get("AI_PORT", DEFAULT_PORT)),
                   config.get("client_id", "robot"), **kw)

    def request(self, command: str, content: str = "", parameters: dict = None) -> dict:
        """One request; the reply dict (`content`, `parameters`). Raises AIServerError."""
        rid = str(uuid.uuid4())
        msg = {"robot_id": self.robot_id, "request_id": rid, "command": [command],
               "parameters": parameters or {}, "content": content}
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                _send(sock, msg)
                reply = _recv(sock)
        except (OSError, ValueError) as e:
            raise AIServerError(f"{command} on {self.host}:{self.port}: {e}") from e
        if reply.get("command") == ["ERR"]:
            raise AIServerError(f"{command}: {reply.get('content')}")
        if reply.get("request_id") != rid:
            raise AIServerError(f"{command}: reply to another request")
        return reply

    def llm(self, prompt: str) -> str:
        return self.request("LLM", prompt)["content"]

    def tts(self, text: str, voice: str = "af_heart") -> bytes:
        """A WAV file's bytes."""
        return base64.b64decode(self.request("TTS", text, {"voice": voice})["content"])

    def stt(self, wav: bytes) -> str:
        """Transcript of a WAV file's bytes (PCM, up to 120 s)."""
        with wave.open(io.BytesIO(wav)) as w:
            params = {"format": "raw_pcm", "dtype": f"int{8 * w.getsampwidth()}",
                      "sample_rate": w.getframerate(), "channels": w.getnchannels()}
            pcm = w.readframes(w.getnframes())
        return self.request("STT", base64.b64encode(pcm).decode("ascii"), params)["content"]

    def face(self, image: bytes, operation: str = "detect", name: str = None, fmt: str = "jpeg") -> list:
        """operation: detect | recognize | enroll (with `name`). The detections."""
        params = {"operation": operation, "format": fmt, **({"name": name} if name else {})}
        reply = self.request("FACE", base64.b64encode(image).decode("ascii"), params)
        return reply.get("parameters", {}).get("detections", [])

    def emotion(self, image: bytes, fmt: str = "jpeg") -> list:
        """[{"emotion", "confidence" (percent), "box"}, ...]"""
        reply = self.request("EMOTION", base64.b64encode(image).decode("ascii"), {"format": fmt})
        return reply.get("parameters", {}).get("detections", [])
