"""
LLM client for the lab's AI server (TCP, length-prefixed JSON). Stdlib only, no
imports from other repos or ChatBox modules, so it can be dropped in beside
`LLMClient` in webcam_loop.

Wire protocol (the server's): the sender writes the payload length as ASCII digits,
waits for the server to echo those digits back, then writes the JSON payload; the
reply comes back the same way. One request per connection.

    request : {robot_id, request_id, command: ["LLM"], parameters, content}
    reply   : same keys, command == [] on success, ["ERR"] with the message in content

The server takes ONE prompt string, so system prompt + history are flattened into
"System: … / User: … / Assistant:" turns (same as the server's own example client).

Same surface as webcam_loop.LLMClient — `model`, `available`, `connect()`,
`respond(system_prompt, user_msg, history, max_tokens)` — so it is a drop-in.
"""
from __future__ import annotations

import json
import os
import re
import socket
import uuid
from typing import Optional

DEFAULT_HOST = os.environ.get("AI_HOST", "")
DEFAULT_PORT = int(os.environ.get("AI_PORT", 7898))
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024

# The model may keep writing the next turn, or (for reasoning models) emit its
# thinking before "</think>". Keep only the answer.
_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)
_NEXT_TURN_RE = re.compile(r"\n\s*(User|Assistant|System):")


class AIServerError(RuntimeError):
    """Server unreachable, protocol violation, or an ERR reply."""


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(min(size - len(buf), 65536))
        if not chunk:
            raise EOFError("server closed the connection mid-message")
        buf.extend(chunk)
    return bytes(buf)


def _send_json(sock: socket.socket, message: dict) -> None:
    payload = json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if not 0 < len(payload) <= _MAX_MESSAGE_BYTES:
        raise AIServerError("message exceeds the 8 MiB protocol limit")
    header = str(len(payload)).encode("ascii")
    sock.sendall(header)
    if _recv_exact(sock, len(header)) != header:
        raise AIServerError("server did not echo the message length")
    sock.sendall(payload)


def _recv_json(sock: socket.socket) -> dict:
    header = sock.recv(32)
    if not header:
        raise EOFError("server closed the connection")
    if not header.isdigit() or not 0 < int(header) <= _MAX_MESSAGE_BYTES:
        raise AIServerError("invalid reply length header")
    sock.sendall(header)   # the sender waits for this echo before the body
    return json.loads(_recv_exact(sock, int(header)).decode("utf-8"))


def to_prompt(messages: list[dict]) -> str:
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    return "\n\n".join(lines + ["Assistant:"])


def clean_reply(text: Optional[str]) -> str:
    s = _THINK_RE.sub("", text or "").strip()
    m = _NEXT_TURN_RE.search(s)
    return (s[:m.start()] if m else s).strip()


class AIServerLLM:
    """Drop-in for webcam_loop.LLMClient that talks to the AI server."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 300.0, robot_id: str = "") -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.robot_id = robot_id or socket.gethostname()
        self.model = f"ai-server@{host}:{port}"
        self.available = False

    def connect(self) -> bool:
        try:
            socket.create_connection((self.host, self.port), timeout=5).close()
            self.available = True
            print(f"[LLM] connected — {self.model}")
        except OSError as exc:
            print(f"[LLM] AI server unavailable at {self.host}:{self.port}: {exc}")
        return self.available

    def _request(self, prompt: str, parameters: dict) -> str:
        msg = dict(robot_id=self.robot_id, request_id=str(uuid.uuid4()),
                   command=["LLM"], parameters=parameters, content=prompt)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            s.settimeout(self.timeout)
            _send_json(s, msg)
            reply = _recv_json(s)
        for key in ("robot_id", "request_id"):
            if reply.get(key) != msg[key]:
                raise AIServerError(f"reply {key} does not match the request")
        if reply.get("command") == ["ERR"]:
            raise AIServerError(f"server error: {reply.get('content')}")
        if reply.get("command") != [] or not isinstance(reply.get("content"), str):
            raise AIServerError("server returned an incomplete LLM reply")
        return reply["content"]

    def respond(self, system_prompt: str, user_msg: str,
                history: list[tuple[str, str]] | None = None,
                max_tokens: int = 140, json_mode: bool = False,
                temperature: float | None = None,
                seed: int | None = None) -> str:
        if not self.available:
            return "[LLM not connected — run with --ai-server]"
        messages = [{"role": "system", "content": system_prompt}]
        for u, b in (history or []):
            messages += [{"role": "user", "content": u},
                         {"role": "assistant", "content": b}]
        messages.append({"role": "user", "content": user_msg})
        # ponytail: the server currently ignores these (fixed cap, no JSON mode);
        # sent anyway so a server that honours them needs no client change.
        params: dict = {"max_tokens": max_tokens}
        if json_mode:
            params["json_mode"] = True
        if temperature is not None:
            params["temperature"] = temperature
        if seed is not None:
            params["seed"] = seed
        try:
            raw = self._request(to_prompt(messages), params)
            # JSON keeps everything after the think block: a "\nUser:" inside a
            # string value must not cut the object short.
            return _THINK_RE.sub("", raw).strip() if json_mode else clean_reply(raw)
        except (OSError, EOFError, ValueError, AIServerError) as exc:
            return f"[LLM error: {exc}]"


def from_arg(spec: str) -> AIServerLLM:
    """`--ai-server HOST[:PORT]` -> a connected client (port defaults to AI_PORT/7898)."""
    host, _, port = spec.partition(":")
    llm = AIServerLLM(host or DEFAULT_HOST, int(port) if port else DEFAULT_PORT)
    llm.connect()
    return llm
