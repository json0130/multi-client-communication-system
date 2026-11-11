"""
conversation/method_b.py
========================
Method B in the server: one gesture tag per Navel sentence, chosen here instead of by a planner service on
the robot's side (modules/action_tag_generator.py is the planner; this is how a streamed reply uses it).

Each sentence of a reply is tagged as soon as the model has written it, while the sentences before it are
still being sent and spoken, and goes out in order with the tag as its `emotion_tag`, which is what a Navel
client without a planner turns into the gesture. Speech never waits on the planner: a sentence whose tag is
not ready in time, or whose planning failed, is sent exactly as the model wrote it.

"In time" is `timeout` seconds for the first sentence, and for each later one until the sentences already sent
have been spoken (estimated from their length), because it cannot be spoken before then anyway. That is what
makes this work on a single Ollama: while the reply is still being written, a tag request queues behind it
(measured: 5 s, against 0.4 s when idle), and arrives while the first sentence is being said.
"""
from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional


def is_navel(instance) -> bool:
    return "".join(str(getattr(instance, "robot_name", "")).lower().split()) == "navel"


class _Strict:
    """A persona_chat backend that raises on its own error text ("[LLM error: ...]"), so a failed call is a
    failure here and not a junk answer the planner would turn into a DEFAULT gesture."""

    def __init__(self, backend):
        self.backend = backend

    @property
    def available(self):
        return getattr(self.backend, "available", True)

    def respond(self, *args, **kwargs):
        out = self.backend.respond(*args, **kwargs)
        if str(out).startswith("[LLM"):
            raise RuntimeError(out)
        return out


MAX_HOLD_SEC = 8.0     # no sentence is held longer than this, whatever is still being said before it


class MethodB:
    def __init__(self, tagger, timeout: float = 2.0):
        self._tagger, self.timeout = tagger, timeout
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="method-b")
        self._last: dict = {}           # robot id -> its previous tag. The planner's "same gesture twice" filter
        self._lock = threading.Lock()   # only sees one call, and every sentence is its own call.

    def _plan(self, sentence: str, user_message: str) -> Optional[str]:
        items = self._tagger.generate(sentence, user_message)
        return next((i["tag"] for i in items if i["tag"]), None)

    def stream(self, robot_id: str, user_message: str, send: Callable[[dict], None]) -> "_Stream":
        return _Stream(self, robot_id, user_message, send)

    def _no_repeat(self, robot_id: str, tag: Optional[str]) -> Optional[str]:
        with self._lock:
            if tag is not None and tag == self._last.get(robot_id):
                tag = None
            self._last[robot_id] = tag
            return tag


class _Stream:
    """One reply: `put` each chat_sentence payload as the model finishes it, `close` when the reply is over."""

    def __init__(self, mb: MethodB, robot_id: str, user_message: str, send):
        self._mb, self._robot, self._user, self._send = mb, robot_id, user_message, send
        self._q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._dropped = False
        self._speech_end = 0.0          # when the sentences sent so far should have been spoken (monotonic)
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"method-b-{robot_id}")
        self._thread.start()

    def put(self, payload: dict) -> None:
        fut = self._mb._pool.submit(self._mb._plan, payload["text"], self._user)
        self._q.put((payload, fut, time.monotonic()))

    def cancel(self) -> None:
        self._dropped = True

    def close(self) -> None:
        self._q.put(None)
        self._thread.join()

    def _run(self) -> None:
        from gateway.websocket_gateway import _speaking_seconds
        while (item := self._q.get()) is not None:
            payload, fut, submitted = item
            deadline = min(max(submitted + self._mb.timeout, self._speech_end), submitted + MAX_HOLD_SEC)
            try:
                tag = self._mb._no_repeat(self._robot, fut.result(timeout=max(0.0, deadline - time.monotonic())))
                payload = {**payload, "emotion_tag": tag or ""}
            except Exception as exc:        # timed out or failed: the sentence goes out as it was
                fut.cancel()
                print(f"[MethodB] {self._robot}: untagged ({type(exc).__name__}: {exc})"[:160])
            if not self._dropped:
                self._send(payload)
                self._speech_end = max(self._speech_end, time.monotonic()) + _speaking_seconds(payload["text"])


def build_method_b(method_b_cfg, ollama_url: str) -> Optional[MethodB]:
    """None (Method B off, plain speech) when METHOD_B_OLLAMA_MODEL is "off" and there is no AI server."""
    if method_b_cfg.ollama_model.lower() in ("", "off", "none") and not method_b_cfg.ai_server:
        return None
    from modules.action_tag_generator import ChatLLM, navel_tagger
    from modules.persona_chat import OllamaChat, ai_server_chat
    backend = (ai_server_chat(method_b_cfg.ai_server) if method_b_cfg.ai_server
               else OllamaChat(method_b_cfg.ollama_model, ollama_url))
    # overrun 10: Navel's gestures are measured at 3-4 s, longer than most sentences; the next one replaces it.
    return MethodB(navel_tagger(ChatLLM(_Strict(backend)), overrun_seconds=10), method_b_cfg.timeout_sec)
