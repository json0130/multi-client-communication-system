"""
tests/test_ai_server_backends.py — LLM_AI_SERVER / STT_AI_SERVER: the server's conversation and
transcription on the lab AI server, against a fake one speaking its wire protocol (localhost only).
"""
from __future__ import annotations

import base64
import io
import json
import socket
import threading
import wave

import pytest

from modules.llm.ai_server_provider import AIServerProvider


def _wav(seconds=0.5, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(b"\x00\x01" * int(rate * seconds))
    return buf.getvalue()


def _exact(conn, n):
    buf = b""
    while len(buf) < n:
        buf += conn.recv(n - len(buf))
    return buf


@pytest.fixture
def fake():
    """A fake AI server: answers each request with reply(req) -> (command, content); keeps the requests."""
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen()
    seen, state = [], {"reply": lambda req: ([], "")}

    def serve():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                header = conn.recv(32)
                if not header:           # connect() only checks the port
                    continue
                conn.sendall(header)
                req = json.loads(_exact(conn, int(header)))
                seen.append(req)
                command, content = state["reply"](req)
                body = json.dumps({**req, "command": command, "content": content}).encode()
                conn.sendall(str(len(body)).encode()); conn.recv(32); conn.sendall(body)

    threading.Thread(target=serve, daemon=True).start()
    yield f"127.0.0.1:{srv.getsockname()[1]}", seen, state
    srv.close()


def _unused_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


def test_provider_sends_role_and_history_and_keeps_the_tag(fake):
    spec, seen, state = fake
    state["reply"] = lambda req: ([], "[WAVE] Hello there. The books are upstairs!")
    p = AIServerProvider(spec)
    assert p.setup() and p.get_status()["provider"] == "ai_server"
    r = p.generate_with_history("You are Stuart.", [{"role": "user", "content": "hi"},
                                                    {"role": "assistant", "content": "Hello!"}], "Where are the books?")
    assert (r.emotion_tag, r.clean_text) == ("WAVE", "Hello there. The books are upstairs!")
    prompt = seen[-1]["content"]
    assert seen[-1]["command"] == ["LLM"]
    assert prompt.index("System: You are Stuart.") < prompt.index("User: hi") < prompt.index("User: Where are the books?")
    assert list(p.stream_with_history("s", [], "q")) == ["[WAVE] Hello there.", "The books are upstairs!"]


def test_provider_error_is_a_spoken_apology_not_a_crash(fake):
    spec, _, state = fake
    state["reply"] = lambda req: (["ERR"], "model not loaded")
    p = AIServerProvider(spec)
    p.setup()
    assert p.generate("s", "q").clean_text == "Sorry, I encountered an error."


def test_llm_module_prefers_the_ai_server_and_falls_back(fake, monkeypatch):
    import modules.llm.llm_module as llm_module

    class _Off:
        def setup(self): return False

    class _Ollama:
        def setup(self): return True
        def is_available(self): return True
        def get_status(self): return {"provider": "ollama"}

    monkeypatch.setattr(llm_module, "OllamaProvider", _Ollama)
    monkeypatch.setattr(llm_module, "OpenAIProvider", _Off)
    spec, _, _ = fake
    for ai_server, want in ((spec, "ai_server"), (f"127.0.0.1:{_unused_port()}", "ollama"), ("", "ollama")):
        monkeypatch.setattr(llm_module, "cfg", type("C", (), {"llm": type("L", (), {"ai_server": ai_server})})())
        m = llm_module.LLMModule()
        assert m.initialize() and m.provider_name == want


@pytest.fixture
def speech(monkeypatch):
    import modules.speech.speech_module as speech_module

    def make(ai_server):
        sp = type("S", (), dict(whisper_model_size="base", whisper_device="cpu", whisper_compute_type="int8",
                                max_audio_length_sec=30, sample_rate=16000, language="en", ai_server=ai_server))
        monkeypatch.setattr(speech_module, "cfg", type("C", (), {"speech": sp})())
        return speech_module.SpeechModule()
    return make


def test_speech_module_transcribes_on_the_ai_server(fake, speech):
    spec, seen, state = fake
    state["reply"] = lambda req: ([], " where are the space books ")
    m = speech(spec)
    assert m.initialize() and m.is_available()
    r = m.transcribe_b64(base64.b64encode(_wav()).decode())
    assert r.success and r.transcription == "where are the space books"
    assert seen[-1]["command"] == ["STT"]
    assert seen[-1]["parameters"] == {"format": "raw_pcm", "dtype": "int16", "sample_rate": 16000, "channels": 1}


def test_speech_module_reports_an_ai_server_failure(fake, speech):
    spec, _, state = fake
    state["reply"] = lambda req: (["ERR"], "audio too long")
    m = speech(spec)
    m.initialize()
    r = m.transcribe_b64(base64.b64encode(_wav()).decode())
    assert not r.success and "audio too long" in r.error
