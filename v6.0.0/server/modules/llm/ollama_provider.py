"""
modules/llm/ollama_provider.py
==============================
LLM backend using a locally running Ollama server.
Communicates via Ollama's OpenAI-compatible /v1 endpoint.

Requires: pip install openai
Requires: Ollama running on localhost (or configured host/port)
"""

from __future__ import annotations
from openai import OpenAI

from core.config import cfg
from modules.llm.llm_provider import LLMProvider, LLMResponse, parse_response


class OllamaProvider(LLMProvider):

    def __init__(self):
        self._model = cfg.llm.ollama_model
        self._base_url = f"http://{cfg.llm.ollama_host}:{cfg.llm.ollama_port}/v1"
        self._client: OpenAI | None = None
        self._available = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Ping Ollama by listing models. Sets _available."""
        try:
            self._client = OpenAI(base_url=self._base_url, api_key="ollama")
            self._client.models.list()   # raises if Ollama is unreachable
            self._available = True
            print(f"[OllamaProvider] Connected — model: {self._model}")
            return True
        except Exception as e:
            print(f"[OllamaProvider] Could not connect to Ollama at {self._base_url}: {e}")
            self._available = False
            return False

    def is_available(self) -> bool:
        return self._available

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Single-turn: system prompt + one user message."""
        return self._call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ]
        )

    def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
    ) -> LLMResponse:
        """Multi-turn: system prompt + past turns + new user message."""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self._call(messages)

    def stream_with_history(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
    ):
        """Stream sentences from Ollama as they complete. Yields clean sentence strings."""
        import re
        if not self._available or self._client is None:
            yield "Local LLM is not available."
            return
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.6,
                timeout=30,
                stream=True,
            )
            buffer = ""
            for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                buffer += delta
                parts = re.split(r'(?<=[.!?])\s+', buffer)
                for sentence in parts[:-1]:
                    s = sentence.strip()
                    if s:
                        yield s
                buffer = parts[-1]
            if buffer.strip():
                yield buffer.strip()
        except Exception as e:
            print(f"[OllamaProvider] Streaming error: {e}")
            yield "Sorry, I encountered an error."

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, messages: list[dict]) -> LLMResponse:
        if not self._available or self._client is None:
            return LLMResponse(
                text="[DEFAULT] Local LLM is not available.",
                emotion_tag="DEFAULT",
                clean_text="Local LLM is not available.",
            )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.6,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            return parse_response(raw)
        except Exception as e:
            print(f"[OllamaProvider] Generation error: {e}")
            return LLMResponse(
                text="[DEFAULT] Sorry, I encountered an error.",
                emotion_tag="DEFAULT",
                clean_text="Sorry, I encountered an error.",
            )

    def get_status(self) -> dict:
        return {
            "provider": "ollama",
            "model": self._model,
            "base_url": self._base_url,
            "available": self._available,
        }