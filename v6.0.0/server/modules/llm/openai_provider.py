"""
modules/llm/openai_provider.py
==============================
LLM backend using the OpenAI cloud API (gpt-4o-mini by default).
Used as a fallback when Ollama is unavailable, or explicitly configured.

Requires: pip install openai
Requires: OPENAI_API_KEY set in .env
"""

from __future__ import annotations
from openai import OpenAI

from core.config import cfg
from modules.llm.llm_provider import LLMProvider, LLMResponse, parse_response


class OpenAIProvider(LLMProvider):

    def __init__(self):
        self._model = cfg.llm.openai_model
        self._client: OpenAI | None = None
        self._available = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> bool:
        """Initialise client. No network call needed — key validation happens on first generate()."""
        if not cfg.llm.openai_api_key:
            print("[OpenAIProvider] OPENAI_API_KEY not set — provider disabled.")
            return False
        try:
            self._client = OpenAI(api_key=cfg.llm.openai_api_key)
            self._available = True
            print(f"[OpenAIProvider] Ready — model: {self._model}")
            return True
        except Exception as e:
            print(f"[OpenAIProvider] Setup failed: {e}")
            return False

    def is_available(self) -> bool:
        return self._available

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
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
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self._call(messages)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, messages: list[dict]) -> LLMResponse:
        if not self._available or self._client is None:
            return LLMResponse(
                text="[DEFAULT] OpenAI client is not available.",
                emotion_tag="DEFAULT",
                clean_text="OpenAI client is not available.",
            )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                timeout=20,
            )
            raw = resp.choices[0].message.content.strip()
            return parse_response(raw)
        except Exception as e:
            print(f"[OpenAIProvider] Generation error: {e}")
            return LLMResponse(
                text="[DEFAULT] Sorry, I encountered an error.",
                emotion_tag="DEFAULT",
                clean_text="Sorry, I encountered an error.",
            )

    def get_status(self) -> dict:
        return {
            "provider": "openai",
            "model": self._model,
            "available": self._available,
        }