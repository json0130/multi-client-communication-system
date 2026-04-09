"""
modules/llm/llm_module.py
=========================
The LLM module that robot_instance.py holds.

Wraps whichever provider is configured (Ollama first, OpenAI fallback).
The robot layer calls llm_module.generate() and never knows which provider is active.

Auto-fallback logic:
  1. Try Ollama (local, fast, free)
  2. If Ollama unavailable AND OpenAI key is set → fall back to OpenAI
  3. If both unavailable → is_available() returns False; robot responds with an error message
"""

from __future__ import annotations

from modules.base import BaseModule
from modules.llm.llm_provider import LLMProvider, LLMResponse
from modules.llm.ollama_provider import OllamaProvider
from modules.llm.openai_provider import OpenAIProvider


class LLMModule(BaseModule):

    def __init__(self):
        self._provider: LLMProvider | None = None

    # ── BaseModule interface ───────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Try Ollama first. Fall back to OpenAI if Ollama is unreachable
        and an API key is configured.
        """
        ollama = OllamaProvider()
        if ollama.setup():
            self._provider = ollama
            print("[LLMModule] Using Ollama provider.")
            return True

        openai = OpenAIProvider()
        if openai.setup():
            self._provider = openai
            print("[LLMModule] Ollama unavailable — using OpenAI fallback.")
            return True

        print("[LLMModule] No LLM provider available. "
              "Start Ollama or set OPENAI_API_KEY.")
        return False

    def is_available(self) -> bool:
        return self._provider is not None and self._provider.is_available()

    def get_status(self) -> dict:
        if self._provider:
            return {"module": "llm", **self._provider.get_status()}
        return {"module": "llm", "available": False, "provider": "none"}

    # ── Public API (called by robot_instance.py) ───────────────────────────────

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Single-turn generation."""
        if not self.is_available():
            from modules.llm.llm_provider import LLMResponse
            return LLMResponse(
                text="[DEFAULT] No LLM available.",
                emotion_tag="DEFAULT",
                clean_text="No LLM available.",
            )
        return self._provider.generate(system_prompt, user_message)

    def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict],
        user_message: str,
    ) -> LLMResponse:
        """Multi-turn generation with conversation history."""
        if not self.is_available():
            from modules.llm.llm_provider import LLMResponse
            return LLMResponse(
                text="[DEFAULT] No LLM available.",
                emotion_tag="DEFAULT",
                clean_text="No LLM available.",
            )
        return self._provider.generate_with_history(system_prompt, history, user_message)

    @property
    def provider_name(self) -> str:
        """Convenience for logging."""
        if self._provider:
            return self._provider.get_status().get("provider", "unknown")
        return "none"