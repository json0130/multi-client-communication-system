"""
modules/llm/llm_provider.py
============================
Abstract interface for LLM backends.

Both OllamaProvider and OpenAIProvider implement this.
The robot layer only imports LLMProvider — it never knows which backend is active.

LLMResponse carries everything the robot needs back from any LLM call:
  - text        : full raw response string (may include [EMOTION] tag)
  - emotion_tag : extracted bracketed tag, e.g. "WAVE" (without brackets)
  - clean_text  : response with the tag stripped, ready for TTS
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re


@dataclass
class LLMResponse:
    text: str             # raw full response, e.g. "[WAVE] Hello there!"
    emotion_tag: str      # e.g. "WAVE"  (empty string if none found)
    clean_text: str       # e.g. "Hello there!"


def parse_response(raw: str) -> LLMResponse:
    """
    Shared parser used by both providers.
    Extracts [EMOTION] tag from the start of the response.
    """
    raw = raw.strip()
    match = re.match(r"^\[([A-Z_]+)\]", raw)
    tag = match.group(1) if match else ""
    clean = re.sub(r"^\[[A-Z_]+\]\s*", "", raw).strip()
    return LLMResponse(text=raw, emotion_tag=tag, clean_text=clean)


class LLMProvider(ABC):
    """
    Implement this to add a new LLM backend.
    Both providers share the same two-method interface so robot_instance.py
    never changes when you swap backends.
    """

    @abstractmethod
    def setup(self) -> bool:
        """
        Connect / verify the backend is reachable.
        Returns True if ready.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """True if setup() succeeded and the backend is still reachable."""
        ...

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        """
        Send a system prompt + user message to the LLM.
        Returns a parsed LLMResponse.
        This is the ONLY method the robot layer calls for normal chat.
        """
        ...

    @abstractmethod
    def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict],   # [{"role": "user"|"assistant", "content": "..."}]
        user_message: str,
    ) -> LLMResponse:
        """
        Like generate() but includes conversation history.
        Used when the robot needs memory across a session.
        """
        ...

    @abstractmethod
    def get_status(self) -> dict:
        """Return provider name, model, and availability as a plain dict."""
        ...