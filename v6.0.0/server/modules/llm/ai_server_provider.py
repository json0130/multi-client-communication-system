"""
modules/llm/ai_server_provider.py
=================================
The lab AI server's LLM (LLM_AI_SERVER=HOST[:PORT]) behind the same interface as
Ollama and OpenAI, so robot_instance.py never knows which one answers.

The AI server returns a whole reply, not a stream: stream_with_history() yields its
sentences once it has arrived, the first one still carrying the reply's tag.
"""

from __future__ import annotations

import re

from modules.ai_server_llm import AIServerError, from_arg
from modules.llm.llm_provider import LLMProvider, LLMResponse, parse_response

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_FAILED = LLMResponse(text="[DEFAULT] Sorry, I encountered an error.", emotion_tag="DEFAULT",
                      clean_text="Sorry, I encountered an error.")


class AIServerProvider(LLMProvider):

    def __init__(self, spec: str):
        self._spec = spec
        self._llm = None

    def setup(self) -> bool:
        self._llm = from_arg(self._spec, robot_id="cares-server")
        return self._llm.available

    def is_available(self) -> bool:
        return bool(self._llm and self._llm.available)

    def get_status(self) -> dict:
        return {"provider": "ai_server", "model": self._llm.model if self._llm else self._spec,
                "available": self.is_available()}

    def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        return self.generate_with_history(system_prompt, [], user_message)

    def generate_with_history(self, system_prompt: str, history: list[dict], user_message: str) -> LLMResponse:
        messages = [{"role": "system", "content": system_prompt},
                    *({"role": m["role"], "content": m["content"]} for m in history),
                    {"role": "user", "content": user_message}]
        try:
            return parse_response(self._llm.chat(messages))
        except (OSError, EOFError, ValueError, AIServerError) as e:
            print(f"[AIServerProvider] {e}")
            return _FAILED

    def stream_with_history(self, system_prompt: str, history: list[dict], user_message: str):
        reply = self.generate_with_history(system_prompt, history, user_message).text
        for sentence in _SENTENCE_END.split(reply):
            if sentence.strip():
                yield sentence.strip()
