"""
tools/check_llm.py
==================
Checkpoint 3 — verifies the LLM module (base class + providers + module).

Run from project root:
    python -m tools.check_llm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — Ollama running locally:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 3 — LLM module
=======================================================
  PASS  modules/base.py imports (BaseModule ABC)
  PASS  LLM provider interface imports (LLMProvider, LLMResponse, parse_response)
  PASS  OllamaProvider imports
  PASS  OpenAIProvider imports
  PASS  LLMModule imports
  PASS  parse_response extracts emotion tag correctly
  PASS  parse_response strips tag from clean_text
  PASS  parse_response handles missing tag gracefully
  [OllamaProvider] Connected — model: qwen2.5:7b
  PASS  LLMModule.initialize() succeeded
  PASS  LLMModule.is_available() is True
  PASS  Active provider is 'ollama'
  [LLMModule] Sending test message...
  PASS  generate() returns an LLMResponse
  PASS  Response text is not empty
  PASS  emotion_tag is a non-empty string
  PASS  clean_text does not start with '['
  [LLMModule] Raw response : [WAVE] Hello! How can I help you today?
  [LLMModule] Emotion tag  : WAVE
  [LLMModule] Clean text   : Hello! How can I help you today?

Result: 15/15 checks passed
LLM module is ready. Proceed to Step 4 (Speech module).
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — Ollama NOT running, no OpenAI key:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  (all import + parse checks still pass)
  [OllamaProvider] Could not connect to Ollama at ...: ...
  [OpenAIProvider] OPENAI_API_KEY not set — provider disabled.
  [LLMModule] No LLM provider available.
  FAIL  LLMModule.initialize() succeeded
         → Start Ollama (`ollama serve`) or add OPENAI_API_KEY to .env

Result: 8/15 checks passed   ← import + parse checks pass, runtime checks fail
=======================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 3 — LLM module")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Imports ────────────────────────────────────────────
    total += 1
    try:
        from modules.base import BaseModule
        ok = True
    except Exception as e:
        ok = False
    if check("modules/base.py imports (BaseModule ABC)", ok):
        passed += 1

    total += 1
    try:
        from modules.llm.llm_provider import LLMProvider, LLMResponse, parse_response
        ok = True
    except Exception as e:
        ok = False
    if check("LLM provider interface imports (LLMProvider, LLMResponse, parse_response)", ok):
        passed += 1

    total += 1
    try:
        from modules.llm.ollama_provider import OllamaProvider
        ok = True
    except Exception as e:
        ok = False
    if check("OllamaProvider imports", ok):
        passed += 1

    total += 1
    try:
        from modules.llm.openai_provider import OpenAIProvider
        ok = True
    except Exception as e:
        ok = False
    if check("OpenAIProvider imports", ok):
        passed += 1

    total += 1
    try:
        from modules.llm.llm_module import LLMModule
        ok = True
    except Exception as e:
        ok = False
    if check("LLMModule imports", ok):
        passed += 1

    if passed < 5:
        print("\n  Fix import errors before continuing.")
        _summary(passed, total)
        return

    from modules.llm.llm_provider import parse_response
    from modules.llm.llm_module import LLMModule

    # ── 2. parse_response unit tests ─────────────────────────
    total += 1
    result = parse_response("[WAVE] Hello there!")
    if check("parse_response extracts emotion tag correctly",
             result.emotion_tag == "WAVE"):
        passed += 1

    total += 1
    if check("parse_response strips tag from clean_text",
             result.clean_text == "Hello there!"):
        passed += 1

    total += 1
    no_tag = parse_response("Hello, I have no tag.")
    if check("parse_response handles missing tag gracefully",
             no_tag.emotion_tag == "" and no_tag.clean_text == "Hello, I have no tag."):
        passed += 1

    # ── 3. LLMModule lifecycle ────────────────────────────────
    total += 1
    module = LLMModule()
    init_ok = module.initialize()
    if check("LLMModule.initialize() succeeded", init_ok,
             "Start Ollama (`ollama serve`) or add OPENAI_API_KEY to .env"):
        passed += 1

    total += 1
    if check("LLMModule.is_available() is True", module.is_available()):
        passed += 1

    total += 1
    provider = module.provider_name
    if check("Active provider is 'ollama' or 'openai'",
             provider in ("ollama", "openai"),
             f"Got: {provider}"):
        passed += 1

    if not module.is_available():
        print("\n  Skipping generation tests — no LLM provider available.")
        _summary(passed, total)
        return

    # ── 4. Generation ─────────────────────────────────────────
    print("  [LLMModule] Sending test message...")
    system = (
        "You are a friendly robot. "
        "ALWAYS start your response with exactly one emotion tag in square brackets "
        "from this list: [WAVE], [HAPPY], [DEFAULT]. "
        "Keep your answer to one sentence."
    )
    user = "Say hello."

    total += 1
    try:
        from modules.llm.llm_provider import LLMResponse
        response = module.generate(system, user)
        is_response = isinstance(response, LLMResponse)
    except Exception as e:
        is_response = False
        print(f"         Error: {e}")
    if check("generate() returns an LLMResponse", is_response):
        passed += 1

    total += 1
    if check("Response text is not empty",
             bool(response.text.strip())):
        passed += 1

    total += 1
    if check("emotion_tag is a non-empty string",
             isinstance(response.emotion_tag, str) and len(response.emotion_tag) > 0,
             f"Got: '{response.emotion_tag}' — check your system prompt"):
        passed += 1

    total += 1
    if check("clean_text does not start with '['",
             not response.clean_text.startswith("["),
             f"Got: '{response.clean_text[:40]}'"):
        passed += 1

    # Print the actual response so you can visually inspect it
    print(f"  [LLMModule] Raw response : {response.text}")
    print(f"  [LLMModule] Emotion tag  : {response.emotion_tag}")
    print(f"  [LLMModule] Clean text   : {response.clean_text}")

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("LLM module is ready. Proceed to Step 4 (Speech module).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()