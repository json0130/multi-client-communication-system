"""
tools/check_robot.py
====================
Checkpoint 7 — verifies the robot layer (prompt_builder, robot_instance, robot_registry).

Run from project root:
    python3 tools/check_robot.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — Ollama running + robot registered in DB:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 7 — Robot layer
=======================================================
  PASS  prompt_builder imports
  PASS  robot_instance imports
  PASS  robot_registry imports

  -- Prompt builder tests --
  PASS  build_delegation_prompt returns (system, user) strings
  PASS  System prompt contains robot_id
  PASS  System prompt contains allowed tags
  PASS  System prompt contains peer robot info
  PASS  RAG context appears in system prompt
  PASS  build_execution_prompt returns (system, user) strings
  PASS  Execution prompt contains task message

  -- RobotRegistry tests --
  [Registry] _checkpoint_robot_001 not found in DB — register it first.
  PASS  connect() returns None for unknown robot (not registered)
  PASS  get() returns None for unknown robot

  -- DB-connected registry tests --
  (registers a test robot in Supabase then connects it)
  [OllamaProvider] Connected — model: qwen2.5:7b
  [Registry] _checkpoint_robot_001 connected. Modules: ['gpt']
  PASS  connect() returns a RobotInstance for registered robot
  PASS  instance.client_id matches
  PASS  instance.llm is set and available
  PASS  get() retrieves the instance
  PASS  is_connected() returns True
  PASS  process_chat() returns a ChatResult
  PASS  ChatResult has non-empty response
  PASS  ChatResult has an emotion_tag
  PASS  disconnect() marks robot inactive
  PASS  instance removed from registry after disconnect

  Cleaning up test robot...
  OK  Test robot deleted

Result: 21/21 checks passed
Robot layer is ready. Proceed to Step 8 (Gateway layer).
=======================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT — Ollama NOT running:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  (all import + prompt builder tests pass)
  PASS  connect() returns a RobotInstance for registered robot
  [OllamaProvider] Could not connect...
  [LLMModule] No LLM provider available.
  PASS  instance.llm is None (LLM failed gracefully)
  PASS  process_chat() returns fallback ChatResult
  ...

Result: 19/21 checks passed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_CLIENT_ID = "_checkpoint_robot_001"
TEST_ROBOT_NAME = "Checkpoint Robot"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 7 — Robot layer")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Imports ────────────────────────────────────────────
    for label, module_path in [
        ("prompt_builder imports",  "robot.prompt_builder"),
        ("robot_instance imports",  "robot.robot_instance"),
        ("robot_registry imports",  "robot.robot_registry"),
    ]:
        total += 1
        try:
            __import__(module_path)
            ok = True
        except Exception as e:
            ok = False
            print(f"         Error: {e}")
        if check(label, ok):
            passed += 1

    if passed < 3:
        _summary(passed, total)
        return

    from robot.prompt_builder import build_delegation_prompt, build_execution_prompt
    from robot.robot_instance import RobotInstance, ChatResult
    from robot.robot_registry import RobotRegistry

    # ── 2. Prompt builder ─────────────────────────────────────
    print("\n  -- Prompt builder tests --")

    # RAG context must be RBAC-cleared before prompt assembly will accept it.
    from datetime import datetime, timezone
    from core.rbac import (
        AccessLevel, ClearanceError, MemoryRecord, RBACFilter,
        RobotIdentity, Visibility,
    )

    _rbac = RBACFilter()
    _identity = RobotIdentity("bot_01", "check", "sess", AccessLevel.LOCAL)
    _cleared = _rbac.filter_records(
        _identity,
        [MemoryRecord(
            record_id="check:0",
            content="I like oat milk in my coffee",
            source_robot_id="bot_01",
            scenario_id="check",
            visibility=Visibility.LOCAL,
        )],
        (),
        datetime.now(timezone.utc),
    )

    total += 1
    system, user = build_delegation_prompt(
        robot_name="bot_01",
        robot_role="You are a front desk assistant.",
        allowed_tags=["[WAVE]", "[HAPPY]", "[DEFAULT]"],
        user_message="Can you bring me a coffee?",
        active_robots=[{
            "client_id": "bot_02",
            "robot_name": "CoffeeBot",
            "robot_role": "I make and deliver coffee.",
        }],
        rag_context=_cleared,
    )
    if check("build_delegation_prompt returns (system, user) strings",
             isinstance(system, str) and isinstance(user, str)):
        passed += 1

    total += 1
    if check("System prompt contains robot name", "bot_01" in system):
        passed += 1

    total += 1
    if check("System prompt contains allowed tags", "[WAVE]" in system):
        passed += 1

    total += 1
    if check("System prompt contains peer robot info", "bot_02" in system):
        passed += 1

    total += 1
    if check("RAG context appears in system prompt",
             "oat milk" in system):
        passed += 1

    total += 1
    exec_sys, exec_user = build_execution_prompt(
        robot_name="bot_02",
        robot_role="I make and deliver coffee.",
        allowed_tags=["[DEFAULT]"],
        task_message="Please make one oat milk latte.",
    )
    if check("build_execution_prompt returns (system, user) strings",
             isinstance(exec_sys, str) and isinstance(exec_user, str)):
        passed += 1

    total += 1
    if check("Execution prompt contains task message",
             "oat milk latte" in exec_user):
        passed += 1

    # ── RBAC: prompt assembly refuses unfiltered memory ───────
    total += 1
    try:
        build_delegation_prompt(
            robot_name="bot_01", robot_role="desk",
            allowed_tags=["[DEFAULT]"], user_message="hi",
            active_robots=[], rag_context=["an unfiltered raw string"],
        )
        refused = False
    except ClearanceError:
        refused = True
    if check("Prompt assembly refuses records with no RBAC clearance", refused):
        passed += 1

    # ── 3. Registry — no DB robot ─────────────────────────────
    print("\n  -- RobotRegistry tests --")
    registry = RobotRegistry()

    total += 1
    result = registry.connect(TEST_CLIENT_ID)
    if check("connect() returns None for unknown robot (not registered)",
             result is None):
        passed += 1

    total += 1
    if check("get() returns None for unknown robot",
             registry.get(TEST_CLIENT_ID) is None):
        passed += 1

    # ── 4. Registry — with DB robot ───────────────────────────
    print("\n  -- DB-connected registry tests --")

    # Register a test robot in Supabase
    from data import robot_repo
    robot_repo.upsert_robot(
        client_id=TEST_CLIENT_ID,
        robot_name=TEST_ROBOT_NAME,
        robot_role="You are a checkpoint test robot. Be brief.",
        allowed_tags=["[DEFAULT]", "[WAVE]"],
        modules=["gpt"],   # only LLM — no speech/emotion/rag to keep test fast
        ip_address="127.0.0.1",
        ws_port=8765,
    )

    total += 1
    instance = registry.connect(TEST_CLIENT_ID)
    if check("connect() returns a RobotInstance for registered robot",
             instance is not None):
        passed += 1

    if instance is None:
        _cleanup()
        _summary(passed, total)
        return

    total += 1
    if check("instance.client_id matches",
             instance.client_id == TEST_CLIENT_ID):
        passed += 1

    total += 1
    llm_ok = instance.llm is not None and instance.llm.is_available()
    if check("instance.llm is set and available", llm_ok,
             "Start Ollama (`ollama serve`) for LLM functionality"):
        passed += 1

    total += 1
    if check("get() retrieves the instance",
             registry.get(TEST_CLIENT_ID) is instance):
        passed += 1

    total += 1
    if check("is_connected() returns True",
             registry.is_connected(TEST_CLIENT_ID)):
        passed += 1

    # ── 5. Chat through the instance ──────────────────────────
    total += 1
    try:
        result = instance.process_chat("Say hello in one word.")
        chat_ok = isinstance(result, ChatResult)
    except Exception as e:
        chat_ok = False
        print(f"         Error: {e}")
    if check("process_chat() returns a ChatResult", chat_ok):
        passed += 1

    if chat_ok:
        total += 1
        if check("ChatResult has non-empty response",
                 bool(result.response.strip())):
            passed += 1

        total += 1
        if check("ChatResult has an emotion_tag",
                 isinstance(result.emotion_tag, str)):
            passed += 1

        print(f"  Response    : {result.response}")
        print(f"  Emotion tag : {result.emotion_tag}")
        print(f"  Clean text  : {result.clean_text}")
        print(f"  Delegation  : {result.is_delegation}")

    # ── 6. Disconnect ─────────────────────────────────────────
    total += 1
    registry.disconnect(TEST_CLIENT_ID)
    robot = robot_repo.get_robot(TEST_CLIENT_ID)
    if check("disconnect() marks robot inactive",
             robot is not None and not robot.is_active):
        passed += 1

    total += 1
    if check("instance removed from registry after disconnect",
             registry.get(TEST_CLIENT_ID) is None):
        passed += 1

    _cleanup()
    _summary(passed, total)


def _cleanup():
    print("\n  Cleaning up test robot...")
    try:
        from data.connection import get_client
        get_client().table("robots").delete().eq(
            "client_id", TEST_CLIENT_ID
        ).execute()
        print("  OK  Test robot deleted")
    except Exception as e:
        print(f"  WARN  Cleanup failed: {e}")


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Robot layer is ready. Proceed to Step 8 (Gateway layer).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()