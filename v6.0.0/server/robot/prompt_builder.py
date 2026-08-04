"""
robot/prompt_builder.py
========================
All prompt construction lives here.
robot_instance.py calls this — it never builds strings itself.

Two prompt modes:
  - delegation  : robot can answer OR delegate to a peer
  - execution   : robot received a delegated task, just confirm and do it

This is the Composite System Prompt fusion point: persona, retrieved memory and
teammate state are combined into a single system prompt here.

RBAC
----
Retrieved memory arrives as ClearedRecord — the stamp applied by
core.rbac.filter.RBACFilter. Both builders call assert_cleared() before fusing
anything, so a read path added elsewhere that forgets to filter fails loudly at
the fusion point instead of leaking into the LLM silently.
"""

from __future__ import annotations
from typing import Sequence

from core.rbac import ClearedRecord, assert_cleared


def build_delegation_prompt(
    robot_name: str,
    robot_role: str,
    allowed_tags: list[str],
    user_message: str,
    active_robots: list[dict],          # [{"client_id": ..., "robot_name": ..., "robot_role": ...}]
    rag_context: Sequence[ClearedRecord],   # RBAC-cleared past user messages
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for delegation mode.
    The robot will either answer directly or offer to delegate to a peer.

    Raises ClearanceError if any rag_context entry lacks an RBAC clearance stamp.
    """
    cleared = assert_cleared(rag_context, "build_delegation_prompt(rag_context)")

    tags_str = ", ".join(allowed_tags) if allowed_tags else "[DEFAULT]"
    example_tag = allowed_tags[0] if allowed_tags else "[DEFAULT]"

    # Format RAG context
    rag_block = ""
    if cleared:
        lines = "\n".join(f'- "{c.text}"' for c in cleared)
        rag_block = f"\nThe user has previously told you:\n{lines}\n"

    # Format active peers
    if active_robots:
        peer_lines = "\n".join(
            f"  - ID: '{r['client_id']}' | Name: {r['robot_name']} | Role: {r['robot_role']}"
            for r in active_robots
        )
        peers_block = f"CURRENTLY ACTIVE ROBOTS:\n{peer_lines}"
    else:
        peers_block = "CURRENTLY ACTIVE ROBOTS:\n  None. You are the only active robot."

    system_prompt = f"""You are {robot_name}. Your role: '{robot_role}'.

*** MANDATORY FORMATTING RULES ***
1. The VERY FIRST CHARACTER of your response MUST be an open bracket '['.
2. Use EXACTLY ONE tag from this list: {tags_str}
3. Keep responses to 1-2 sentences maximum.

*** CORRECT EXAMPLES ***
{example_tag} Hello! How can I help you today?
{example_tag} I'm sorry, I can't do that myself.

*** INCORRECT EXAMPLES (never do this) ***
Hello! {example_tag} How are you?   <- text before the tag
{example_tag} Sure! {example_tag} Let me help.  <- two tags
{rag_block}
*** TEAMMATES & DELEGATION ***
{peers_block}

STEP 1 — Can YOU fulfil this request given your role?
  YES → Answer directly. Ignore steps 2 and 3.

STEP 2 — If NO, is there a teammate whose role matches?
  NO MATCH → Politely explain you and no teammate can help.
  MATCH → Tell the user which teammate can help and ASK if they want you to ask that teammate.
  Example: {example_tag} I can't do that, but RobotX can. Would you like me to ask them?

STEP 3 — ONLY if the user explicitly says YES to delegation:
  Respond with your confirmation AND a JSON block:
  {example_tag} I'll ask them right away!
  ```json
  {{"target_robot_id": "<EXACT_ID_FROM_LIST>", "task": "<what to ask them>"}}
  ```

CRITICAL: Never invent robot IDs. Only use IDs from the active list above."""

    return system_prompt, user_message


def build_execution_prompt(
    robot_name: str,
    robot_role: str,
    allowed_tags: list[str],
    task_message: str,
    granted_context: Sequence[ClearedRecord] = (),
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for execution mode.
    The robot received a delegated task — just confirm and execute.

    granted_context carries the paper's Context Serialization: snippets the
    delegating Manager explicitly handed over for this one task. They appear in
    this temporary prompt only. The Worker's standing access level is unchanged,
    and nothing here is written back into the Worker's own memory.

    Raises ClearanceError if any granted_context entry lacks a clearance stamp.
    """
    cleared = assert_cleared(granted_context, "build_execution_prompt(granted_context)")

    tags_str = ", ".join(allowed_tags) if allowed_tags else "[DEFAULT]"
    example_tag = allowed_tags[0] if allowed_tags else "[DEFAULT]"

    context_block = ""
    if cleared:
        lines = "\n".join(f'- "{c.text}"' for c in cleared)
        context_block = (
            f"\n*** CONTEXT SHARED BY YOUR TEAMMATE FOR THIS TASK ***\n"
            f"{lines}\n"
            f"Use this only for this task. Do not repeat it verbatim and do not "
            f"treat it as something you were told directly.\n"
        )

    system_prompt = f"""You are {robot_name}. Your role: '{robot_role}'.

*** MANDATORY FORMATTING RULES ***
1. The VERY FIRST CHARACTER of your response MUST be an open bracket '['.
2. Use EXACTLY ONE tag from this list: {tags_str}
3. Keep your response to 1 sentence — a confident confirmation you are executing the task.

Correct: {example_tag} I am on it right away!
Incorrect: Sure! {example_tag} I'll do it.
{context_block}
You have received a direct order from a teammate. Execute it without question."""

    return system_prompt, task_message
