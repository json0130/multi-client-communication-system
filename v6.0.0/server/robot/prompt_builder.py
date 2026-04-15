"""
robot/prompt_builder.py
========================
All prompt construction lives here.
robot_instance.py calls this — it never builds strings itself.

Two prompt modes:
  - delegation  : robot can answer OR delegate to a peer
  - execution   : robot received a delegated task, just confirm and do it
"""

from __future__ import annotations


def build_delegation_prompt(
    robot_id: str,
    robot_role: str,
    allowed_tags: list[str],
    user_message: str,
    active_robots: list[dict],   # [{"client_id": ..., "robot_name": ..., "robot_role": ...}]
    rag_context: list[str],      # past user messages from RAG search
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for delegation mode.
    The robot will either answer directly or offer to delegate to a peer.
    """
    tags_str = ", ".join(allowed_tags) if allowed_tags else "[DEFAULT]"
    example_tag = allowed_tags[0] if allowed_tags else "[DEFAULT]"

    # Format RAG context
    rag_block = ""
    if rag_context:
        lines = "\n".join(f'- "{t}"' for t in rag_context)
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

    system_prompt = f"""You are {robot_id}. Your role: '{robot_role}'.

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
    robot_id: str,
    robot_role: str,
    allowed_tags: list[str],
    task_message: str,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for execution mode.
    The robot received a delegated task — just confirm and execute.
    """
    tags_str = ", ".join(allowed_tags) if allowed_tags else "[DEFAULT]"
    example_tag = allowed_tags[0] if allowed_tags else "[DEFAULT]"

    system_prompt = f"""You are {robot_id}. Your role: '{robot_role}'.

*** MANDATORY FORMATTING RULES ***
1. The VERY FIRST CHARACTER of your response MUST be an open bracket '['.
2. Use EXACTLY ONE tag from this list: {tags_str}
3. Keep your response to 1 sentence — a confident confirmation you are executing the task.

Correct: {example_tag} I am on it right away!
Incorrect: Sure! {example_tag} I'll do it.

You have received a direct order from a teammate. Execute it without question."""

    return system_prompt, task_message