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
    grounded_facts: Sequence[str] = (),
    standing_in_for: str = "",
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

    # Verified detail about the topic being asked about — see
    # data/demo_topic_facts and decision/grounding.py. Without it a robot has
    # nothing but its one-paragraph role, and a live run showed what fills
    # that gap: invented model names ("Extended Kalman Filters", "BM25")
    # stated confidently to visitors. The closing rule is the important half
    # — grounding only helps if NOT having a fact changes the answer.
    if grounded_facts:
        listed = "\n".join(f"  - {f}" for f in grounded_facts)
        facts_block = (
            f"\n*** WHAT IS ACTUALLY TRUE OF YOUR WORK ***\n{listed}\n"
            "USE THESE. They are the specifics a visitor came for, and they "
            "are confirmed — name the method, the model, the index, the "
            "numbers, rather than paraphrasing them into something vaguer. "
            "Use the exact names from the list; do not borrow a method from "
            "anywhere else.\n"
            "They are also the ONLY specifics you may state. Anything marked "
            "UNVERIFIED is a draft nobody has confirmed — describe it in "
            "general terms and do not quote it as a precise result.\n"
        )
    else:
        facts_block = ""

    honesty_rule = (
        "\n*** NEVER INVENT SPECIFICS ***\n"
        "Do not name a model, algorithm, dataset, number or paper unless it "
        "appears above. A visitor told you would need to check is a good "
        "outcome; a visitor told a plausible-sounding name that turns out to "
        "be wrong is not.\n"
        "But LEAD WITH WHAT YOU DO KNOW, and mention the gap briefly at the "
        "end if at all. Answer the question as far as the facts above take "
        "you, then stop. Do not open with a disclaimer, do not describe what "
        "you are unable to say, and do not offer to go and check — a visitor "
        "asked what technique you use wants to hear the technique, not a "
        "sentence about the limits of your knowledge.\n"
    )

    # Format active peers
    if active_robots:
        # Declared topics, where the graph knows them, because ownership is
        # what makes a hand-off decidable. Given only prose roles, a robot
        # asked about a peer's subject answered it itself.
        peer_lines = "\n".join(
            f"  - ID: '{r['client_id']}' | Name: {r['robot_name']} | Role: {r['robot_role']}"
            + (f"\n      OWNS: {', '.join(r['declared_topics'])}"
               if r.get("declared_topics") else "")
            for r in active_robots
        )
        peers_block = (
            f"CURRENTLY ACTIVE ROBOTS:\n{peer_lines}\n"
            "If a question is about a subject another robot OWNS, it is theirs "
            "to answer, not yours — hand it over even if you could attempt it."
        )
    else:
        peers_block = "CURRENTLY ACTIVE ROBOTS:\n  None. You are the only active robot."

    stand_in_block = (
        f"\n*** You are standing in for {standing_in_for} ***\n"
        f"{standing_in_for} is at another station in the lab, so you answer this "
        f"question yourself, for them. Do not hand it over and do not suggest "
        f"asking {standing_in_for}. The facts above are {standing_in_for}'s work: "
        f"say \"{standing_in_for}'s project...\", not \"we\". If the visitor only "
        f"asks whether they may ask, say yes and give one fact to start.\n"
    ) if standing_in_for else ""

    system_prompt = f"""You are {robot_name}. Your role: '{robot_role}'.

*** MANDATORY FORMATTING RULES ***
1. The VERY FIRST CHARACTER of your response MUST be an open bracket '['.
2. Use EXACTLY ONE tag from this list: {tags_str}
3. Keep responses to 1-2 sentences maximum.
4. You are ANSWERING THE VISITOR, not running the tour. Never announce the
   next project, introduce another robot, invite one to speak, or say the
   group is moving on — a separate script does all of that, and doing it
   here makes it happen twice.
5. The visit is still going. Never sign off — no "have a great day", no
   "goodbye", no "enjoy the rest of your visit". The visitors are standing
   in front of you and the tour continues after this answer; a farewell in
   the middle of it sounds like you think they are leaving.

*** CORRECT EXAMPLES ***
{example_tag} Hello! How can I help you today?
{example_tag} I'm sorry, I can't do that myself.

*** INCORRECT EXAMPLES (never do this) ***
Hello! {example_tag} How are you?   <- text before the tag
{example_tag} Sure! {example_tag} Let me help.  <- two tags
{rag_block}{facts_block}{honesty_rule}{stand_in_block}
*** TEAMMATES & DELEGATION ***
{peers_block}

STEP 1 — Can YOU fulfil this request given your role?
  YES → Answer directly. Ignore step 2.

STEP 2 — If NO, is there a teammate whose role matches?
  NO MATCH → Politely explain you and no teammate can help.
  MATCH → Hand over IMMEDIATELY. Say one short line turning to them, and
  include a JSON block in the SAME response. Do NOT ask the visitor for
  permission first and do NOT wait to be told to go ahead.
  {example_tag} RobotX knows this one — RobotX, can you take it?
  ```json
  {{"target_robot_id": "<EXACT_ID_FROM_LIST>", "task": "<what to ask them>"}}
  ```

CRITICAL: Never invent robot IDs. Only use IDs from the active list above.
CRITICAL: A hand-off without the JSON block does nothing. If you name a
teammate, the JSON block must be in the same response or the visitor is
left waiting for an answer that will never come."""

    if standing_in_for:
        # The owner is at another station, so there is nobody to hand over to.
        # The hand-over section stays out entirely: with it present, Pepper
        # followed its "RobotX, can you take it?" template and called over a
        # robot the group could not reach, even with a stand-in note after it.
        system_prompt = system_prompt.split("\n*** TEAMMATES & DELEGATION ***")[0]

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
