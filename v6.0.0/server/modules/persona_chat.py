"""
Persona chat — talk to the robot as the viz currently has it set up, to compare how
different roles / OCEAN personas change its replies.

Deliberately about the ROBOT only: the prompt is IDENTITY + a few general reply rules
+ the role/persona blocks (roles.role_persona_blocks, the same ones the webcam loop
uses). There is no WHO block, no person memory, no RAG, no culture, no tier: it does
not know or use who the visitor is.

Stateless per message: the KG file and persona_assignment.json are re-read every
time into a PRIVATE in-memory store, so the latest Apply takes effect immediately
and the webcam loop does not need to be running. Nothing is written back.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import List, Optional, Tuple

from modules.chatbox_actions import normalize_tag, tag_rule
from modules.graph_relationship.roles import (
    apply_assignment, attached_persona, attached_role, role_identity_block,
    role_persona_blocks,
)
from modules.graph_relationship.schema import Embodiment, RobotNode
from modules.graph_relationship.store import InMemoryGraphStore
from modules.role_seed import seed_role_library

# The lab AI server the robot talks to by default ($AI_HOST overrides; the viz server's
# --ai-server HOST[:PORT] overrides both). Used when no chat backend flag is given.
DEFAULT_AI_SERVER = os.environ.get("AI_HOST") or "172.24.47.255"

MAX_HISTORY_TURNS = 6          # (message, reply) pairs sent back to the model
MAX_MESSAGE_CHARS = 600

# The general reply-style rules from the loop's HOW TO REPLY block (verbatim; a drift
# test asserts they are still in the loop's prompt) minus the webcam-memory rules.
# The tag line differs on purpose: it offers only the actions the ChatBox can perform
# (modules/chatbox_actions.py) instead of any emotion word.
REPLY_RULES = (
    "━━━ HOW TO REPLY ━━━\n"
    "• Reply in ENGLISH, in one or two short, warm, spoken sentences. Output ONLY your "
    "single reply — never write the user's next turn.\n"
    + tag_rule() +
    "• Do NOT state a specific fact — a name, number, title, place, or a 'did you know…' "
    "claim — unless it appears in the memory below (including the cultural facts and your "
    "role knowledge). You may chat about a topic in general terms, but never make up "
    "specifics to sound knowledgeable; if you don't have a concrete fact, say so plainly "
    "or ask.\n"
    "• Stay on the topic they actually referenced. If they say 'tell me more about it', "
    "continue the SAME thing from the previous turn — do not switch to an unrelated topic, "
    "and do not force a connection between unrelated topics.\n"
    "• Reply to what they actually said or asked. Do not comment on how they seem to feel "
    "or offer emotional support unless they bring up their feelings themselves."
)

_TAG = re.compile(r"^\s*\[([A-Za-z][A-Za-z _-]{0,20})\]\s*")


class ChatError(RuntimeError):
    """The LLM backend failed (shown to the user as-is)."""


# ── LLM backends (stdlib only, same `respond` shape as the loop's clients) ───────

class OllamaChat:
    def __init__(self, model: str = "qwen2.5:7b", url: str = "http://127.0.0.1:11434"):
        self.model, self.url = model, url.rstrip("/")
        self.label = f"ollama:{model}"

    def respond(self, system_prompt: str, user_msg: str, history, max_tokens: int = 140) -> str:
        msgs = [{"role": "system", "content": system_prompt}]
        for u, b in history:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": user_msg})
        body = json.dumps({"model": self.model, "messages": msgs, "stream": False,
                           "options": {"temperature": 0.7, "num_predict": max_tokens}}).encode()
        req = urllib.request.Request(self.url + "/api/chat", body,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return str(json.load(r).get("message", {}).get("content", "")).strip()
        except (OSError, ValueError) as exc:
            return f"[LLM error: {exc}]"


def ai_server_chat(spec: str):
    """`HOST[:PORT]` → modules.ai_server_llm client (lazy: it is optional)."""
    from modules.ai_server_llm import AIServerLLM, DEFAULT_PORT
    host, _, port = spec.partition(":")
    llm = AIServerLLM(host, int(port) if port else DEFAULT_PORT)
    llm.connect()                       # False if unreachable: reply() retries per message
    llm.label = f"ai-server:{llm.host}:{llm.port}"
    return llm


# ── prompt + reply ───────────────────────────────────────────────────────────────

def split_tag(text: str) -> Tuple[str, str]:
    """'[HAPPY] hi there' → ('HAPPY', 'hi there'); no tag → ('', text)."""
    m = _TAG.match(text or "")
    return (m.group(1).strip().upper(), text[m.end():].strip()) if m else ("", (text or "").strip())


def load_assigned(assignment_path: Optional[str], robot_id: str) -> Optional[dict]:
    try:
        with open(assignment_path, encoding="utf-8") as fh:
            return json.load(fh).get(robot_id)
    except (OSError, ValueError, TypeError):
        return None


def build_store(kg_path: str, assignment_path: Optional[str], robot_id: str):
    """Private store = the KG file (or a fresh one) + the role library + the robot's
    latest assignment. Returns (store, assigned_entry_or_None)."""
    store = InMemoryGraphStore()
    try:
        store.load(kg_path, quiet=True)
    except Exception:                                  # missing / half-written file
        pass
    if store.get_node(robot_id) is None:
        store.upsert_node(RobotNode(id=robot_id, name=robot_id.capitalize(),
                                    embodiment=Embodiment.CAT))
    assigned = load_assigned(assignment_path, robot_id)
    if assigned:
        if store.get_node(assigned["role"]) is None:   # library not seeded in this file yet
            seed_role_library(store)
        apply_assignment(store, robot_id, assigned["role"], assigned["ocean"], source="chat")
    return store, assigned


def build_prompt(store, robot_id: str) -> Tuple[str, dict]:
    robot = store.get_node(robot_id)
    role, persona = attached_role(store, robot_id), attached_persona(store, robot_id)
    blocks = [role_identity_block(store, robot_id, robot.name, role, persona), REPLY_RULES]
    blocks += role_persona_blocks(store, role, persona)
    who = {"robot": robot_id, "name": robot.name,
           "role": role.descriptor if role else "", "persona": persona.descriptor if persona else ""}
    return "\n\n".join(blocks), who


class PersonaChat:
    def __init__(self, kg_path: str, assignment_path: Optional[str], llm):
        self.kg_path, self.assignment_path, self.llm = kg_path, assignment_path, llm

    @property
    def backend(self) -> str:
        return getattr(self.llm, "label", type(self.llm).__name__)

    def reply(self, robot_id: str, message: str, history: List[Tuple[str, str]]) -> dict:
        message = (message or "").strip()[:MAX_MESSAGE_CHARS]
        if not message:
            raise ValueError("empty message")
        if getattr(self.llm, "available", True) is False:
            self.llm.connect()          # the server may have come up since start-up
        store, assigned = build_store(self.kg_path, self.assignment_path, robot_id)
        system, who = build_prompt(store, robot_id)
        hist = [(str(u)[:MAX_MESSAGE_CHARS], str(b)[:MAX_MESSAGE_CHARS])
                for u, b in history[-MAX_HISTORY_TURNS:]]
        raw = self.llm.respond(system, message, hist)
        if raw.startswith("[LLM error"):
            raise ChatError(raw.strip("[]"))
        if raw.startswith("[LLM not connected"):
            raise ChatError(f"cannot reach {self.backend} — is the AI server up and on this "
                            "network? (use --ollama for a local model)")
        raw_tag, text = split_tag(raw)
        tag = normalize_tag(raw_tag)       # only tags the ChatBox can perform reach the robot
        return {"reply": text or raw, "tag": tag, **who,
                "raw_tag": raw_tag, "assigned_ts": (assigned or {}).get("ts"), "backend": self.backend}
