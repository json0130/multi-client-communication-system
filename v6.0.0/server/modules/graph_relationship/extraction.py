"""
Knowledge extraction from conversation history.

Reads a session's transcript, asks an LLM to distill it into a structured
KnowledgeUpdate, runs deterministic guards over that output, then writes it back
to the graph:

  * closeness  — rapport/trust deltas applied to the pair's InteractionNode
  * interests  — new person Interest nodes + about-edges to shared Topics

Design contract
---------------
* Imports ONLY schema/store + the sibling interactions/topics helpers. The LLM
  is INJECTED as a callable `llm_fn(system_prompt, user_message) -> str`, so this
  module has NO dependency on any LLM library and graph_relationship/ stays
  copy-pasteable. No PAD, no kg_bridge.
* The LLM only PROPOSES; deterministic guards decide what actually lands in the
  graph (clamped deltas, normalized labels, capped counts, malformed → no-op).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .interactions import adjust_closeness
from .store import GraphStore
from .topics import Matcher, add_person_interest, link_capability_to_topic

# Type of the injected LLM call — matches demo_harness LLMClient.respond.
LLMFn = Callable[[str, str], str]

# Guards
_MAX_DELTA = 0.2          # per-session cap on rapport/trust change
_MAX_INTERESTS = 6
_MAX_TOPICS_PER_INTEREST = 6

_SYSTEM_PROMPT = (
    "You extract structured knowledge from a conversation between a CHILD and a "
    "companion ROBOT. Output ONLY a single JSON object, no prose, no code fences.\n"
    "Schema:\n"
    '{"interests": [{"label": "<short interest area, e.g. music>", '
    '"topics": ["<specific topic the child mentioned, e.g. jazz>"], '
    '"summary": "<one short sentence on what the child said about this>"}], '
    '"rapport_delta": <number between -0.2 and 0.2>, '
    '"trust_delta": <number between -0.2 and 0.2>}\n'
    "Rules: include ONLY interests/topics the CHILD actually expressed. The "
    "summary is a brief note about what the child shared on that interest. "
    "rapport_delta rises with warmth and positive affect; trust_delta rises with "
    "the child sharing personal things. Use values near 0 if the exchange was "
    "neutral or too short. If nothing was expressed, return empty interests and "
    "zero deltas."
)


@dataclass
class KnowledgeUpdate:
    """Validated, ready-to-apply extraction result.

    Each interest is (label, [topics], summary); the summary is attached as a
    per-person note on each of the interest's topic nodes.
    """
    interests: List[Tuple[str, List[str], str]] = field(default_factory=list)
    rapport_delta: float = 0.0
    trust_delta: float = 0.0

    @property
    def is_empty(self) -> bool:
        return (not self.interests
                and self.rapport_delta == 0.0 and self.trust_delta == 0.0)


# --- transcript formatting -------------------------------------------------

def format_transcript(turns: List[dict]) -> str:
    """Render a session's turns as 'child: ... / robot: ...' lines."""
    lines: List[str] = []
    for t in turns or []:
        child = (t.get("child") or "").strip()
        reply = (t.get("reply") or "").strip()
        emotion = t.get("emotion")
        if child:
            tag = f"child ({emotion})" if emotion else "child"
            lines.append(f"{tag}: {child}")
        if reply:
            lines.append(f"robot: {reply}")
    return "\n".join(lines)


# --- parsing + guards ------------------------------------------------------

def _extract_json_object(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of a possibly-noisy LLM response."""
    if not raw:
        return None
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _clamp_delta(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(-_MAX_DELTA, min(_MAX_DELTA, v))


def normalize(obj: Optional[dict]) -> KnowledgeUpdate:
    """Deterministic guards over raw LLM JSON → a safe KnowledgeUpdate."""
    if not isinstance(obj, dict):
        return KnowledgeUpdate()

    interests: List[Tuple[str, List[str], str]] = []
    seen_labels = set()
    for item in (obj.get("interests") or [])[:_MAX_INTERESTS]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label or label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        raw_topics = item.get("topics") or []
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        topics, seen_topics = [], set()
        for tp in raw_topics[:_MAX_TOPICS_PER_INTEREST]:
            tp = str(tp).strip()
            if tp and tp.lower() not in seen_topics:
                seen_topics.add(tp.lower())
                topics.append(tp)
        summary = str(item.get("summary", "")).strip()[:280]
        interests.append((label, topics, summary))

    return KnowledgeUpdate(
        interests=interests,
        rapport_delta=_clamp_delta(obj.get("rapport_delta")),
        trust_delta=_clamp_delta(obj.get("trust_delta")),
    )


# --- extract + apply -------------------------------------------------------

def extract(turns: List[dict], llm_fn: LLMFn) -> KnowledgeUpdate:
    """LLM-propose + guard: turn a transcript into a validated KnowledgeUpdate.

    Never raises on bad LLM output — returns an empty update instead.
    """
    transcript = format_transcript(turns)
    if not transcript.strip():
        return KnowledgeUpdate()
    try:
        raw = llm_fn(_SYSTEM_PROMPT, transcript)
    except Exception:
        return KnowledgeUpdate()
    return normalize(_extract_json_object(raw))


def apply_update(
    store: GraphStore, person_id: str, robot_id: str, update: KnowledgeUpdate,
    *, matcher: Optional[Matcher] = None, source: str = "extraction",
) -> dict:
    """Write a KnowledgeUpdate to the graph. Returns a summary dict.

    After adding each interest → topic, tries to link the robot's capability to
    that topic (via `matcher`, keyword by default) so a topic the child raised
    that the robot can cover becomes shared.
    """
    if update.rapport_delta or update.trust_delta:
        adjust_closeness(store, person_id, robot_id,
                         d_rapport=update.rapport_delta, d_trust=update.trust_delta,
                         source=source)
    added, capability_links = [], []
    for label, topics, summary in update.interests:
        node = add_person_interest(store, person_id, label, topics,
                                   summary=summary or None, source=source)
        if node is None:
            continue
        added.append((label, topics, summary))
        for tl in topics:
            item = link_capability_to_topic(store, robot_id, tl,
                                            matcher=matcher, source=source)
            if item is not None:
                capability_links.append((item, tl))
    return {
        "interests_added": added,
        "capability_links": capability_links,
        "rapport_delta": update.rapport_delta,
        "trust_delta": update.trust_delta,
    }


def extract_and_apply(
    store: GraphStore, person_id: str, robot_id: str, turns: List[dict], llm_fn: LLMFn,
    *, matcher: Optional[Matcher] = None, source: str = "extraction",
) -> Tuple[KnowledgeUpdate, dict]:
    """Convenience: extract from a transcript and apply it in one call."""
    update = extract(turns, llm_fn)
    summary = apply_update(store, person_id, robot_id, update, matcher=matcher, source=source)
    return update, summary
