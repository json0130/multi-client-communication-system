"""
Interaction + Session helpers.

The whole person↔robot relationship is abstracted into ONE InteractionNode
(deterministic id per pair). Both participants link to it (has_interaction);
its Session subnodes (has_session) hold the conversation history:

    person --has_interaction--> Interaction <--has_interaction-- robot
    Interaction { rapport, trust, interaction_count } --has_session--> Session {turns}

The InteractionNode holds "how close they are" (rapport, trust) and how much
they have interacted (interaction_count = total turns across sessions).

Design contract: imports ONLY schema.py + store.py — no PAD, no kg_bridge.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from .schema import (
    HasInteractionEdge,
    HasSessionEdge,
    InteractionNode,
    Provenance,
    SessionNode,
)
from .store import GraphStore


def _prov(source: str) -> Provenance:
    return Provenance(source=source, confidence=1.0,
                      timestamp=datetime.now(timezone.utc))


def interaction_id(person_id: str, robot_id: str) -> str:
    """Deterministic id — one InteractionNode per (person, robot) pair."""
    return f"interaction:{person_id}:{robot_id}"


# --- Interaction node ------------------------------------------------------

def get_or_create_interaction(
    store: GraphStore, person_id: str, robot_id: str, *, source: Optional[str] = None,
) -> InteractionNode:
    """Return the pair's InteractionNode, creating + linking it if absent.

    Person and robot nodes must already exist.
    """
    iid = interaction_id(person_id, robot_id)
    node = store.get_node(iid)
    if node is not None and node.node_type == "interaction":
        return node
    node = InteractionNode(id=iid)
    prov = _prov(source or robot_id)
    store.apply_delta(
        nodes=[node],
        edges=[
            HasInteractionEdge(source_id=person_id, target_id=iid, provenance=prov),
            HasInteractionEdge(source_id=robot_id, target_id=iid, provenance=prov),
        ],
    )
    return node


def get_interaction(store: GraphStore, person_id: str, robot_id: str) -> Optional[InteractionNode]:
    node = store.get_node(interaction_id(person_id, robot_id))
    return node if node is not None and node.node_type == "interaction" else None


def set_closeness(
    store: GraphStore, person_id: str, robot_id: str, *,
    rapport: Optional[float] = None, trust: Optional[float] = None,
    source: Optional[str] = None,
) -> InteractionNode:
    """Set rapport and/or trust on the pair's InteractionNode (SET, not add)."""
    node = get_or_create_interaction(store, person_id, robot_id, source=source)
    update = {}
    if rapport is not None:
        update["rapport"] = max(0.0, min(1.0, rapport))
    if trust is not None:
        update["trust"] = max(0.0, min(1.0, trust))
    if update:
        node = node.model_copy(update=update)
        store.upsert_node(node)
    return node


def adjust_closeness(
    store: GraphStore, person_id: str, robot_id: str, *,
    d_rapport: float = 0.0, d_trust: float = 0.0, source: Optional[str] = None,
) -> InteractionNode:
    """Add a delta to rapport/trust on the InteractionNode, clamped to [0, 1]."""
    node = get_or_create_interaction(store, person_id, robot_id, source=source)
    node = node.model_copy(update={
        "rapport": max(0.0, min(1.0, node.rapport + d_rapport)),
        "trust":   max(0.0, min(1.0, node.trust + d_trust)),
    })
    store.upsert_node(node)
    return node


# --- Sessions --------------------------------------------------------------

def sessions_of(store: GraphStore, interaction_id_: str) -> List[SessionNode]:
    return [
        n for _e, n in store.query_neighbors(interaction_id_, "has_session")
        if n.node_type == "session"
    ]


def start_session(
    store: GraphStore, *, interaction_id_: str, label: Optional[str] = None,
    source: Optional[str] = None,
) -> SessionNode:
    """Create a new Session under an interaction (has_session edge)."""
    session = SessionNode(label=label or "session", turn_count=0, turns=[])
    store.apply_delta(
        nodes=[session],
        edges=[HasSessionEdge(
            source_id=interaction_id_, target_id=session.id, provenance=_prov(source or "session"),
        )],
    )
    return session


def unextracted_turns(session: SessionNode) -> List[dict]:
    """Turns of a session not yet fed to knowledge extraction."""
    return list(session.turns[session.extracted_turns:])


def mark_session_extracted(store: GraphStore, session_id: str) -> Optional[SessionNode]:
    """Record that all current turns of a session have been extracted."""
    session = store.get_node(session_id)
    if session is None or session.node_type != "session":
        return None
    updated = session.model_copy(update={"extracted_turns": len(session.turns)})
    store.upsert_node(updated)
    return updated


def append_turn(
    store: GraphStore, *, session_id: str,
    emotion: Optional[str] = None, child_message: Optional[str] = None,
    reply: Optional[str] = None,
) -> Optional[SessionNode]:
    """Append one turn to a Session's transcript and bump its turn_count."""
    session = store.get_node(session_id)
    if session is None or session.node_type != "session":
        return None
    turns = list(session.turns)
    turns.append({
        "turn": session.turn_count + 1,
        "ts": datetime.now(timezone.utc).isoformat(),
        "emotion": emotion,
        "child": child_message,
        "reply": reply,
    })
    updated = session.model_copy(update={"turns": turns, "turn_count": session.turn_count + 1})
    store.upsert_node(updated)
    return updated


# --- Counts (index-based) --------------------------------------------------

def count_person_sessions(store: GraphStore, person_id: str) -> int:
    """Number of meetups: sessions under all of a person's interactions."""
    total = 0
    for _e, inter in store.query_neighbors(person_id, "has_interaction"):
        if inter.node_type == "interaction":
            total += len(sessions_of(store, inter.id))
    return total


def count_person_turns(store: GraphStore, person_id: str) -> int:
    """Total turns across a person's sessions (the interaction count)."""
    total = 0
    for _e, inter in store.query_neighbors(person_id, "has_interaction"):
        if inter.node_type == "interaction":
            total += sum(s.turn_count for s in sessions_of(store, inter.id))
    return total


def sync_interaction_count(store: GraphStore, person_id: str, robot_id: str) -> InteractionNode:
    """Recompute interaction_count (total turns) onto the InteractionNode."""
    node = get_or_create_interaction(store, person_id, robot_id)
    total = sum(s.turn_count for s in sessions_of(store, node.id))
    if total != node.interaction_count:
        node = node.model_copy(update={"interaction_count": total})
        store.upsert_node(node)
    return node


def set_interaction_count(store: GraphStore, person_id: str, robot_id: str,
                          count: int, *, source: Optional[str] = None) -> InteractionNode:
    """Set interaction_count directly (e.g. from an external transcript store when
    SessionNodes no longer live in the graph). Pure; no LLM/DB imports here."""
    node = get_or_create_interaction(store, person_id, robot_id, source=source)
    count = max(0, int(count))
    if count != node.interaction_count:
        node = node.model_copy(update={"interaction_count": count})
        store.upsert_node(node)
    return node
