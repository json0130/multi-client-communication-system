"""
PAD ↔ KG bridge — reads graph state before each turn, writes PAD output after.

Dependency direction
--------------------
  kg_bridge → graph_relationship (store + schema)
  kg_bridge reads from pad_result dict (string contract only; no import of pad_persona)
  pad_persona     does NOT import kg_bridge
  graph_relationship  does NOT import kg_bridge

This is the ONLY module that couples the two subsystems.

D-axis invariant
----------------
D (Dominance) is derived fresh from KG edges each turn via derive_tier()
and is NEVER written back to the graph.  V and A come from the face model only
(blended with the graph's FAST MoodEdge to soften frame spikes).  Long-term
state enters the PAD adapter exclusively as text via BridgeInput.structured_memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .schema import (
    AnyEdge,
    AttentionEdge,
    Embodiment,
    MoodEdge,
    PersonNode,
    Provenance,
    RobotNode,
)
from .store import GraphStore, InMemoryGraphStore
from .interactions import (
    append_turn,
    count_person_sessions,
    count_person_turns,
    get_interaction,
    get_or_create_interaction,
    set_closeness,
    start_session,
    sync_interaction_count,
)

# ---------------------------------------------------------------------------
# Emotion → (valence, arousal) — Russell (1980) circumplex model
# Mirrors pad_persona.pipeline_adapter.EMOTION_VA; kept separate so this
# module has no runtime dependency on pad_persona.
# ---------------------------------------------------------------------------
_EMOTION_VA: dict[str, tuple[float, float]] = {
    "happy":    ( 0.8,  0.6),
    "neutral":  ( 0.0,  0.0),
    "sad":      (-0.7, -0.4),
    "angry":    (-0.6,  0.7),
    "fear":     (-0.5,  0.8),
    "disgust":  (-0.6,  0.3),
    "surprise": ( 0.1,  0.8),
}

# Used when auto-creating a robot node on first post_turn.
# Robot node id is set to robot_id (not a UUID) for stable cross-session lookup.
_ROBOT_EMBODIMENT: dict[str, Embodiment] = {
    "chatbox": Embodiment.CAT,
    "ellebot": Embodiment.ELEPHANT,
}


# ---------------------------------------------------------------------------
# Public result type — frozen so callers can never alias into the store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BridgeInput:
    """
    Snapshot produced by pre_turn and consumed by PADPipelineAdapter.process_turn.

    Frozen so that a subsequent post_turn write cannot mutate a caller's reference —
    this is the structural guarantee for same-turn blend isolation.
    """
    valence: float
    arousal: float
    tier: str
    structured_memory: str
    rapport: float = 0.0
    trust: float = 0.0
    interaction_count: int = 0


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def emotion_label_to_va(label: Optional[str]) -> tuple[float, float]:
    """Return (valence, arousal) for a camera emotion label. Unknown/None → (0, 0)."""
    if label is None:
        return (0.0, 0.0)
    return _EMOTION_VA.get(label.lower(), (0.0, 0.0))


def _tier_from_scores(rapport: float, trust: float, count: int) -> str:
    """
    Core tier thresholds (unchanged behaviour).

    score = (rapport + trust) / 2
      score > 0.70  → "close"
      score > 0.45  → "known"
      count > 5     → "known"   (even with low score — seen enough turns)
      count > 0     → "visitor"
      else          → "unknown"
    """
    score = (rapport + trust) / 2.0
    if score > 0.70:
        return "close"
    if score > 0.45:
        return "known"
    if count > 5:
        return "known"
    if count > 0:
        return "visitor"
    return "unknown"


def _tier_from_edges(relationship_edges: List[AnyEdge]) -> str:
    """Tier from a pre-fetched edge list (used by tests that build edges directly)."""
    rapport = 0.0
    trust = 0.0
    count = 0
    for edge in relationship_edges:
        if edge.edge_type == "rapport":
            rapport = edge.weight
        elif edge.edge_type == "trust":
            trust = edge.weight
        elif edge.edge_type == "interaction_count":
            count = edge.count
    return _tier_from_scores(rapport, trust, count)


def derive_tier(person_id: str, robot_id: str, store: GraphStore) -> str:
    """
    Derive the relationship tier from the pair's InteractionNode.

    Closeness (rapport, trust) and the interaction count are now FIELDS on the
    single InteractionNode — there are no direct person→robot rapport/trust
    edges. Scoring thresholds are unchanged (_tier_from_scores).
    """
    interaction = get_interaction(store, person_id, robot_id)
    if interaction is None:
        return "unknown"
    return _tier_from_scores(
        interaction.rapport, interaction.trust, interaction.interaction_count,
    )


def format_slow_edges(attribute_edges: List[AnyEdge]) -> str:
    """
    Render SLOW (trait / preference) edges as prompt-ready text.

    Output: "[trait: shy] [prefers: <topic_id>]"
    Only SLOW-timescale edges are included; FAST edges (mood, attention,
    current_topic) are intentionally excluded — they must never enter the
    prompt text path that could influence tier derivation or V/A blending.
    Topic label resolution (target_id → TopicNode.label) is deferred to the
    server wiring step.
    """
    parts: list[str] = []
    for edge in attribute_edges:
        if edge.edge_type == "trait":
            parts.append(f"[trait: {edge.value}]")
        elif edge.edge_type == "preference":
            parts.append(f"[prefers: {edge.target_id}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prov(source: str) -> Provenance:
    return Provenance(source=source, confidence=1.0, timestamp=datetime.now(timezone.utc))


def _ensure_robot_node(store: GraphStore, robot_id: str) -> str:
    """Return robot_id after ensuring a robot node with id=robot_id exists."""
    if store.get_node(robot_id) is None:
        store.upsert_node(RobotNode(
            id=robot_id,
            name=robot_id,
            embodiment=_ROBOT_EMBODIMENT.get(robot_id.lower(), Embodiment.CAT),
        ))
    return robot_id


def _ensure_person_node(store: GraphStore, person_id: str) -> None:
    if store.get_node(person_id) is None:
        store.upsert_node(PersonNode(id=person_id))


# ---------------------------------------------------------------------------
# KGBridge
# ---------------------------------------------------------------------------

class KGBridge:
    """
    Thin stateful bridge: reads the KG before each PAD turn, writes back after.

    One instance per server session.  Thread-safety is the caller's responsibility.
    """

    def __init__(self, store: GraphStore) -> None:
        self._store = store
        # One session (meetup) per person for the lifetime of this bridge.
        # A fresh bridge = a fresh run = a new session, so next-meetup turns
        # land on a new Session node under the pair's InteractionNode.
        # person_id -> current session node id.
        self._session: dict[str, str] = {}

    def current_sessions(self) -> dict:
        """person_id -> current session node id for this bridge's lifetime.

        Used by end-of-session hooks (e.g. knowledge extraction) to find the
        transcript(s) produced during this meetup.
        """
        return dict(self._session)

    def pre_turn(
        self,
        person_id: Optional[str],
        robot_id: str,
        camera_emotion: Optional[str],
        camera_va: Optional[tuple] = None,
    ) -> BridgeInput:
        """
        Collect graph context for this turn and return frozen PAD inputs.

        camera_va: optional (valence, arousal) from a weighted softmax blend;
                   when provided, the emotion-label lookup table is bypassed.

        Cold-start / null-person path: does not touch the store; returns camera
        VA unblended, tier "unknown", empty structured_memory.
        """
        if camera_va is not None:
            camera_v, camera_a = camera_va
        else:
            camera_v, camera_a = emotion_label_to_va(camera_emotion)

        if person_id is None or self._store.get_node(person_id) is None:
            return BridgeInput(valence=camera_v, arousal=camera_a,
                               tier="unknown", structured_memory="")

        ctx = self._store.get_person_context(person_id)

        # Tier + closeness now come from the pair's InteractionNode.
        # D is never read from the graph here.
        tier = derive_tier(person_id, robot_id, self._store)
        interaction = get_interaction(self._store, person_id, robot_id)
        rapport = interaction.rapport if interaction else 0.0
        trust = interaction.trust if interaction else 0.0
        count = interaction.interaction_count if interaction else 0

        # Valence blend: camera is primary (0.7); graph MoodEdge softens spikes (0.3).
        # Arousal is NOT blended — only the camera frame contributes A.
        graph_mood = next(
            (e.value for e in ctx.person_attribute_edges if e.edge_type == "mood"),
            None,
        )
        blended_v = (0.7 * camera_v + 0.3 * graph_mood) if graph_mood is not None else camera_v

        # Structured memory: SLOW edges only — text path, never numeric.
        structured_memory = format_slow_edges(ctx.person_attribute_edges)

        return BridgeInput(
            valence=blended_v,
            arousal=camera_a,
            tier=tier,
            structured_memory=structured_memory,
            rapport=rapport,
            trust=trust,
            interaction_count=count,
        )

    def post_turn(
        self,
        person_id: Optional[str],
        robot_id: str,
        pad_result: dict,
        *,
        emotion: Optional[str] = None,
        child_message: Optional[str] = None,
        reply: Optional[str] = None,
    ) -> None:
        """
        Write PAD turn output to the KG.

        Writes two self-attribute edges (MoodEdge, AttentionEdge), then appends
        this turn to the current SESSION under the pair's InteractionNode
        (creating the interaction and/or session as needed) and refreshes the
        interaction's turn count. Closeness (rapport/trust) is updated separately
        by set_closeness — not here.

        The optional emotion / child_message / reply are stored on the session's
        turn list, so the graph holds the conversation (no separate transcript).

        D (Dominance) is NOT written — it is re-derived each turn and must never
        be persisted as its own edge.
        """
        if person_id is None:
            return

        _ensure_person_node(self._store, person_id)
        robot_node_id = _ensure_robot_node(self._store, robot_id)

        p, a, _d = pad_result["pad_state"]
        prov = _prov(robot_id)

        # AttentionEdge expects [0, 1]; rescale PAD arousal from [-1, 1]
        attention_value = max(0.0, min(1.0, (a + 1.0) / 2.0))

        # Self-attribute edges (mood + attention). No graph scan (apply_delta contract).
        self._store.apply_delta(
            edges=[
                MoodEdge(
                    source_id=person_id,
                    target_id=person_id,
                    provenance=prov,
                    value=max(-1.0, min(1.0, p)),
                ),
                AttentionEdge(
                    source_id=person_id,
                    target_id=person_id,
                    provenance=prov,
                    value=attention_value,
                ),
            ]
        )

        # Ensure the pair's InteractionNode, then append this turn to the
        # current per-meetup session hanging under it (start one on first turn).
        interaction = get_or_create_interaction(
            self._store, person_id, robot_node_id, source=robot_id,
        )
        session_id = self._session.get(person_id)
        # Recreate if we have no session yet, or ours was deleted externally.
        if session_id is None or self._store.get_node(session_id) is None:
            label = f"session {count_person_sessions(self._store, person_id) + 1}"
            session = start_session(
                self._store, interaction_id_=interaction.id, label=label, source=robot_id,
            )
            session_id = session.id
            self._session[person_id] = session_id
        append_turn(
            self._store, session_id=session_id,
            emotion=emotion, child_message=child_message, reply=reply,
        )
        # Keep interaction_count (total turns) in sync on the InteractionNode.
        sync_interaction_count(self._store, person_id, robot_node_id)


# ---------------------------------------------------------------------------
# KG — scripting / REPL façade over InMemoryGraphStore
# ---------------------------------------------------------------------------

class KG:
    """
    Thin convenience wrapper for scripts and notebooks.

    Closeness (rapport / trust) and interaction_count are SET semantics and live
    as fields on the pair's InteractionNode (see interactions.py).
    """

    def __init__(self) -> None:
        self._store = InMemoryGraphStore()

    def _ensure(self, person_id: str, robot_id: str) -> None:
        if self._store.get_node(person_id) is None:
            self._store.upsert_node(PersonNode(id=person_id))
        if self._store.get_node(robot_id) is None:
            self._store.upsert_node(RobotNode(
                id=robot_id, name=robot_id,
                embodiment=_ROBOT_EMBODIMENT.get(robot_id.lower(), Embodiment.CAT),
            ))

    def set_rapport(self, person_id: str, robot_id: str, weight: float) -> None:
        self._ensure(person_id, robot_id)
        set_closeness(self._store, person_id, robot_id, rapport=weight, source="kg-facade")

    def set_trust(self, person_id: str, robot_id: str, weight: float) -> None:
        self._ensure(person_id, robot_id)
        set_closeness(self._store, person_id, robot_id, trust=weight, source="kg-facade")
