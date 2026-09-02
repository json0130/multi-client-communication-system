"""Unit tests for the graph_relationship storage layer (InMemoryGraphStore)."""

from datetime import datetime, timedelta, timezone

import pytest

from .schema import (
    AttentionEdge,
    Embodiment,
    InteractionCountEdge,
    MoodEdge,
    PersonNode,
    PreferenceEdge,
    Provenance,
    RapportEdge,
    RobotNode,
    TopicNode,
    TraitEdge,
    TrustEdge,
)
from .store import InMemoryGraphStore, PersonContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_T1 = _T0 + timedelta(seconds=10)


def _prov(source: str = "robot:cat", confidence: float = 0.9, ts: datetime = _T0) -> Provenance:
    return Provenance(source=source, confidence=confidence, timestamp=ts)


def _make_base():
    """Return a store pre-loaded with one robot and one person."""
    store = InMemoryGraphStore()
    robot = store.upsert_node(RobotNode(name="CHATBOX", embodiment=Embodiment.CAT))
    person = store.upsert_node(PersonNode(display_name="Alex"))
    return store, robot, person


# ---------------------------------------------------------------------------
# Test 1 — upsert + get round-trip (node and edge)
# ---------------------------------------------------------------------------

def test_upsert_node_round_trip():
    store = InMemoryGraphStore()
    node = PersonNode(display_name="Sam")
    store.upsert_node(node)
    fetched = store.get_node(node.id)
    assert fetched is not None
    assert fetched.display_name == "Sam"
    assert fetched.node_type == "person"


def test_upsert_edge_round_trip():
    store, robot, person = _make_base()
    edge = store.upsert_edge(
        RapportEdge(
            source_id=robot.id,
            target_id=person.id,
            provenance=_prov(),
            weight=0.75,
        )
    )
    fetched = store.get_edge(robot.id, person.id, "rapport")
    assert fetched is not None
    assert fetched.weight == 0.75
    assert fetched.provenance.source == "robot:cat"
    assert fetched.id == edge.id


# ---------------------------------------------------------------------------
# Test 2 — get_person_context returns only that person's edges
# ---------------------------------------------------------------------------

def test_get_person_context_is_scoped():
    store, robot, alex = _make_base()
    sam = store.upsert_node(PersonNode(display_name="Sam"))

    # Alex's edges
    store.upsert_edge(
        RapportEdge(source_id=robot.id, target_id=alex.id, provenance=_prov(), weight=0.6)
    )
    store.upsert_edge(
        MoodEdge(source_id=alex.id, target_id=alex.id, provenance=_prov(), value=0.4)
    )

    # Sam's edges — must not appear in Alex's context
    store.upsert_edge(
        RapportEdge(source_id=robot.id, target_id=sam.id, provenance=_prov(), weight=0.3)
    )
    store.upsert_edge(
        MoodEdge(source_id=sam.id, target_id=sam.id, provenance=_prov(), value=-0.2)
    )

    ctx = store.get_person_context(alex.id)

    assert len(ctx.relationship_edges) == 1
    assert ctx.relationship_edges[0].edge_type == "rapport"
    assert ctx.relationship_edges[0].target_id == alex.id

    assert len(ctx.person_attribute_edges) == 1
    assert ctx.person_attribute_edges[0].edge_type == "mood"
    assert ctx.person_attribute_edges[0].source_id == alex.id


# ---------------------------------------------------------------------------
# Test 3 — apply_delta touches only the delta; pre-existing data is unchanged
# ---------------------------------------------------------------------------

def test_apply_delta_only_touches_delta():
    store, robot, person = _make_base()

    # Pre-existing edge
    existing_rapport = store.upsert_edge(
        RapportEdge(source_id=robot.id, target_id=person.id, provenance=_prov(), weight=0.5)
    )

    # Delta: only a mood edge for the person — no update to rapport
    store.apply_delta(
        edges=[
            MoodEdge(source_id=person.id, target_id=person.id, provenance=_prov(), value=0.8)
        ]
    )

    # Rapport must be unchanged
    rapport_after = store.get_edge(robot.id, person.id, "rapport")
    assert rapport_after is not None
    assert rapport_after.weight == 0.5
    assert rapport_after.id == existing_rapport.id

    # Mood must exist
    mood = store.get_edge(person.id, person.id, "mood")
    assert mood is not None
    assert mood.value == 0.8


# ---------------------------------------------------------------------------
# Test 4 — interaction_count increments rather than overwrites
# ---------------------------------------------------------------------------

def test_interaction_count_increments():
    store, robot, person = _make_base()

    store.upsert_edge(
        InteractionCountEdge(
            source_id=robot.id, target_id=person.id,
            provenance=_prov(ts=_T0), count=3
        )
    )
    store.upsert_edge(
        InteractionCountEdge(
            source_id=robot.id, target_id=person.id,
            provenance=_prov(ts=_T1), count=2
        )
    )

    edge = store.get_edge(robot.id, person.id, "interaction_count")
    assert edge is not None
    assert edge.count == 5  # 3 + 2, not 2


# ---------------------------------------------------------------------------
# Test 5 — query_neighbors is scoped by edge_type
# ---------------------------------------------------------------------------

def test_query_neighbors_scoped_by_edge_type():
    store, robot, person = _make_base()
    topic = store.upsert_node(TopicNode(label="school stress"))

    store.upsert_edge(
        RapportEdge(source_id=robot.id, target_id=person.id, provenance=_prov(), weight=0.7)
    )
    store.upsert_edge(
        TrustEdge(source_id=robot.id, target_id=person.id, provenance=_prov(), weight=0.5)
    )
    store.upsert_edge(
        PreferenceEdge(source_id=person.id, target_id=topic.id, provenance=_prov(), weight=0.9)
    )

    # Unfiltered: robot has two outgoing edges to person
    robot_neighbors = store.query_neighbors(robot.id)
    assert len(robot_neighbors) == 2

    # Filtered to trust only
    trust_neighbors = store.query_neighbors(robot.id, edge_type="trust")
    assert len(trust_neighbors) == 1
    edge, neighbor = trust_neighbors[0]
    assert edge.edge_type == "trust"
    assert neighbor.id == person.id

    # Person's neighbors: robot (via rapport/trust — indexed on both sides) + topic
    person_neighbors = store.query_neighbors(person.id, edge_type="preference")
    assert len(person_neighbors) == 1
    edge, neighbor = person_neighbors[0]
    assert edge.edge_type == "preference"
    assert neighbor.id == topic.id


# ---------------------------------------------------------------------------
# Test 6 — stale write (older timestamp) does not overwrite stored value
# ---------------------------------------------------------------------------

def test_stale_write_discarded():
    store, robot, person = _make_base()

    store.upsert_edge(
        RapportEdge(
            source_id=robot.id, target_id=person.id,
            provenance=_prov(ts=_T1),  # stored at T1
            weight=0.9,
        )
    )

    # Incoming write has an older timestamp — should be discarded
    store.upsert_edge(
        RapportEdge(
            source_id=robot.id, target_id=person.id,
            provenance=_prov(ts=_T0),  # older than stored
            weight=0.1,
        )
    )

    edge = store.get_edge(robot.id, person.id, "rapport")
    assert edge.weight == 0.9  # original value preserved
