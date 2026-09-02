"""
Tests for the shared-TopicNode / Interest layer (topics.py + seeding).
"""

from datetime import timezone

from .schema import (
    AboutEdge, CapabilityNode, Embodiment, HasCapabilityEdge, HasInterestEdge,
    InterestNode, PersonNode, Provenance, RobotNode,
)
from .store import InMemoryGraphStore
from .topics import (
    interest_id, keyword_match, link_capability_to_topic, person_interests,
    resolve_topic, robot_topics, shared_topics, topic_id,
)

_CAP_ID = "chatbox:capability"


def _prov():
    return Provenance(source="test", confidence=1.0)


def _seed_shared(store):
    """robot reaches jazz via capability list; person via interest 'music'."""
    store.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    store.upsert_node(PersonNode(id="jay", display_name="Jay"))
    # robot: has_capability -> capability(items) --about[label]--> topic
    cap = CapabilityNode(id=_CAP_ID, items=["tells stories", "knows jazz"])
    store.upsert_node(cap)
    store.upsert_edge(HasCapabilityEdge(source_id="chatbox", target_id=cap.id, provenance=_prov()))
    t1 = resolve_topic(store, "jazz")
    store.upsert_edge(AboutEdge(source_id=cap.id, target_id=t1.id, label="knows jazz", provenance=_prov()))
    # person: has_interest -> interest -> about -> topic
    inode = InterestNode(id=interest_id("jay", "music"), label="music")
    store.upsert_node(inode)
    store.upsert_edge(HasInterestEdge(source_id="jay", target_id=inode.id, provenance=_prov()))
    t2 = resolve_topic(store, "jazz")
    store.upsert_edge(AboutEdge(source_id=inode.id, target_id=t2.id, provenance=_prov()))
    return t1, t2


def test_keyword_match():
    assert keyword_match("knows jazz", "jazz")
    assert keyword_match("good at math", "math")
    assert keyword_match("knows about space", "space")
    assert not keyword_match("good at math", "addition")


def test_link_capability_to_topic_labels_edge():
    store = InMemoryGraphStore()
    store.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    store.upsert_node(CapabilityNode(id=_CAP_ID, items=["good at math"]))
    store.upsert_edge(HasCapabilityEdge(source_id="chatbox", target_id=_CAP_ID, provenance=_prov()))
    resolve_topic(store, "math")
    item = link_capability_to_topic(store, "chatbox", "math")
    assert item == "good at math"
    edge = store.get_edge(_CAP_ID, "topic:math", "about")
    assert edge is not None and edge.label == "good at math"
    assert [t.label for t in robot_topics(store, "chatbox")] == ["math"]
    # keyword can't bridge 'addition' to 'good at math'
    resolve_topic(store, "addition")
    assert link_capability_to_topic(store, "chatbox", "addition") is None


def test_resolve_topic_is_one_shared_node():
    store = InMemoryGraphStore()
    t1, t2 = _seed_shared(store)
    assert t1.id == t2.id == topic_id("jazz") == "topic:jazz"
    topics = [n for n in store._nodes.values() if n.node_type == "topic"]
    assert len(topics) == 1  # robot 'jazz' and interest-about 'jazz' collapse to one


def test_shared_topics_traversal():
    store = InMemoryGraphStore()
    _seed_shared(store)
    assert shared_topics(store, "jay", "chatbox") == ["jazz"]


def test_shared_topics_empty_when_human_interest_removed():
    store = InMemoryGraphStore()
    _seed_shared(store)
    store.delete_edge(interest_id("jay", "music"), "topic:jazz", "about")
    assert shared_topics(store, "jay", "chatbox") == []


def test_shared_topics_empty_when_robot_capability_topic_removed():
    store = InMemoryGraphStore()
    _seed_shared(store)
    store.delete_edge(_CAP_ID, "topic:jazz", "about")   # robot no longer reaches jazz
    assert shared_topics(store, "jay", "chatbox") == []


def test_person_interests():
    store = InMemoryGraphStore()
    _seed_shared(store)
    result = person_interests(store, "jay")
    assert len(result) == 1
    interest, topics = result[0]
    assert interest.label == "music"
    assert [t.label for t in topics] == ["jazz"]


def test_interest_id_deterministic():
    assert interest_id("jay", "Music") == interest_id("jay", "music")
    assert interest_id("jay", "music") == "interest:jay:music"
