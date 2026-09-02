"""
Tests for knowledge extraction (extraction.py) with an injected fake LLM —
no real model needed. Also exercises the deterministic guards.
"""

from .schema import Embodiment, PersonNode, RobotNode
from .store import InMemoryGraphStore
from .extraction import (
    KnowledgeUpdate, apply_update, extract, extract_and_apply, normalize,
    _extract_json_object,
)
from .interactions import get_interaction
from .topics import person_interests, shared_topics


def _fake_llm(response: str):
    """Return an llm_fn(system, user) -> str that always yields `response`."""
    return lambda system, user: response


def _store_with_pair():
    store = InMemoryGraphStore()
    store.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    store.upsert_node(PersonNode(id="jay", display_name="Jay"))
    return store


_TURNS = [
    {"turn": 1, "emotion": "happy", "child": "I love playing jazz on piano", "reply": "Cool!"},
]


# --- parsing / guards ------------------------------------------------------

def test_extract_json_from_noisy_response():
    raw = "Sure! Here is the JSON:\n```json\n{\"rapport_delta\": 0.1}\n```\nHope it helps"
    assert _extract_json_object(raw) == {"rapport_delta": 0.1}


def test_normalize_clamps_deltas():
    u = normalize({"rapport_delta": 5.0, "trust_delta": -9.0, "interests": []})
    assert u.rapport_delta == 0.2 and u.trust_delta == -0.2


def test_normalize_dedupes_and_shapes_interests():
    u = normalize({"interests": [
        {"label": "Music", "topics": ["jazz", "jazz", "piano"], "summary": "loves jazz"},
        {"label": "music", "topics": []},        # duplicate label (case-insensitive)
        {"label": "", "topics": ["x"]},          # empty label dropped
        "not a dict",                             # ignored
    ]})
    assert u.interests == [("Music", ["jazz", "piano"], "loves jazz")]


def test_normalize_bad_input_is_empty():
    assert normalize(None).is_empty
    assert normalize({"interests": "nope", "rapport_delta": "abc"}).is_empty


def test_extract_malformed_llm_is_noop():
    assert extract(_TURNS, _fake_llm("total nonsense, no json")).is_empty


def test_extract_empty_transcript_skips_llm():
    called = []
    def llm(s, u):
        called.append(1); return "{}"
    assert extract([], llm).is_empty
    assert not called  # never called the LLM on an empty transcript


# --- apply -----------------------------------------------------------------

def test_apply_update_writes_closeness_and_interest():
    store = _store_with_pair()
    update = KnowledgeUpdate(interests=[("music", ["jazz"], "loves jazz piano")],
                             rapport_delta=0.1, trust_delta=0.05)
    summary = apply_update(store, "jay", "chatbox", update)

    inter = get_interaction(store, "jay", "chatbox")
    assert abs(inter.rapport - 0.1) < 1e-9 and abs(inter.trust - 0.05) < 1e-9
    assert [(i.label, [t.label for t in ts]) for i, ts in person_interests(store, "jay")] \
        == [("music", ["jazz"])]
    assert summary["interests_added"] == [("music", ["jazz"], "loves jazz piano")]
    # the summary is attached as a note on the jazz topic
    jazz = store.get_node("topic:jazz")
    assert jazz.notes and jazz.notes[0]["person"] == "jay"
    assert jazz.notes[0]["text"] == "loves jazz piano"


def test_extracted_interest_shares_robot_topic():
    """An extracted interest that matches the robot's capability topic is shared."""
    store = _store_with_pair()
    # robot knows jazz via capability(items) -> about -> topic
    from .schema import CapabilityNode, HasCapabilityEdge, AboutEdge, Provenance
    from .topics import resolve_topic
    prov = Provenance(source="t", confidence=1.0)
    cap = CapabilityNode(id="chatbox:capability", items=["knows jazz"])
    store.upsert_node(cap)
    store.upsert_edge(HasCapabilityEdge(source_id="chatbox", target_id=cap.id, provenance=prov))
    store.upsert_edge(AboutEdge(source_id=cap.id, target_id=resolve_topic(store, "jazz").id,
                                label="knows jazz", provenance=prov))

    llm = _fake_llm('{"interests":[{"label":"music","topics":["jazz"]}],'
                    '"rapport_delta":0.1,"trust_delta":0.0}')
    extract_and_apply(store, "jay", "chatbox", _TURNS, llm)
    assert shared_topics(store, "jay", "chatbox") == ["jazz"]


def test_apply_is_idempotent():
    store = _store_with_pair()
    llm = _fake_llm('{"interests":[{"label":"music","topics":["jazz"]}],'
                    '"rapport_delta":0.0,"trust_delta":0.0}')
    extract_and_apply(store, "jay", "chatbox", _TURNS, llm)
    n1, e1 = len(store._nodes), len(store._edges)
    extract_and_apply(store, "jay", "chatbox", _TURNS, llm)  # same again
    assert (len(store._nodes), len(store._edges)) == (n1, e1)
