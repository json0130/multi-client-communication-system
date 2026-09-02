"""
Headless verification for the Bayesian preference overlay (Command B).

Run:  python3 test_preference_model.py
No camera, no LLM, no network. Synthetic in-memory store for the unit checks.

Asserts the overlay is READ-ONLY, deterministic, propagates over related_topic
links, excludes observed topics, and degrades gracefully to prior-only ranking.
"""

from __future__ import annotations

import json
import os
import tempfile

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import (
    PersonNode, RobotNode, Embodiment,
)
from modules.graph_relationship.topics import (
    resolve_topic, add_person_interest, link_related_topic, topic_id,
)
from modules.culture_seed import assign_person_culture
from modules.graph_relationship.cultures import (
    ensure_culture, ensure_culture_topic, knows_culture, set_culture_prior,
)
from modules.preference_model import rank_suggestions

_ROBOT = "chatbox"
_PERSON = "kid"


def _base_store(culture_priors: dict) -> InMemoryGraphStore:
    """Robot + person tagged Korean, with the given {label: prior} culture topics.
    No person interests, no related links (the caller adds those)."""
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id=_PERSON, display_name=_PERSON))
    c = ensure_culture(s, "Korean")
    knows_culture(s, _ROBOT, c.id)
    for label, prior in culture_priors.items():
        ct = ensure_culture_topic(s, c.id, label)
        set_culture_prior(s, c.id, ct.id, prior)
    assign_person_culture(s, _PERSON, "Korean")
    return s


# ── 1. Prior-only: no interests, no related links → culture priors desc ───────

def test_prior_only():
    s = _base_store({"kimchi": 0.80, "kpop": 0.70, "taekwondo": 0.45})
    got = rank_suggestions(s, _PERSON, k=3, floor=0.35)
    labels = [s.get_node(i).label for i, _ in got]
    posts = [p for _, p in got]
    assert labels == ["kimchi", "kpop", "taekwondo"], labels
    assert posts == [0.80, 0.70, 0.45], posts
    print("1. prior-only: ranking == culture priors desc "
          f"{list(zip(labels, posts))} ✓")


# ── 2. Propagation raises a linked topic above its raw prior ──────────────────

def test_propagation():
    # kpop has a LOW culture prior (0.30) so propagation from kdrama can raise it.
    s = _base_store({"kdrama": 0.65, "kpop": 0.30, "taekwondo": 0.45})
    # person observes kdrama; person-topic nodes for kdrama + kpop; related link.
    add_person_interest(s, _PERSON, "media", ["kdrama"], affinity=0.9)
    resolve_topic(s, "kpop")                       # unobserved person topic node
    link_related_topic(s, topic_id("kdrama"), topic_id("kpop"), 0.65)

    got = dict((s.get_node(i).label, p) for i, p in
               rank_suggestions(s, _PERSON, k=5, floor=0.30))
    # kpop rises above its raw prior 0.30 via 0.90*0.8*0.65 = 0.468
    assert abs(got["kpop"] - 0.468) < 1e-6, got
    assert got["kpop"] > 0.30
    # taekwondo (unlinked) stays at its prior
    assert abs(got["taekwondo"] - 0.45) < 1e-9, got
    print(f"2. propagation: kpop 0.30 → {got['kpop']:.3f} (linked); "
          f"taekwondo unchanged {got['taekwondo']:.2f} ✓")


# ── 3. Observed topics never appear in suggestions ────────────────────────────

def test_observed_excluded():
    s = _base_store({"kdrama": 0.65, "kpop": 0.30})
    add_person_interest(s, _PERSON, "media", ["kdrama"], affinity=0.9)
    resolve_topic(s, "kpop")
    link_related_topic(s, topic_id("kdrama"), topic_id("kpop"), 0.65)
    labels = [s.get_node(i).label for i, _ in rank_suggestions(s, _PERSON, k=9, floor=0.0)]
    assert "kdrama" not in labels, labels
    print(f"3. observed exclusion: kdrama absent from {labels} ✓")


# ── 4. Read-only: store unchanged (counts + save bytes) ───────────────────────

def test_read_only():
    s = _base_store({"kimchi": 0.80, "kpop": 0.30, "kdrama": 0.65})
    add_person_interest(s, _PERSON, "media", ["kdrama"], affinity=0.9)
    resolve_topic(s, "kpop")
    link_related_topic(s, topic_id("kdrama"), topic_id("kpop"), 0.65)

    n0, e0 = len(s._nodes), len(s._edges)
    f0 = tempfile.mktemp(suffix=".json"); s.save(f0); before = open(f0).read()
    for _ in range(3):
        rank_suggestions(s, _PERSON, k=4)
    f1 = tempfile.mktemp(suffix=".json"); s.save(f1); after = open(f1).read()
    os.remove(f0); os.remove(f1)
    assert (len(s._nodes), len(s._edges)) == (n0, e0), "node/edge count changed"
    assert before == after, "store bytes changed — overlay is not read-only"
    print(f"4. read-only: counts stable ({n0}n/{e0}e), save bytes identical ✓")


# ── 5. Determinism: identical calls → identical rankings ──────────────────────

def test_determinism():
    s = _base_store({"kimchi": 0.80, "kpop": 0.50, "kdrama": 0.65, "bibimbap": 0.65})
    add_person_interest(s, _PERSON, "media", ["kdrama"], affinity=0.9)
    resolve_topic(s, "kpop")
    link_related_topic(s, topic_id("kdrama"), topic_id("kpop"), 0.60)
    a = rank_suggestions(s, _PERSON, k=4)
    b = rank_suggestions(s, _PERSON, k=4)
    assert a == b, (a, b)
    # tie-break: equal-posterior kimchi/bibimbap? kimchi(0.80) leads; check order stable
    print(f"5. determinism: two calls identical → {[s.get_node(i).label for i,_ in a]} ✓")


# ── 6. Graceful degradation: no related links → prior-only, no crash ──────────

def test_graceful_degradation():
    s = _base_store({"kimchi": 0.80, "kpop": 0.30, "kdrama": 0.65})
    add_person_interest(s, _PERSON, "media", ["kdrama"], affinity=0.9)
    resolve_topic(s, "kpop")
    link_related_topic(s, topic_id("kdrama"), topic_id("kpop"), 0.65)
    # delete ALL related_topic edges
    for e in [e for e in list(s._edges.values()) if e.edge_type == "related_topic"]:
        s.delete_edge(e.source_id, e.target_id, "related_topic")
    got = dict((s.get_node(i).label, p) for i, p in
               rank_suggestions(s, _PERSON, k=5, floor=0.0))
    # kpop falls back to its raw prior (no propagation)
    assert abs(got["kpop"] - 0.30) < 1e-9, got
    assert "kdrama" not in got            # still observed-excluded
    print(f"6. graceful degradation: no related edges → prior-only "
          f"(kpop {got['kpop']:.2f}), no crash ✓")


if __name__ == "__main__":
    test_prior_only()
    test_propagation()
    test_observed_excluded()
    test_read_only()
    test_determinism()
    test_graceful_degradation()
    print("\nALL PREFERENCE-MODEL TESTS PASSED")
