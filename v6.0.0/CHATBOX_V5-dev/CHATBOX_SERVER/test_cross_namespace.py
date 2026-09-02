"""
Headless verification for Approach-1 STEP 2: cross-namespace culture↔person bridges.

Run:  python3 test_cross_namespace.py
No camera, no LLM, no network. Synthetic store; a FAKE embed_fn with controllable
cosine drives the embedding-bridge pass deterministically.

Proves: bridge edges let the EXISTING 2-round noisy-OR carry a person's observed
interest into a culturally-adjacent culture topic (and, with Step 1, a dislike pulls
it down); consolidation never merges `ck:` ↔ `topic:`; the overlay stays read-only,
deterministic, and degrades gracefully.
"""

from __future__ import annotations

import math
import os
import tempfile

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment
from modules.graph_relationship.topics import (
    add_person_topic, resolve_topic, link_related_cross, topic_id,
)
from modules.graph_relationship.cultures import (
    ensure_culture, ensure_culture_topic, knows_culture, set_culture_prior,
    culture_topic_id,
)
from modules.culture_seed import assign_person_culture
from modules.kg_extraction import (
    link_cross_namespace_bridges, consolidate_topics,
)
from modules.preference_model import rank_suggestions

_ROBOT = "chatbox"
_PERSON = "kid"
_CULTURE = "Korean"
_CID = "culture:korean"


def _store(priors: dict) -> InMemoryGraphStore:
    """Robot + person tagged Korean + given {ck_label: prior} culture topics."""
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id=_PERSON, display_name=_PERSON))
    c = ensure_culture(s, _CULTURE)
    knows_culture(s, _ROBOT, c.id)
    for label, (prior, cat) in priors.items():
        ct = ensure_culture_topic(s, c.id, label, category=cat)
        set_culture_prior(s, c.id, ct.id, prior)
    assign_person_culture(s, _PERSON, _CULTURE)
    return s


def _fake_embed(pairs: dict):
    """embed_fn where `pairs` (label -> equal-length vector) drives controlled
    cosines; every other label gets a unique high-dim one-hot, so it is orthogonal
    to the pair vectors (different length → cosine 0) and to other one-hots."""
    counter = {"n": 0}

    def fn(label: str):
        if label in pairs:
            return list(pairs[label])
        i = counter["n"]; counter["n"] += 1
        v = [0.0] * 64
        v[i % 64] = 1.0
        return v
    return fn


def _ns_counts(s) -> tuple:
    t = sum(1 for n in s._nodes.values() if n.node_type == "topic")
    ck = sum(1 for n in s._nodes.values() if n.node_type == "culture_topic")
    return t, ck


def _post(s, floor=0.0) -> dict:
    return {s.get_node(i).label: p for i, p in
            rank_suggestions(s, _PERSON, k=20, floor=floor)}


# ── 1. Embedding bridge drives the lift (delete → fallback) ───────────────────

def test_embedding_bridge_drives_lift():
    s = _store({"kpop": (0.30, "music")})
    add_person_topic(s, _PERSON, "jazz", "music", affinity=1.0, confidence=0.9)
    ck_kpop = culture_topic_id(_CID, "kpop")
    # explicit bridge topic:jazz ~ ck:korean:kpop (distinct slugs), weight 0.62
    assert link_related_cross(s, topic_id("jazz"), ck_kpop, 0.62)

    with_bridge = _post(s)
    assert with_bridge["kpop"] > 0.30 + 1e-9, with_bridge      # lifted by evidence

    # remove the bridge → kpop falls back to its raw prior (bridge caused the lift)
    s.delete_edge(*sorted((topic_id("jazz"), ck_kpop)), "related_topic")
    without = _post(s)
    assert abs(without["kpop"] - 0.30) < 1e-9, without
    print(f"1. embedding bridge: kpop 0.30 → {with_bridge['kpop']:.3f} with bridge; "
          f"{without['kpop']:.2f} after delete (fallback) ✓")


# ── 2. Exact-slug bridge: created, idempotent, merge-safe, traversable ────────

def test_exact_slug_bridge():
    s = _store({"hiking": (0.40, "activity")})
    add_person_topic(s, _PERSON, "hiking", "activity", affinity=0.9, confidence=0.9)
    embed = _fake_embed({})            # no embedding links — isolate the exact-slug pass

    r1 = link_cross_namespace_bridges(s, embed)
    exact = r1["exact"]
    assert len(exact) == 1 and exact[0][2] == 1.0, exact
    tid, ck = topic_id("hiking"), culture_topic_id(_CID, "hiking")
    edge = s.get_edge(*sorted((tid, ck)), "related_topic")
    assert edge is not None and edge.weight == 1.0, "exact-slug bridge missing"
    # the two same-slug nodes stay DISTINCT and the join is now a traversable edge
    assert s.get_node(tid).node_type == "topic"
    assert s.get_node(ck).node_type == "culture_topic"
    nbrs = [n.id for _e, n in s.query_neighbors(tid, "related_topic")]
    assert ck in nbrs, "bridge not traversable from the person topic"
    # idempotent — re-run adds no edge, reports it as existing
    e_before = len(s._edges)
    r2 = link_cross_namespace_bridges(s, embed)
    assert r2["exact"] == [] and len(s._edges) == e_before, r2
    print("2. exact-slug bridge: topic:hiking ~ ck:korean:hiking weight 1.0; distinct "
          "nodes; traversable; idempotent ✓")


# ── 3. Merge invariant: consolidation never merges ck ↔ topic ─────────────────

def test_merge_invariant():
    s = _store({"hiking": (0.40, "activity")})
    add_person_topic(s, _PERSON, "hiking", "activity", affinity=0.9)
    resolve_topic(s, "camping", category="activity")     # a 2nd distinct person topic
    before = _ns_counts(s)
    e_before = len(s._edges)

    # consolidation operates only on `topic` nodes — never touches ck: nodes.
    consolidate_topics(s, _fake_embed({}), source="test")
    assert _ns_counts(s) == before, "consolidation changed per-namespace node counts"

    # linking then only ADDS a related_topic edge (exact-slug hiking), no node change.
    link_cross_namespace_bridges(s, _fake_embed({}), source="test")
    assert _ns_counts(s) == before, "bridge linking changed node counts"
    assert len(s._edges) == e_before + 1, "expected exactly one new bridge edge"
    assert s.get_node(culture_topic_id(_CID, "hiking")).node_type == "culture_topic"
    assert s.get_node(topic_id("hiking")).node_type == "topic"
    print(f"3. merge invariant: ck/topic never merged (counts {before} unchanged); "
          "only +1 related_topic edge ✓")


# ── 4. Dislike crosses the bridge too (Step-1 × Step-2 compose) ───────────────

def test_dislike_crosses():
    s = _store({"kpop": (0.30, "music")})
    add_person_topic(s, _PERSON, "jazz", "music", affinity=0.1, confidence=0.9)  # dislike
    link_related_cross(s, topic_id("jazz"), culture_topic_id(_CID, "kpop"), 0.62)
    got = _post(s)
    assert got["kpop"] < 0.30 - 1e-9, got        # pulled DOWN below its prior
    print(f"4. dislike crosses: disliked jazz pulls kpop 0.30 → {got['kpop']:.3f} "
          "(signed clamp across the bridge) ✓")


# ── 5. Read-only / deterministic / graceful degradation ───────────────────────

def test_read_only_deterministic_graceful():
    s = _store({"kpop": (0.30, "music"), "kimchi": (0.55, "food")})
    add_person_topic(s, _PERSON, "jazz", "music", affinity=0.95)
    link_related_cross(s, topic_id("jazz"), culture_topic_id(_CID, "kpop"), 0.62)

    n0, e0 = len(s._nodes), len(s._edges)
    f0 = tempfile.mktemp(suffix=".json"); s.save(f0); before = open(f0).read()
    a = rank_suggestions(s, _PERSON, k=5)
    b = rank_suggestions(s, _PERSON, k=5)
    f1 = tempfile.mktemp(suffix=".json"); s.save(f1); after = open(f1).read()
    os.remove(f0); os.remove(f1)
    assert (len(s._nodes), len(s._edges)) == (n0, e0), "graph mutated by read"
    assert before == after, "store bytes changed — not read-only"
    assert a == b, "non-deterministic ranking"

    # graceful: delete ALL related_topic edges (incl. bridges) → prior-only, no crash
    for e in [e for e in list(s._edges.values()) if e.edge_type == "related_topic"]:
        s.delete_edge(e.source_id, e.target_id, "related_topic")
    got = _post(s)
    assert abs(got["kpop"] - 0.30) < 1e-9, got
    print(f"5. read-only ({n0}n/{e0}e, bytes stable), deterministic, graceful "
          f"(kpop→{got['kpop']:.2f} prior-only) ✓")


# ── 6. Embedding-band linker actually creates a cross bridge, idempotently ────

def test_embedding_linker_band():
    s = _store({"kpop": (0.30, "music")})
    add_person_topic(s, _PERSON, "jazz", "music", affinity=0.9)
    # jazz ~ kpop cosine ≈ 0.70 → inside [0.60, 0.86)
    embed = _fake_embed({"jazz": [1.0, 0.0], "kpop": [0.7, math.sqrt(1 - 0.49)]})
    r = link_cross_namespace_bridges(s, embed)
    assert len(r["links"]) == 1, r
    assert r["links"][0][2] and 0.60 <= r["links"][0][2] < 0.86, r
    edge = s.get_edge(*sorted((topic_id("jazz"), culture_topic_id(_CID, "kpop"))),
                      "related_topic")
    assert edge is not None, "embedding bridge not created"
    e_before = len(s._edges)
    r2 = link_cross_namespace_bridges(s, embed)          # idempotent
    assert r2["links"] == [] and len(s._edges) == e_before, r2
    print(f"6. embedding linker: jazz ~ kpop @ {r['links'][0][2]} bridged; idempotent ✓")


if __name__ == "__main__":
    test_embedding_bridge_drives_lift()
    test_exact_slug_bridge()
    test_merge_invariant()
    test_dislike_crosses()
    test_read_only_deterministic_graceful()
    test_embedding_linker_band()
    print("\nALL CROSS-NAMESPACE (STEP 2) TESTS PASSED")
