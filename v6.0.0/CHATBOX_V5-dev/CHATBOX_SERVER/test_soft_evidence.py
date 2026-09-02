"""
Headless verification for Approach-1 STEP 3: confidence-weighted BN clamp.

Run:  python3 test_soft_evidence.py
No camera, no LLM, no network. Synthetic store; bridges created directly.

Proves the clamp `0.5 + (affinity-0.5)*confidence` is a strict GENERALIZATION of
Step 2: confidence 1.0 reproduces the exact Step-2 lift, lower confidence pulls the
clamp toward neutral so the observation moves the posterior less (and a weak signal
can't stomp a strong culture prior). Rounds/damping/floor/bridge traversal unchanged.
"""

from __future__ import annotations

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
from modules.preference_model import rank_suggestions, clamp_from

_ROBOT, _PERSON, _CULTURE, _CID = "chatbox", "kid", "Korean", "culture:korean"
_W = 0.62   # bridge weight used throughout


def _store(priors: dict) -> InMemoryGraphStore:
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


def _bridged(observed_label, cat, affinity, confidence, ck_label, ck_prior):
    """Person observes `observed_label` (affinity/confidence), bridged to a culture
    topic `ck_label` (prior ck_prior). Returns the ck topic's posterior for the person."""
    s = _store({ck_label: (ck_prior, cat)})
    add_person_topic(s, _PERSON, observed_label, cat,
                     affinity=affinity, confidence=confidence)
    link_related_cross(s, topic_id(observed_label), culture_topic_id(_CID, ck_label), _W)
    post = {s.get_node(i).label: p for i, p in
            rank_suggestions(s, _PERSON, k=20, floor=0.0)}
    return post.get(ck_label), s


# ── 1. Full trust reproduces the exact Step-2 lift ────────────────────────────

def test_full_trust_unchanged():
    assert clamp_from(1.0, 1.0) == 1.0
    kpop, _ = _bridged("jazz", "music", 1.0, 1.0, "kpop", 0.30)
    step2 = 1.0 * 0.8 * _W          # Step-2 formula: clamp(=affinity 1.0) * damping * w
    assert abs(kpop - step2) < 1e-9, (kpop, step2)
    assert abs(kpop - 0.496) < 1e-9, kpop
    print(f"1. full trust: clamp 1.0 → kpop {kpop:.4f} == Step-2 number {step2:.4f} ✓")


# ── 2. Uncertainty shrinks the lift ───────────────────────────────────────────

def test_uncertainty_shrinks_lift():
    assert abs(clamp_from(1.0, 0.6) - 0.8) < 1e-9
    high, _ = _bridged("jazz", "music", 1.0, 1.0, "kpop", 0.30)   # clamp 1.0
    low,  _ = _bridged("jazz", "music", 1.0, 0.6, "kpop", 0.30)   # clamp 0.8
    assert low < high, (low, high)
    assert abs(low - 0.8 * 0.8 * _W) < 1e-9, low
    print(f"1↔2. same affinity 1.0: lift(conf 1.0)={high:.4f} > lift(conf 0.6)={low:.4f} "
          "(uncertainty shrinks the lift) ✓")


# ── 3. Confident dislike still drags a neighbour down ─────────────────────────

def test_confident_dislike_drags_down():
    assert abs(clamp_from(0.05, 0.9) - 0.095) < 1e-9
    kpop, _ = _bridged("jazz", "music", 0.05, 0.9, "kpop", 0.30)
    assert kpop < 0.30 - 1e-9, kpop
    print(f"3. confident dislike: clamp 0.095 → kpop 0.30 → {kpop:.4f} (dragged down) ✓")


# ── 4. Weak signal can't stomp a strong prior ─────────────────────────────────

def test_weak_signal_spares_strong_prior():
    assert abs(clamp_from(0.4, 0.3) - 0.47) < 1e-9
    # strong culture prior 0.60; a weak, low-confidence dislike bridged to it
    weak,   _ = _bridged("outdoors", "activity", 0.4, 0.3, "baseball", 0.60)  # clamp 0.47
    strong, _ = _bridged("outdoors", "activity", 0.4, 1.0, "baseball", 0.60)  # clamp 0.40
    assert weak > 0.35, weak                 # prior survives the shaky evidence
    assert weak > strong, (weak, strong)     # confident evidence moves it more
    print(f"4. weak vs strong: prior 0.60 → weak(conf 0.3)={weak:.4f} (> floor 0.35, "
          f"survives) vs confident(conf 1.0)={strong:.4f} (moves more) ✓")


# ── 5. Neutral is a fixed point at any confidence ─────────────────────────────

def test_neutral_fixed_point():
    for conf in (0.0, 0.3, 0.7, 1.0):
        assert clamp_from(0.5, conf) == 0.5, conf
    a, _ = _bridged("jazz", "music", 0.5, 0.3, "kpop", 0.30)
    b, _ = _bridged("jazz", "music", 0.5, 1.0, "kpop", 0.30)
    assert abs(a - 0.30) < 1e-9 and abs(b - 0.30) < 1e-9, (a, b)
    print("5. neutral fixed point: affinity 0.5 at any confidence → clamp 0.5, kpop "
          "stays at prior 0.30 ✓")


# ── 6. Read-only / deterministic / graceful degradation ───────────────────────

def test_read_only_deterministic_graceful():
    s = _store({"kpop": (0.30, "music"), "kimchi": (0.55, "food")})
    add_person_topic(s, _PERSON, "jazz", "music", affinity=0.95, confidence=0.7)
    link_related_cross(s, topic_id("jazz"), culture_topic_id(_CID, "kpop"), _W)

    n0, e0 = len(s._nodes), len(s._edges)
    f0 = tempfile.mktemp(suffix=".json"); s.save(f0); before = open(f0).read()
    a = rank_suggestions(s, _PERSON, k=5)
    b = rank_suggestions(s, _PERSON, k=5)
    f1 = tempfile.mktemp(suffix=".json"); s.save(f1); after = open(f1).read()
    os.remove(f0); os.remove(f1)
    assert (len(s._nodes), len(s._edges)) == (n0, e0), "graph mutated by read"
    assert before == after, "store bytes changed — not read-only"
    assert a == b, "non-deterministic ranking"

    for e in [e for e in list(s._edges.values()) if e.edge_type == "related_topic"]:
        s.delete_edge(e.source_id, e.target_id, "related_topic")
    got = {s.get_node(i).label: p for i, p in rank_suggestions(s, _PERSON, k=9, floor=0.0)}
    assert abs(got["kpop"] - 0.30) < 1e-9, got     # prior-only fallback, no crash
    print(f"6. read-only ({n0}n/{e0}e, bytes stable), deterministic, graceful "
          f"(kpop→{got['kpop']:.2f} prior-only) ✓")


if __name__ == "__main__":
    test_full_trust_unchanged()
    test_uncertainty_shrinks_lift()
    test_confident_dislike_drags_down()
    test_weak_signal_spares_strong_prior()
    test_neutral_fixed_point()
    test_read_only_deterministic_graceful()
    print("\nALL SOFT-EVIDENCE (STEP 3) TESTS PASSED")
