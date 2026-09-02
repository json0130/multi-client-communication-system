"""
Headless verification for Approach-1 STEP 4: thin per-culture STATIC style hint.

Run:  python3 test_style_hint.py
No camera, no LLM, no network.

Proves the manner hint is: a single static string on CultureNode (default ""),
seeded idempotently, injected into HOW-TO-REPLY only when the person is tagged to a
culture with a non-empty hint, kept SEPARATE from the content topic-offer block, and
identical regardless of tier/interaction (no dynamics — that is Approach 2).
"""

from __future__ import annotations

import json
import os
import tempfile

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment
from modules.graph_relationship.cultures import ensure_culture, person_culture_style_hint
from modules.culture_seed import seed_korean_demo, assign_person_culture, _KOREAN_STYLE_HINT
from modules.face_webcam.webcam_loop import WebcamKGLoop

_ROBOT = "chatbox"
_MANNER = "Cultural manner (soft guidance"


def _loop(store) -> WebcamKGLoop:
    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store = store; L.robot_id = _ROBOT; L._robot_display = "ChatBox"
    return L


def _robot_store() -> InMemoryGraphStore:
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    return s


def _manner_line(prompt: str) -> str:
    for ln in prompt.splitlines():
        if _MANNER in ln:
            return ln.strip()
    return ""


# ── 1. Backward-compat: pre-feature CultureNode loads as style_hint="" ────────

def test_backward_compat():
    s = _robot_store()
    ensure_culture(s, "Korean")                 # default style_hint == ""
    full = tempfile.mktemp(suffix=".json"); s.save(full)
    data = json.load(open(full))
    stripped = 0
    for n in data["nodes"]:
        if n.get("node_type") == "culture":
            n.pop("style_hint", None); stripped += 1     # simulate an OLD file
    old = tempfile.mktemp(suffix=".json")
    json.dump(data, open(old, "w"), indent=2, default=str)
    assert stripped, "no culture node in fixture"

    s2 = InMemoryGraphStore(); s2.load(old, quiet=True)
    cnode = s2.get_node("culture:korean")
    assert cnode.style_hint == "", cnode.style_hint
    a = tempfile.mktemp(suffix=".json"); s2.save(a)
    s3 = InMemoryGraphStore(); s3.load(a, quiet=True)
    b = tempfile.mktemp(suffix=".json"); s3.save(b)
    same = open(a).read() == open(b).read()
    for f in (full, old, a, b):
        os.remove(f)
    assert same, "save→load→save not byte-identical for empty-default case"
    print("1. backward-compat: pre-feature culture loads style_hint=''; round-trip "
          "byte-identical ✓")


# ── 2. Seed sets it, idempotently ─────────────────────────────────────────────

def test_seed_idempotent():
    s = _robot_store()
    seed_korean_demo(s, robot_id=_ROBOT)
    c = s.get_node("culture:korean")
    assert c.style_hint == _KOREAN_STYLE_HINT and c.style_hint, "hint not seeded"
    n0, e0 = len(s._nodes), len(s._edges)
    seed_korean_demo(s, robot_id=_ROBOT)         # re-seed
    # Counts + the hint are unchanged (re-seed only refreshes prior provenance
    # timestamps, a pre-existing behaviour — the STYLE HINT itself is idempotent).
    assert (len(s._nodes), len(s._edges)) == (n0, e0), "re-seed changed counts"
    assert s.get_node("culture:korean").style_hint == _KOREAN_STYLE_HINT, "hint changed"
    print("2. seed: Korean style_hint set; re-seed idempotent (counts + hint unchanged) ✓")


# ── 3. Injected when tagged; SEPARATE from the topic-offer block ──────────────

def test_injected_when_tagged():
    s = _robot_store()
    seed_korean_demo(s, robot_id=_ROBOT)
    s.upsert_node(PersonNode(id="kid", display_name="kid"))
    assign_person_culture(s, "kid", "Korean")
    p = _loop(s)._build_system_prompt("kid")

    assert _MANNER in p, "manner line missing"
    # manner line sits in HOW TO REPLY (before the WHO block)
    assert p.index(_MANNER) < p.index("WHO YOU'RE TALKING TO"), "manner not in HOW TO REPLY"
    # the content topic-offer block is a SEPARATE block, later, under CULTURAL BACKGROUND
    assert "CULTURAL BACKGROUND" in p and "things you could bring up" in p
    assert p.index("WHO YOU'RE TALKING TO") < p.index("CULTURAL BACKGROUND")
    # manner (how to talk) and offers (what to talk about) are two different blocks
    assert p.index(_MANNER) < p.index("things you could bring up")
    assert _KOREAN_STYLE_HINT in p
    print("3. tagged: manner line in HOW TO REPLY, topic-offer block separate under "
          "CULTURAL BACKGROUND — both present, not merged ✓")


# ── 4. Not injected when untagged ─────────────────────────────────────────────

def test_not_injected_when_untagged():
    s = _robot_store()
    seed_korean_demo(s, robot_id=_ROBOT)
    s.upsert_node(PersonNode(id="stranger", display_name="stranger"))   # NO culture tag
    p = _loop(s)._build_system_prompt("stranger")
    assert _MANNER not in p, "manner line leaked for an untagged person"
    assert "Cultural manner" not in p
    print("4. untagged: no manner line, no empty header, no leakage ✓")


# ── 5. Empty hint injects nothing ─────────────────────────────────────────────

def test_empty_hint_injects_nothing():
    s = _robot_store()
    ensure_culture(s, "Nowhere")                 # culture with style_hint == ""
    s.upsert_node(PersonNode(id="p", display_name="p"))
    assign_person_culture(s, "p", "Nowhere")
    assert person_culture_style_hint(s, "p") == ""
    p = _loop(s)._build_system_prompt("p")
    assert _MANNER not in p, "manner line injected for an empty hint"
    print("5. empty hint: tagged but hint '' → no manner line ✓")


# ── 6. Static: identical for a 'visitor' vs a 'close' tagged person ───────────

def test_static_regardless_of_tier():
    from modules.graph_relationship.interactions import adjust_closeness
    s = _robot_store()
    seed_korean_demo(s, robot_id=_ROBOT)
    for pid in ("visitor", "close"):
        s.upsert_node(PersonNode(id=pid, display_name=pid))
        assign_person_culture(s, pid, "Korean")
    # make 'close' a high-rapport/trust, many-interaction person; 'visitor' stays new
    adjust_closeness(s, "close", _ROBOT, d_rapport=0.9, d_trust=0.9, source="test")
    L = _loop(s)
    hint_visitor = _manner_line(L._build_system_prompt("visitor"))
    hint_close   = _manner_line(L._build_system_prompt("close"))
    assert hint_visitor and hint_visitor == hint_close, (hint_visitor, hint_close)
    print("6. static: manner line identical for visitor vs close (no tier/affect "
          "dynamics crept in) ✓")


if __name__ == "__main__":
    test_backward_compat()
    test_seed_idempotent()
    test_injected_when_tagged()
    test_not_injected_when_untagged()
    test_empty_hint_injects_nothing()
    test_static_regardless_of_tier()
    print("\nALL STYLE-HINT (STEP 4) TESTS PASSED")
