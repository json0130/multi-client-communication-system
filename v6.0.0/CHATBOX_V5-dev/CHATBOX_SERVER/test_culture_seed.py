"""
Headless verification for the culture layer (ChatBox-owned prior knowledge +
prompt injection). Command A.

Run:  python3 test_culture_seed.py
No camera, no LLM, no network. Uses a COPY of the real kg_state.json (never the
live file) for the reuse/prompt checks.

Model under test:
    chatbox --knows_culture--> Korean --culture_prior--> CultureTopic(ck:korean:…)
    person  --belongs_to_culture--> Korean            (tag only, no topic edges)
Culture topics are SEPARATE from shared person-interest topics, so seeding never
touches a person and tagging one person never couples another.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment
from modules.graph_relationship.topics import (
    normalize_label, person_interests, add_person_interest,
)
from modules.graph_relationship.cultures import (
    culture_priors, person_culture, culture_knowers, culture_topic_id,
)
from modules.culture_seed import seed_korean_demo, assign_person_culture, _KOREAN_DEMO

# Prefer the culture-FREE pre-culture backup as the "real KG" fixture, so these
# tests are independent of whether the live kg_state.json has been seeded.
_REAL_KG = ("kg_state.pre-culture.bak" if os.path.exists("kg_state.pre-culture.bak")
            else "kg_state.json")
_ROBOT = "chatbox"


def _counts(store) -> tuple:
    return len(store._nodes), len(store._edges)


def _robot_store() -> InMemoryGraphStore:
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    return s


# ── 1. Empty seed → ChatBox-owned nodes + exact counts + round-trip ───────────

def test_empty_seed_counts_and_roundtrip():
    s = _robot_store()
    info = seed_korean_demo(s, robot_id=_ROBOT)
    cultures  = [n for n in s._nodes.values() if n.node_type == "culture"]
    ctopics   = [n for n in s._nodes.values() if n.node_type == "culture_topic"]
    person_ts = [n for n in s._nodes.values() if n.node_type == "topic"]
    priors    = [e for e in s._edges.values() if e.edge_type == "culture_prior"]
    knows     = [e for e in s._edges.values() if e.edge_type == "knows_culture"]
    assert len(cultures) == 1, cultures
    assert len(ctopics) == 12, len(ctopics)
    assert len(person_ts) == 0, "seed must NOT create person-interest topics"
    assert len(priors) == 12, len(priors)
    assert len(knows) == 1 and info["robot"] == _ROBOT      # ChatBox owns it
    assert culture_knowers(s, info["culture"]) == [_ROBOT]
    # every culture topic carries shareable facts
    assert all(n.facts for n in ctopics), "culture topics must be seeded with facts"

    f1 = tempfile.mktemp(suffix=".json"); s.save(f1)
    s2 = InMemoryGraphStore(); s2.load(f1)
    f2 = tempfile.mktemp(suffix=".json"); s2.save(f2)
    a, b = json.load(open(f1)), json.load(open(f2))
    os.remove(f1); os.remove(f2)
    assert a == b, "save→load→save not identical"
    print("1. empty seed: chatbox--knows-->Korean, 12 culture_topics/priors, 0 "
          "person topics, round-trips identically ✓")


# ── 2. Idempotent re-run ──────────────────────────────────────────────────────

def test_idempotent():
    s = _robot_store()
    seed_korean_demo(s, robot_id=_ROBOT); c1 = _counts(s)
    seed_korean_demo(s, robot_id=_ROBOT); c2 = _counts(s)
    assert c1 == c2, (c1, c2)
    assert len([e for e in s._edges.values() if e.edge_type == "culture_prior"]) == 12
    assert len([e for e in s._edges.values() if e.edge_type == "knows_culture"]) == 1
    print(f"2. idempotent re-run: identical counts {c1} ✓")


# ── 3. Seed on a COPY of the real KG → person topics UNCHANGED (no reuse) ─────

def test_no_person_topic_coupling_on_real_kg():
    if not os.path.exists(_REAL_KG):
        print("3. SKIP — no real kg_state.json present")
        return
    tmp = tempfile.mktemp(suffix=".json"); shutil.copy(_REAL_KG, tmp)
    s = InMemoryGraphStore(); s.load(tmp)
    before_topics = len([n for n in s._nodes.values() if n.node_type == "topic"])
    before_ct     = len([n for n in s._nodes.values() if n.node_type == "culture_topic"])
    seed_korean_demo(s, robot_id=_ROBOT)
    after_topics = len([n for n in s._nodes.values() if n.node_type == "topic"])
    after_ct     = len([n for n in s._nodes.values() if n.node_type == "culture_topic"])
    os.remove(tmp)
    # Person-interest topics must be untouched; culture topics are their own nodes.
    assert after_topics == before_topics, (before_topics, after_topics)
    assert after_ct - before_ct == 12, (before_ct, after_ct)
    print(f"3. real-KG copy: person topics unchanged ({after_topics}), +12 culture "
          "topics — no person-topic reuse/coupling ✓")


# ── 4. Assign jay; ChatBox owns culture; prompt block correct ────────────────

def test_prompt_block_and_ownership():
    if not os.path.exists(_REAL_KG):
        print("4. SKIP — no real kg_state.json present")
        return
    from modules.face_webcam.webcam_loop import WebcamKGLoop

    tmp = tempfile.mktemp(suffix=".json"); shutil.copy(_REAL_KG, tmp)
    s = InMemoryGraphStore(); s.load(tmp)
    seed_korean_demo(s, robot_id=_ROBOT)
    assign_person_culture(s, "jay", "Korean")

    # jay likes kpop (a culture topic) → must be EXCLUDED from offers (by slug).
    add_person_interest(s, "jay", "music", ["kpop"])
    assert person_culture(s, "jay") == "culture:korean"
    assert culture_knowers(s, "culture:korean") == [_ROBOT]   # ChatBox owns it

    loop = WebcamKGLoop.__new__(WebcamKGLoop)
    loop.store = s; loop.robot_id = _ROBOT; loop._robot_display = "ChatBox"
    prompt = loop._build_system_prompt("jay")

    assert "CULTURAL BACKGROUND" in prompt
    # jay is MANUALLY assigned (not self-declared) here → the robot's knowledge-lens
    # framing (active-culture model), never asserted as a fact about them.
    assert "Cultural knowledge lens: Korean" in prompt
    assert "not a fact about them" in prompt
    assert "may politely offer ONE" in prompt and "Never assert what they like" in prompt

    import re
    # Offers are bulleted "  – <label>: <fact>" lines under the culture block.
    block = prompt[prompt.index("CULTURAL BACKGROUND"):]
    offers = re.findall(r"^  – ([^:\n]+?):", block, re.MULTILINE)
    offers = [o.strip() for o in offers]
    assert 0 < len(offers) <= 4, offers
    assert "kpop" not in offers, f"observed topic leaked into offers: {offers}"
    assert offers[0] == "kimchi", offers
    # each offered culture topic carries a fact
    assert "fermented" in block, "kimchi fact missing from prompt"
    # The learned-memory block (now rendered as signed, hedged topic lines) still
    # leads before the weak cultural-background hint.
    assert prompt.index("How they feel about topics:") < prompt.index("CULTURAL BACKGROUND")
    os.remove(tmp)
    print(f"4. prompt+ownership: ChatBox owns Korean; block present, offers={offers} "
          "(≤4, kpop excluded, memory leads) ✓")


# ── 5. HJ decoupling: sharing a topic label ≠ connection to the culture ───────

def test_hj_decoupled():
    s = _robot_store()
    # HJ likes hiking (a real person-interest topic).
    s.upsert_node(PersonNode(id="HJ", display_name="HJ"))
    add_person_interest(s, "HJ", "outdoors", ["hiking"])
    # Seed Korean (which also has a 'hiking' culture topic) + tag ONLY jay.
    seed_korean_demo(s, robot_id=_ROBOT)
    s.upsert_node(PersonNode(id="jay", display_name="jay"))
    assign_person_culture(s, "jay", "Korean")

    # HJ has NO culture edge of any kind.
    hj_edges = [e.edge_type for e, _n in s.query_neighbors("HJ")]
    assert "belongs_to_culture" not in hj_edges
    assert all(not et.startswith("culture") and et != "knows_culture"
               for et in hj_edges), hj_edges
    # The culture's hiking and HJ's hiking are DIFFERENT nodes.
    ck_hiking = culture_topic_id("culture:korean", "hiking")
    assert ck_hiking == "ck:korean:hiking"
    assert s.get_node(ck_hiking).node_type == "culture_topic"
    assert s.get_node("topic:hiking").node_type == "topic"
    # No edge touches both HJ and the culture (no coupling path of length 1).
    assert person_culture(s, "HJ") is None
    print("5. HJ decoupled: shares 'hiking' label but has zero culture edges; "
          "ck:korean:hiking ≠ topic:hiking ✓")


# ── 5b. Self-declared culture is recallable; manual stays a tentative hint ────

def test_self_declared_vs_manual_framing():
    from modules.face_webcam.webcam_loop import WebcamKGLoop
    from modules.graph_relationship.cultures import person_culture_self_declared

    s = _robot_store()
    seed_korean_demo(s, robot_id=_ROBOT)
    s.upsert_node(PersonNode(id="dec", display_name="dec"))
    s.upsert_node(PersonNode(id="man", display_name="man"))
    # dec said it themselves; man was seed/manually assigned.
    assign_person_culture(s, "dec", "Korean", source="self-declared:sess-1")
    assign_person_culture(s, "man", "Korean")               # default 'culture-seed'
    assert person_culture_self_declared(s, "dec") is True
    assert person_culture_self_declared(s, "man") is False

    loop = WebcamKGLoop.__new__(WebcamKGLoop)
    loop.store = s; loop.robot_id = _ROBOT; loop._robot_display = "ChatBox"

    dec_block = loop._culture_block("dec")
    man_block = loop._culture_block("man")
    # self-declared → recallable fact wording, NOT the tentative hedge.
    assert "they told you themselves" in dec_block.lower(), dec_block
    assert "recall it as a fact" in dec_block.lower(), dec_block
    assert "knowledge lens" not in dec_block.lower(), dec_block
    # manual → tentative hint wording, NOT the recall permission.
    assert "knowledge lens" in man_block.lower() and "not a fact about them" in man_block, man_block
    assert "recall it as a fact" not in man_block.lower(), man_block
    # both still refuse to ASSUME preferences from the background.
    assert "ask" in dec_block.lower() and "ask" in man_block.lower()
    print("5b. framing: self-declared culture recallable as fact; manual stays a "
          "tentative hint; neither assumes preferences ✓")


# ── 6. Purity — graph_relationship pure modules stay clean ────────────────────

def test_purity():
    import pathlib, re
    pkg = pathlib.Path("modules/graph_relationship")
    _EXCLUDE_NAMES = {"embedding.py", "demo_harness.py"}
    forbidden = re.compile(
        r"^\s*(from|import)\s+"
        r"(modules(?!\.graph_relationship)|.*ollama|.*torch|.*pad_module|"
        r"modules\.pad_persona|.*llm_processor|.*emotion_processor)",
        re.MULTILINE)
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name in _EXCLUDE_NAMES or "tests" in py.parts or "viz" in py.parts:
            continue
        for mobj in forbidden.finditer(py.read_text()):
            offenders.append(f"{py}: {mobj.group(0).strip()}")
    assert not offenders, "purity violations:\n" + "\n".join(offenders)
    ctext = (pkg / "cultures.py").read_text()
    import_lines = [l.strip() for l in ctext.splitlines()
                    if re.match(r"(from \S+ import|import \S+)", l.strip())]
    assert all(("from .schema" in l or "from .store" in l or "from .topics" in l
                or "from __future__" in l or l.startswith("from datetime")
                or l.startswith("from typing")) for l in import_lines), import_lines
    print("5. purity: pure graph_relationship/ modules clean; cultures.py imports "
          "only schema/store/topics ✓")


if __name__ == "__main__":
    test_empty_seed_counts_and_roundtrip()
    test_idempotent()
    test_no_person_topic_coupling_on_real_kg()
    test_prompt_block_and_ownership()
    test_hj_decoupled()
    test_self_declared_vs_manual_framing()
    test_purity()
    print("\nALL CULTURE-SEED TESTS PASSED")
