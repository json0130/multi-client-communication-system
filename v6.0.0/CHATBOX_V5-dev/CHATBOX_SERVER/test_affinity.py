"""
Headless verification for Approach-1 STEP 1: continuous affinity (0–10) +
confidence hedging.

Run:  python3 test_affinity.py
No camera, no LLM, no network — a fake llm_fn returns fixed JSON. Covers the two
consumers of the signal: the BN clamp (affinity → signed clamp) and the memory
prompt (affinity → word, confidence → hedge), plus schema backward-compat, the
scale helpers, and graph_relationship purity.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import (
    PersonNode, RobotNode, Embodiment,
)
from modules.graph_relationship.scales import aff01_from_10, aff10_from_01
from modules.graph_relationship.topics import (
    add_person_topic, resolve_topic, link_related_topic, topic_id,
    person_topic_affinity,
)
from modules.kg_extraction import extract_and_apply_topics
from modules.affinity_phrasing import topic_memory_line
from modules.preference_model import rank_suggestions

_ROBOT = "chatbox"
_PERSON = "kid"
_REAL_KG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg_state.json")


def _person_store() -> InMemoryGraphStore:
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id=_PERSON, display_name=_PERSON))
    return s


# ── 1. Backward-compat: old kg_state.json loads as neutral; round-trip stable ──

def test_backward_compat():
    # Build a realistic graph, then STRIP the new affinity/confidence keys from its
    # about edges to reconstruct a genuine PRE-FEATURE file. (The live kg_state.json
    # can't be used as the "old" fixture any more — the new extraction code now
    # writes real affinity/confidence into it.)
    src = _person_store()
    add_person_topic(src, _PERSON, "jazz", "music", affinity=0.9, confidence=0.8)
    add_person_topic(src, _PERSON, "baseball", "sport", affinity=0.1, confidence=0.7)
    full = tempfile.mktemp(suffix=".json"); src.save(full)
    data = json.load(open(full))
    n_about = 0
    for e in data["edges"]:
        if e.get("edge_type") == "about":
            e.pop("affinity", None); e.pop("confidence", None)   # simulate old file
            n_about += 1
    old = tempfile.mktemp(suffix=".json")
    with open(old, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    assert n_about, "fixture should contain about edges"

    # Old file loads: every about edge reports the schema defaults (neutral/trusted).
    s = InMemoryGraphStore(); s.load(old, quiet=True)
    about = [e for e in s._edges.values() if e.edge_type == "about"]
    for e in about:
        assert e.affinity == 0.5, (e.source_id, e.target_id, e.affinity)
        assert e.confidence == 1.0, (e.source_id, e.target_id, e.confidence)

    # save → load → save must be byte-identical (our serialisation is stable).
    a = tempfile.mktemp(suffix=".json"); s.save(a)
    s2 = InMemoryGraphStore(); s2.load(a, quiet=True)
    b = tempfile.mktemp(suffix=".json"); s2.save(b)
    same = open(a).read() == open(b).read()

    # Sanity: the REAL kg_state.json (whatever state it's in) still loads cleanly.
    real_ok = True
    if os.path.exists(_REAL_KG):
        try:
            InMemoryGraphStore().load(_REAL_KG, quiet=True)
        except Exception:
            real_ok = False

    for f in (full, old, a, b):
        os.remove(f)
    assert same, "save→load→save not byte-identical"
    assert real_ok, "live kg_state.json failed to load"
    print(f"1. backward-compat: {len(about)} old-format about edges load as "
          "affinity=0.5/confidence=1.0; save→load→save byte-identical; live KG loads ✓")


# ── 2. Extraction sentiment → affinity mapping (fake LLM) ─────────────────────

def test_extraction_mapping():
    s = _person_store()
    fixed = {
        "existing_topics_discussed": [],
        "new_topics": [
            {"label": "jazz",     "category": "music", "sentiment": 10, "confidence": 0.9},
            {"label": "baseball", "category": "sport", "sentiment": 1,  "confidence": 0.9},
            {"label": "pasta",    "category": "food",  "confidence": 0.9},   # no sentiment
            {"label": "curling",  "category": "sport", "sentiment": 99, "confidence": 0.9},  # out of range
        ],
        "relations": [],
    }
    llm = lambda _sys, _usr: json.dumps(fixed)
    turns = [{"child": "I love jazz but I can't stand baseball"}]
    res = extract_and_apply_topics(s, _PERSON, _ROBOT, turns, llm, session_id="t2")
    assert res["applied"], res

    aff = {t.label: a for t, a, _c in person_topic_affinity(s, _PERSON)}
    assert abs(aff["jazz"] - 1.0) < 1e-9, aff
    assert aff["baseball"] <= 0.1 + 1e-9, aff          # "can't stand" → very low
    assert abs(aff["pasta"] - 0.5) < 1e-9, aff         # missing sentiment → neutral, KEPT
    assert abs(aff["curling"] - 0.5) < 1e-9, aff       # out-of-range → neutral, KEPT
    assert {"pasta", "curling"} <= set(aff), "neutral/oob items were dropped"

    # Idempotent: identical input applied again does not change stored affinity.
    extract_and_apply_topics(s, _PERSON, _ROBOT, turns, llm, session_id="t2")
    aff2 = {t.label: a for t, a, _c in person_topic_affinity(s, _PERSON)}
    assert aff2 == aff, (aff, aff2)
    print("2. extraction mapping: jazz→{:.2f}, baseball→{:.2f}, pasta(no sent.)→{:.2f}, "
          "curling(oob)→{:.2f}; both kept; idempotent ✓".format(
              aff["jazz"], aff["baseball"], aff["pasta"], aff["curling"]))


# ── 3. Scale helpers round-trip exactly at 0 / 5 / 10 ─────────────────────────

def test_helpers_roundtrip():
    for human, internal in [(0, 0.0), (5, 0.5), (10, 1.0)]:
        assert aff01_from_10(human) == internal, human
        assert aff10_from_01(internal) == human, internal
    # clamping
    assert aff01_from_10(-3) == 0.0 and aff01_from_10(42) == 1.0
    assert aff10_from_01(-1) == 0.0 and aff10_from_01(9) == 10.0
    print("3. helpers: 0/5/10 ↔ 0.0/0.5/1.0 round-trip exact; clamps out-of-range ✓")


# ── 4. BN clamp: dislike pulls a neighbour DOWN, like pushes UP ───────────────

def test_bn_signed_clamp():
    s = _person_store()
    # two observed topics with opposite affinity, each related to an UNOBSERVED
    # neighbour that starts at the default prior (0.30).
    add_person_topic(s, _PERSON, "baseball", "sport", affinity=0.10, confidence=0.9)
    add_person_topic(s, _PERSON, "jazz",     "music", affinity=0.95, confidence=0.9)
    resolve_topic(s, "softball", category="sport")   # unobserved neighbour
    resolve_topic(s, "blues",    category="music")   # unobserved neighbour
    link_related_topic(s, topic_id("baseball"), topic_id("softball"), 0.65)
    link_related_topic(s, topic_id("jazz"),     topic_id("blues"),    0.65)

    # read-only baseline
    n0, e0 = len(s._nodes), len(s._edges)
    f0 = tempfile.mktemp(suffix=".json"); s.save(f0); before = open(f0).read()

    got = dict((s.get_node(i).label, p) for i, p in
               rank_suggestions(s, _PERSON, k=9, floor=0.0))
    a2 = dict((s.get_node(i).label, p) for i, p in rank_suggestions(s, _PERSON, k=9, floor=0.0))

    f1 = tempfile.mktemp(suffix=".json"); s.save(f1); after = open(f1).read()
    os.remove(f0); os.remove(f1)

    # disliked baseball drags softball BELOW its prior; liked jazz lifts blues ABOVE
    assert got["softball"] < 0.30 - 1e-9, got
    assert got["blues"] > 0.30 + 1e-9, got
    # observed topics excluded from suggestions
    assert "baseball" not in got and "jazz" not in got, got
    # read-only + deterministic
    assert (len(s._nodes), len(s._edges)) == (n0, e0), "graph mutated"
    assert before == after, "store bytes changed — not read-only"
    assert got == a2, "non-deterministic"
    print("4. BN clamp: softball {:.3f} < 0.30 (dislike pull), blues {:.3f} > 0.30 "
          "(like push); baseball/jazz excluded; read-only + deterministic ✓".format(
              got["softball"], got["blues"]))


# ── 5. Prompt wording: verb (affinity) × hedge (confidence) ───────────────────

def test_prompt_wording():
    cases = [
        ("jazz",     0.95, 0.90, "They clearly like jazz."),
        ("baseball", 0.05, 0.70, "They probably dislike baseball — avoid raising it."),
        ("pasta",    0.50, 0.90, "They may be neutral on pasta."),
        ("rap",      0.90, 0.50, "They possibly like rap."),           # low-conf like
        ("opera",    0.05, 0.95, "They clearly dislike opera — avoid raising it."),
    ]
    for label, aff, conf, expected in cases:
        got = topic_memory_line(label, aff, conf)
        assert got == expected, (label, got, expected)
    print("5. prompt wording: verb×hedge lines correct — "
          f'e.g. {topic_memory_line("jazz", 0.95, 0.9)!r} / '
          f'{topic_memory_line("baseball", 0.05, 0.7)!r} ✓')


# ── 6. Purity: graph_relationship/ imports no LLM/PAD/app/embedding modules ───

def test_purity():
    import pathlib
    root = pathlib.Path(__file__).parent / "modules" / "graph_relationship"
    banned = ("openai", "ollama", "requests", "torch", "transformers",
              "sentence_transformers", "cv2", "modules.pad_persona",
              "modules.llm_processor", "modules.kg_extraction",
              "modules.face_webcam", "modules.emotion_processor")
    # The pure library surface only. `demo_harness.py` is a dev/demo runner that
    # lazily imports openai/pad_persona, and tests_*.py are test files — neither is
    # part of the importable pure layer, so they are excluded from the guard.
    def _is_library(p) -> bool:
        return not (p.name == "demo_harness.py"
                    or p.name.startswith("tests_")
                    or p.name.startswith("test_"))
    offenders = []
    for py in root.rglob("*.py"):
        if not _is_library(py):
            continue
        tree = ast.parse(py.read_text(), filename=str(py))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        for n in names:
            if any(n == b or n.startswith(b + ".") for b in banned):
                offenders.append((py.name, n))
    assert not offenders, f"graph_relationship/ has forbidden imports: {offenders}"
    print("6. purity: graph_relationship/ imports no LLM/PAD/app/embedding modules ✓")


if __name__ == "__main__":
    test_backward_compat()
    test_extraction_mapping()
    test_helpers_roundtrip()
    test_bn_signed_clamp()
    test_prompt_wording()
    test_purity()
    print("\nALL AFFINITY (STEP 1) TESTS PASSED")
