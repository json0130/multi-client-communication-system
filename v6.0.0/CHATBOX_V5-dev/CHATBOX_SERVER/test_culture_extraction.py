"""
Headless verification for self-declared culture detection + tagging.

Run:  python3 test_culture_extraction.py
No real LLM — a fake llm_fn returns canned outputs so we test the guard/parse
logic and the end-to-end "tag only on explicit self-declaration" behaviour.
"""

from __future__ import annotations

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import PersonNode, RobotNode, Embodiment
from modules.graph_relationship.cultures import person_culture
from modules.culture_extraction import detect_self_declared_culture
from modules.culture_seed import assign_person_culture


def _turns(*msgs):
    return [{"child": m, "reply": "you said something"} for m in msgs]


# ── 1. Parse/guard logic (fake LLM returns a fixed label) ─────────────────────

def test_parse_and_guards():
    def fake(label):
        return lambda _sys, _user: label
    assert detect_self_declared_culture(_turns("x"), fake("Korean")) == "Korean"
    assert detect_self_declared_culture(_turns("x"), fake("korean")) == "Korean"   # title-cased
    assert detect_self_declared_culture(_turns("x"), fake("NONE")) is None
    assert detect_self_declared_culture(_turns("x"), fake("none")) is None
    assert detect_self_declared_culture(_turns("x"), fake("")) is None
    assert detect_self_declared_culture(_turns("x"), fake("[HAPPY] Korean")) is None  # tag junk
    assert detect_self_declared_culture(_turns("x"), fake(
        "They said they like Korean food but did not state it")) is None            # sentence
    assert detect_self_declared_culture(_turns("x"), fake("Latino Hispanic")) == "Latino Hispanic"
    # empty transcript (no person turns) → None, LLM never consulted
    called = {"n": 0}
    def counting(_s, _u): called["n"] += 1; return "Korean"
    assert detect_self_declared_culture([{"child": "", "reply": "hi"}], counting) is None
    assert called["n"] == 0
    print("1. parse/guards: label title-cased; NONE/empty/sentence/tag rejected; "
          "empty transcript short-circuits ✓")


# ── 2. Only the person's OWN words are read (robot reply ignored) ─────────────

def test_ignores_robot_turns():
    seen = {}
    def capture(_sys, user): seen["user"] = user; return "NONE"
    turns = [{"child": "hello", "reply": "I know a lot about Korean culture!"}]
    detect_self_declared_culture(turns, capture)
    assert "Korean" not in seen["user"], seen["user"]     # robot text not fed
    assert "hello" in seen["user"]
    print("2. transcript: only the person's words are sent (robot reply excluded) ✓")


# ── 3. End-to-end: tag ONLY on self-declaration, idempotent, no double-tag ────

def _store():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id="chatbox", name="chatbox", embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id="kid", display_name="kid"))
    return s

def _maybe_tag(store, pid, turns, llm):
    declared = detect_self_declared_culture(turns, llm)
    if declared and person_culture(store, pid) is None:
        assign_person_culture(store, pid, declared)
    return declared

def test_end_to_end_tagging():
    # (a) liking Korean food → NO tag
    s = _store()
    _maybe_tag(s, "kid", _turns("i love kimchi"), lambda _s, _u: "NONE")
    assert person_culture(s, "kid") is None, "liking food must not tag culture"

    # (b) explicit self-declaration → tag
    _maybe_tag(s, "kid", _turns("i'm korean"), lambda _s, _u: "Korean")
    assert person_culture(s, "kid") == "culture:korean"
    n_edges = len([e for e in s._edges.values() if e.edge_type == "belongs_to_culture"])

    # (c) re-run declaration → idempotent (no second edge, no flip)
    _maybe_tag(s, "kid", _turns("i'm korean"), lambda _s, _u: "Korean")
    assert person_culture(s, "kid") == "culture:korean"
    assert len([e for e in s._edges.values()
                if e.edge_type == "belongs_to_culture"]) == n_edges
    print("3. end-to-end: kimchi-lover NOT tagged; explicit 'I'm Korean' tags once "
          "(idempotent) ✓")


if __name__ == "__main__":
    test_parse_and_guards()
    test_ignores_robot_turns()
    test_end_to_end_tagging()
    print("\nALL CULTURE-EXTRACTION TESTS PASSED")
