"""
Headless verification for the FIRST-IMPRESSION integration on the culture branch:
auto-enrol an unknown face as a provisional guest, then re-key that guest to their
real name across every store when they introduce themselves — while the FULL culture
/interest pipeline (and viz highlight) works on them as a normal person.

Run:  python3 test_first_impression.py   (no camera / no LLM / no network)
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading

import numpy as np

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import RobotNode, PersonNode, Embodiment
from modules.graph_relationship.interactions import set_interaction_count
from modules.graph_relationship.topics import (
    add_person_topic, update_conversation, get_conversation, interest_id,
)
from modules.graph_relationship.rename import rename_person
from modules.culture_seed import seed_all_cultures, assign_person_culture
from modules.graph_relationship.cultures import person_culture
from modules.session_store import SessionStore
from modules.face_webcam.face_id import FaceIdentifier
from modules.face_webcam.webcam_loop import (
    WebcamKGLoop, _extract_name, _slug_name,
)

_ROBOT = "chatbox"


# ── 1. Name extraction: intros yes, filler/feelings no ────────────────────────

def test_name_extraction():
    yes = {"my name is Jay": "jay", "i'm Alice": "alice", "this is Casey": "casey",
           "call me Bob-Lee": "boblee"}
    for text, slug in yes.items():
        assert _slug_name(_extract_name(text)) == slug, text
    for text in ("i'm fine thanks", "i am korean", "hello there", "i'm not sure"):
        assert _extract_name(text) is None, text
    print("1. name extraction: intros parsed, feelings/filler/culture-words rejected ✓")


# ── 2. graph rename_person: re-key person + interaction + conversation ────────

def test_graph_rename():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id="guest_1", display_name="guest_1"))
    set_interaction_count(s, "guest_1", _ROBOT, 3, source="t")   # interaction:guest_1:chatbox
    update_conversation(s, "guest_1", _ROBOT, topic="jazz", create=True)  # conversation node
    add_person_topic(s, "guest_1", "jazz", "music", affinity=0.9)
    seed_all_cultures(s, robot_id=_ROBOT)
    assign_person_culture(s, "guest_1", "Korean", source="self-declared:s1")

    n0, e0 = len(s._nodes), len(s._edges)
    assert rename_person(s, "guest_1", "jay", _ROBOT, display_name="Jay")

    # old ids gone, new ids present
    assert s.get_node("guest_1") is None
    assert s.get_node("jay") is not None and s.get_node("jay").display_name == "Jay"
    assert s.get_node("interaction:jay:chatbox") is not None
    assert s.get_node("interaction:guest_1:chatbox") is None
    assert get_conversation(s, "jay", _ROBOT) is not None
    # relationship + culture edge followed the person
    assert person_culture(s, "jay") == "culture:korean"
    # no dangling guest_1 anywhere in edges
    dangling = [e for e in s._edges.values()
                if "guest_1" in (e.source_id, e.target_id)]
    assert not dangling, dangling
    # node/edge counts unchanged (pure re-key, nothing added/lost)
    assert (len(s._nodes), len(s._edges)) == (n0, e0), (n0, e0, len(s._nodes), len(s._edges))
    print("2. graph rename: person/interaction/conversation re-keyed, culture + "
          "relationship edges followed, no dangling guest id, counts stable ✓")


# ── 2b. rename onto an EXISTING person merges, never clobbers ─────────────────

def test_graph_rename_merges_into_existing_person():
    """A returning person whose face is not recognised gets auto-enrolled as a new
    guest; folding that guest back must not wipe the relationship they already had.
    (Blind upsert previously demoted them from 'known' to 'visitor'.)"""
    from modules.graph_relationship.interactions import set_closeness
    from modules.graph_relationship.kg_bridge import derive_tier

    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id="jay", display_name="Jay"))
    s.upsert_node(PersonNode(id="guest_1", display_name="guest_1"))

    set_closeness(s, "jay", _ROBOT, rapport=1.0, trust=0.10)   # established bond
    set_interaction_count(s, "jay", _ROBOT, 3, source="t")
    set_closeness(s, "guest_1", _ROBOT, rapport=0.10, trust=0.0)   # fresh guest
    set_interaction_count(s, "guest_1", _ROBOT, 1, source="t")
    assert derive_tier("jay", _ROBOT, s) == "known"

    assert rename_person(s, "guest_1", "jay", _ROBOT, display_name="Jay")

    n = s.get_node("interaction:jay:chatbox")
    assert n.rapport == 1.0 and n.trust == 0.10, "closeness must not be clobbered"
    assert n.interaction_count == 4, "interaction counts should add up"
    assert derive_tier("jay", _ROBOT, s) == "known", "must not be demoted to visitor"
    assert s.get_node("guest_1") is None
    print("2b. rename onto an existing person MERGES closeness (max) and counts "
          "(sum) — no demotion when a known face is re-enrolled as a guest ✓")


# ── 3. SessionStore.rename_person ─────────────────────────────────────────────

def test_session_rename():
    db = tempfile.mktemp(suffix=".db"); ss = SessionStore(db)
    ss.append_turn(session_id="s", person_id="guest_1", robot_id=_ROBOT,
                   child="hi", reply="hello")
    ss.append_turn(session_id="s", person_id="guest_1", robot_id=_ROBOT,
                   child="i'm jay", reply="nice")
    moved = ss.rename_person("guest_1", "jay")
    assert moved == 2, moved
    assert ss.person_turn_count("guest_1", _ROBOT) == 0
    assert ss.person_turn_count("jay", _ROBOT) == 2
    os.remove(db)
    print("3. session rename: transcript rows re-keyed guest_1 → jay ✓")


# ── 4. FaceIdentifier.rename (move + gallery concat) ─────────────────────────

def test_face_rename():
    f = FaceIdentifier.__new__(FaceIdentifier)   # skip model init
    f.k_anchor, f.k_adapt = 12, 6
    f._protos = {"guest_1": np.array([[1.0, 0.0, 0.0]], dtype=np.float32)}
    f._meta   = {"guest_1": np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)}
    f._counts = {"guest_1": 2}
    # move (new name absent)
    assert f.rename("guest_1", "jay")
    assert "guest_1" not in f._protos and "jay" in f._protos
    assert f._counts["jay"] == 2
    # merge (new name present) — galleries CONCATENATE. Averaging the two would
    # destroy exactly the pose diversity the multi-view gallery exists to build.
    f._protos["guest_2"] = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    f._meta["guest_2"]   = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    f._counts["guest_2"] = 1
    assert f.rename("guest_2", "jay")
    assert "guest_2" not in f._protos
    assert f._protos["jay"].shape == (2, 3)                 # both views survive
    assert np.allclose(np.linalg.norm(f._protos["jay"], axis=1), 1.0)  # unit rows
    assert f._counts["jay"] == 3
    assert f.rename("nobody", "x") is False
    print("4. face rename: move + gallery concat (per-row unit-norm), missing key → False ✓")


# ── 5. _learn_name flow: re-key across all stores + in-memory maps ────────────

class _MockFace:
    def __init__(self): self.renamed = None
    def rename(self, a, b): self.renamed = (a, b); return True
    def save(self, path): pass

def test_learn_name_flow():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    s.upsert_node(PersonNode(id="guest_1", display_name="guest_1"))
    set_interaction_count(s, "guest_1", _ROBOT, 1, source="t")
    db = tempfile.mktemp(suffix=".db"); ss = SessionStore(db)
    ss.append_turn(session_id="s", person_id="guest_1", robot_id=_ROBOT, child="hi", reply="yo")

    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store = s; L.robot_id = _ROBOT; L.faces_path = "x.npz"; L.kg_path = None
    L._store_lock = threading.RLock(); L._session_store = ss
    L._run_sessions = {"guest_1": "sess"}; L._last_mood = {"guest_1": (0.1, "happy")}
    L._chat_history = {"guest_1": ["prev"]}
    L._kg_dirty = False; L._last_save_t = 0.0; L._save_min_interval = 1.0
    L.face_id = _MockFace()

    ok = L._learn_name("guest_1", "jay", "Jay")
    assert ok and L.face_id.renamed == ("guest_1", "jay")
    assert s.get_node("jay") is not None and s.get_node("guest_1") is None   # graph
    assert ss.person_turn_count("jay", _ROBOT) == 1                          # transcripts
    for d in (L._run_sessions, L._last_mood, L._chat_history):               # in-memory
        assert "jay" in d and "guest_1" not in d
    os.remove(db)
    print("5. _learn_name: re-keys face DB + graph + transcripts + in-memory maps ✓")


# ── 6. A guest is a normal person → full culture pipeline + prompt works ──────

def test_guest_gets_culture_pipeline():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    seed_all_cultures(s, robot_id=_ROBOT)
    # a freshly auto-enrolled guest, tagged via (mid-session) self-declaration
    s.upsert_node(PersonNode(id="guest_2", display_name="guest_2"))
    add_person_topic(s, "guest_2", "jazz", "music", affinity=0.9, confidence=0.9)
    assign_person_culture(s, "guest_2", "Korean", source="self-declared:s1")

    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store = s; L.robot_id = _ROBOT; L._robot_display = "ChatBox"; L.kg_path = None
    p = L._build_system_prompt("guest_2")
    assert "Cultural knowledge lens: Korean" in p or "you're Korean" in p
    assert "jazz" in p                                    # their interest is in memory
    print("6. guest pipeline: an auto-enrolled guest gets the full culture/interest "
          "prompt (and viz highlight) like any person ✓")


if __name__ == "__main__":
    test_name_extraction()
    test_graph_rename()
    test_graph_rename_merges_into_existing_person()
    test_session_rename()
    test_face_rename()
    test_learn_name_flow()
    test_guest_gets_culture_pipeline()
    print("\nALL FIRST-IMPRESSION INTEGRATION TESTS PASSED")
