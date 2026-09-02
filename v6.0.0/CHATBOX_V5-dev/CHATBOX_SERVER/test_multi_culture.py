"""
Headless verification for the MULTI-CULTURE selector feature:
  * Korean + Māori both seeded (robot knows both);
  * active culture resolved per turn = override → the person's tag → generic;
  * mid-session auto-attach on self-declaration (only while untagged);
  * viz culture button ↔ webcam loop via culture_override.json.

Run:  python3 test_multi_culture.py   (no camera / no real LLM / no network)
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time

from modules.graph_relationship.store import InMemoryGraphStore
from modules.graph_relationship.schema import RobotNode, PersonNode, Embodiment
from modules.culture_seed import (
    seed_all_cultures, assign_person_culture, _KOREAN_STYLE_HINT, _MAORI_STYLE_HINT,
)
from modules.graph_relationship.cultures import (
    person_culture, person_culture_self_declared,
)
from modules.face_webcam.webcam_loop import WebcamKGLoop
from modules.graph_relationship.viz.server import GraphState

_ROBOT = "chatbox"


class _FakeLLM:
    available = True
    def __init__(self, reply="Korean"):
        self._reply = reply
    def respond(self, system, user, history=None, max_tokens=140, json_mode=False):
        return self._reply


def _loop(store, kg_path=None, llm=None) -> WebcamKGLoop:
    L = WebcamKGLoop.__new__(WebcamKGLoop)
    L.store = store; L.robot_id = _ROBOT; L._robot_display = "ChatBox"; L.kg_path = kg_path
    L.llm = llm
    L._store_lock = threading.RLock()
    L._pending_culture = queue.Queue()
    L._kg_dirty = False; L._last_save_t = 0.0; L._save_min_interval = 1.0
    return L


def _base_store():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id=_ROBOT, name=_ROBOT, embodiment=Embodiment.CAT))
    seed_all_cultures(s, robot_id=_ROBOT)
    return s


# ── 1. Both cultures seeded; robot knows both; idempotent ─────────────────────

def test_seed_both():
    s = _base_store()
    knows = sorted(n.label for _e, n in s.query_neighbors(_ROBOT, "knows_culture"))
    assert knows == ["Korean", "Maori"], knows
    assert s.get_node("culture:korean").style_hint == _KOREAN_STYLE_HINT
    assert s.get_node("culture:maori").style_hint == _MAORI_STYLE_HINT
    n0, e0 = len(s._nodes), len(s._edges)
    seed_all_cultures(s, robot_id=_ROBOT)
    assert (len(s._nodes), len(s._edges)) == (n0, e0), "re-seed changed counts"
    print(f"1. seed both: robot knows {knows}; hints set; re-seed idempotent ✓")


# ── 2. Person-driven: each person's tag drives their own lens + manner ────────

def test_person_driven():
    s = _base_store()
    s.upsert_node(PersonNode(id="kj", display_name="kj")); assign_person_culture(s, "kj", "Korean")
    s.upsert_node(PersonNode(id="mj", display_name="mj")); assign_person_culture(s, "mj", "Maori")
    L = _loop(s)
    pk, pm = L._build_system_prompt("kj"), L._build_system_prompt("mj")
    assert "Cultural knowledge lens: Korean" in pk and _KOREAN_STYLE_HINT in pk
    assert "Cultural knowledge lens: Maori" in pm and _MAORI_STYLE_HINT in pm
    assert "Maori" not in pk and "Korean" not in pm    # no cross-leak
    print("2. person-driven: Korean person → Korean lens+manner; Māori → Māori; no leak ✓")


# ── 3. Override resolver: auto → generic → forced ─────────────────────────────

def test_override_resolver():
    d = tempfile.mkdtemp(); kg = os.path.join(d, "kg_state.json")
    s = _base_store()
    s.upsert_node(PersonNode(id="kj", display_name="kj")); assign_person_culture(s, "kj", "Korean")
    s.save(kg)
    L = _loop(s, kg_path=kg)
    gs = GraphState(kg)

    assert L._active_culture_id("kj") == "culture:korean"      # auto (no file)
    gs.set_culture_override("maori")
    assert L._active_culture_id("kj") == "culture:maori"       # override wins
    gs.set_culture_override("generic")
    assert L._active_culture_id("kj") is None                  # culture OFF (A/B)
    gs.set_culture_override("auto")
    assert L._active_culture_id("kj") == "culture:korean"      # back to person
    assert gs.culture_labels() == ["Korean", "Maori"]
    print("3. resolver: auto→person, 'maori'→forced, 'generic'→off; viz labels correct ✓")


# ── 4. Mid-session auto-attach (self-declaration only, while untagged) ────────

def test_mid_session_attach():
    s = _base_store()
    s.upsert_node(PersonNode(id="kid", display_name="kid"))
    L = _loop(s, llm=_FakeLLM("Korean"))

    # untagged + origin cue → detect + attach
    assert person_culture(s, "kid") is None
    L._spawn_culture_detect("i am from korea", "kid", "sess1"); time.sleep(0.3)
    L._drain_pending_culture()
    assert person_culture(s, "kid") == "culture:korean"
    assert person_culture_self_declared(s, "kid")

    # already tagged → no re-detection (queue stays empty)
    L._spawn_culture_detect("i am from korea", "kid", "sess2"); time.sleep(0.15)
    assert L._pending_culture.empty()

    # a different untagged person, no origin cue → skipped (no LLM)
    s.upsert_node(PersonNode(id="p2", display_name="p2"))
    L._spawn_culture_detect("i like pizza", "p2", "sess1"); time.sleep(0.15)
    L._drain_pending_culture()
    assert person_culture(s, "p2") is None
    print("4. mid-session attach: cue+untagged→attach; tagged→skip; no-cue→skip ✓")


# ── 5. Viz button ↔ loop via culture_override.json ────────────────────────────

def test_viz_roundtrip():
    d = tempfile.mkdtemp(); kg = os.path.join(d, "kg_state.json")
    s = _base_store()
    s.upsert_node(PersonNode(id="kj", display_name="kj")); assign_person_culture(s, "kj", "Korean")
    s.save(kg)
    gs = GraphState(kg)
    L = _loop(s, kg_path=kg)
    gs.set_culture_override("maori")
    p = L._build_system_prompt("kj")           # Korean person, but robot forced Māori
    assert "Cultural knowledge lens: Maori" in p and _MAORI_STYLE_HINT in p
    gs.set_culture_override("generic")
    p = L._build_system_prompt("kj")
    assert "CULTURAL BACKGROUND" not in p and "Cultural manner" not in p   # off
    print("5. viz round-trip: button sets override → loop's prompt switches lens / turns off ✓")


# ── 6. Active-state sidecar: loop writes current focus → viz reads it ─────────

def test_active_state_for_viz():
    d = tempfile.mkdtemp(); kg = os.path.join(d, "kg_state.json")
    s = _base_store()
    s.upsert_node(PersonNode(id="jay", display_name="jay")); assign_person_culture(s, "jay", "Korean")
    s.upsert_node(PersonNode(id="mia", display_name="mia")); assign_person_culture(s, "mia", "Maori")
    s.save(kg)
    L = _loop(s, kg_path=kg)
    L._last_active_written = None; L._last_active_write_t = 0.0
    L._active_person = None; L._active_person_t = 0.0
    gs = GraphState(kg)

    def _act():
        a = gs.read()["active"]
        return (a.get("person"), a.get("culture"), a.get("present"), a.get("live"))

    L._write_active_state("jay", present=True)
    assert _act() == ("jay", "culture:korean", True, True), _act()
    # STICKY through a true dropout (no face present) → keeps jay
    L._write_active_state(None, present=False)
    assert _act()[0] == "jay", _act()
    # an UNKNOWN face on camera (present, unrecognised) drops jay IMMEDIATELY
    L._write_active_state(None, present=True)
    assert _act() == (None, None, True, True), _act()      # → viz: ChatBox only
    # switching to a recognised person changes focus immediately
    L._write_active_state("mia", present=True)
    assert _act()[:2] == ("mia", "culture:maori"), _act()
    # a SUSTAINED no-face dropout (past the grace window) clears to nobody, present False
    L._active_person_t = 0.0; L._last_active_write_t = 0.0
    L._write_active_state(None, present=False)
    assert _act() == (None, None, False, True), _act()     # → viz: no dim (idle)
    print("6. active-state: sticky, unknown-face→ChatBox-only, no-face→idle, live "
          "heartbeat + present flag drive the viz dim/no-dim ✓")


if __name__ == "__main__":
    test_seed_both()
    test_person_driven()
    test_override_resolver()
    test_mid_session_attach()
    test_viz_roundtrip()
    test_active_state_for_viz()
    print("\nALL MULTI-CULTURE TESTS PASSED")
