"""
tools/check_phase1.py
=====================
Phase 1 checkpoint — verifies persona DB layer + HTTP endpoints.

Run from project root with the server RUNNING:
    python3 app.py &
    python3 tools/check_phase1.py

EXPECTED OUTPUT:
=======================================================
Phase 1 checkpoint — Persona DB + endpoints
=======================================================
  PASS  persona_repo imports
  PASS  persona_gateway imports
  PASS  GET /personas returns 200
  PASS  Default persona exists in DB
  PASS  Default persona has correct name
  PASS  Default persona has personality keys O,C,E,A,N
  PASS  POST /personas creates a test persona
  PASS  GET /personas/<id> retrieves it
  PASS  PUT /personas/<id> updates the name
  PASS  POST /robots/jay_mock_001/persona assigns persona
  PASS  Assignment returns live_updated field
  PASS  DELETE /personas/<id> removes test persona
  PASS  Cannot delete default persona (returns 400)

Result: 13/13 checks passed
Phase 1 complete. Move to Phase 2 (React dashboard).
=======================================================
"""

import sys
import os
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://127.0.0.1:5000"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Phase 1 checkpoint — Persona DB + endpoints")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Imports ────────────────────────────────────────────
    total += 1
    try:
        from data import persona_repo
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("persona_repo imports", ok):
        passed += 1

    total += 1
    try:
        from gateway.persona_gateway import create_persona_gateway
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("persona_gateway imports", ok):
        passed += 1

    # ── 2. GET /personas ──────────────────────────────────────
    total += 1
    try:
        resp = requests.get(f"{BASE}/personas", timeout=5)
        ok = resp.status_code == 200
    except Exception as e:
        ok = False
        print(f"         Error: {e} — is the server running?")
    if check("GET /personas returns 200", ok,
             "Start the server: python3 app.py"):
        passed += 1

    if not ok:
        _summary(passed, total)
        return

    personas = resp.json().get("personas", [])

    # ── 3. Default persona ────────────────────────────────────
    total += 1
    default = next((p for p in personas if p.get("is_default")), None)
    if check("Default persona exists in DB", default is not None,
             "Run the SQL in the Phase 1 instructions to seed it"):
        passed += 1

    total += 1
    if check("Default persona has correct name",
             default is not None and "Friendly" in default.get("name", ""),
             f"Got: {default.get('name') if default else 'None'}"):
        passed += 1

    total += 1
    personality = (default or {}).get("personality", {})
    has_ocean = all(k in personality for k in ["O", "C", "E", "A", "N"])
    if check("Default persona has personality keys O,C,E,A,N", has_ocean,
             f"Got keys: {list(personality.keys())}"):
        passed += 1

    # ── 4. Create test persona ────────────────────────────────
    total += 1
    resp = requests.post(f"{BASE}/personas", json={
        "name": "_Test Persona (delete me)",
        "description": "Checkpoint test persona",
        "robot_role": "You are a test robot.",
        "allowed_tags": ["[DEFAULT]"],
        "modules": ["gpt"],
        "voice_config": {"gender": "male", "language": "en", "rate": "+0%"},
        "capabilities": {"navigation": False},
        "personality": {"O": 0.5, "C": 0.5, "E": 0.5, "A": 0.5, "N": 0.5},
    }, timeout=5)
    test_id = None
    if resp.status_code == 200:
        test_id = resp.json().get("persona", {}).get("id")
    if check("POST /personas creates a test persona",
             resp.status_code == 200 and test_id is not None,
             f"Got {resp.status_code}: {resp.text[:80]}"):
        passed += 1

    # ── 5. Get by ID ──────────────────────────────────────────
    if test_id:
        total += 1
        resp = requests.get(f"{BASE}/personas/{test_id}", timeout=5)
        if check("GET /personas/<id> retrieves it",
                 resp.status_code == 200 and resp.json().get("id") == test_id):
            passed += 1

        # ── 6. Update ─────────────────────────────────────────
        total += 1
        resp = requests.put(f"{BASE}/personas/{test_id}",
                            json={"name": "_Test Persona UPDATED"},
                            timeout=5)
        updated_name = resp.json().get("persona", {}).get("name", "")
        if check("PUT /personas/<id> updates the name",
                 resp.status_code == 200 and "UPDATED" in updated_name):
            passed += 1

    # ── 7. Assign persona to robot ────────────────────────────
    if default:
        total += 1
        resp = requests.post(
            f"{BASE}/robots/jay_mock_001/persona",
            json={"persona_id": default["id"]},
            timeout=5,
        )
        ok = resp.status_code == 200
        body = resp.json() if ok else {}
        if check("POST /robots/jay_mock_001/persona assigns persona",
                 ok, f"Got {resp.status_code}: {resp.text[:80]}"):
            passed += 1

        total += 1
        if check("Assignment returns live_updated field",
                 "live_updated" in body):
            passed += 1

    # ── 8. Delete test persona ────────────────────────────────
    if test_id:
        total += 1
        resp = requests.delete(f"{BASE}/personas/{test_id}", timeout=5)
        if check("DELETE /personas/<id> removes test persona",
                 resp.status_code == 200):
            passed += 1

    # ── 9. Cannot delete default ──────────────────────────────
    if default:
        total += 1
        resp = requests.delete(f"{BASE}/personas/{default['id']}", timeout=5)
        if check("Cannot delete default persona (returns 400)",
                 resp.status_code == 400):
            passed += 1

    _summary(passed, total)


def _summary(passed, total):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Phase 1 complete. Move to Phase 2 (React dashboard).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()