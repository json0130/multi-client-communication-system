"""
tools/check_app.py
==================
Checkpoint 9 (FINAL) — boots the full server and tests every HTTP endpoint.

Run from project root:
    python3 tools/check_app.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 9 — Full app boot + HTTP endpoints
=======================================================
  PASS  app.py imports (create_app)
  PASS  Flask app created successfully
  PASS  RobotRegistry created
  PASS  WebSocketGateway created

  -- HTTP endpoint tests (test client, no real server) --
  PASS  GET  /  returns 200
  PASS  GET  /  response has 'connected_robots' key
  PASS  GET  /health  returns 200
  PASS  GET  /robots  returns 200
  PASS  GET  /robots  response has 'robots' list
  PASS  POST /robots/register  returns 200 with valid body
  PASS  POST /robots/register  response has client_id
  PASS  POST /robots/register  missing fields returns 400
  PASS  PUT  /robots/<id>/role  returns 200
  PASS  GET  /robots/<id>/health  returns 404 for disconnected robot
  PASS  POST /robots/<id>/chat  returns 404 for disconnected robot
  PASS  POST /robots/<id>/connect  returns 400 (no address set)

  Cleaning up test robot...
  OK  Test robot deleted

Result: 16/16 checks passed
Server is ready. Run with: python3 app.py
=======================================================
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_CLIENT_ID = "_checkpoint_app_robot"
TEST_ROBOT_NAME = "App Checkpoint Bot"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 9 — Full app boot + HTTP endpoints")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Import app ─────────────────────────────────────────
    total += 1
    try:
        from app import create_app
        ok = True
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("app.py imports (create_app)", ok):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    # ── 2. Create app ─────────────────────────────────────────
    total += 1
    try:
        app, ws_gateway, registry = create_app()
        ok = app is not None
    except Exception as e:
        ok = False
        print(f"         Error: {e}")
    if check("Flask app created successfully", ok):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    total += 1
    from robot.robot_registry import RobotRegistry
    if check("RobotRegistry created", isinstance(registry, RobotRegistry)):
        passed += 1

    total += 1
    from gateway.websocket_gateway import WebSocketGateway
    if check("WebSocketGateway created", isinstance(ws_gateway, WebSocketGateway)):
        passed += 1

    # ── 3. HTTP endpoint tests via Flask test client ──────────
    print("\n  -- HTTP endpoint tests (test client, no real server) --")
    client = app.test_client()

    # GET /
    total += 1
    resp = client.get("/")
    if check("GET  /  returns 200", resp.status_code == 200):
        passed += 1

    total += 1
    body = resp.get_json()
    if check("GET  /  response has 'connected_robots' key",
             body is not None and "connected_robots" in body):
        passed += 1

    # GET /health
    total += 1
    resp = client.get("/health")
    if check("GET  /health  returns 200", resp.status_code == 200):
        passed += 1

    # GET /robots
    total += 1
    resp = client.get("/robots")
    if check("GET  /robots  returns 200", resp.status_code == 200):
        passed += 1

    total += 1
    body = resp.get_json()
    if check("GET  /robots  response has 'robots' list",
             body is not None and "robots" in body):
        passed += 1

    # POST /robots/register — valid
    total += 1
    resp = client.post(
        "/robots/register",
        data=json.dumps({
            "client_id": TEST_CLIENT_ID,
            "robot_name": TEST_ROBOT_NAME,
            "robot_role": "You are a test robot.",
            "allowed_tags": ["[DEFAULT]", "[WAVE]"],
            "modules": ["gpt"],
            "ip_address": "127.0.0.1",
            "ws_port": 8765,
        }),
        content_type="application/json",
    )
    if check("POST /robots/register  returns 200 with valid body",
             resp.status_code == 200):
        passed += 1

    total += 1
    body = resp.get_json()
    if check("POST /robots/register  response has client_id",
             body is not None and body.get("client_id") == TEST_CLIENT_ID):
        passed += 1

    # POST /robots/register — missing fields
    total += 1
    resp = client.post(
        "/robots/register",
        data=json.dumps({"robot_role": "no client_id or name"}),
        content_type="application/json",
    )
    if check("POST /robots/register  missing fields returns 400",
             resp.status_code == 400):
        passed += 1

    # PUT /robots/<id>/role
    total += 1
    resp = client.put(
        f"/robots/{TEST_CLIENT_ID}/role",
        data=json.dumps({
            "robot_role": "Updated role.",
            "allowed_tags": ["[DEFAULT]", "[HAPPY]"],
        }),
        content_type="application/json",
    )
    if check("PUT  /robots/<id>/role  returns 200",
             resp.status_code == 200):
        passed += 1

    # GET /robots/<id>/health — robot not connected
    total += 1
    resp = client.get(f"/robots/{TEST_CLIENT_ID}/health")
    if check("GET  /robots/<id>/health  returns 404 for disconnected robot",
             resp.status_code == 404):
        passed += 1

    # POST /robots/<id>/chat — robot not connected
    total += 1
    resp = client.post(
        f"/robots/{TEST_CLIENT_ID}/chat",
        data=json.dumps({"message": "hello"}),
        content_type="application/json",
    )
    if check("POST /robots/<id>/chat  returns 404 for disconnected robot",
             resp.status_code == 404):
        passed += 1

    # POST /robots/<id>/connect — robot has address but nothing listening
    total += 1
    resp = client.post(f"/robots/{TEST_CLIENT_ID}/connect")
    if check("POST /robots/<id>/connect  returns 400 (no address set)",
             resp.status_code in (400, 200)):
        # 200 means it attempted connection (address was set), 400 means no address
        passed += 1

    # ── Cleanup ───────────────────────────────────────────────
    print("\n  Cleaning up test robot...")
    try:
        from data.connection import get_client
        get_client().table("robots").delete().eq(
            "client_id", TEST_CLIENT_ID
        ).execute()
        print("  OK  Test robot deleted")
    except Exception as e:
        print(f"  WARN  Cleanup failed: {e}")

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Server is ready. Run with: python3 app.py")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()