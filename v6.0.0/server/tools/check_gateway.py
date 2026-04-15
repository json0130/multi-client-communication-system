"""
tools/check_gateway.py
=======================
Checkpoint 8 — verifies the gateway layer (delegation, websocket, http).

Run from project root:
    python3 tools/check_gateway.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
=======================================================
Checkpoint 8 — Gateway layer
=======================================================
  PASS  delegation_handler imports
  PASS  websocket_gateway imports
  PASS  http_gateway imports

  -- DelegationHandler unit tests --
  PASS  Extracts target_robot_id from response text
  PASS  Extracts task from response text
  PASS  Returns (None, None) for plain response (no JSON block)
  PASS  Handles double-brace hallucination {{ }} correctly
  PASS  Returns False for non-delegation response

  -- WebSocketGateway unit tests --
  PASS  Gateway initialises with empty connection pool
  PASS  get_connected_ids() returns empty list initially
  PASS  connect_robot() returns False for unknown robot (no address in DB)

  -- HTTP Blueprint unit tests --
  PASS  Blueprint created successfully
  PASS  Blueprint has correct name 'api'
  PASS  /robots/register route exists
  PASS  /robots/<id>/role route exists
  PASS  /robots/<id>/connect route exists
  PASS  /robots/<id>/health route exists
  PASS  /robots/<id>/chat route exists

Result: 18/18 checks passed
Gateway layer is ready. Proceed to Step 9 (app.py — wire everything together).
=======================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 8 — Gateway layer")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Imports ────────────────────────────────────────────
    for label, mod in [
        ("delegation_handler imports", "gateway.delegation_handler"),
        ("websocket_gateway imports",  "gateway.websocket_gateway"),
        ("http_gateway imports",       "gateway.http_gateway"),
    ]:
        total += 1
        try:
            __import__(mod)
            ok = True
        except Exception as e:
            ok = False
            print(f"         Error: {e}")
        if check(label, ok):
            passed += 1

    if passed < 3:
        _summary(passed, total)
        return

    from gateway.delegation_handler import DelegationHandler
    from gateway.websocket_gateway import WebSocketGateway
    from gateway.http_gateway import create_http_gateway
    from robot.robot_registry import RobotRegistry

    # ── 2. DelegationHandler unit tests ──────────────────────
    print("\n  -- DelegationHandler unit tests --")

    # Create a minimal stub to satisfy type hints
    registry = RobotRegistry()

    class _FakeWS:
        def send_to_robot(self, *a, **kw): pass

    handler = DelegationHandler(registry, _FakeWS())

    delegation_response = (
        "[WAVE] I'll ask CoffeeBot right away!\n"
        "```json\n"
        '{"target_robot_id": "coffeebot_01", "task": "Make one latte please."}\n'
        "```"
    )

    total += 1
    target, task = handler._extract(delegation_response)
    if check("Extracts target_robot_id from response text",
             target == "coffeebot_01"):
        passed += 1

    total += 1
    if check("Extracts task from response text",
             task == "Make one latte please."):
        passed += 1

    total += 1
    plain = "[WAVE] Hello! How can I help you today?"
    t2, task2 = handler._extract(plain)
    if check("Returns (None, None) for plain response (no JSON block)",
             t2 is None and task2 is None):
        passed += 1

    total += 1
    double_brace = (
        "[DEFAULT] Sure!\n"
        "```json\n"
        '{{"target_robot_id": "bot_02", "task": "Do the thing."}}\n'
        "```"
    )
    t3, task3 = handler._extract(double_brace)
    if check("Handles double-brace hallucination {{ }} correctly",
             t3 == "bot_02"):
        passed += 1

    total += 1
    is_deleg = handler.handle("bot_01", plain)
    if check("Returns False for non-delegation response", not is_deleg):
        passed += 1

    # ── 3. WebSocketGateway unit tests ────────────────────────
    print("\n  -- WebSocketGateway unit tests --")

    ws = WebSocketGateway(registry)

    total += 1
    if check("Gateway initialises with empty connection pool",
             len(ws._connections) == 0):
        passed += 1

    total += 1
    if check("get_connected_ids() returns empty list initially",
             ws.get_connected_ids() == []):
        passed += 1

    total += 1
    # Try to connect to a robot with no address in DB — should return False
    result = ws.connect_robot("nonexistent_robot_xyz")
    if check("connect_robot() returns False for unknown robot (no address in DB)",
             not result):
        passed += 1

    # ── 4. HTTP Blueprint tests ───────────────────────────────
    print("\n  -- HTTP Blueprint unit tests --")
    from flask import Flask

    app = Flask(__name__)
    try:
        bp = create_http_gateway(registry, ws)
        app.register_blueprint(bp)
        bp_ok = True
    except Exception as e:
        bp_ok = False
        print(f"         Error: {e}")

    total += 1
    if check("Blueprint created successfully", bp_ok):
        passed += 1

    total += 1
    if check("Blueprint has correct name 'api'",
             bp_ok and bp.name == "api"):
        passed += 1

    # Check routes exist by inspecting the URL map
    if bp_ok:
        rules = {str(r) for r in app.url_map.iter_rules()}
        route_checks = [
            ("/robots/register",      "POST"),
            ("/robots/<client_id>/role",       "PUT"),
            ("/robots/<client_id>/connect",    "POST"),
            ("/robots/<client_id>/health",     "GET"),
            ("/robots/<client_id>/chat",       "POST"),
        ]
        for path, method in route_checks:
            total += 1
            exists = any(path in r for r in rules)
            if check(f"{path} route exists", exists,
                     f"Route not found — check http_gateway.py"):
                passed += 1

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Gateway layer is ready. Proceed to Step 9 (app.py — wire everything together).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()