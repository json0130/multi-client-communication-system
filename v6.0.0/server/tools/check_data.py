"""
tools/check_data.py
===================
Checkpoint 2b — Full CRUD test against your real Supabase DB.
Run AFTER check_supabase.py passes.

Run from project root:
    python -m tools.check_data

EXPECTED OUTPUT (when everything is correct):
=======================================================
Checkpoint 2b — data layer CRUD
=======================================================
  PASS  All data modules import
  PASS  get_client() returns a client
  PASS  robot_repo.upsert_robot() returns a record
  PASS  robot_repo.get_robot() retrieves it
  PASS  robot_repo.set_active(True) succeeds
  PASS  robot_repo.get_all_active_robots() includes test robot
  PASS  robot_repo.update_role_and_tags() persists correctly
  PASS  robot_repo.get_robot_address() returns (ip, port)
  PASS  user_repo.create_user() returns an int user_id
  PASS  user_repo.get_user() retrieves it
  PASS  user_repo.update_interests_and_conditions() persists
  PASS  chat_log_repo.insert() returns an id
  PASS  chat_log_repo.get_recent_messages() returns messages

  Cleaning up test data...
  OK  User + chat logs deleted
  OK  Test robot deleted

Result: 13/13 checks passed
Data layer is ready. Proceed to Step 3 (modules).
=======================================================

NOTE: This test creates then deletes real rows in your Supabase DB.
      Safe to run on a dev project. Run check_supabase.py first.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_CLIENT_ID = "_checkpoint_test_robot"
TEST_ROBOT_NAME = "Checkpoint Bot"


def check(label: str, condition: bool, detail: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and detail:
        print(f"         → {detail}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 2b — data layer CRUD")
    print("=" * 55)
    passed = 0
    total = 0

    # ── 1. Imports ────────────────────────────────────────────
    total += 1
    try:
        from data.connection import get_client
        from data import robot_repo, user_repo, chat_log_repo
        ok = True
    except Exception as e:
        ok = False
        print(f"  FAIL  Imports failed: {e}")
    if check("All data modules import", ok):
        passed += 1
    if not ok:
        _summary(passed, total)
        return

    # ── 2. Connection ─────────────────────────────────────────
    total += 1
    detail = ""
    try:
        client = get_client()
        ok = client is not None
    except Exception as e:
        ok = False
        detail = str(e)
    if check("get_client() returns a client", ok, detail):
        passed += 1
    if not ok:
        print("\n  Run check_supabase.py first to diagnose the connection.")
        _summary(passed, total)
        return

    # ── 3. robot_repo ─────────────────────────────────────────
    total += 1
    robot = robot_repo.upsert_robot(
        client_id=TEST_CLIENT_ID,
        robot_name=TEST_ROBOT_NAME,
        robot_role="Test robot for checkpoint",
        allowed_tags=["[DEFAULT]", "[WAVE]"],
        modules=["gpt", "speech"],
        ip_address="127.0.0.1",
        ws_port=8765,
    )
    if check("robot_repo.upsert_robot() returns a record",
             robot is not None and robot.client_id == TEST_CLIENT_ID):
        passed += 1

    total += 1
    fetched = robot_repo.get_robot(TEST_CLIENT_ID)
    if check("robot_repo.get_robot() retrieves it",
             fetched is not None and fetched.robot_name == TEST_ROBOT_NAME):
        passed += 1

    total += 1
    ok = robot_repo.set_active(TEST_CLIENT_ID, True)
    if check("robot_repo.set_active(True) succeeds", ok):
        passed += 1

    total += 1
    active = robot_repo.get_all_active_robots()
    if check("robot_repo.get_all_active_robots() includes test robot",
             TEST_CLIENT_ID in [r.client_id for r in active]):
        passed += 1

    total += 1
    ok = robot_repo.update_role_and_tags(
        TEST_CLIENT_ID,
        robot_role="Updated role",
        allowed_tags=["[DEFAULT]", "[HAPPY]", "[SAD]"],
    )
    refetched = robot_repo.get_robot(TEST_CLIENT_ID)
    if check("robot_repo.update_role_and_tags() persists correctly",
             ok and refetched is not None and refetched.robot_role == "Updated role"):
        passed += 1

    total += 1
    addr = robot_repo.get_robot_address(TEST_CLIENT_ID)
    if check("robot_repo.get_robot_address() returns (ip, port)",
             addr == ("127.0.0.1", 8765)):
        passed += 1

    # ── 4. user_repo ──────────────────────────────────────────
    user_id = None
    total += 1
    try:
        user_id = user_repo.create_user(name="Checkpoint Test User")
        if check("user_repo.create_user() returns an int user_id",
                 isinstance(user_id, int) and user_id > 0):
            passed += 1
    except Exception as e:
        check("user_repo.create_user()", False, str(e))

    if user_id:
        total += 1
        user = user_repo.get_user(user_id)
        if check("user_repo.get_user() retrieves it",
                 user is not None and user.name == "Checkpoint Test User"):
            passed += 1

        total += 1
        ok = user_repo.update_interests_and_conditions(
            user_id,
            interests=["robotics", "AI"],
            health_conditions=["none"],
        )
        updated = user_repo.get_user(user_id)
        if check("user_repo.update_interests_and_conditions() persists",
                 ok and updated is not None and "robotics" in updated.interests):
            passed += 1

        # ── 5. chat_log_repo ──────────────────────────────────
        total += 1
        log_id = chat_log_repo.insert(user_id, "Hello robot", "Hello human!")
        if check("chat_log_repo.insert() returns an id",
                 log_id is not None and log_id > 0):
            passed += 1

        total += 1
        messages = chat_log_repo.get_recent_messages(user_id, limit=10)
        if check("chat_log_repo.get_recent_messages() returns messages",
                 "Hello robot" in messages):
            passed += 1

        # ── 6. Cleanup ────────────────────────────────────────
        print("\n  Cleaning up test data...")
        try:
            client.table("chat_logs").delete().eq("user_id", user_id).execute()
            client.table("users").delete().eq("user_id", user_id).execute()
            print("  OK  User + chat logs deleted")
        except Exception as e:
            print(f"  WARN  Cleanup failed (delete manually): {e}")

    try:
        client.table("robots").delete().eq("client_id", TEST_CLIENT_ID).execute()
        print("  OK  Test robot deleted")
    except Exception as e:
        print(f"  WARN  Robot cleanup failed: {e}")

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Data layer is ready. Proceed to Step 3 (modules).")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()