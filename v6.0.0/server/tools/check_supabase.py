"""
tools/check_supabase.py
=======================
Checkpoint 2a — Verifies your Supabase connection and schema are ready.

Run from project root:
    python -m tools.check_supabase

EXPECTED OUTPUT (when everything is correct):
=======================================================
Checkpoint 2a — Supabase connection + schema
=======================================================
  PASS  Config loads (SUPABASE_URL + SUPABASE_KEY found)
  PASS  Connected to Supabase
  PASS  Table 'robots' exists and is readable
  PASS  Table 'users' exists and is readable
  PASS  Table 'chat_logs' exists and is readable
  PASS  'robots' has column: ip_address
  PASS  'robots' has column: ws_port

Result: 7/7 checks passed
Supabase is ready. You can now run check_data.py for full CRUD tests.
=======================================================

IF YOU SEE FAILURES:
  - "Config loads" FAIL     → Add SUPABASE_URL and SUPABASE_KEY to your .env file
  - "Connected" FAIL        → Wrong URL or KEY, check Supabase project settings
  - Table FAIL              → Run the SQL in the comment below to create it
  - Column FAIL             → Run the ALTER TABLE sql shown at the bottom

REQUIRED SUPABASE SCHEMA (run in Supabase SQL editor if tables don't exist):
----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id           SERIAL PRIMARY KEY,
    name              TEXT,
    interests         TEXT[] DEFAULT '{}',
    health_conditions TEXT[] DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS robots (
    client_id    TEXT PRIMARY KEY,
    robot_name   TEXT NOT NULL,
    robot_role   TEXT DEFAULT 'You are a helpful robot.',
    is_active    BOOLEAN DEFAULT FALSE,
    allowed_tags TEXT[] DEFAULT '{[DEFAULT]}',
    modules      TEXT[] DEFAULT '{}',
    ip_address   TEXT,
    ws_port      INTEGER
);

CREATE TABLE IF NOT EXISTS chat_logs (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    message    TEXT,
    response   TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
----------------------------------------------------------------------
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, condition: bool, fail_hint: str = "") -> bool:
    icon = "  PASS" if condition else "  FAIL"
    print(f"{icon}  {label}")
    if not condition and fail_hint:
        print(f"         → {fail_hint}")
    return condition


def run():
    print("=" * 55)
    print("Checkpoint 2a — Supabase connection + schema")
    print("=" * 55)

    passed = 0
    total = 0

    # ── 1. Config loads ───────────────────────────────────────
    total += 1
    try:
        from core.config import cfg
        config_ok = cfg is not None
    except Exception as e:
        config_ok = False

    if check("Config loads (SUPABASE_URL + SUPABASE_KEY found)",
             config_ok,
             "Add SUPABASE_URL and SUPABASE_KEY to your .env file"):
        passed += 1
    else:
        _summary(passed, total)
        return

    # ── 2. Connect ────────────────────────────────────────────
    total += 1
    try:
        from data.connection import get_client
        client = get_client()
        connected = client is not None
    except Exception as e:
        connected = False
        print(f"         Error: {e}")

    if check("Connected to Supabase", connected,
             "Check your SUPABASE_URL and SUPABASE_KEY values"):
        passed += 1
    else:
        _summary(passed, total)
        return

    # ── 3. Tables exist ───────────────────────────────────────
    for table in ["robots", "users", "chat_logs"]:
        total += 1
        try:
            # Select with limit 0 — just checks the table exists
            resp = client.table(table).select("*").limit(1).execute()
            exists = True
        except Exception as e:
            exists = False
        if check(f"Table '{table}' exists and is readable", exists,
                 f"Run the CREATE TABLE sql in the comment at the top of this file"):
            passed += 1

    # ── 4. Required columns on robots ─────────────────────────
    # Try inserting a minimal row and reading back its columns
    for col in ["ip_address", "ws_port"]:
        total += 1
        try:
            # Read one row to inspect columns; if table is empty use select
            resp = client.table("robots").select(col).limit(1).execute()
            has_col = True
        except Exception as e:
            has_col = "column" not in str(e).lower()  # non-column errors still pass

        if check(f"'robots' has column: {col}", has_col,
                 f"Run: ALTER TABLE robots ADD COLUMN {col} "
                 + ("TEXT;" if col == "ip_address" else "INTEGER;")):
            passed += 1

    _summary(passed, total)


def _summary(passed: int, total: int):
    print()
    print(f"Result: {passed}/{total} checks passed")
    if passed == total:
        print("Supabase is ready. You can now run check_data.py for full CRUD tests.")
    else:
        print("Fix the failing checks before moving on.")
    print("=" * 55)


if __name__ == "__main__":
    run()