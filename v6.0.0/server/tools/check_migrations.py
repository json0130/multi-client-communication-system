"""
tools/check_migrations.py
=========================
Which migrations have actually landed on the live database.

Written because the answer turned out to be "none of them": the RBAC code has
been running against a schema with no access_level, no visibility and no audit
table since the branch was written. Nothing failed loudly — the filter degrades,
the audit sink warns once — so the gap was invisible until something asked.

Also probes an actual INSERT on each log table. Existence is not enough: 003 and
004 both landed with RLS on and no policy, so every write was rejected while the
sinks degraded it to a warning — a demo could run and record nothing. A checker
that only looked for tables reported all-clear through exactly that.

Writes a probe row and deletes it. Safe to run any time, against any environment:

    python3 tools/check_migrations.py

Exits 0 when every migration is present, 1 otherwise, so it can gate a demo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.connection import get_client   # noqa: E402


# What each migration adds, as things we can probe over PostgREST.
# (label, table, columns that must exist)  — a missing table is reported as such.
CHECKS = [
    ("002_rbac", "robots", ["access_level", "scenario_id"]),
    ("002_rbac", "chat_logs", ["source_robot_id", "scenario_id", "session_id",
                               "subject_user_id", "visibility"]),
    ("003_rbac_audit", "rbac_audit_log", []),
    ("004_demo_decisions", "demo_decision_log", []),
    ("004_demo_decisions", "demo_correction_log", []),
    ("006_demo_kg", "demo_topics", []),
    ("006_demo_kg", "demo_topic_links", []),
    ("006_demo_kg", "demo_robot_topic", []),
    ("007_kg_displaced", "demo_robot_topic", ["n_displaced"]),
    ("008_demo_durations", "demo_step_durations", []),
    ("008_demo_durations", "demo_qa_durations", []),
]

# Tables the server must be able to INSERT into, with a minimal valid row.
WRITE_PROBES = [
    ("003_rbac_audit", "rbac_audit_log", "id", {
        "requester_robot_id": "_probe", "record_id": "_probe:0", "allowed": True,
        "reason": "_probe", "store": "_probe",
    }),
    ("004_demo_decisions", "demo_decision_log", "decision_id", {
        "decision_point": "_probe", "action_kind": "_probe",
        "mechanism": "_probe", "step_idx": 0,
    }),
    ("004_demo_decisions", "demo_correction_log", "correction_id", {
        "decision_point": "_probe", "corrected_to_kind": "_probe",
        "source": "_probe", "step_idx": 0,
    }),
]

# Views are probed separately: a missing view means the aggregates the analysis
# depends on are absent even though the tables landed.
VIEWS = [
    ("003_rbac_audit", "rbac_denials_by_robot"),
    ("003_rbac_audit", "rbac_denials_by_reason"),
    ("004_demo_decisions", "demo_corrections_by_mechanism"),
    ("004_demo_decisions", "demo_decisions_by_step"),
    ("004_demo_decisions", "demo_plan_revisions"),
    ("006_demo_kg", "demo_kg_edges"),
    ("008_demo_durations", "demo_step_duration_stats"),
    ("008_demo_durations", "demo_qa_duration_stats"),
]


def _probe_write(client, table: str, key: str, row: dict):
    """(ok, detail). Inserts a probe row and removes it again."""
    import uuid
    row = dict(row)
    if key != "id":
        row[key] = str(uuid.uuid4())
    try:
        res = client.table(table).insert([row]).execute()
    except Exception as e:
        msg = str(e)
        if "row-level security" in msg or "42501" in msg:
            return False, "BLOCKED BY RLS — no insert policy"
        return False, msg[:70]
    # Clean up, then CONFIRM it worked. A denied DELETE is not an error over
    # PostgREST — RLS just matches zero rows and reports success — so trusting
    # the delete call left a fake row in the real data on every run, showing up
    # in demo_corrections_by_mechanism as its own mechanism.
    try:
        written = (res.data or [{}])[0]
        ident = written.get(key) or row.get(key)
        if ident is None:
            return True, "insert ok (probe row could not be identified to remove)"
        client.table(table).delete().eq(key, ident).execute()
        still = client.table(table).select(key).eq(key, ident).execute()
        if still.data:
            return False, ("insert ok BUT probe row could not be deleted — "
                           "re-run 005_log_rls.sql for the probe DELETE policy")
    except Exception as e:
        return False, f"insert ok, cleanup failed: {str(e)[:50]}"
    return True, "insert ok"


def _probe(client, relation: str):
    """(exists, columns, rowcount). columns is [] when the table is empty."""
    try:
        r = client.table(relation).select("*", count="exact").limit(1).execute()
    except Exception as e:
        return False, [], str(e)[:80]
    cols = sorted(r.data[0].keys()) if r.data else []
    return True, cols, r.count


def main() -> int:
    try:
        client = get_client()
    except Exception as e:
        print(f"Cannot reach Supabase: {e}")
        return 1

    print("=" * 62)
    print("  Migration status")
    print("=" * 62)

    missing: set = set()

    for label, table, needed in CHECKS:
        exists, cols, info = _probe(client, table)
        if not exists:
            print(f"  [{label}] {table:24} MISSING TABLE")
            missing.add(label)
            continue
        if not needed:
            print(f"  [{label}] {table:24} ok   rows={info}")
            continue
        # An empty table hides its columns from PostgREST's row output, so fall
        # back to asking for the columns by name — a missing one is rejected.
        # Reporting "unverifiable" here was itself a blind spot: an unapplied
        # migration on an empty table came back as all-clear.
        if not cols:
            gone = []
            for c in needed:
                try:
                    client.table(table).select(c).limit(1).execute()
                except Exception:
                    gone.append(c)
            if gone:
                print(f"  [{label}] {table:24} MISSING COLUMNS: {', '.join(gone)}")
                missing.add(label)
            else:
                print(f"  [{label}] {table:24} ok   (empty, columns verified)")
            continue
        gone = [c for c in needed if c not in cols]
        if gone:
            print(f"  [{label}] {table:24} MISSING COLUMNS: {', '.join(gone)}")
            missing.add(label)
        else:
            print(f"  [{label}] {table:24} ok   rows={info}")

    print()
    for label, table, key, row in WRITE_PROBES:
        exists, _c, _i = _probe(client, table)
        if not exists:
            continue   # already reported as a missing table above
        ok, detail = _probe_write(client, table, key, row)
        print(f"  [{label}] write {table:24} {'ok' if ok else detail}")
        if not ok:
            missing.add("005_log_rls" if "RLS" in detail else label)

    print()
    for label, view in VIEWS:
        exists, _cols, info = _probe(client, view)
        if exists:
            print(f"  [{label}] view {view:29} ok")
        else:
            print(f"  [{label}] view {view:29} MISSING")
            missing.add(label)

    print()
    if missing:
        print(f"INCOMPLETE — not applied: {', '.join(sorted(missing))}")
        print()
        print("Apply them by pasting this file into the Supabase SQL editor:")
        print("    v6.0.0/server/data/migrations/apply_all.sql")
        print("It is idempotent, so running it twice is harmless.")
        return 1

    print("All migrations present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
