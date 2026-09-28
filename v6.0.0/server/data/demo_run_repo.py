"""
data/demo_run_repo.py
=====================
All database operations that touch the 'demo_runs' table — one row per tour,
schema in data/migrations/014_v2_add.sql.

`mode` is the column that matters: only 'live' runs may feed the planner's
duration estimates, and a run with no row here reads as mode 'unknown'.

Writes are best-effort like every write on a demo's path: they return False and
print, never raise into a live tour.
"""

from __future__ import annotations

from data.connection import get_client

RUNS = "demo_runs"


def upsert_run(row: dict) -> bool:
    """Insert or update one run by id. Called at start, and at the end with
    ended_at added; both carry the whole row, so either order leaves the same
    result.

    scenario_id and supervisor_id are foreign keys. A run naming one the
    database does not know is retried without them — losing the row would
    make the run's durations read as 'unknown' mode, which is worse than
    losing who supervised it.
    """
    try:
        get_client().table(RUNS).upsert(row, on_conflict="id").execute()
        return True
    except Exception as e:
        slim = {k: v for k, v in row.items() if k not in ("scenario_id", "supervisor_id")}
        if slim == row:
            print(f"[demo_run_repo] upsert_run error: {e}")
            return False
        try:
            get_client().table(RUNS).upsert(slim, on_conflict="id").execute()
            print(f"[demo_run_repo] run {row.get('id')} recorded without "
                  f"scenario/supervisor ({e})")
            return True
        except Exception as e2:
            print(f"[demo_run_repo] upsert_run error: {e2}")
            return False


def run_modes() -> dict:
    """{run_id: {"mode", "format"}} for every run — how duration rows are split
    into live evidence and simulated evidence."""
    try:
        rows = get_client().table(RUNS).select("id, mode, format").limit(10000).execute().data or []
        return {r["id"]: r for r in rows}
    except Exception as e:
        print(f"[demo_run_repo] run_modes error: {e}")
        return {}
