"""
data/supervisor_repo.py
=======================
The 'supervisors' table (migration 014): who can supervise a demo run. A run
names one in demo_runs.supervisor_id and every correction in it carries it.
Rows are added by hand in SQL; the server only reads them.
"""

from __future__ import annotations

from typing import Optional

from data.connection import get_client

SUPERVISORS = "supervisors"


def list_supervisors() -> list[dict]:
    """[{id, name}] by name. Empty on failure, so the dashboard's list is blank
    rather than broken."""
    try:
        return (get_client().table(SUPERVISORS).select("id, name")
                .order("name").execute().data or [])
    except Exception as e:
        print(f"[supervisor_repo] list_supervisors error: {e}")
        return []


def supervisor_exists(supervisor_id: str) -> Optional[bool]:
    """True / False, or None when the table could not be read — a caller
    decides whether "can't tell" should stop anything."""
    try:
        rows = (get_client().table(SUPERVISORS).select("id")
                .eq("id", supervisor_id).limit(1).execute().data or [])
        return bool(rows)
    except Exception as e:
        print(f"[supervisor_repo] supervisor_exists error: {e}")
        return None
