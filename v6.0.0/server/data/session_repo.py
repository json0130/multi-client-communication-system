"""
data/session_repo.py
====================
All database operations that touch the 'sessions' table — one row per robot
connection, schema in data/migrations/014_v2_add.sql. The id is the robot
instance's identity.session_id ('{client_id}:{epoch}'), the same value every
log row's session_id column carries.

legacy_user_id links the users row created for the same connection, while
users is still written (dual write).

Best-effort: failures print and return False, never raise into a connect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from data.connection import get_client

SESSIONS = "sessions"


def start_session(
    session_id: str,
    robot_id: str,
    scenario_id: Optional[str] = None,
    legacy_user_id: Optional[int] = None,
    started_at: Optional[float] = None,
) -> bool:
    """Upsert the row for a new connection. A scenario_id the scenarios table
    does not know is dropped and the row retried: losing the session would
    leave every log row of the connection dangling."""
    row = {
        "id": session_id,
        "robot_id": robot_id,
        "scenario_id": scenario_id,
        "legacy_user_id": legacy_user_id or None,
        "started_at": datetime.fromtimestamp(
            started_at if started_at is not None else datetime.now().timestamp(),
            timezone.utc).isoformat(),
    }
    try:
        get_client().table(SESSIONS).upsert(row, on_conflict="id").execute()
        return True
    except Exception as e:
        if row["scenario_id"] is None:
            print(f"[session_repo] start_session error: {e}")
            return False
        try:
            get_client().table(SESSIONS).upsert(
                {**row, "scenario_id": None}, on_conflict="id").execute()
            print(f"[session_repo] session {session_id} recorded without scenario ({e})")
            return True
        except Exception as e2:
            print(f"[session_repo] start_session error: {e2}")
            return False


def end_session(session_id: str) -> bool:
    """Stamp ended_at on a connection that closed."""
    try:
        (get_client().table(SESSIONS)
            .update({"ended_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", session_id).execute())
        return True
    except Exception as e:
        print(f"[session_repo] end_session error: {e}")
        return False
