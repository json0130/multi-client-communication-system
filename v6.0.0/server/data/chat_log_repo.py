"""
data/chat_log_repo.py
=====================
All database operations that touch the 'chat_logs' table.

Supabase schema expected (see data/migrations/002_rbac.sql for the RBAC columns):
  chat_logs (
    id              serial PRIMARY KEY,
    user_id         integer REFERENCES users(user_id),
    message         text,
    response        text,
    created_at      timestamptz DEFAULT now(),
    -- RBAC provenance + visibility
    source_robot_id text,
    scenario_id     text,
    session_id      text,
    subject_user_id integer,
    visibility      text NOT NULL DEFAULT 'local'
  )

Reads that feed a prompt must go through get_recent_records(), which filters via
core.rbac. get_recent_messages() is kept for admin/diagnostic callers and does
not filter — it is not a retrieval path.
"""

from __future__ import annotations
from typing import Optional, Sequence

from data.connection import get_client
from core.rbac import (
    ClearedRecord,
    DelegationGrant,
    MemoryRecord,
    RBACFilter,
    RobotIdentity,
    make_record_id,
)

STORE_NAME = "chat_logs"

# Columns the RBAC layer needs to make a decision.
_RBAC_COLUMNS = (
    "id, message, response, source_robot_id, scenario_id, session_id, "
    "visibility, subject_user_id"
)


def insert(
    user_id: int,
    message: str,
    response: str,
    source_robot_id: Optional[str] = None,
    scenario_id: Optional[str] = None,
    session_id: Optional[str] = None,
    visibility: str = "local",
    subject_user_id: Optional[int] = None,
) -> Optional[int]:
    """
    Persist a user message + bot response pair with its provenance.
    Returns the new row id, or None on failure.

    visibility defaults to 'local' — fail closed. Widening is a deployment
    decision expressed in the scenario profile, never an implicit default.
    """
    try:
        payload = {
            "user_id": user_id,
            "message": message,
            "response": response,
            "source_robot_id": source_robot_id,
            "scenario_id": scenario_id,
            "session_id": session_id,
            "visibility": visibility,
            "subject_user_id": subject_user_id if subject_user_id is not None else user_id,
        }
        resp = get_client().table("chat_logs").insert(payload).execute()
        return resp.data[0]["id"]
    except Exception as e:
        print(f"[chat_log_repo] insert error: {e}")
        return None


def get_recent_records(
    requester: RobotIdentity,
    rbac: RBACFilter,
    grants: Sequence[DelegationGrant] = (),
    user_id: Optional[int] = None,
    limit: int = 100,
) -> list[ClearedRecord]:
    """
    Return recent interaction-log entries this robot is allowed to read.

    Over-fetches and filters on metadata, so a Worker is not left empty-handed
    when the most recent rows belong to other robots. Pass user_id to narrow to
    one subject; omit it for a cross-client query (only a Manager will get
    anything back that it did not write itself).
    """
    try:
        query = get_client().table("chat_logs").select(_RBAC_COLUMNS)
        if user_id is not None:
            query = query.eq("user_id", user_id)
        # Over-fetch: the accessible subset is a fraction of the newest rows.
        resp = query.order("id", desc=True).limit(max(limit * 4, limit)).execute()
        rows = resp.data or []
    except Exception as e:
        print(f"[chat_log_repo] get_recent_records error: {e}")
        return []

    candidates = [_to_memory_record(r) for r in rows if (r.get("message") or "").strip()]
    cleared = rbac.filter_records(requester, candidates, grants, store=STORE_NAME)
    return cleared[:limit]


def get_recent_messages(user_id: int, limit: int = 100) -> list[str]:
    """
    Return the most recent <limit> user message strings for a given user.

    UNFILTERED — admin and diagnostic use only. Anything that feeds a prompt
    must use get_recent_records() instead.
    """
    try:
        resp = (
            get_client()
            .table("chat_logs")
            .select("message")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return [r["message"] for r in (resp.data or []) if r.get("message")]
    except Exception as e:
        print(f"[chat_log_repo] get_recent_messages error: {e}")
        return []


def _to_memory_record(row: dict) -> MemoryRecord:
    """Adapt a chat_logs row into the store-agnostic type the policy operates on."""
    return MemoryRecord(
        record_id=make_record_id(STORE_NAME, row.get("id")),
        content=row.get("message") or "",
        source_robot_id=row.get("source_robot_id"),
        scenario_id=row.get("scenario_id"),
        session_id=row.get("session_id"),
        visibility=row.get("visibility"),
        subject_user_id=row.get("subject_user_id"),
    )
