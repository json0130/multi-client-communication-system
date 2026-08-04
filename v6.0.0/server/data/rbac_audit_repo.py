"""
data/rbac_audit_repo.py
=======================
Persistence for RBAC access decisions.

Writes to `rbac_audit_log`, a table entirely separate from `chat_logs` — the
interaction log records what was said, this records what was allowed. Schema in
data/migrations/003_rbac_audit.sql, along with the denial-count views.

This module is the Supabase-backed writer plugged into
core.rbac.audit.BatchingAuditSink, which batches calls onto a daemon thread and
degrades a write failure to a warning. Nothing here is on the retrieval path.
"""

from __future__ import annotations
from typing import Optional, Sequence

from data.connection import get_client
from core.rbac import AuditEvent

TABLE = "rbac_audit_log"


def write_events(events: Sequence[AuditEvent]) -> None:
    """
    Insert a batch of decisions.

    Raises on failure — BatchingAuditSink catches and warns. Raising here rather
    than swallowing keeps the failure visible in exactly one place.
    """
    if not events:
        return
    get_client().table(TABLE).insert([e.as_row() for e in events]).execute()


# ── Queries ───────────────────────────────────────────────────────────────────
# These back the "enforcement effectiveness" measurements: at minimum, denial
# counts by robot and by reason.

def denials_by_robot(scenario_id: Optional[str] = None) -> list[dict]:
    """[{requester_robot_id, denials}] — reads the rbac_denials_by_robot view."""
    try:
        query = get_client().table("rbac_denials_by_robot").select("*")
        if scenario_id:
            query = query.eq("scenario_id", scenario_id)
        return query.execute().data or []
    except Exception as e:
        print(f"[rbac_audit_repo] denials_by_robot error: {e}")
        return []


def denials_by_reason(scenario_id: Optional[str] = None) -> list[dict]:
    """[{reason, denials}] — reads the rbac_denials_by_reason view."""
    try:
        query = get_client().table("rbac_denials_by_reason").select("*")
        if scenario_id:
            query = query.eq("scenario_id", scenario_id)
        return query.execute().data or []
    except Exception as e:
        print(f"[rbac_audit_repo] denials_by_reason error: {e}")
        return []


def recent(limit: int = 100, allowed: Optional[bool] = None) -> list[dict]:
    """Most recent decisions, newest first. Pass allowed=False for denials only."""
    try:
        query = get_client().table(TABLE).select("*")
        if allowed is not None:
            query = query.eq("allowed", allowed)
        return query.order("decided_at", desc=True).limit(limit).execute().data or []
    except Exception as e:
        print(f"[rbac_audit_repo] recent error: {e}")
        return []
