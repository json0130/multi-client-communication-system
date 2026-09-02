"""
data/demo_decision_repo.py
==========================
Persistence for demo orchestration decisions and supervisor corrections.

Writes to `demo_decision_log` and `demo_correction_log`. Schema in
data/migrations/004_demo_decisions.sql, along with the aggregate views.

This module is the Supabase-backed writer plugged into
decision.recorder.BatchingDecisionSink, which batches calls onto a daemon thread
and degrades a write failure to a warning. Nothing here runs on the demo's
critical path.

Same split as data/rbac_audit_repo.py: writes raise so the sink can warn once
with a count, reads swallow and return empty so a dashboard panel degrades to
blank instead of 500-ing.
"""

from __future__ import annotations
from typing import Optional, Sequence

from data.connection import get_client
from decision.models import CorrectionEvent, DecisionEvent

DECISIONS = "demo_decision_log"
CORRECTIONS = "demo_correction_log"


# ── Writes ────────────────────────────────────────────────────────────────────

def write_events(events: Sequence[DecisionEvent]) -> None:
    """
    Insert a batch of decisions.

    Raises on failure — BatchingDecisionSink catches and warns. Raising here
    rather than swallowing keeps the failure visible in exactly one place.
    """
    if not events:
        return
    get_client().table(DECISIONS).insert([e.as_row() for e in events]).execute()


def write_corrections(events: Sequence[CorrectionEvent]) -> None:
    """
    Insert a batch of supervisor corrections.

    A correction whose decision_id is not present — the decision batch was
    dropped, or none was logged — still inserts: the FK is nullable and
    ON DELETE SET NULL. Losing the parent must not lose the label.
    """
    if not events:
        return
    get_client().table(CORRECTIONS).insert([e.as_row() for e in events]).execute()


# ── Queries ───────────────────────────────────────────────────────────────────
# These back the phase-one measurements: which mechanism actually decides, and
# how often a supervisor disagrees with it.

def corrections_by_mechanism(scenario_id: Optional[str] = None) -> list[dict]:
    """
    [{mechanism, decision_point, decisions, corrections, correction_rate}]
    — reads the demo_corrections_by_mechanism view.

    The headline number for phase one: a mechanism with a high correction rate
    is the one worth replacing with a learned policy first.
    """
    try:
        query = get_client().table("demo_corrections_by_mechanism").select("*")
        if scenario_id:
            query = query.eq("scenario_id", scenario_id)
        return query.execute().data or []
    except Exception as e:
        print(f"[demo_decision_repo] corrections_by_mechanism error: {e}")
        return []


def decisions_by_step(session_id: Optional[str] = None) -> list[dict]:
    """[{step_idx, step_id, decision_point, decisions, ...}] — per-step breakdown."""
    try:
        query = get_client().table("demo_decisions_by_step").select("*")
        if session_id:
            query = query.eq("session_id", session_id)
        return query.order("step_idx").execute().data or []
    except Exception as e:
        print(f"[demo_decision_repo] decisions_by_step error: {e}")
        return []


def plan_revisions(session_id: Optional[str] = None) -> list[dict]:
    """[{step_id, mechanism, op_kind, op_robot_id, decided_at}] — how the tour changed."""
    try:
        query = get_client().table("demo_plan_revisions").select("*")
        if session_id:
            query = query.eq("session_id", session_id)
        return query.order("decided_at", desc=True).execute().data or []
    except Exception as e:
        print(f"[demo_decision_repo] plan_revisions error: {e}")
        return []


def recent_decisions(
    limit: int = 100,
    session_id: Optional[str] = None,
    decision_point: Optional[str] = None,
) -> list[dict]:
    """Most recent decisions, newest first."""
    try:
        query = get_client().table(DECISIONS).select("*")
        if session_id:
            query = query.eq("session_id", session_id)
        if decision_point:
            query = query.eq("decision_point", decision_point)
        return query.order("decided_at", desc=True).limit(limit).execute().data or []
    except Exception as e:
        print(f"[demo_decision_repo] recent_decisions error: {e}")
        return []


def recent_corrections(limit: int = 100, session_id: Optional[str] = None) -> list[dict]:
    """Most recent supervisor corrections, newest first — the training set so far."""
    try:
        query = get_client().table(CORRECTIONS).select("*")
        if session_id:
            query = query.eq("session_id", session_id)
        return query.order("corrected_at", desc=True).limit(limit).execute().data or []
    except Exception as e:
        print(f"[demo_decision_repo] recent_corrections error: {e}")
        return []
