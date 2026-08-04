"""
data/project_repo.py
====================
All database operations on the 'projects' and 'robot_project_access' tables.

RDAC (Robot Database Access Control):
  - get_all()            : unrestricted — used by server/admin only
  - get_for_robot(id)    : returns only projects the robot has been granted access to
  - grant_access / revoke_access : manage the junction table
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from data.connection import get_client


@dataclass
class ProjectRecord:
    id: str
    name: str
    description: str
    researcher: str
    robot_id: str           # primary assigned robot (client_id)
    keywords: list[str]
    details: str
    created_at: str


def _row(r: dict) -> ProjectRecord:
    return ProjectRecord(
        id=r["id"],
        name=r.get("name", ""),
        description=r.get("description", "") or "",
        researcher=r.get("researcher", "") or "",
        robot_id=r.get("robot_id", "") or "",
        keywords=r.get("keywords") or [],
        details=r.get("details", "") or "",
        created_at=str(r.get("created_at", "")),
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_all() -> list[ProjectRecord]:
    """Return all projects ordered by created_at."""
    try:
        resp = (
            get_client()
            .table("projects")
            .select("*")
            .order("created_at")
            .execute()
        )
        return [_row(r) for r in (resp.data or [])]
    except Exception as e:
        print(f"[project_repo] get_all error: {e}")
        return []


def get_by_id(project_id: str) -> Optional[ProjectRecord]:
    """Fetch a single project by UUID."""
    try:
        resp = (
            get_client()
            .table("projects")
            .select("*")
            .eq("id", project_id)
            .single()
            .execute()
        )
        return _row(resp.data) if resp.data else None
    except Exception:
        return None


def get_for_robot(robot_id: str, audit_sink=None) -> list[ProjectRecord]:
    """
    RDAC-filtered fetch — returns only projects the robot has access to
    via a row in robot_project_access.

    This predates core.rbac and keeps its own junction-table model: project
    access is an explicit per-robot grant, not a function of access level. Pass
    audit_sink to route its decisions into the same rbac_audit_log, so denial
    counts cover every access path rather than just the memory layer.
    """
    try:
        access_resp = (
            get_client()
            .table("robot_project_access")
            .select("project_id")
            .eq("robot_id", robot_id)
            .execute()
        )
        granted = {r["project_id"] for r in (access_resp.data or [])}

        if audit_sink is not None:
            _audit_rdac(robot_id, granted, audit_sink)

        if not granted:
            return []
        resp = (
            get_client()
            .table("projects")
            .select("*")
            .in_("id", list(granted))
            .order("created_at")
            .execute()
        )
        return [_row(r) for r in (resp.data or [])]
    except Exception as e:
        print(f"[project_repo] get_for_robot error: {e}")
        return []


def _audit_rdac(robot_id: str, granted: set, audit_sink) -> None:
    """
    Record one decision per existing project, so denials are countable.
    Never raises — an audit failure must not break the fetch.
    """
    try:
        from datetime import datetime, timezone
        from core.rbac import AuditEvent, make_record_id

        all_resp = get_client().table("projects").select("id").execute()
        now = datetime.now(timezone.utc)
        for row in (all_resp.data or []):
            pid = row["id"]
            allowed = pid in granted
            audit_sink.record(AuditEvent(
                requester_robot_id=robot_id,
                record_id=make_record_id("projects", pid),
                allowed=allowed,
                reason="rdac_grant" if allowed else "rdac_no_grant",
                matched_grant_id=None,
                scenario_id=None,
                session_id=None,
                store="projects",
                decided_at=now,
            ))
    except Exception as e:
        import warnings
        warnings.warn(f"[project_repo] RDAC audit failed (ignored): {e}", RuntimeWarning)


# ── Write ─────────────────────────────────────────────────────────────────────

def create(
    name: str,
    description: str = "",
    researcher: str = "",
    robot_id: str = "",
    keywords: list[str] | None = None,
    details: str = "",
) -> Optional[ProjectRecord]:
    """Insert a new project and return the created record."""
    try:
        payload = {
            "name": name,
            "description": description,
            "researcher": researcher,
            "robot_id": robot_id,
            "keywords": keywords or [],
            "details": details,
        }
        resp = get_client().table("projects").insert(payload).execute()
        return _row(resp.data[0]) if resp.data else None
    except Exception as e:
        print(f"[project_repo] create error: {e}")
        return None


def update(project_id: str, updates: dict) -> Optional[ProjectRecord]:
    """Partial update — only keys present in `updates` are changed."""
    try:
        resp = (
            get_client()
            .table("projects")
            .update(updates)
            .eq("id", project_id)
            .execute()
        )
        return _row(resp.data[0]) if resp.data else None
    except Exception as e:
        print(f"[project_repo] update error: {e}")
        return None


def delete(project_id: str) -> bool:
    """Delete a project (cascades to robot_project_access). Returns True on success."""
    try:
        get_client().table("projects").delete().eq("id", project_id).execute()
        return True
    except Exception as e:
        print(f"[project_repo] delete error: {e}")
        return False


# ── RDAC management ───────────────────────────────────────────────────────────

def grant_access(robot_id: str, project_id: str) -> bool:
    """Grant a robot read access to a project. Idempotent."""
    try:
        get_client().table("robot_project_access").upsert(
            {"robot_id": robot_id, "project_id": project_id}
        ).execute()
        return True
    except Exception as e:
        print(f"[project_repo] grant_access error: {e}")
        return False


def revoke_access(robot_id: str, project_id: str) -> bool:
    """Revoke a robot's access to a project."""
    try:
        (
            get_client()
            .table("robot_project_access")
            .delete()
            .eq("robot_id", robot_id)
            .eq("project_id", project_id)
            .execute()
        )
        return True
    except Exception as e:
        print(f"[project_repo] revoke_access error: {e}")
        return False
