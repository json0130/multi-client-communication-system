"""
data/robot_repo.py
==================
All database operations that touch the 'robots' table.
No other layer queries this table directly.

Supabase schema expected:
  robots (
    client_id       text PRIMARY KEY,
    robot_name      text NOT NULL,
    robot_role      text,
    is_active       boolean DEFAULT false,
    allowed_tags    text[],         -- e.g. ['[WAVE]', '[HAPPY]', '[DEFAULT]']
    modules         text[],         -- e.g. ['gpt', 'speech', 'emotion', 'rag']
    ip_address      text,
    ws_port         integer
  )
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from data.connection import get_client


@dataclass
class RobotRecord:
    client_id: str
    robot_name: str
    robot_role: str
    is_active: bool
    allowed_tags: list[str]
    modules: list[str]
    ip_address: Optional[str] = None
    ws_port: Optional[int] = None


def _row_to_record(row: dict) -> RobotRecord:
    return RobotRecord(
        client_id=row["client_id"],
        robot_name=row.get("robot_name", ""),
        robot_role=row.get("robot_role", "You are a helpful robot."),
        is_active=row.get("is_active", False),
        allowed_tags=row.get("allowed_tags") or ["[DEFAULT]"],
        modules=row.get("modules") or [],
        ip_address=row.get("ip_address"),
        ws_port=row.get("ws_port"),
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_robot(client_id: str) -> Optional[RobotRecord]:
    """Fetch a single robot by its client_id. Returns None if not found."""
    try:
        resp = (
            get_client()
            .table("robots")
            .select("*")
            .eq("client_id", client_id)
            .single()
            .execute()
        )
        return _row_to_record(resp.data) if resp.data else None
    except Exception as e:
        # .single() raises if no row found — treat as not found
        return None


def get_all_active_robots(exclude_id: Optional[str] = None) -> list[RobotRecord]:
    """
    Return all robots currently marked is_active=True.
    Pass exclude_id to omit the calling robot from its own peer list.
    """
    try:
        query = get_client().table("robots").select("*").eq("is_active", True)
        if exclude_id:
            query = query.neq("client_id", exclude_id)
        resp = query.execute()
        return [_row_to_record(r) for r in (resp.data or [])]
    except Exception as e:
        print(f"[robot_repo] get_all_active_robots error: {e}")
        return []


def get_robot_address(client_id: str) -> Optional[tuple[str, int]]:
    """
    Return (ip_address, ws_port) for a robot, or None if not stored.
    Used by the server when it needs to open a WebSocket connection to a robot.
    """
    robot = get_robot(client_id)
    if robot and robot.ip_address and robot.ws_port:
        return (robot.ip_address, robot.ws_port)
    return None


# ── Write ─────────────────────────────────────────────────────────────────────

def upsert_robot(
    client_id: str,
    robot_name: str,
    robot_role: str,
    allowed_tags: list[str],
    modules: list[str],
    ip_address: Optional[str] = None,
    ws_port: Optional[int] = None,
) -> Optional[RobotRecord]:
    """
    Register a new robot or update an existing one.
    Called from the HTTP management API when a robot is registered via the web UI.
    """
    try:
        payload = {
            "client_id": client_id,
            "robot_name": robot_name,
            "robot_role": robot_role,
            "allowed_tags": allowed_tags,
            "modules": modules,
            "is_active": False,  # becomes True when the server connects
        }
        if ip_address:
            payload["ip_address"] = ip_address
        if ws_port:
            payload["ws_port"] = ws_port

        resp = (
            get_client()
            .table("robots")
            .upsert(payload, on_conflict="client_id")
            .execute()
        )
        return _row_to_record(resp.data[0]) if resp.data else None
    except Exception as e:
        print(f"[robot_repo] upsert_robot error: {e}")
        return None


def set_active(client_id: str, active: bool) -> bool:
    """
    Flip the is_active flag. Called when the server opens/closes a connection.
    Returns True on success.
    """
    try:
        get_client().table("robots").update(
            {"is_active": active}
        ).eq("client_id", client_id).execute()
        return True
    except Exception as e:
        print(f"[robot_repo] set_active error: {e}")
        return False


def update_role_and_tags(
    client_id: str,
    robot_role: Optional[str] = None,
    allowed_tags: Optional[list[str]] = None,
) -> bool:
    """
    Update role/tags from the web management UI. Returns True on success.
    """
    updates: dict = {}
    if robot_role is not None:
        updates["robot_role"] = robot_role
    if allowed_tags is not None:
        updates["allowed_tags"] = allowed_tags
    if not updates:
        return True  # nothing to do

    try:
        get_client().table("robots").update(updates).eq("client_id", client_id).execute()
        return True
    except Exception as e:
        print(f"[robot_repo] update_role_and_tags error: {e}")
        return False