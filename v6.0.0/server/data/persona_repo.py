"""
data/persona_repo.py
====================
All database operations on the 'personas' table.
No other layer queries this table directly.
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field

from data.connection import get_client


@dataclass
class PersonaRecord:
    id: str
    name: str
    description: str
    robot_role: str
    allowed_tags: list[str]
    modules: list[str]
    voice_config: dict
    capabilities: dict
    personality: dict
    is_default: bool
    created_at: str


def _row(r: dict) -> PersonaRecord:
    return PersonaRecord(
        id=r["id"],
        name=r.get("name", ""),
        description=r.get("description", ""),
        robot_role=r.get("robot_role", ""),
        allowed_tags=r.get("allowed_tags") or [],
        modules=r.get("modules") or [],
        voice_config=r.get("voice_config") or {},
        capabilities=r.get("capabilities") or {},
        personality=r.get("personality") or {},
        is_default=r.get("is_default", False),
        created_at=str(r.get("created_at", "")),
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_all() -> list[PersonaRecord]:
    """Return all personas ordered by created_at."""
    try:
        resp = (
            get_client()
            .table("personas")
            .select("*")
            .order("created_at")
            .execute()
        )
        return [_row(r) for r in (resp.data or [])]
    except Exception as e:
        print(f"[persona_repo] get_all error: {e}")
        return []


def get_by_id(persona_id: str) -> Optional[PersonaRecord]:
    """Fetch a single persona by ID."""
    try:
        resp = (
            get_client()
            .table("personas")
            .select("*")
            .eq("id", persona_id)
            .single()
            .execute()
        )
        return _row(resp.data) if resp.data else None
    except Exception:
        return None


def get_default() -> Optional[PersonaRecord]:
    """Return the persona marked is_default=True, or None."""
    try:
        resp = (
            get_client()
            .table("personas")
            .select("*")
            .eq("is_default", True)
            .limit(1)
            .execute()
        )
        return _row(resp.data[0]) if resp.data else None
    except Exception as e:
        print(f"[persona_repo] get_default error: {e}")
        return None


# ── Write ─────────────────────────────────────────────────────────────────────

def create(
    name: str,
    description: str,
    robot_role: str,
    allowed_tags: list[str],
    modules: list[str],
    voice_config: dict,
    capabilities: dict,
    personality: dict,
    is_default: bool = False,
) -> Optional[PersonaRecord]:
    """Insert a new persona and return the created record."""
    try:
        payload = {
            "name": name,
            "description": description,
            "robot_role": robot_role,
            "allowed_tags": allowed_tags,
            "modules": modules,
            "voice_config": voice_config,
            "capabilities": capabilities,
            "personality": personality,
            "is_default": is_default,
        }
        resp = get_client().table("personas").insert(payload).execute()
        return _row(resp.data[0]) if resp.data else None
    except Exception as e:
        print(f"[persona_repo] create error: {e}")
        return None


def update(persona_id: str, updates: dict) -> Optional[PersonaRecord]:
    """Partial update — only keys present in `updates` are changed."""
    try:
        resp = (
            get_client()
            .table("personas")
            .update(updates)
            .eq("id", persona_id)
            .execute()
        )
        return _row(resp.data[0]) if resp.data else None
    except Exception as e:
        print(f"[persona_repo] update error: {e}")
        return None


def delete(persona_id: str) -> bool:
    """Delete a persona. Returns True on success."""
    try:
        get_client().table("personas").delete().eq("id", persona_id).execute()
        return True
    except Exception as e:
        print(f"[persona_repo] delete error: {e}")
        return False