"""
data/user_repo.py
=================
All database operations that touch the 'users' table.

Supabase schema expected:
  users (
    user_id          serial PRIMARY KEY,
    name             text,
    interests        text[] DEFAULT '{}',
    health_conditions text[] DEFAULT '{}'
  )
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from data.connection import get_client


@dataclass
class UserRecord:
    user_id: int
    name: Optional[str]
    interests: list[str]
    health_conditions: list[str]


def _row_to_record(row: dict) -> UserRecord:
    return UserRecord(
        user_id=row["user_id"],
        name=row.get("name"),
        interests=row.get("interests") or [],
        health_conditions=row.get("health_conditions") or [],
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def get_user(user_id: int) -> Optional[UserRecord]:
    """Fetch a user by ID. Returns None if not found."""
    try:
        resp = (
            get_client()
            .table("users")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return _row_to_record(resp.data) if resp.data else None
    except Exception:
        return None


# ── Write ─────────────────────────────────────────────────────────────────────

def create_user(name: Optional[str] = None) -> int:
    """
    Insert a new user row and return its user_id.
    Called when a robot connects for the first time.
    """
    resp = (
        get_client()
        .table("users")
        .insert({"name": name, "interests": [], "health_conditions": []})
        .execute()
    )
    return resp.data[0]["user_id"]


def update_interests_and_conditions(
    user_id: int,
    interests: list[str],
    health_conditions: list[str],
) -> bool:
    """Overwrite the full interests and health_conditions arrays."""
    try:
        get_client().table("users").update(
            {"interests": interests, "health_conditions": health_conditions}
        ).eq("user_id", user_id).execute()
        return True
    except Exception as e:
        print(f"[user_repo] update error: {e}")
        return False