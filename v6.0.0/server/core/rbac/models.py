"""
core/rbac/models.py
===================
The value types the policy operates on.

These are deliberately plain frozen dataclasses with no I/O and no knowledge of
Supabase, FAISS or any particular store. The memory layer adapts its own rows
into MemoryRecord before asking for a decision, which is what keeps
core/rbac application-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RobotIdentity:
    """
    Who is asking. Assembled from the robot's "Social Identity" at connect time:
    its DB row (access_level, scenario_id) plus the live session.

    access_level is intentionally typed as the raw value rather than AccessLevel
    so that an unparseable level from the database reaches the policy and is
    turned into an auditable deny, instead of blowing up at construction.
    """

    robot_id: str
    scenario_id: Optional[str] = None
    session_id: Optional[str] = None
    access_level: object = None     # AccessLevel | str | None
    role: Optional[str] = None      # persona role, e.g. "Receptionist" — not used for decisions


@dataclass(frozen=True)
class MemoryRecord:
    """
    One retrievable unit of memory, from any store — a FAISS vector, an
    interaction-log row, or (in future) a key point.

    record_id must be stable and store-scoped, because delegation grants name
    records by ID. See filter.make_record_id.
    """

    record_id: str
    content: str
    source_robot_id: Optional[str] = None
    scenario_id: Optional[str] = None
    session_id: Optional[str] = None
    visibility: object = None            # Visibility | str | None
    subject_user_id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class Decision:
    """
    The outcome of one access check.

    `reason` is part of the contract, not a debugging aid: the audit log stores
    it and the tests assert on it, so denials can be counted by cause. Reasons
    are stable identifiers — see policy.Reason.
    """

    allowed: bool
    reason: str
    matched_grant_id: Optional[str] = None

    def __bool__(self) -> bool:
        return self.allowed
