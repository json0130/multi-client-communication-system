"""
core/rbac/grants.py
===================
Delegation grants — the paper's Context Serialization.

When a Manager delegates to a Worker it injects specific global context snippets
into the Worker's *temporary* prompt. The Worker gains situational awareness for
one hand-off without its standing access level changing at all.

Design constraints, all enforced here rather than by convention:
  - A grant names explicit snippet IDs. Never a wildcard, never a query. There
    is no API on this module that accepts a pattern.
  - Grants are short-lived (expires_at) and scoped to one task (task_id).
  - Grants are revoked on task completion.
  - Grants are prompt-time only. Nothing in this module writes to a store, so a
    granted snippet can never be persisted into the Worker's own memory.

The store is in-memory and process-local, which matches the lifetime of a
hand-off: RobotRegistry holds live instances in memory too, and a grant that
outlived a server restart would be a grant that outlived its task.
"""

from __future__ import annotations
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence


# How long a grant lives if the caller doesn't specify. Deliberately short — a
# hand-off is a single turn of conversation, not a session.
DEFAULT_GRANT_TTL_SEC = 180


@dataclass(frozen=True)
class DelegationGrant:
    """
    One Manager -> Worker context hand-off.

    snippet_ids is a tuple of explicit MemoryRecord.record_id values. An empty
    tuple is legal (the Manager had no accessible context to share) and grants
    nothing.
    """

    grant_id: str
    snippet_ids: tuple[str, ...]
    granted_to: str          # robot_id of the Worker
    granted_by: str          # robot_id of the Manager
    session_id: Optional[str]
    task_id: str
    expires_at: datetime

    def is_active(self, now: datetime) -> bool:
        """A grant with no expiry, or an expired one, is not active. Fail closed."""
        if self.expires_at is None:
            return False
        return now < self.expires_at

    def covers(self, record_id: str) -> bool:
        return record_id in self.snippet_ids


def new_grant(
    snippet_ids: Sequence[str],
    granted_to: str,
    granted_by: str,
    task_id: str,
    session_id: Optional[str] = None,
    ttl_sec: int = DEFAULT_GRANT_TTL_SEC,
    now: Optional[datetime] = None,
) -> DelegationGrant:
    """Build a grant expiring ttl_sec from now. Snippet IDs are frozen into a tuple."""
    now = now or datetime.now(timezone.utc)
    return DelegationGrant(
        grant_id=str(uuid.uuid4()),
        snippet_ids=tuple(snippet_ids),
        granted_to=granted_to,
        granted_by=granted_by,
        session_id=session_id,
        task_id=task_id,
        expires_at=now + timedelta(seconds=ttl_sec),
    )


class GrantStore:
    """
    Thread-safe registry of live grants.

    Thread safety matters: DelegationHandler runs hand-offs on daemon threads
    (delegation_handler.py handle() -> threading.Thread), so issue and revoke can
    race with a Worker's retrieval.
    """

    def __init__(self):
        self._by_task: dict[str, list[DelegationGrant]] = {}
        self._lock = threading.RLock()

    def issue(self, grant: DelegationGrant) -> DelegationGrant:
        with self._lock:
            self._by_task.setdefault(grant.task_id, []).append(grant)
        return grant

    def active_for(
        self,
        robot_id: str,
        now: Optional[datetime] = None,
        task_id: Optional[str] = None,
    ) -> list[DelegationGrant]:
        """
        Every unexpired grant issued to this robot, optionally narrowed to one task.

        Expired grants are never returned, so a caller cannot accidentally pass a
        stale grant into the policy.
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            if task_id is not None:
                candidates = list(self._by_task.get(task_id, ()))
            else:
                candidates = [g for gs in self._by_task.values() for g in gs]
        return [
            g for g in candidates
            if g.granted_to == robot_id and g.is_active(now)
        ]

    def revoke_task(self, task_id: str) -> int:
        """
        Drop every grant for a completed task. Returns how many were revoked.
        Called from the hand-off's finally block, so it must never raise.
        """
        with self._lock:
            revoked = self._by_task.pop(task_id, [])
        return len(revoked)

    def purge_expired(self, now: Optional[datetime] = None) -> int:
        """Housekeeping for grants whose task never signalled completion."""
        now = now or datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            for task_id in list(self._by_task):
                keep = [g for g in self._by_task[task_id] if g.is_active(now)]
                removed += len(self._by_task[task_id]) - len(keep)
                if keep:
                    self._by_task[task_id] = keep
                else:
                    del self._by_task[task_id]
        return removed

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._by_task.values())
