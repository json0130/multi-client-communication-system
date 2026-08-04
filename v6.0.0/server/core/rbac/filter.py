"""
core/rbac/filter.py
===================
The enforcement filter the memory layer calls.

This sits between the stores and the Composite System Prompt (the paper's Fig. 4
ordering): every read path — FAISS episodic retrieval, key-point lookup,
interaction-log queries — passes its candidate records through here before
anything reaches prompt assembly.

Cleared records carry a clearance stamp. `ClearedRecord` can only be constructed
by this module, so a read path that forgets to filter cannot hand raw rows to
the prompt builder and have them accepted — prompt assembly calls
`assert_cleared()` and fails loudly instead of leaking silently.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional, Sequence

from core.rbac.audit import AuditSink, NullAuditSink, build_event
from core.rbac.grants import DelegationGrant
from core.rbac.models import Decision, MemoryRecord, RobotIdentity
from core.rbac.policy import can_read


class ClearanceError(RuntimeError):
    """
    Raised when something that has not been through the RBAC filter reaches
    prompt assembly. This is a programming error, not a permission error — it
    means a read path bypassed filter.py.
    """


# Module-private sentinel. Nothing outside this module can obtain it, so
# ClearedRecord cannot be forged elsewhere.
_CLEARANCE_TOKEN = object()


@dataclass(frozen=True)
class ClearedRecord:
    """
    A record that has been granted access, plus the decision that granted it.

    Construct via RBACFilter only. Direct construction raises ClearanceError.
    """

    record: MemoryRecord
    decision: Decision
    _token: object = None

    def __post_init__(self):
        if self._token is not _CLEARANCE_TOKEN:
            raise ClearanceError(
                "ClearedRecord cannot be constructed directly — it is the RBAC "
                "clearance stamp. Route the read through core.rbac.filter.RBACFilter."
            )

    @property
    def text(self) -> str:
        return self.record.content

    @property
    def record_id(self) -> str:
        return self.record.record_id


def make_record_id(store: str, key: object) -> str:
    """
    Build a stable, store-scoped record ID.

    Delegation grants name records by ID, so these must be reproducible across
    the Manager's retrieval and the Worker's lookup within one hand-off.
    """
    return f"{store}:{key}"


class RBACFilter:
    """
    Applies the policy to candidate records and audits every decision.

    Stateless apart from its audit sink, so one instance can be shared across
    threads — which it is: hand-offs run on daemon threads.
    """

    def __init__(self, audit_sink: Optional[AuditSink] = None):
        self._audit = audit_sink or NullAuditSink()

    @property
    def audit_sink(self) -> AuditSink:
        return self._audit

    def decide(
        self,
        requester: RobotIdentity,
        record: MemoryRecord,
        grants: Sequence[DelegationGrant] = (),
        now: Optional[datetime] = None,
        store: str = "unknown",
    ) -> Decision:
        """Decide on a single record and audit the outcome."""
        now = now or datetime.now(timezone.utc)
        try:
            decision = can_read(requester, record, grants, now)
        except Exception as e:
            # The policy is pure and should not raise, but if it ever does the
            # answer is deny, not crash.
            decision = Decision(False, "policy_error")
            _warn_policy_error(e)

        self._audit_safely(requester, record, decision, store, now)
        return decision

    def filter_records(
        self,
        requester: RobotIdentity,
        records: Iterable[MemoryRecord],
        grants: Sequence[DelegationGrant] = (),
        now: Optional[datetime] = None,
        store: str = "unknown",
    ) -> list[ClearedRecord]:
        """
        Return only the records this requester may read, each stamped.

        Order is preserved, so a relevance-ranked candidate list stays ranked.
        """
        now = now or datetime.now(timezone.utc)
        cleared: list[ClearedRecord] = []
        for record in records:
            decision = self.decide(requester, record, grants, now, store)
            if decision.allowed:
                cleared.append(
                    ClearedRecord(record=record, decision=decision, _token=_CLEARANCE_TOKEN)
                )
        return cleared

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audit_safely(self, requester, record, decision, store, now) -> None:
        """A failing audit sink must never break retrieval."""
        try:
            self._audit.record(build_event(requester, record, decision, store, now))
        except Exception as e:
            _warn_audit_error(e)


def assert_cleared(items: Sequence[object], where: str) -> list[ClearedRecord]:
    """
    Belt-and-braces check at the fusion point.

    The composite prompt builder calls this before fusing anything into the
    system prompt. If a future read path is added that skips the filter, this
    raises rather than letting unfiltered memory reach the LLM.
    """
    for item in items or ():
        if not isinstance(item, ClearedRecord):
            raise ClearanceError(
                f"{where}: refusing to fuse a record with no RBAC clearance stamp "
                f"(got {type(item).__name__}). Every read path must go through "
                f"core.rbac.filter.RBACFilter before prompt assembly."
            )
    return list(items or ())


def _warn_policy_error(e: Exception) -> None:
    import warnings
    warnings.warn(f"[rbac.filter] policy raised, denying: {e}", RuntimeWarning)


def _warn_audit_error(e: Exception) -> None:
    import warnings
    warnings.warn(f"[rbac.filter] audit sink error (ignored): {e}", RuntimeWarning)
