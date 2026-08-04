"""
core/rbac/audit.py
==================
Decision logging.

Every access decision is recorded — requester, record, allow/deny, reason,
matched grant, timestamp — to a sink separate from the interaction log. This is
what makes "enforcement effectiveness under real-world conditions" measurable:
denial counts by robot and by reason are a query, not a guess.

Two properties matter more than completeness here:

  Cheap.   Writes are queued and flushed in batches on a background thread, so a
           retrieval never blocks on the audit store.
  Safe.    A failing sink degrades to a warning. Retrieval must not break
           because the audit database is unreachable — that would turn an
           observability outage into a service outage.

This module stays application-agnostic: BatchingAuditSink takes a `writer`
callable. The Supabase-backed writer lives in data/rbac_audit_repo.py.
"""

from __future__ import annotations
import atexit
import queue
import threading
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol, Sequence

from core.rbac.models import Decision, MemoryRecord, RobotIdentity


@dataclass(frozen=True)
class AuditEvent:
    """One recorded access decision."""

    requester_robot_id: str
    record_id: str
    allowed: bool
    reason: str
    matched_grant_id: Optional[str]
    scenario_id: Optional[str]
    session_id: Optional[str]
    store: str                  # which memory store was queried, e.g. "faiss"
    decided_at: datetime

    def as_row(self) -> dict:
        d = asdict(self)
        d["decided_at"] = self.decided_at.isoformat()
        return d


def build_event(
    requester: RobotIdentity,
    record: MemoryRecord,
    decision: Decision,
    store: str,
    now: Optional[datetime] = None,
) -> AuditEvent:
    return AuditEvent(
        requester_robot_id=requester.robot_id,
        record_id=record.record_id,
        allowed=decision.allowed,
        reason=decision.reason,
        matched_grant_id=decision.matched_grant_id,
        scenario_id=requester.scenario_id,
        session_id=requester.session_id,
        store=store,
        decided_at=now or datetime.now(timezone.utc),
    )


class AuditSink(Protocol):
    """Anything that can accept decision events."""

    def record(self, event: AuditEvent) -> None: ...

    def flush(self) -> None: ...


class NullAuditSink:
    """Discards everything. The default, and what tests use unless asserting on audit."""

    def record(self, event: AuditEvent) -> None:
        return None

    def flush(self) -> None:
        return None


class MemoryAuditSink:
    """
    Keeps events in a list. Used by tests and by tools/check_rbac.py, and useful
    as a local fallback when no database is configured.
    """

    def __init__(self):
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self.events.append(event)

    def flush(self) -> None:
        return None

    def denials_by_reason(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self.events:
                if not e.allowed:
                    counts[e.reason] = counts.get(e.reason, 0) + 1
            return counts

    def denials_by_robot(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self.events:
                if not e.allowed:
                    counts[e.requester_robot_id] = counts.get(e.requester_robot_id, 0) + 1
            return counts


class BatchingAuditSink:
    """
    Queues events and flushes them in batches from a daemon thread.

    `writer` receives a list of AuditEvent and persists it. Any exception it
    raises is swallowed into a warning — see the module docstring.

    If the queue is full the event is dropped with a warning rather than blocking
    the caller; losing an audit line is strictly better than stalling retrieval.
    """

    def __init__(
        self,
        writer: Callable[[Sequence[AuditEvent]], None],
        batch_size: int = 25,
        flush_interval_sec: float = 5.0,
        max_queue: int = 10_000,
    ):
        self._writer = writer
        self._batch_size = batch_size
        self._flush_interval = flush_interval_sec
        self._queue: "queue.Queue[AuditEvent]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._warned_full = False

        self._thread = threading.Thread(
            target=self._run, name="rbac-audit", daemon=True
        )
        self._thread.start()
        atexit.register(self.shutdown)

    def record(self, event: AuditEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Warn once per sink, not once per dropped event.
            if not self._warned_full:
                self._warned_full = True
                warnings.warn(
                    "[rbac.audit] queue full — dropping audit events. "
                    "The audit sink is not keeping up; retrieval is unaffected.",
                    RuntimeWarning,
                )

    def flush(self) -> None:
        """Drain and write everything currently queued, synchronously."""
        self._drain(block=False)

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the worker and make a best-effort final flush."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._thread.join(timeout=timeout)
        except RuntimeError:
            pass
        self._drain(block=False)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain(block=True)
        # One last pass so events queued during shutdown are not lost.
        self._drain(block=False)

    def _drain(self, block: bool) -> None:
        batch: list[AuditEvent] = []
        try:
            if block:
                try:
                    batch.append(self._queue.get(timeout=self._flush_interval))
                except queue.Empty:
                    return
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
        except Exception:  # pragma: no cover - queue ops are not expected to raise
            return

        if not batch:
            return

        try:
            self._writer(batch)
        except Exception as e:
            warnings.warn(
                f"[rbac.audit] sink write failed, {len(batch)} event(s) dropped: {e}",
                RuntimeWarning,
            )
