"""
decision/recorder.py
====================
Where decisions and corrections go.

Structurally this is core/rbac/audit.py's BatchingAuditSink with two event types
instead of one, and for the same reason: a live demo in front of visitors must
never stall or crash because a logging database is slow or unreachable.

  Cheap.   Writes are queued and flushed in batches on a daemon thread. A
           decision never waits on the network.
  Safe.    A failing writer degrades to a warning. Losing a training label is
           bad; stopping the demo to report it is worse.

Decisions and corrections share ONE queue, in the order they were recorded.
An earlier version gave them a queue each and drained decisions first, reasoning
that this kept a correction from being written before the decision it
references. It does not: the two queues are filled concurrently, so a correction
enqueued just after a drain pass reaches the database before its parent, and the
foreign key rejects it. Observed in practice — corrections were dropped with
23503 while decisions wrote fine.

A single FIFO fixes it by construction. DecisionRecorder always records a
decision before any correction that names it, so insertion order IS a valid
write order. Batches are split into runs of consecutive same-type events, which
preserves that order while still writing in bulk.
"""

from __future__ import annotations

import atexit
import queue
import threading
import warnings
from typing import Callable, Optional, Protocol, Sequence

from decision.models import CorrectionEvent, DecisionEvent


class DecisionSink(Protocol):
    """Anything that can accept decision and correction events."""

    def record(self, event: DecisionEvent) -> None: ...

    def record_correction(self, event: CorrectionEvent) -> None: ...

    def flush(self) -> None: ...


class NullDecisionSink:
    """Discards everything. The default, and what tests use unless asserting."""

    def record(self, event: DecisionEvent) -> None:
        return None

    def record_correction(self, event: CorrectionEvent) -> None:
        return None

    def flush(self) -> None:
        return None


class MemoryDecisionSink:
    """
    Keeps events in lists. Used by tests, and a usable local fallback when no
    database is configured — a demo run still yields inspectable labels.
    """

    def __init__(self):
        self.decisions: list[DecisionEvent] = []
        self.corrections: list[CorrectionEvent] = []
        self._lock = threading.Lock()

    def record(self, event: DecisionEvent) -> None:
        with self._lock:
            self.decisions.append(event)

    def record_correction(self, event: CorrectionEvent) -> None:
        with self._lock:
            self.corrections.append(event)

    def flush(self) -> None:
        return None

    def decisions_by_mechanism(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self.decisions:
                counts[e.mechanism] = counts.get(e.mechanism, 0) + 1
            return counts

    def corrections_by_point(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self.corrections:
                counts[e.decision_point] = counts.get(e.decision_point, 0) + 1
            return counts


class BatchingDecisionSink:
    """
    Queues events and flushes them in batches from a daemon thread.

    `decision_writer` and `correction_writer` each receive a list of events and
    persist it. Any exception either raises is swallowed into a warning.

    If a queue is full the event is dropped with a warning rather than blocking
    the caller — see the module docstring.
    """

    def __init__(
        self,
        decision_writer: Callable[[Sequence[DecisionEvent]], None],
        correction_writer: Callable[[Sequence[CorrectionEvent]], None],
        batch_size: int = 25,
        flush_interval_sec: float = 5.0,
        max_queue: int = 10_000,
    ):
        self._decision_writer = decision_writer
        self._correction_writer = correction_writer
        self._batch_size = batch_size
        self._flush_interval = flush_interval_sec

        # One queue, insertion-ordered — see the module docstring for why two
        # was wrong.
        self._queue: "queue.Queue" = queue.Queue(maxsize=max_queue)

        self._stop = threading.Event()
        self._warned_full = False

        self._thread = threading.Thread(
            target=self._run, name="demo-decisions", daemon=True
        )
        self._thread.start()
        atexit.register(self.shutdown)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, event: DecisionEvent) -> None:
        self._put(event)

    def record_correction(self, event: CorrectionEvent) -> None:
        self._put(event)

    def flush(self) -> None:
        """Drain and write everything currently queued, synchronously."""
        while self._drain(block=False):
            pass

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the worker and make a best-effort final flush."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._thread.join(timeout=timeout)
        except RuntimeError:
            pass
        self.flush()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _put(self, event) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Warn once per sink, not once per dropped event.
            if not self._warned_full:
                self._warned_full = True
                warnings.warn(
                    "[decision.recorder] queue full — dropping decision events. "
                    "The sink is not keeping up; the demo is unaffected.",
                    RuntimeWarning,
                )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain(block=True)
        # One last pass so events queued during shutdown are not lost.
        self.flush()

    def _drain(self, block: bool) -> bool:
        """Write one batch. Returns True if anything was written."""
        batch = self._collect(block)
        if not batch:
            return False

        # Split into runs of consecutive same-type events and write each run in
        # order. Grouping by type instead would reorder them and reintroduce the
        # foreign-key failure this design exists to prevent.
        run: list = [batch[0]]
        for event in batch[1:]:
            if type(event) is type(run[0]):
                run.append(event)
                continue
            self._write(run)
            run = [event]
        self._write(run)
        return True

    def _write(self, run: list) -> None:
        if not run:
            return
        is_decision = isinstance(run[0], DecisionEvent)
        writer = self._decision_writer if is_decision else self._correction_writer
        kind = "decision" if is_decision else "correction"
        try:
            writer(run)
        except Exception as e:
            warnings.warn(
                f"[decision.recorder] {kind} write failed, "
                f"{len(run)} event(s) dropped: {e}",
                RuntimeWarning,
            )

    def _collect(self, block: bool) -> list:
        batch: list = []
        try:
            if block:
                try:
                    batch.append(self._queue.get(timeout=self._flush_interval))
                except queue.Empty:
                    return batch
            while len(batch) < self._batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
        except Exception:  # pragma: no cover - queue ops are not expected to raise
            return batch
        return batch


class DecisionRecorder:
    """
    Thin facade the gateway and orchestrator hold.

    It exists so callers never touch a sink directly, and so the "which decision
    is currently live for this step" bookkeeping lives in one place. That
    bookkeeping is what turns an operator's button press into a *correction of a
    specific decision* rather than an orphan row.

    Live decisions are keyed by (step_id, decision_point), not step_id alone.
    Several points are decided on the same step — a visitor turn produces a
    QA_ADVANCE, a PLAN_REVISE and a QA_ROUTE — and keying on the step alone made
    every correction attach to whichever happened to be logged last. An operator
    clicking "Move On" would then be recorded as overriding the routing choice,
    and the correction rate per mechanism, which is the number this layer exists
    to produce, measured nothing.
    """

    def __init__(self, sink: Optional[DecisionSink] = None):
        self._sink: DecisionSink = sink if sink is not None else NullDecisionSink()
        self._lock = threading.Lock()
        # (step_id, decision_point) -> most recent decision_id for that pair
        self._live: dict[tuple, str] = {}

    @property
    def sink(self) -> DecisionSink:
        return self._sink

    def record(self, event: DecisionEvent) -> str:
        """Persist a decision and remember it as the live one for its
        (step, decision point)."""
        if event.step_id:
            with self._lock:
                self._live[(event.step_id, event.decision_point)] = event.decision_id
        self._sink.record(event)
        return event.decision_id

    def record_correction(self, event: CorrectionEvent) -> str:
        self._sink.record_correction(event)
        return event.correction_id

    def live_decision_id(
        self, step_id: Optional[str], decision_point: Optional[str] = None,
    ) -> Optional[str]:
        """The last decision of this point logged for this step, if any.

        `decision_point` is optional only so an older caller cannot break; pass
        it. Without it this falls back to the most recent decision of ANY point
        on the step, which is the behaviour that made correction rates
        meaningless in the first place.
        """
        if not step_id:
            return None
        with self._lock:
            if decision_point is not None:
                return self._live.get((step_id, decision_point))
            for (sid, _point), did in reversed(list(self._live.items())):
                if sid == step_id:
                    return did
            return None

    def clear(self) -> None:
        """Forget live decisions — called when a demo run starts or stops."""
        with self._lock:
            self._live.clear()

    def flush(self) -> None:
        self._sink.flush()
