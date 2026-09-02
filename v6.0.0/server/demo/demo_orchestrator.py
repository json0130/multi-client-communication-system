"""
demo/demo_orchestrator.py
==========================
Server-side state machine that drives the CARES lab demo.

State flow:
    IDLE ──start()──► RUNNING ──send step──► WAITING_ACK
                          ▲                       │ ACK / manual_next
                          └───────────────────────┘
                          │
                          └──(step has qa_window=True)──► QA_WINDOW
                                 ▲ resume() / manual_next()     │ timeout / qa_end()
                                 └───────────────────────────────┘

Any state ──qa_interrupt()──► QA_WINDOW  (ad-hoc Q&A from dashboard)
Any state ──pause()──────────► PAUSED ──resume()──► (previous state)
Any state ──stop()───────────► IDLE
Last step done ──────────────► COMPLETED
Timeout on WAITING_ACK ──────► ERROR  (recover with manual_next)
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Sequence, TYPE_CHECKING

from decision.models import (
    Action,
    DecisionPoint,
    PlanOp,
    PlanOpKind,
    build_correction,
)

if TYPE_CHECKING:
    from gateway.websocket_gateway import WebSocketGateway

logger = logging.getLogger(__name__)


# ── States ─────────────────────────────────────────────────────────────────────

class DemoState(str, Enum):
    IDLE        = "idle"
    RUNNING     = "running"
    WAITING_ACK = "waiting_ack"
    QA_WINDOW   = "qa_window"    # timed Q&A pause between demo steps
    PAUSED      = "paused"
    COMPLETED   = "completed"
    ERROR       = "error"


# ── Step definition ────────────────────────────────────────────────────────────

@dataclass
class DemoStep:
    """
    One step in the demo sequence.

    Args:
        step_id     : Unique identifier — used for ACK matching and logs.
        robot_id    : client_id of the robot that executes this step.
        text        : Speech text (emotion tags like [GREETING] preserved).
        require_ack : Wait for ACK before advancing. False = fire-and-forget.
        timeout_sec : Seconds to wait for ACK before entering ERROR.
        qa_window   : After this step's ACK, open a Q&A window automatically.
        qa_timeout  : Seconds the Q&A window stays open before auto-advancing.
                      Set to 0 to require manual close (POST /demo/next).

    Block metadata (block_robot_id / role) exists so revise_script() can act on
    "ChatBox's part of the tour" without pattern-matching step_id strings. It is
    optional: a hand-written script that omits it still runs normally, but its
    steps cannot be skipped or compressed by robot, only dropped wholesale.
    """
    step_id:     str
    robot_id:    str
    text:        str
    require_ack: bool  = True
    timeout_sec: float = 30.0
    qa_window:   bool  = False
    qa_timeout:  float = 60.0
    generate:    bool  = False   # if True, client generates speech via LLM instead of verbatim

    # Which project robot's block this step belongs to. None for opening/closing.
    block_robot_id: Optional[str] = None
    # This step's function within its block. See StepRole.
    role:           str = ""


class StepRole:
    """
    What a step is for. Plain string constants rather than an Enum so an
    untagged script (role="") stays valid and comparisons never raise.

    COMPRESS drops INTRO/HANDOFF/GREETING/PROMPT and keeps PROJECT/QA — the
    research content survives, the social scaffolding is what gets trimmed.
    """

    OPENING    = "opening"      # greeting, lab_intro, overview
    INTRO      = "intro"        # guide introduces the project concept
    HANDOFF    = "handoff"      # guide points at the robot
    GREETING   = "greeting"     # robot says hello
    PROMPT     = "prompt"       # guide asks the robot to present
    PROJECT    = "project"      # the robot presents — never trimmed
    QA         = "qa"           # Q&A window
    TRANSITION = "transition"   # guide signs off, moves on
    CLOSING    = "closing"      # wrap_up, open_floor — survives DROP_REMAINING

    # Dropped by COMPRESS. PROJECT and QA are deliberately absent.
    COMPRESSIBLE = frozenset({INTRO, HANDOFF, GREETING, PROMPT})


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DemoOrchestrator:
    """
    Drives the demo step by step. All public methods are thread-safe.
    """

    def __init__(
        self,
        ws_gateway: "WebSocketGateway",
        transition_delay: float = 0.5,
        recorder=None,
        session_context: Optional[Callable[[], dict]] = None,
        duration_sink: Optional[Callable[[str, dict], None]] = None,
    ):
        self._ws     = ws_gateway
        self._script: list[DemoStep] = []

        self._state  = DemoState.IDLE
        self._idx    = 0

        # Pause between consecutive steps — gives TTS and hardware time to settle.
        # Set to 0 to disable. Skipped immediately on stop() or manual_next().
        self._transition_delay = transition_delay

        # ── Decision logging ──────────────────────────────────────────────────
        # Optional. Without a recorder the orchestrator behaves exactly as before
        # — a demo must never depend on the training pipeline being wired up.
        # session_context() supplies scenario_id/session_id, which the
        # orchestrator has no other way to know and which are what make a
        # correction row joinable to rbac_audit_log.
        self._recorder = recorder
        self._session_context = session_context

        # Timing. Called as duration_sink("step"|"qa", row). Optional, and
        # failures are swallowed — a tour must not stop because a stopwatch did.
        # Scripted steps and Q&A windows go to SEPARATE streams and are never
        # averaged together: step length is a property of the content, Q&A
        # length is a property of the operator and the group, and mixing them
        # makes every step estimate worse as more runs arrive.
        self._duration_sink = duration_sink
        self._run_id: Optional[str] = None
        self._step_started_at: Optional[float] = None

        # Run clock. Started by start(), read by get_status() so Observation has
        # a real budget rather than an estimate made at decision time.
        self._started_at: Optional[float] = None
        self._time_budget_sec: Optional[float] = None
        self._revisions: list[dict] = []

        self._lock           = threading.Lock()
        self._ack_event      = threading.Event()   # set when ACK arrives or manual_next()
        self._pause_event    = threading.Event()   # cleared when paused
        self._pause_event.set()
        self._qa_end_event   = threading.Event()   # set to close Q&A window early
        self._advance_event  = threading.Event()   # set by stop/manual_next to skip transition delay
        self._skip_next_qa   = False               # set by manual_next to skip upcoming qa_window
        self._qa_closed_by: Optional[str] = None   # who ended the current window

        self._runner: Optional[threading.Thread] = None

        # Holds the LLM-generated text for the current step (replaces raw instruction
        # for display in get_status and as what's actually sent to the robot).
        self._current_step_text: Optional[str] = None

    # ── Script ────────────────────────────────────────────────────────────────

    def load_script(self, steps: list[DemoStep]):
        with self._lock:
            if self._state not in (DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR):
                logger.warning("[Demo] Cannot reload while running. Stop first.")
                return
            self._script = list(steps)
            self._idx    = 0
        logger.info(f"[Demo] Script loaded — {len(steps)} steps.")

    # ── Controls ──────────────────────────────────────────────────────────────

    def start(self, robot_ids: list = None, time_budget_sec: Optional[float] = None):
        """
        Start the demo.  If *robot_ids* is provided (non-empty list), a script is
        built dynamically: first entry is the guide/host, the rest are project
        robots in the requested order.  If omitted, the pre-loaded script is used.

        *time_budget_sec* is how long the whole tour is supposed to take. It is
        optional, and without it PLAN_REVISE can still act on an explicit visitor
        request but never on the clock — an inferred "we are running late" needs
        something to be late against.
        """
        with self._lock:
            if self._state == DemoState.RUNNING:
                logger.warning("[Demo] Already running.")
                return
            if robot_ids:
                from demo.demo_script import build_script  # local import avoids circular dependency
                guide    = robot_ids[0]
                projects = robot_ids[1:]
                self._script = build_script(guide, projects)
                logger.info(f"[Demo] Dynamic script built — guide={guide}, projects={projects}")
            if not self._script:
                logger.error("[Demo] No script loaded.")
                return
            self._idx                = 0
            self._state              = DemoState.RUNNING
            self._skip_next_qa       = False
            self._current_step_text  = None
            self._started_at         = time.time()
            self._time_budget_sec    = time_budget_sec
            self._run_id             = f"run-{int(self._started_at)}"
            self._revisions          = []
            self._ack_event.clear()
            self._qa_end_event.clear()
            self._pause_event.set()
        if self._recorder is not None:
            self._recorder.clear()
        self._runner = threading.Thread(target=self._run_loop, daemon=True, name="demo-runner")
        self._runner.start()
        logger.info(
            f"[Demo] Started."
            + (f" Budget: {time_budget_sec:.0f}s." if time_budget_sec else "")
        )

    def stop(self):
        with self._lock:
            self._state             = DemoState.IDLE
            self._idx               = 0
            self._skip_next_qa      = False
            self._current_step_text = None
            self._started_at        = None
        self._ack_event.set()
        self._qa_end_event.set()
        self._pause_event.set()
        self._advance_event.set()   # skip any in-progress transition delay
        if self._recorder is not None:
            # Flush before the process can exit — the corrections from this run
            # are the training signal and are not reconstructable.
            self._recorder.clear()
            self._recorder.flush()
        logger.info("[Demo] Stopped.")

    def pause(self, source: str = "auto", reason: str = ""):
        """
        Pause at the current step.

        `source` is recorded in the log line only. Pausing is not a correction:
        it is not an alternative to any action in the decision space, it just
        stops the clock. manual_next / qa_end / qa_interrupt are the ones that
        express disagreement with a decision.
        """
        with self._lock:
            if self._state in (DemoState.RUNNING, DemoState.WAITING_ACK, DemoState.QA_WINDOW):
                self._state = DemoState.PAUSED
                self._pause_event.clear()
                logger.info(f"[Demo] Paused ({source}).")

    def resume(self, source: str = "auto", reason: str = ""):
        with self._lock:
            if self._state == DemoState.PAUSED:
                # Determine correct state to return to
                self._state = DemoState.RUNNING
                self._pause_event.set()
                logger.info(f"[Demo] Resumed ({source}).")

    def manual_next(self, source: str = "auto", reason: str = ""):
        """
        Force-advance past current wait (ACK timeout, Q&A window, or transition delay).

        When an operator triggers this, it is a supervisor correction: the demo
        should have moved on by now and did not. That is recorded against
        whatever QA_ADVANCE decision is live for the current step — or as an
        orphan correction if none was, which is itself the label.
        """
        with self._lock:
            # Only pre-arm the skip flag when we are NOT already inside a Q&A window.
            # If we ARE inside one, _qa_end_event.set() below is enough to close it;
            # setting the flag here would incorrectly skip the *next* scripted Q&A step.
            if self._state not in (DemoState.QA_WINDOW, DemoState.IDLE,
                                   DemoState.COMPLETED, DemoState.ERROR):
                self._skip_next_qa = True
        logger.info(f"[Demo] Manual next ({source}).")
        self._record_correction(DecisionPoint.QA_ADVANCE, Action.advance(), source, reason)
        self._ack_event.set()
        self._qa_end_event.set()
        self._pause_event.set()
        self._advance_event.set()   # skip transition delay if currently waiting

    def qa_interrupt(self, message: str = "", source: str = "auto", reason: str = ""):
        """
        Ad-hoc Q&A — works at ANY point during the demo (running, waiting_ack,
        paused, or even during the scripted Q&A window).

        The run loop unblocks from whatever wait it is in (ACK wait, transition
        delay, pause) and enters QA_WINDOW.  Call qa_end() or manual_next() to
        close and resume exactly where the demo left off (same step index).

        From an operator this is the opposite correction to manual_next: the demo
        kept going when it should have stopped and listened. Recorded as
        QA_ADVANCE → stay.
        """
        with self._lock:
            if self._state == DemoState.QA_WINDOW:
                logger.info("[Demo] Already in Q&A window.")
                return
            if self._state not in (
                DemoState.RUNNING, DemoState.WAITING_ACK, DemoState.PAUSED
            ):
                logger.warning(f"[Demo] qa_interrupt ignored in state {self._state}")
                return
            self._state = DemoState.QA_WINDOW
            self._qa_end_event.clear()

        # Unblock every possible blocking call in _run_loop so it wakes up
        # immediately and detects the QA_WINDOW state.
        self._ack_event.set()       # unblocks _ack_event.wait()
        self._advance_event.set()   # unblocks transition-delay wait
        self._pause_event.set()     # unblocks pause if currently paused

        if message:
            default_robot = self._script[0].robot_id if self._script else None
            if default_robot:
                self._ws.send_to_robot(default_robot, {
                    "event":       "demo_step",
                    "step_id":     "_qa_interrupt",
                    "text":        message,
                    "require_ack": False,
                })
        logger.info(f"[Demo] Entered ad-hoc Q&A window (interrupt, {source}).")
        self._record_correction(DecisionPoint.QA_ADVANCE, Action.stay(), source, reason)

    def qa_end(self, source: str = "auto", reason: str = ""):
        """End the current Q&A window and advance the demo."""
        self._qa_closed_by = source
        self._qa_end_event.set()
        logger.info(f"[Demo] Q&A window closed ({source}).")
        self._record_correction(DecisionPoint.QA_ADVANCE, Action.advance(), source, reason)

    # ── ACK reception ─────────────────────────────────────────────────────────

    def receive_ack(self, step_id: str):
        with self._lock:
            current = self._script[self._idx] if self._idx < len(self._script) else None
        if current and current.step_id == step_id:
            logger.info(f"[Demo] ACK for '{step_id}'.")
            self._ack_event.set()
        else:
            logger.debug(f"[Demo] Ignored ACK '{step_id}' "
                         f"(expected '{current.step_id if current else None}').")

    # ── Plan revision ─────────────────────────────────────────────────────────

    def revise_script(
        self,
        ops: Sequence[PlanOp],
        source: str = "auto",
        reason: str = "",
    ) -> dict:
        """
        Change the *remaining* tour: skip a project, reorder, trim, extend a Q&A,
        or cut to the wrap-up.

        THE INVARIANT: only steps at index > self._idx are ever touched. The step
        currently executing always completes, and `self._idx` keeps pointing at
        the same step object afterwards. Everything below re-splices the tail and
        then re-derives the index by identity, never by arithmetic — a revision
        that shifted the index would strand the ACK the run loop is waiting on
        and silently skip or repeat a step in front of visitors.

        Returns a summary dict: {applied: [...], ignored: [...], total: n}.
        """
        applied: list[dict] = []
        ignored: list[dict] = []

        with self._lock:
            if self._state in (DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR):
                logger.warning(f"[Demo] revise_script ignored in state {self._state}.")
                return {"applied": [], "ignored": [o.payload() for o in ops],
                        "total": len(self._script)}

            idx = self._idx
            current = self._script[idx] if idx < len(self._script) else None
            head = self._script[: idx + 1]
            tail = self._script[idx + 1:]

            for op in ops:
                new_tail, note = self._apply_op(op, tail, current)
                if new_tail is None:
                    ignored.append({**op.payload(), "why": note})
                    continue
                tail = new_tail
                applied.append({**op.payload(), "effect": note})

            self._script = head + tail

            # Re-derive the index by identity. If `current` is somehow gone the
            # arithmetic index is still correct, because head was never altered.
            if current is not None:
                try:
                    self._idx = self._script.index(current)
                except ValueError:      # pragma: no cover - head is never mutated
                    self._idx = idx

            if applied:
                self._revisions.append({
                    "at_step": current.step_id if current else None,
                    "source": source,
                    "reason": reason,
                    "ops": applied,
                })
            total = len(self._script)

        if applied:
            logger.info(
                f"[Demo] Script revised ({source}): "
                f"{', '.join(a['kind'] for a in applied)} — now {total} steps."
            )
        if ignored:
            logger.info(f"[Demo] Revision ops ignored: {ignored}")

        if applied:
            self._record_correction(
                DecisionPoint.PLAN_REVISE, Action.revise(ops), source, reason
            )
        return {"applied": applied, "ignored": ignored, "total": total}

    def _apply_op(
        self,
        op: PlanOp,
        tail: list[DemoStep],
        current: Optional[DemoStep],
    ) -> tuple[Optional[list[DemoStep]], str]:
        """
        Apply one op to the remaining steps. Caller holds the lock.

        Returns (new_tail, note), or (None, why) when the op does not apply —
        an op naming a robot whose block has already been presented is a no-op,
        not an error: by the time a visitor says "skip that one", it may already
        be over.
        """
        if op.kind is PlanOpKind.DROP_REMAINING:
            kept = [s for s in tail if s.role == StepRole.CLOSING]
            if len(kept) == len(tail):
                return None, "nothing left to drop"
            return kept, f"dropped {len(tail) - len(kept)} step(s), kept closing"

        if not op.robot_id:
            return None, "op requires robot_id"

        block = [s for s in tail if s.block_robot_id == op.robot_id]
        if not block and op.kind is not PlanOpKind.EXTEND_QA:
            return None, f"no remaining steps for '{op.robot_id}'"

        if op.kind is PlanOpKind.SKIP:
            return (
                [s for s in tail if s.block_robot_id != op.robot_id],
                f"skipped {len(block)} step(s)",
            )

        if op.kind is PlanOpKind.COMPRESS:
            dropped = [
                s for s in block if s.role in StepRole.COMPRESSIBLE
            ]
            if not dropped:
                return None, f"'{op.robot_id}' already compressed"
            dropped_ids = {id(s) for s in dropped}
            return (
                [s for s in tail if id(s) not in dropped_ids],
                f"trimmed {len(dropped)} step(s)",
            )

        if op.kind is PlanOpKind.SET_QA_BUDGET:
            return self._set_qa_budget(op.robot_id, op.seconds, tail)

        if op.kind is PlanOpKind.EXTEND_QA:
            return self._extend_qa(op.robot_id, tail, current)

        if op.kind is PlanOpKind.REORDER:
            return self._reorder(op.robot_id, op.position, tail, current)

        return None, f"unhandled op '{op.kind.value}'"

    def _set_qa_budget(
        self,
        robot_id: str,
        seconds: Optional[float],
        tail: list[DemoStep],
    ) -> tuple[Optional[list[DemoStep]], str]:
        """
        Allocate this project's upcoming Q&A window a time budget.

        The first rung of the compression ladder. Q&A is the largest share of
        tour time and the least noticed when shortened — a visitor remembers a
        robot that never spoke, not a question round that ended a minute early.
        So a planner facing 15 minutes for a 25-minute tour tightens windows
        before it drops anything.

        seconds <= 0 restores manual advance (qa_timeout = 0), which is the
        right default when there is no time pressure at all.
        """
        if seconds is None:
            return None, "set_qa_budget requires seconds"
        budget = max(0.0, float(seconds))
        out, changed = [], 0
        for s in tail:
            if s.block_robot_id == robot_id and s.role == StepRole.QA:
                out.append(replace(s, qa_timeout=budget, qa_window=True))
                changed += 1
            else:
                out.append(s)
        if not changed:
            return None, f"no upcoming Q&A window for '{robot_id}'"
        how = f"{budget:.0f}s" if budget > 0 else "manual advance"
        return out, f"Q&A for '{robot_id}' set to {how}"

    def _extend_qa(
        self,
        robot_id: str,
        tail: list[DemoStep],
        current: Optional[DemoStep],
    ) -> tuple[Optional[list[DemoStep]], str]:
        """
        Give a project more Q&A time.

        If its block is still ahead, widen the Q&A window it already has. If the
        block is behind us — the usual case, since a visitor asks for more after
        hearing something — insert a fresh Q&A step at the front of the tail so
        it opens next, rather than making them wait for the tour to end.
        """
        for i, s in enumerate(tail):
            if s.block_robot_id == robot_id and s.role == StepRole.QA:
                new_tail = list(tail)
                new_tail[i] = replace(s, qa_timeout=0.0, qa_window=True)
                return new_tail, f"widened upcoming Q&A for '{robot_id}'"

        guide = current.robot_id if current else robot_id
        extra = DemoStep(
            step_id=f"qa_extend_{robot_id}_{len(tail)}",
            robot_id=guide,
            text=(
                f"The visitors want to hear more about {robot_id}'s project. "
                f"Invite them to ask {robot_id} further questions. "
                "1-2 sentences. Use [DEFAULT]."
            ),
            generate=True,
            timeout_sec=50,
            qa_window=True,
            qa_timeout=0.0,
            block_robot_id=robot_id,
            role=StepRole.QA,
        )
        return [extra] + list(tail), f"inserted extra Q&A for '{robot_id}'"

    def _reorder(
        self,
        robot_id: str,
        position: Optional[int],
        tail: list[DemoStep],
        current: Optional[DemoStep],
    ) -> tuple[Optional[list[DemoStep]], str]:
        """
        Move a project block among the blocks that have not started yet.

        `position` indexes the remaining project blocks, not the flat step list —
        callers think in projects, and the step expansion is this module's
        business. Out-of-range positions clamp rather than fail: a visitor
        request should not 400 because there were fewer projects left than they
        assumed.

        The block currently being presented is pinned to the front and cannot be
        moved. Its own steps are still in the tail — a block's farewell comes
        after its Q&A — and letting another project jump ahead of them would
        have the guide thank a robot the visitors stopped hearing from two
        projects ago.
        """
        pinned_id = current.block_robot_id if current else None
        pinned = [s for s in tail if s.block_robot_id == pinned_id] if pinned_id else []
        movable = tail[len(pinned):] if pinned else list(tail)

        # Defensive: pinned steps must be contiguous at the front of the tail.
        # If they are not, the script was hand-edited into a shape this cannot
        # reason about, and reordering it would do more harm than refusing.
        if pinned and any(s.block_robot_id == pinned_id for s in movable):
            return None, f"'{pinned_id}' block is not contiguous — refusing to reorder"

        order: list[str] = []
        for s in movable:
            if s.block_robot_id and s.block_robot_id not in order:
                order.append(s.block_robot_id)
        if robot_id not in order:
            if robot_id == pinned_id:
                return None, f"'{robot_id}' is currently presenting and cannot be moved"
            return None, f"'{robot_id}' is not in the remaining blocks"

        target = 0 if position is None else max(0, min(position, len(order) - 1))
        order.remove(robot_id)
        order.insert(target, robot_id)

        blocks: dict[str, list[DemoStep]] = {r: [] for r in order}
        loose: list[DemoStep] = []
        for s in movable:
            if s.block_robot_id in blocks:
                blocks[s.block_robot_id].append(s)
            else:
                loose.append(s)

        rebuilt: list[DemoStep] = list(pinned)
        for r in order:
            rebuilt.extend(blocks[r])
        # Closing steps have no block and must stay at the end.
        rebuilt.extend(loose)
        return rebuilt, f"moved '{robot_id}' to position {target}"

    # ── Correction recording ──────────────────────────────────────────────────

    def _record_correction(
        self,
        point: DecisionPoint,
        corrected_to: Action,
        source: str,
        reason: str,
    ) -> None:
        """
        Log a supervisor override.

        Only 'operator' is recorded. The system calling qa_end() on its own is a
        decision, already logged where it was made — recording it here too would
        double-count it as its own correction and make the correction rate
        meaningless.
        """
        if self._recorder is None or source != "operator":
            return
        try:
            with self._lock:
                idx = self._idx
                step = self._script[idx] if idx < len(self._script) else None
            step_id = step.step_id if step else None

            ctx = {}
            if self._session_context is not None:
                ctx = self._session_context() or {}

            self._recorder.record_correction(build_correction(
                point=point,
                corrected_to=corrected_to,
                source=source,
                reason=reason,
                decision_id=self._recorder.live_decision_id(step_id, point.value),
                supervisor_id=ctx.get("supervisor_id"),
                step_id=step_id,
                step_idx=idx,
                scenario_id=ctx.get("scenario_id"),
                session_id=ctx.get("session_id"),
            ))
        except Exception as e:
            # A logging failure must never propagate into a live demo.
            logger.warning(f"[Demo] Could not record correction: {e}")

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            idx    = self._idx
            total  = len(self._script)
            step   = self._script[idx] if idx < total else None
            # Show the LLM-generated text when available; fall back to raw instruction
            # Return None while generation is in progress so the dashboard waits
            # rather than displaying the raw instruction prompt.
            display_text = self._current_step_text
            elapsed = (time.time() - self._started_at) if self._started_at else 0.0
            return {
                "state":       self._state.value,
                "step_idx":    idx,
                "total":       total,
                "step_id":     step.step_id  if step else None,
                "robot_id":    step.robot_id if step else None,
                "text":        display_text,
                "qa_window":   step.qa_window if step else False,
                # Run clock — feeds Observation's time budget, and lets the
                # dashboard show projected overrun before an operator has to
                # notice it themselves.
                "elapsed_sec":     round(elapsed, 1),
                "time_budget_sec": self._time_budget_sec,
                "revisions":       list(self._revisions),
                # Full step list for the dashboard timeline
                "steps": [
                    {
                        "step_id":   s.step_id,
                        "robot_id":  s.robot_id,
                        "text":      s.text[:80] + ("..." if len(s.text) > 80 else ""),
                        "qa_window": s.qa_window,
                        "role":      s.role,
                        "block_robot_id": s.block_robot_id,
                    }
                    for s in self._script
                ],
            }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _wait_if_interrupted_qa(self) -> bool:
        """
        If a qa_interrupt() was called while the loop was in a blocking wait,
        the state is now QA_WINDOW.  Block here until the operator closes it
        (qa_end / manual_next), then restore RUNNING so the loop continues
        from the same step index without skipping anything.

        Returns False if the demo was stopped (should exit the loop).
        """
        with self._lock:
            if self._state != DemoState.QA_WINDOW:
                return self._state not in (
                    DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR
                )

        logger.info("[Demo] Ad-hoc Q&A — waiting for operator to close...")
        self._notify_window("open")
        self._qa_end_event.wait()          # blocks until qa_end() / manual_next()
        self._notify_window("close")

        with self._lock:
            if self._state in (DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR):
                return False
            self._state = DemoState.RUNNING

        logger.info("[Demo] Ad-hoc Q&A closed — resuming demo.")
        return True

    def _record_step_duration(self, step: DemoStep) -> None:
        """Time a scripted step, from send to ACK.

        Q&A steps are excluded here and recorded by _open_qa_window instead. A
        Q&A step's ACK time is the guide finishing its invitation, not the
        window; conflating the two would put operator-driven variance into the
        content averages, which is the whole thing this split avoids.
        """
        if self._duration_sink is None or step.role == StepRole.QA:
            return
        with self._lock:
            started = self._step_started_at
            text = self._current_step_text or ""
        if not started:
            return
        try:
            self._duration_sink("step", {
                "run_id": self._run_id or "unknown",
                "step_id": step.step_id,
                "robot_id": step.robot_id,
                "block_robot_id": step.block_robot_id,
                "role": step.role or "",
                "seconds": round(time.time() - started, 3),
                "text_chars": len(text),
                "generated": bool(step.generate),
            })
        except Exception as e:
            logger.warning(f"[Demo] step duration not recorded: {e}")

    def _record_qa_duration(self, step: DemoStep, seconds: float,
                            closed_by: str, turns: int = 0) -> None:
        """Time a Q&A window, into its own stream.

        budget_sec is None when the window was manual-advance only, which stays
        the default for an unhurried tour. A number means the planner allocated
        one, and that is what makes "how often does the operator overrun the
        allocation" answerable.
        """
        if self._duration_sink is None:
            return
        try:
            self._duration_sink("qa", {
                "run_id": self._run_id or "unknown",
                "step_id": step.step_id,
                "block_robot_id": step.block_robot_id,
                "seconds": round(seconds, 3),
                "turns": turns,
                "budget_sec": step.qa_timeout if step.qa_timeout > 0 else None,
                "closed_by": closed_by,
            })
        except Exception as e:
            logger.warning(f"[Demo] Q&A duration not recorded: {e}")

    def _notify_window(self, edge: str) -> None:
        """
        Tell the gateway a Q&A window opened or closed so its run tracker can
        time the window and count turns within it.

        Best-effort and duck-typed: the orchestrator predates the decision layer
        and must keep working against a gateway that has no such hook.
        """
        hook = getattr(self._ws, f"on_qa_window_{edge}", None)
        if hook is None:
            return
        try:
            hook()
        except Exception as e:
            logger.warning(f"[Demo] Q&A window {edge} hook failed: {e}")

    def _run_loop(self):
        while True:
            self._pause_event.wait()  # blocks when paused

            # A qa_interrupt() during the transition delay lands here.
            if not self._wait_if_interrupted_qa():
                return

            with self._lock:
                state = self._state
                idx   = self._idx

            if state in (DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR):
                return

            if idx >= len(self._script):
                with self._lock:
                    self._state = DemoState.COMPLETED
                logger.info("[Demo] All steps completed.")
                return

            step = self._script[idx]
            logger.info(f"[Demo] Step {idx+1}/{len(self._script)}: "
                        f"'{step.step_id}' → {step.robot_id}")

            self._send_step(step)

            if step.require_ack:
                with self._lock:
                    self._state = DemoState.WAITING_ACK
                self._ack_event.clear()

                got_ack = self._ack_event.wait(timeout=step.timeout_sec)
                if got_ack:
                    self._record_step_duration(step)

                with self._lock:
                    if self._state == DemoState.IDLE:
                        return
                    if self._state == DemoState.QA_WINDOW:
                        pass   # handled by _wait_if_interrupted_qa below
                    elif not got_ack:
                        self._state = DemoState.ERROR
                        logger.error(f"[Demo] Timeout on '{step.step_id}'. "
                                     "POST /demo/next to continue.")
                        return
                    else:
                        self._state = DemoState.RUNNING

            # A qa_interrupt() during the ACK wait lands here.
            if not self._wait_if_interrupted_qa():
                return

            # Q&A window configured on this step (scripted pause)
            if step.qa_window:
                self._open_qa_window(step)

            with self._lock:
                if self._state == DemoState.IDLE:
                    return
                self._idx += 1
                self._current_step_text = None

            # Pause between steps so TTS/hardware settle before the next cue.
            # Skipped immediately if stop() or manual_next() or qa_interrupt() fires.
            if self._transition_delay > 0:
                self._advance_event.clear()
                self._advance_event.wait(timeout=self._transition_delay)

    def _send_step(self, step: DemoStep):
        text = step.text
        if step.generate:
            logger.info(f"[Demo] Calling generate_demo_step for '{step.step_id}' → {step.robot_id}")
            generated = self._ws.generate_demo_step(step.robot_id, step.text)
            if generated and generated != step.text:
                text = generated
                logger.info(f"[Demo] Generated text ready for '{step.step_id}'")
            else:
                logger.warning(f"[Demo] Generation returned fallback for '{step.step_id}' "
                               f"— speaking instruction as-is")
        with self._lock:
            self._current_step_text = text
            self._step_started_at = time.time()
        self._ws.send_to_robot(step.robot_id, {
            "event":       "demo_step",
            "step_id":     step.step_id,
            "text":        text,
            "require_ack": step.require_ack,
        })
        logger.info(f"[Demo] Sent '{step.step_id}' to '{step.robot_id}'")

    def _open_qa_window(self, step: DemoStep):
        """Block in QA_WINDOW state until timeout or manual close."""
        with self._lock:
            # If manual_next() already fired before we entered the window, skip it entirely.
            if self._skip_next_qa:
                self._skip_next_qa = False
                logger.info("[Demo] Q&A window skipped — manual advance already pending.")
                return
            self._state = DemoState.QA_WINDOW
        self._qa_end_event.clear()

        timeout = step.qa_timeout if step.qa_timeout > 0 else None
        logger.info(f"[Demo] Q&A window open "
                    f"({'auto-closes in ' + str(timeout) + 's' if timeout else 'manual close only'}).")

        self._notify_window("open")
        opened_at = time.time()
        closed_naturally = self._qa_end_event.wait(timeout=timeout)
        elapsed = time.time() - opened_at
        self._notify_window("close")

        # 'timeout' means the allocated budget ran out with nobody closing it —
        # distinct from an operator or the policy deciding to move on, and the
        # distinction is what tells you whether budgets are being respected.
        self._record_qa_duration(
            step, elapsed,
            closed_by=(self._qa_closed_by or "unknown") if closed_naturally else "timeout")
        self._qa_closed_by = None

        with self._lock:
            if self._state == DemoState.IDLE:
                return
            self._state = DemoState.RUNNING
        logger.info(f"[Demo] Q&A window closed after {elapsed:.0f}s.")
