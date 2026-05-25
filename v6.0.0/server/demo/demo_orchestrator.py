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
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

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
    """
    step_id:     str
    robot_id:    str
    text:        str
    require_ack: bool  = True
    timeout_sec: float = 30.0
    qa_window:   bool  = False
    qa_timeout:  float = 60.0
    generate:    bool  = False   # if True, client generates speech via LLM instead of verbatim


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DemoOrchestrator:
    """
    Drives the demo step by step. All public methods are thread-safe.
    """

    def __init__(self, ws_gateway: "WebSocketGateway", transition_delay: float = 1.5):
        self._ws     = ws_gateway
        self._script: list[DemoStep] = []

        self._state  = DemoState.IDLE
        self._idx    = 0

        # Pause between consecutive steps — gives TTS and hardware time to settle.
        # Set to 0 to disable. Skipped immediately on stop() or manual_next().
        self._transition_delay = transition_delay

        self._lock           = threading.Lock()
        self._ack_event      = threading.Event()   # set when ACK arrives or manual_next()
        self._pause_event    = threading.Event()   # cleared when paused
        self._pause_event.set()
        self._qa_end_event   = threading.Event()   # set to close Q&A window early
        self._advance_event  = threading.Event()   # set by stop/manual_next to skip transition delay
        self._skip_next_qa   = False               # set by manual_next to skip upcoming qa_window

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

    def start(self, robot_ids: list = None):
        """
        Start the demo.  If *robot_ids* is provided (non-empty list), a script is
        built dynamically: first entry is the guide/host, the rest are project
        robots in the requested order.  If omitted, the pre-loaded script is used.
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
            self._ack_event.clear()
            self._qa_end_event.clear()
            self._pause_event.set()
        self._runner = threading.Thread(target=self._run_loop, daemon=True, name="demo-runner")
        self._runner.start()
        logger.info("[Demo] Started.")

    def stop(self):
        with self._lock:
            self._state             = DemoState.IDLE
            self._idx               = 0
            self._skip_next_qa      = False
            self._current_step_text = None
        self._ack_event.set()
        self._qa_end_event.set()
        self._pause_event.set()
        self._advance_event.set()   # skip any in-progress transition delay
        logger.info("[Demo] Stopped.")

    def pause(self):
        with self._lock:
            if self._state in (DemoState.RUNNING, DemoState.WAITING_ACK, DemoState.QA_WINDOW):
                self._state = DemoState.PAUSED
                self._pause_event.clear()
                logger.info("[Demo] Paused.")

    def resume(self):
        with self._lock:
            if self._state == DemoState.PAUSED:
                # Determine correct state to return to
                self._state = DemoState.RUNNING
                self._pause_event.set()
                logger.info("[Demo] Resumed.")

    def manual_next(self):
        """Force-advance past current wait (ACK timeout, Q&A window, or transition delay)."""
        with self._lock:
            # Only pre-arm the skip flag when we are NOT already inside a Q&A window.
            # If we ARE inside one, _qa_end_event.set() below is enough to close it;
            # setting the flag here would incorrectly skip the *next* scripted Q&A step.
            if self._state not in (DemoState.QA_WINDOW, DemoState.IDLE,
                                   DemoState.COMPLETED, DemoState.ERROR):
                self._skip_next_qa = True
        logger.info("[Demo] Manual next.")
        self._ack_event.set()
        self._qa_end_event.set()
        self._pause_event.set()
        self._advance_event.set()   # skip transition delay if currently waiting

    def qa_interrupt(self, message: str = ""):
        """
        Ad-hoc Q&A — works at ANY point during the demo (running, waiting_ack,
        paused, or even during the scripted Q&A window).

        The run loop unblocks from whatever wait it is in (ACK wait, transition
        delay, pause) and enters QA_WINDOW.  Call qa_end() or manual_next() to
        close and resume exactly where the demo left off (same step index).
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
        logger.info("[Demo] Entered ad-hoc Q&A window (interrupt).")

    def qa_end(self):
        """End the current Q&A window and advance the demo."""
        self._qa_end_event.set()
        logger.info("[Demo] Q&A window closed.")

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
            return {
                "state":       self._state.value,
                "step_idx":    idx,
                "total":       total,
                "step_id":     step.step_id  if step else None,
                "robot_id":    step.robot_id if step else None,
                "text":        display_text,
                "qa_window":   step.qa_window if step else False,
                # Full step list for the dashboard timeline
                "steps": [
                    {
                        "step_id":   s.step_id,
                        "robot_id":  s.robot_id,
                        "text":      s.text[:80] + ("..." if len(s.text) > 80 else ""),
                        "qa_window": s.qa_window,
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
        self._qa_end_event.wait()          # blocks until qa_end() / manual_next()

        with self._lock:
            if self._state in (DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR):
                return False
            self._state = DemoState.RUNNING

        logger.info("[Demo] Ad-hoc Q&A closed — resuming demo.")
        return True

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

        self._qa_end_event.wait(timeout=timeout)

        with self._lock:
            if self._state == DemoState.IDLE:
                return
            self._state = DemoState.RUNNING
        logger.info("[Demo] Q&A window closed.")
