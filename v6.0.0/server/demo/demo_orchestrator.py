"""
demo/demo_orchestrator.py
==========================
Server-side state machine that drives the CARES lab demo.

State flow:
    IDLE ──start()──► RUNNING ──send step──► WAITING_ACK
                          ▲                       │
                          └──────ACK received──────┘
                          └──────manual_next()─────┘
                                                   ▼ timeout
                                                 ERROR

Any state ──pause()──► PAUSED ──resume()──► (previous running state)
Any state ──stop()───► IDLE
Last step ACKed      ──────────────────────► COMPLETED
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


# ── State machine states ───────────────────────────────────────────────────────

class DemoState(str, Enum):
    IDLE        = "idle"
    RUNNING     = "running"
    WAITING_ACK = "waiting_ack"
    PAUSED      = "paused"
    COMPLETED   = "completed"
    ERROR       = "error"


# ── Demo step definition (used in demo_script.py) ─────────────────────────────

@dataclass
class DemoStep:
    """
    One step in the demo sequence.

    Args:
        step_id     : Unique identifier (used for ACK matching).
        robot_id    : client_id of the robot that should execute this step.
                      Must match a registered + connected robot.
        text        : Text to speak (emotion tags like [GREETING] are preserved
                      so the client can drive Arduino/animation).
        require_ack : If True the orchestrator waits for an ACK before
                      advancing. Set False for fire-and-forget steps.
        timeout_sec : How long to wait for the ACK before entering ERROR state.
                      Use manual_next() or POST /demo/next to recover.
    """
    step_id:     str
    robot_id:    str
    text:        str
    require_ack: bool  = True
    timeout_sec: float = 30.0


# ── Orchestrator ──────────────────────────────────────────────────────────────

class DemoOrchestrator:
    """
    Drives the demo step-by-step.
    Thread-safe — all public methods acquire self._lock.
    """

    def __init__(self, ws_gateway: "WebSocketGateway"):
        self._ws      = ws_gateway
        self._script: list[DemoStep] = []
        self._state   = DemoState.IDLE
        self._idx     = 0
        self._lock    = threading.Lock()
        self._ack_event   = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()          # unset = paused, set = running
        self._runner: Optional[threading.Thread] = None

    # ── Script loading ────────────────────────────────────────────────────────

    def load_script(self, steps: list[DemoStep]):
        """Load (or reload) the demo script. Call before start()."""
        with self._lock:
            if self._state not in (DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR):
                logger.warning("[Demo] Cannot reload script while demo is running.")
                return
            self._script = list(steps)
            self._idx    = 0
        logger.info(f"[Demo] Script loaded — {len(steps)} steps.")

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self):
        """Start from the beginning (or where load_script last set idx=0)."""
        with self._lock:
            if self._state == DemoState.RUNNING:
                logger.warning("[Demo] Already running.")
                return
            if not self._script:
                logger.error("[Demo] No script loaded.")
                return
            self._idx   = 0
            self._state = DemoState.RUNNING
            self._ack_event.clear()
            self._pause_event.set()

        self._runner = threading.Thread(target=self._run_loop, daemon=True, name="demo-runner")
        self._runner.start()
        logger.info("[Demo] Started.")

    def stop(self):
        """Stop and reset to IDLE. Safe to call at any time."""
        with self._lock:
            self._state = DemoState.IDLE
            self._idx   = 0
        self._ack_event.set()      # unblock any waiting thread
        self._pause_event.set()    # unblock pause
        logger.info("[Demo] Stopped.")

    def pause(self):
        with self._lock:
            if self._state in (DemoState.RUNNING, DemoState.WAITING_ACK):
                self._state = DemoState.PAUSED
                self._pause_event.clear()
                logger.info("[Demo] Paused.")

    def resume(self):
        with self._lock:
            if self._state == DemoState.PAUSED:
                self._state = DemoState.WAITING_ACK if not self._ack_event.is_set() else DemoState.RUNNING
                self._pause_event.set()
                logger.info("[Demo] Resumed.")

    def manual_next(self):
        """Skip ACK wait and force-advance to the next step. Useful for recovery."""
        logger.info("[Demo] Manual next — skipping ACK wait.")
        self._ack_event.set()
        with self._lock:
            if self._state == DemoState.PAUSED:
                self._pause_event.set()

    # ── ACK reception (called by WebSocketGateway on type=="ack") ─────────────

    def receive_ack(self, step_id: str):
        with self._lock:
            current = self._script[self._idx] if self._idx < len(self._script) else None

        if current and current.step_id == step_id:
            logger.info(f"[Demo] ACK received for step '{step_id}'.")
            self._ack_event.set()
        else:
            logger.debug(f"[Demo] Ignored ACK for '{step_id}' "
                         f"(current: '{current.step_id if current else None}').")

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            step = self._script[self._idx] if self._idx < len(self._script) else None
            return {
                "state":    self._state.value,
                "step_idx": self._idx,
                "total":    len(self._script),
                "step_id":  step.step_id  if step else None,
                "robot_id": step.robot_id if step else None,
                "text":     step.text     if step else None,
            }

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _run_loop(self):
        while True:
            # Check pause (blocks here when paused)
            self._pause_event.wait()

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
            logger.info(f"[Demo] Step {idx + 1}/{len(self._script)}: "
                        f"'{step.step_id}' → {step.robot_id}")

            # Send the step to the target robot
            self._send_step(step)

            if step.require_ack:
                with self._lock:
                    self._state = DemoState.WAITING_ACK
                self._ack_event.clear()

                # Block until ACK, manual_next, stop, or timeout
                got_ack = self._ack_event.wait(timeout=step.timeout_sec)

                with self._lock:
                    if self._state == DemoState.IDLE:
                        return   # stop() was called
                    if not got_ack:
                        self._state = DemoState.ERROR
                        logger.error(f"[Demo] Timeout waiting for ACK on '{step.step_id}'. "
                                     "Use POST /demo/next to continue manually.")
                        return
                    self._state = DemoState.RUNNING

            # Advance
            with self._lock:
                self._idx += 1

    def _send_step(self, step: DemoStep):
        """Send a demo_step event directly to the robot (bypasses LLM)."""
        self._ws.send_to_robot(step.robot_id, {
            "event":       "demo_step",
            "step_id":     step.step_id,
            "text":        step.text,
            "require_ack": step.require_ack,
        })
        logger.info(f"[Demo] Sent step '{step.step_id}' to '{step.robot_id}'")
