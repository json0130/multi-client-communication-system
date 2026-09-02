"""
decision/observation.py
=======================
Builds the state the policy sees.

Everything a policy is allowed to know is assembled here and nowhere else. That
constraint is the point: when the learned policy replaces HeuristicPolicy, the
two must be looking at exactly the same inputs or the comparison in the paper
means nothing.

Two sources feed an Observation:

  DemoOrchestrator.get_status()  position in the script, run clock, budget
  DemoRunTracker                 conversational state the orchestrator has no
                                 view of — who spoke, how often, which project
                                 drew questions

The tracker lives here rather than in the gateway because it is state-space
bookkeeping, not transport.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from decision.models import Observation

# Prefixes and marks that make a visitor turn a question. Shared with
# HeuristicPolicy so "was this a question" is decided once.
QUESTION_PREFIXES = (
    "what ", "how ", "why ", "where ", "when ", "who ", "which ",
    "can ", "could ", "tell me", "explain", "describe",
    "is there", "are there", "do you", "does it",
)


def looks_like_question(text: str) -> bool:
    """
    Heuristic pre-filter, lifted verbatim from websocket_gateway._looks_like_question
    so the behaviour is preserved exactly while gaining a single home.
    """
    t = text.lower().strip()
    if "?" in t:
        return True
    return t.startswith(QUESTION_PREFIXES)


class DemoRunTracker:
    """
    Per-run conversational bookkeeping.

    Thread-safe: WebSocket callbacks arrive on the client's reader thread while
    the orchestrator's runner thread advances steps, and both touch this.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        self._window_opened_at: Optional[float] = None
        self._turns_in_window = 0
        self._last_speaker_id: Optional[str] = None
        self._last_robot_utterance = ""
        self._last_user_utterance = ""
        # robot_id -> {"turns": int, "questions": int}
        self._engagement: dict[str, dict[str, int]] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_run(self) -> None:
        with self._lock:
            self._reset()

    def open_window(self) -> None:
        with self._lock:
            self._window_opened_at = time.time()
            self._turns_in_window = 0

    def close_window(self) -> None:
        with self._lock:
            self._window_opened_at = None
            self._turns_in_window = 0

    # ── Turns ─────────────────────────────────────────────────────────────────

    def note_visitor_turn(self, addressed_robot_id: str, text: str) -> None:
        """
        A visitor spoke to `addressed_robot_id`. Counts toward that robot's
        engagement — the signal behind "they clearly want more of project B".
        """
        with self._lock:
            self._turns_in_window += 1
            self._last_speaker_id = "visitor"
            self._last_user_utterance = text or ""
            entry = self._engagement.setdefault(
                addressed_robot_id, {"turns": 0, "questions": 0}
            )
            entry["turns"] += 1
            if looks_like_question(text or ""):
                entry["questions"] += 1

    def note_robot_turn(self, robot_id: str, text: str) -> None:
        with self._lock:
            self._last_speaker_id = robot_id
            self._last_robot_utterance = text or ""

    # ── Read ──────────────────────────────────────────────────────────────────

    def seconds_in_window(self) -> float:
        with self._lock:
            if self._window_opened_at is None:
                return 0.0
            return time.time() - self._window_opened_at

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "seconds_in_window": self.seconds_in_window(),
                "turns_in_window": self._turns_in_window,
                "last_speaker_id": self._last_speaker_id,
                "last_robot_utterance": self._last_robot_utterance,
                "last_user_utterance": self._last_user_utterance,
                "engagement_by_robot": {
                    k: dict(v) for k, v in self._engagement.items()
                },
            }


# ── Builder ───────────────────────────────────────────────────────────────────

def _access_level_str(value: object) -> Optional[str]:
    """
    Render an access level for the log without ever raising.

    parse_access_level fails closed by design, but a decision row is not an
    access check — a malformed level must be *visible* in the audit trail, not
    the cause of a crash mid-demo. So an unparseable value is recorded verbatim.
    """
    if value is None:
        return None
    try:
        from core.rbac import parse_access_level
        return parse_access_level(value).value
    except Exception:
        return str(value)


def _guide_and_presenter(status: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Work out who is hosting and who is currently presenting.

    The guide is the robot on step 0 — build_script() always puts the host
    first. The presenter is the most recent non-guide robot at or before the
    current index, because during a Q&A step the step's own robot_id is the
    guide, not the project robot the visitors are actually asking about.
    """
    steps = status.get("steps") or []
    if not steps:
        return None, None

    guide = steps[0].get("robot_id")
    idx = min(status.get("step_idx", 0), len(steps) - 1)
    for s in reversed(steps[: idx + 1]):
        rid = s.get("robot_id")
        if rid and rid != guide:
            return guide, rid
    return guide, None


def _projected_overrun(
    elapsed: float, budget: Optional[float], step_idx: int, total: int
) -> Optional[float]:
    """
    Seconds the run is expected to exceed its budget by, or None without a budget.

    Deliberately the simplest defensible estimator: average the time actually
    spent per completed step and extend it over the steps that remain. It is
    crude early in a run, which is correct — PLAN_REVISE should not fire on two
    steps of evidence.
    """
    if budget is None or step_idx <= 0:
        return None
    avg = elapsed / step_idx
    remaining = max(total - step_idx, 0)
    return (elapsed + avg * remaining) - budget


def build_observation(
    status: dict,
    registry: Any,
    tracker: DemoRunTracker,
    decider: Any = None,
    user_utterance: str = "",
) -> Observation:
    """
    Assemble the state at a decision point.

    `decider` is the RobotInstance whose identity the decision is made under —
    normally the robot that received the visitor's turn. Its access level is
    recorded so a decision row joins to rbac_audit_log on session_id, which is
    what makes "did this policy widen context exposure?" a query.
    """
    snap = tracker.snapshot()
    guide_id, presenter_id = _guide_and_presenter(status)

    total = int(status.get("total") or 0)
    idx = int(status.get("step_idx") or 0)
    elapsed = float(status.get("elapsed_sec") or 0.0)
    budget = status.get("time_budget_sec")
    budget = float(budget) if budget is not None else None

    peers = []
    try:
        for inst in registry.get_all():
            # role comes off the RBAC identity — RobotInstance keeps it private
            # and refreshes it from the DB before each chat, so this is the one
            # reading that is guaranteed current.
            peers.append({
                "client_id": inst.client_id,
                "robot_name": inst.robot_name,
                # Kept structured, not flattened — the relational KG attaches here.
                "robot_role": getattr(inst.identity, "role", None),
                "access_level": _access_level_str(getattr(inst, "access_level", None)),
            })
    except Exception as e:
        # A registry hiccup must not stop a decision from being made or logged.
        print(f"[decision.observation] peer snapshot failed: {e}")

    identity = getattr(decider, "identity", None)

    return Observation(
        step_id=status.get("step_id"),
        step_idx=idx,
        total_steps=total,
        steps_remaining=max(total - idx, 0),
        demo_state=str(status.get("state") or "idle"),

        seconds_in_window=snap["seconds_in_window"],
        turns_in_window=snap["turns_in_window"],
        last_speaker_id=snap["last_speaker_id"],
        user_utterance=user_utterance or snap["last_user_utterance"],
        last_robot_utterance=snap["last_robot_utterance"],

        elapsed_sec=elapsed,
        time_budget_sec=budget,
        projected_overrun_sec=_projected_overrun(elapsed, budget, idx, total),

        engagement_by_robot=snap["engagement_by_robot"],

        connected_peers=tuple(peers),
        guide_robot_id=guide_id,
        presenting_robot_id=presenter_id,

        decider_robot_id=getattr(decider, "client_id", None),
        decider_access_level=_access_level_str(getattr(decider, "access_level", None)),
        scenario_id=getattr(identity, "scenario_id", None),
        session_id=getattr(identity, "session_id", None),
    )
