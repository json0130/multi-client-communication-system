"""
decision/models.py
==================
The value types the demo decision layer operates on.

Same discipline as core/rbac/models.py: plain frozen dataclasses, no I/O, no
knowledge of Flask, Supabase or the orchestrator. The gateway adapts live state
into an Observation before asking a Policy for an Action, which is what keeps
this package reusable for the simulator later — a rollout and a live demo must
produce the same row shape or the training data is worthless.

Four decision points, one action space:

  QA_ADVANCE           close this Q&A window, or keep it open
  QA_ROUTE             which robot should answer this visitor turn
  PLAN_REVISE          change the *remaining* script (time pressure, interest)
  DELEGATE_INITIATIVE  robot-lead / human-lead / hybrid

DELEGATE_INITIATIVE has no logic behind it yet. It is declared here, and its
column values are accepted by the schema, so adding it later is a code change
and not a migration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Sequence


# ── Decision points ───────────────────────────────────────────────────────────

class DecisionPoint(str, Enum):
    """Which question is being answered."""

    QA_ADVANCE = "qa_advance"
    QA_ROUTE = "qa_route"
    PLAN_REVISE = "plan_revise"
    DELEGATE_INITIATIVE = "delegate_initiative"   # reserved — no policy yet


# ── Actions ───────────────────────────────────────────────────────────────────

class ActionKind(str, Enum):
    ADVANCE = "advance"                   # close the Q&A window, move to next step
    STAY = "stay"                         # keep the window open
    ROUTE_TO = "route_to"                 # this robot answers
    GUIDE_INTERJECT = "guide_interject"   # the guide takes the floor
    REVISE = "revise"                     # apply plan operations


class PlanOpKind(str, Enum):
    SKIP = "skip"                     # drop a project robot's remaining steps
    REORDER = "reorder"               # move a project robot to a new position
    COMPRESS = "compress"             # keep the project talk, drop greeting/prompt
    EXTEND_QA = "extend_qa"           # give this project another Q&A window
    DROP_REMAINING = "drop_remaining" # jump to the wrap-up
    # Allocate a Q&A window a time budget in seconds. THE FIRST RUNG of the
    # compression ladder: Q&A is the largest share of tour time, and a window
    # that runs a little short is something visitors do not notice, while a
    # missing robot is. Reach for this before dropping anything.
    #
    # `seconds` = 0 restores manual advance, which stays the default for an
    # unhurried tour — a budget is a response to time pressure, not a policy.
    SET_QA_BUDGET = "set_qa_budget"


@dataclass(frozen=True)
class PlanOp:
    """
    One edit to the remaining script.

    `robot_id` names the project block to act on. DROP_REMAINING ignores it.
    `position` is only meaningful for REORDER — an index into the remaining
    project blocks, not into the flat step list, because callers think in
    projects and the orchestrator owns the step expansion.
    `seconds` is only meaningful for SET_QA_BUDGET.
    """

    kind: PlanOpKind
    robot_id: Optional[str] = None
    position: Optional[int] = None
    seconds: Optional[float] = None

    def payload(self) -> dict:
        d: dict[str, Any] = {"kind": self.kind.value}
        if self.robot_id is not None:
            d["robot_id"] = self.robot_id
        if self.position is not None:
            d["position"] = self.position
        if self.seconds is not None:
            d["seconds"] = self.seconds
        return d

    @classmethod
    def from_payload(cls, data: dict) -> "PlanOp":
        """Parse one op from an API request body. Raises ValueError on junk."""
        try:
            kind = PlanOpKind(str(data.get("kind", "")).strip().lower())
        except ValueError:
            raise ValueError(
                f"Unknown plan op {data.get('kind')!r}. "
                f"Valid: {', '.join(k.value for k in PlanOpKind)}."
            )
        position = data.get("position")
        seconds = data.get("seconds")
        return cls(
            kind=kind,
            robot_id=data.get("robot_id") or None,
            position=int(position) if position is not None else None,
            seconds=max(0.0, float(seconds)) if seconds is not None else None,
        )


@dataclass(frozen=True)
class Action:
    """
    What the policy decided to do.

    Constructed through the classmethods rather than directly, so an Action can
    never carry an ops list with the wrong kind.
    """

    kind: ActionKind
    robot_id: Optional[str] = None
    ops: tuple[PlanOp, ...] = ()

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def advance(cls) -> "Action":
        return cls(ActionKind.ADVANCE)

    @classmethod
    def stay(cls) -> "Action":
        return cls(ActionKind.STAY)

    @classmethod
    def route_to(cls, robot_id: str) -> "Action":
        return cls(ActionKind.ROUTE_TO, robot_id=robot_id)

    @classmethod
    def guide_interject(cls, robot_id: Optional[str] = None) -> "Action":
        return cls(ActionKind.GUIDE_INTERJECT, robot_id=robot_id)

    @classmethod
    def revise(cls, ops: Sequence[PlanOp]) -> "Action":
        return cls(ActionKind.REVISE, ops=tuple(ops))

    # ── Serialization ─────────────────────────────────────────────────────────

    def payload(self) -> dict:
        d: dict[str, Any] = {"kind": self.kind.value}
        if self.robot_id is not None:
            d["robot_id"] = self.robot_id
        if self.ops:
            d["ops"] = [o.payload() for o in self.ops]
        return d

    @classmethod
    def from_payload(cls, data: dict) -> "Action":
        """Parse an action from an API request body. Raises ValueError on junk."""
        try:
            kind = ActionKind(str(data.get("kind", "")).strip().lower())
        except ValueError:
            raise ValueError(
                f"Unknown action {data.get('kind')!r}. "
                f"Valid: {', '.join(k.value for k in ActionKind)}."
            )
        return cls(
            kind=kind,
            robot_id=data.get("robot_id") or None,
            ops=tuple(PlanOp.from_payload(o) for o in data.get("ops") or ()),
        )

    def describe(self) -> str:
        """Short human-readable form for logs and the dashboard."""
        if self.kind is ActionKind.ROUTE_TO:
            return f"route_to({self.robot_id})"
        if self.kind is ActionKind.REVISE:
            inner = ", ".join(
                f"{o.kind.value}({o.robot_id or ''})".replace("()", "")
                for o in self.ops
            )
            return f"revise[{inner}]"
        return self.kind.value


# ── Observation ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    """
    Everything the policy is allowed to look at. This *is* the state space —
    if a signal is not here, no policy can use it, so adding a feature later
    means adding a field here and nowhere else.

    Two fields carry more weight than they appear to:

    `connected_peers` stays a list of dicts, never a flattened prompt string.
    The relational knowledge graph planned for the next phase attaches to these
    entries; collapsing them into text now would have to be undone.

    `time_budget_sec` / `projected_overrun_sec` are what give PLAN_REVISE any
    grounds to fire. Without a declared budget a visitor saying "we're running
    out of time" is unfalsifiable, so the budget is an input to the run, not a
    guess made at decision time.
    """

    # Position in the script
    step_id: Optional[str] = None
    step_idx: int = 0
    total_steps: int = 0
    steps_remaining: int = 0
    demo_state: str = "idle"

    # The current Q&A window
    seconds_in_window: float = 0.0
    turns_in_window: int = 0
    last_speaker_id: Optional[str] = None
    user_utterance: str = ""
    last_robot_utterance: str = ""

    # Time budget
    elapsed_sec: float = 0.0
    time_budget_sec: Optional[float] = None
    projected_overrun_sec: Optional[float] = None

    # Visitor interest, per project robot: {robot_id: {"turns": n, "questions": n}}
    engagement_by_robot: dict = field(default_factory=dict)

    # The team. Entries: {client_id, robot_name, robot_role, access_level}
    connected_peers: tuple = ()
    guide_robot_id: Optional[str] = None
    presenting_robot_id: Optional[str] = None

    # RBAC context — same identifiers as rbac_audit_log so the tables join
    decider_robot_id: Optional[str] = None
    decider_access_level: Optional[str] = None
    scenario_id: Optional[str] = None
    session_id: Optional[str] = None

    def as_dict(self) -> dict:
        """JSON-safe dict for the observation column."""
        return {
            "step_id": self.step_id,
            "step_idx": self.step_idx,
            "total_steps": self.total_steps,
            "steps_remaining": self.steps_remaining,
            "demo_state": self.demo_state,
            "seconds_in_window": round(self.seconds_in_window, 2),
            "turns_in_window": self.turns_in_window,
            "last_speaker_id": self.last_speaker_id,
            "user_utterance": self.user_utterance,
            "last_robot_utterance": self.last_robot_utterance,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "time_budget_sec": self.time_budget_sec,
            "projected_overrun_sec": (
                round(self.projected_overrun_sec, 2)
                if self.projected_overrun_sec is not None else None
            ),
            "engagement_by_robot": dict(self.engagement_by_robot),
            "connected_peers": [dict(p) for p in self.connected_peers],
            "guide_robot_id": self.guide_robot_id,
            "presenting_robot_id": self.presenting_robot_id,
            "decider_robot_id": self.decider_robot_id,
            "decider_access_level": self.decider_access_level,
            "scenario_id": self.scenario_id,
            "session_id": self.session_id,
        }


# ── Events ────────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class DecisionEvent:
    """
    One recorded decision. Mirrors core.rbac.AuditEvent so both streams can be
    written by the same batching machinery and read with the same repo shape.

    `mechanism` is part of the contract, not a debugging aid — it names which
    rule produced the action ('advance_phrase', 'llm_classifier', 'learned_v1',
    …). Every comparison in the paper is a GROUP BY on this column, so it is
    never allowed to be null.
    """

    decision_id: str
    decision_point: str
    action_kind: str
    action_payload: dict
    mechanism: str
    observation: dict
    decider_robot_id: Optional[str]
    decider_access_level: Optional[str]
    matched_grant_id: Optional[str]
    scenario_id: Optional[str]
    session_id: Optional[str]
    step_id: Optional[str]
    step_idx: int
    decided_at: datetime

    def as_row(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_point": self.decision_point,
            "action_kind": self.action_kind,
            "action_payload": self.action_payload,
            "mechanism": self.mechanism,
            "observation": self.observation,
            "decider_robot_id": self.decider_robot_id,
            "decider_access_level": self.decider_access_level,
            "matched_grant_id": self.matched_grant_id,
            "scenario_id": self.scenario_id,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "step_idx": self.step_idx,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass(frozen=True)
class CorrectionEvent:
    """
    A supervisor overriding what the system did — or doing something the system
    never offered.

    `decision_id` is nullable on purpose. An operator clicking "Move On" during a
    window where no policy decision was logged is still a label: it says the
    window should have closed at that moment. Requiring a parent row would throw
    away exactly the corrections that matter most.
    """

    correction_id: str
    decision_id: Optional[str]
    decision_point: str
    corrected_to_kind: str
    corrected_to_payload: dict
    source: str                  # 'operator' | 'auto' | 'policy'
    reason: str
    supervisor_id: Optional[str]
    step_id: Optional[str]
    step_idx: int
    scenario_id: Optional[str]
    session_id: Optional[str]
    corrected_at: datetime

    def as_row(self) -> dict:
        return {
            "correction_id": self.correction_id,
            "decision_id": self.decision_id,
            "decision_point": self.decision_point,
            "corrected_to_kind": self.corrected_to_kind,
            "corrected_to_payload": self.corrected_to_payload,
            "source": self.source,
            "reason": self.reason,
            "supervisor_id": self.supervisor_id,
            "step_id": self.step_id,
            "step_idx": self.step_idx,
            "scenario_id": self.scenario_id,
            "session_id": self.session_id,
            "corrected_at": self.corrected_at.isoformat(),
        }


def build_decision(
    point: DecisionPoint,
    action: Action,
    mechanism: str,
    observation: Observation,
    matched_grant_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> DecisionEvent:
    return DecisionEvent(
        decision_id=_new_id(),
        decision_point=point.value,
        action_kind=action.kind.value,
        action_payload=action.payload(),
        mechanism=mechanism,
        observation=observation.as_dict(),
        decider_robot_id=observation.decider_robot_id,
        decider_access_level=observation.decider_access_level,
        matched_grant_id=matched_grant_id,
        scenario_id=observation.scenario_id,
        session_id=observation.session_id,
        step_id=observation.step_id,
        step_idx=observation.step_idx,
        decided_at=now or datetime.now(timezone.utc),
    )


def build_correction(
    point: DecisionPoint,
    corrected_to: Action,
    source: str,
    reason: str = "",
    decision_id: Optional[str] = None,
    supervisor_id: Optional[str] = None,
    step_id: Optional[str] = None,
    step_idx: int = 0,
    scenario_id: Optional[str] = None,
    session_id: Optional[str] = None,
    now: Optional[datetime] = None,
) -> CorrectionEvent:
    return CorrectionEvent(
        correction_id=_new_id(),
        decision_id=decision_id,
        decision_point=point.value,
        corrected_to_kind=corrected_to.kind.value,
        corrected_to_payload=corrected_to.payload(),
        source=source,
        reason=reason or "",
        supervisor_id=supervisor_id,
        step_id=step_id,
        step_idx=step_idx,
        scenario_id=scenario_id,
        session_id=session_id,
        corrected_at=now or datetime.now(timezone.utc),
    )
