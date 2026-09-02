"""
decision/
=========
The demo's decision layer: what was observed, what was chosen, which rule chose
it, and what a supervisor corrected it to.

The lab demo's sequence is authored — demo_script.build_script() emits a fixed
block per robot and DemoOrchestrator walks it. The decisions that actually shape
a visitor's experience are made elsewhere and are not recorded anywhere:

  when a Q&A window closes    five overlapping rules in websocket_gateway
  who answers a question      not decided — whoever heard the audio replies
  what the rest of the tour   not decided — a human clicks Move On / Skip
  should look like

This package makes those three explicit, executable and logged, so a policy can
later be learned from the supervisor corrections rather than hand-tuned. Phase
one changes no outcome: HeuristicPolicy reproduces today's precedence chain
exactly, and the only new behaviour is plan revision, which is opt-in.

Layout:
  models.py       DecisionPoint / Action / PlanOp / Observation and the events
  observation.py  the state space, and per-run conversational bookkeeping
  policy.py       the Policy protocol and the current system as a baseline
  recorder.py     non-blocking batched persistence, and live-decision tracking

This package is application-agnostic in the same sense core/rbac is: it must not
import from robot/, gateway/ or data/. Callables are injected instead.
"""

from decision.models import (
    Action,
    ActionKind,
    CorrectionEvent,
    DecisionEvent,
    DecisionPoint,
    Observation,
    PlanOp,
    PlanOpKind,
    build_correction,
    build_decision,
)
from decision.observation import DemoRunTracker, build_observation, looks_like_question
from decision.policy import (
    HeuristicPolicy,
    Mechanism,
    Policy,
    PolicyResult,
    QA_ADVANCE_PHRASES,
    QA_CLOSING_PHRASES,
)
from decision.recorder import (
    BatchingDecisionSink,
    DecisionRecorder,
    DecisionSink,
    MemoryDecisionSink,
    NullDecisionSink,
)

__all__ = [
    "Action",
    "ActionKind",
    "BatchingDecisionSink",
    "CorrectionEvent",
    "DecisionEvent",
    "DecisionPoint",
    "DecisionRecorder",
    "DecisionSink",
    "DemoRunTracker",
    "HeuristicPolicy",
    "Mechanism",
    "MemoryDecisionSink",
    "NullDecisionSink",
    "Observation",
    "PlanOp",
    "PlanOpKind",
    "Policy",
    "PolicyResult",
    "QA_ADVANCE_PHRASES",
    "QA_CLOSING_PHRASES",
    "build_correction",
    "build_decision",
    "build_observation",
    "looks_like_question",
]
