"""
decision/style_fit.py
=====================
How well a robot pitches to a given audience, learned from supervision.

WHAT THIS IS NOT
Not routing. Nothing here decides who answers — a robot that explains
navigation badly to a business audience is still the robot that knows
navigation. Style fit changes HOW the answer is framed, never WHO gives it.
Folding it into decision/kg.py's weight would put an audience-fit judgement
into the number that picks a robot, and the two would then be impossible to
separate afterwards.

WHY IT IS PER ROBOT WHEN THE DIRECTIVE IS NOT
decision/visitor_profile.py says style is "a property of the VISITOR, not of
any one robot", and that remains true of the DIRECTIVE: a technical visitor
needs precise language from every robot equally, so STYLE_FRAMING is injected
uniformly. What is per robot is how well each one ACTS on it. One robot
reliably drops into implementation detail when asked; another keeps giving
the same general answer whatever the framing says. That gap is a fact about
the robot, observable only by watching, and it is what this module holds.

SUPERVISOR EVIDENCE ONLY
No automatic outcome signal, deliberately. A Q&A window closing cleanly means
nobody objected to the ROUTING; it says nothing about whether the answer was
pitched well, and crediting style fit for it would manufacture a judgement
nobody made — the same failure kg_feedback.Segment's silence rule exists to
prevent, in a place where it would be even harder to notice. Style fit moves
only when a person says the pitch was good or bad.

WHAT A LEARNED VALUE DOES
A scalar is not actionable on its own — an LLM cannot be told "be 0.3
better". So the value selects between two framing strengths: the ordinary
directive, or that directive plus an explicit corrective note. Reinforcement
is driven by the CLAMPED value, so one bad night does not permanently change
how a robot is briefed.

Pure, like the rest of decision/: no I/O. Persistence lives in
data/demo_style_repo.py and the arithmetic is reused from decision/kg.py
rather than duplicated, so there is one update rule in this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from decision.kg import (CONFIDENCE_HALFLIFE, NEUTRAL, PRIOR, Evidence,
                         learning_rate)
from decision.visitor_profile import STYLE_FRAMING

REINFORCE_BELOW = 0.40
"""Clamped fit under which a robot gets the corrective note as well as the
directive.

Set from the actual clamp curve rather than by eye. Consecutive poor ratings
land at:

    1 rating   0.4375
    2 ratings  0.3583
    3 ratings  0.2969

so 0.40 is the value that requires TWO independent judgements before a
robot's briefing changes. One is too thin — an operator can judge a single
answer harshly for reasons that have nothing to do with a habit — and three
would make the mechanism almost never fire, since supervision is scarce
enough that this whole system exists to economise on it.

An earlier 0.45 was chosen by eye and silently triggered on a single
rating; the test that asserts one rating is not enough is what caught it."""

# Added ON TOP of STYLE_FRAMING when the robot's record on this audience is
# poor. Phrased as a correction of a known habit rather than a restatement of
# the directive — repeating "be technical" louder does not help a robot that
# has already been told that and stayed vague.
STYLE_REINFORCEMENT = {
    "technical": (
        " Visitors have previously found your answers on this too general: "
        "go deeper into the mechanism — the steps, the trade-offs, what it "
        "assumes — rather than describing the area in the abstract. Still "
        "never name a method, model or number you were not given."
    ),
    "business": (
        " Visitors have previously found your answers on this too technical: "
        "open with what it lets someone DO, and mention mechanism only if it "
        "explains the benefit."
    ),
    "interactive": (
        " Visitors have previously found your answers on this too much like a "
        "monologue: stop after one idea and put a question back to them."
    ),
    "general": "",   # nothing to correct toward — no directive was given
}


@dataclass(frozen=True)
class StyleFit:
    """One robot's record with one kind of audience.

    Frozen and additive, exactly like RobotTopicEdge: `record` returns a new
    value so a caller cannot leave the count disagreeing with the weight it
    justified.
    """

    robot_id: str
    style: str
    weight: float = PRIOR
    n_supervisor: int = 0
    last_updated: Optional[datetime] = None

    @property
    def n_obs(self) -> int:
        return self.n_supervisor

    @property
    def confidence(self) -> float:
        n = self.n_obs
        return n / (n + CONFIDENCE_HALFLIFE) if n else 0.0

    @property
    def clamped(self) -> float:
        """The weight pulled toward neutral by how little is known — the same
        formula RobotTopicEdge uses, for the same reason."""
        return NEUTRAL + (self.weight - NEUTRAL) * self.confidence

    @property
    def needs_reinforcement(self) -> bool:
        return self.clamped < REINFORCE_BELOW

    def record(self, target: float, now: Optional[datetime] = None) -> "StyleFit":
        """Fold in one supervisor judgement. 1.0 = well pitched, 0.0 = not."""
        target = max(0.0, min(1.0, float(target)))
        lr = learning_rate(Evidence.SUPERVISOR, self.n_obs)
        return StyleFit(
            robot_id=self.robot_id,
            style=self.style,
            weight=self.weight + lr * (target - self.weight),
            n_supervisor=self.n_supervisor + 1,
            last_updated=now or datetime.now(timezone.utc),
        )

    def as_row(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "style": self.style,
            "weight": round(self.weight, 6),
            "n_supervisor": self.n_supervisor,
            "last_updated": (self.last_updated
                             or datetime.now(timezone.utc)).isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict) -> "StyleFit":
        from decision.kg import _parse_ts
        return cls(
            robot_id=row["robot_id"],
            style=row["style"],
            weight=float(row.get("weight", PRIOR)),
            n_supervisor=int(row.get("n_supervisor", 0) or 0),
            last_updated=_parse_ts(row.get("last_updated")),
        )


def framing_for(style: Optional[str], fit: Optional[StyleFit] = None) -> str:
    """
    The framing suffix for one generated step.

    `fit` is this robot's record with this audience, or None when nothing is
    known — which is the common case and returns exactly what
    VisitorProfile.framing returned before any of this existed. That is the
    property worth protecting: a deployment with no style feedback behaves
    identically to one that never had the feature.
    """
    base = STYLE_FRAMING.get(style or "", "")
    if not base or fit is None or not fit.needs_reinforcement:
        return base
    return base + STYLE_REINFORCEMENT.get(style or "", "")
