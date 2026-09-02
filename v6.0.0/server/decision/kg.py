"""
decision/kg.py
==============
The robot→topic competence graph, and the arithmetic that learns it.

WHAT THIS MODELS
    robot --handles[w, n]--> topic          learned, corrigible
    topic --related[w]------ topic          semantic, seeds propagation

"How good is this robot at questions about this topic." It is deliberately NOT
robot→robot (twenty ordered pairs over a five-robot fleet is a lookup table, and
nothing useful lives two hops out) and NOT context→action (that is a conditional
policy, which decision/policy.py already is). Only robot→topic earns the graph
structure: a correction on one topic propagates along topic↔topic edges to its
neighbours, which is the only form of generalisation from sparse supervision
any of the three candidates offers.

WHY NOT OVERWRITE
    The CHATBOX KG overwrites affinity on every write, and for a person stating
    a preference that is right — a declaration is authoritative and legitimately
    replaces the prior. A supervisor correcting a demo is not declaring anything.
    It is one noisy observation about competence, made under time pressure, and
    two supervisors (or one supervisor on two tours) will push the same edge in
    opposite directions. Overwrite gives last-correction-wins: the weight
    oscillates and never converges, and repeated rollouts buy nothing — which is
    the whole claimed contribution.

    So: additive, with an observation count.

        w  <-  w + lr(kind, n) * (target - w)

    Because that rearranges to (1 - lr)*w + lr*target, it is a convex
    combination whenever lr is in [0,1]: the weight is self-capping in [0,1] with
    no separate clamp, and repeated conflicting evidence averages rather than
    thrashes. lr decays with n, so an edge with fifty observations barely moves
    while a fresh one moves a lot.

TWO STRENGTHS OF EVIDENCE
    A supervisor pressing a button and a segment finishing inside its budget are
    not the same quality of signal. Same edge, different learning rate, and the
    counts are tracked separately so "how much of this graph came from human
    correction versus rollout outcome" is a query rather than an assumption.

WHAT THIS RULE IS NOT
    It is recency-weighted, not an unbiased estimator. Ten "yes" observations
    followed by ten "no" lands low; the reverse lands high. That is deliberate —
    a plain running mean is order-independent but cannot track a robot whose
    competence genuinely changed, and a retuned platform would stay pinned by
    stale evidence. The cost is real though: the graph reports RECENT competence,
    and a long one-sided run dominates. Alternating disagreement still converges
    (tests/test_kg_update.py asserts both), so the oscillation problem that ruled
    out overwrite does not return; but if lifetime competence is what you want to
    report, this is the wrong statistic and you want a separate running mean
    alongside it rather than a retuned lr.

THE COUNT IS NOT DEAD WEIGHT
    n also drives the confidence fed to the read-time clamp — the same
    0.5 + (w - 0.5) * confidence used by CHATBOX's preference BN. There,
    confidence was an LLM's self-report about a single reading. Here it is
    accumulated agreement, which is what that formula always wanted. An edge with
    one observation reads as barely distinguishable from neutral no matter how
    extreme its weight; the two mechanisms compose instead of substituting.

This module is pure: no I/O, no store, no Supabase. Persistence lives in
data/demo_kg_repo.py, propagation reads in decision/kg_infer.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

PRIOR = 0.5
"""Weight of an edge with no observations. Neutral, matching AboutEdge.affinity —
an unseen robot/topic pair is unknown, not bad. Note this differs from
InteractionNode's 0.0 prior; that models 'no relationship yet', this models 'no
evidence yet', and conflating them would make every new edge look incompetent."""

NEUTRAL = 0.5
"""Fixed point of the read-time clamp. Identical to preference_model._NEUTRAL."""

CONFIDENCE_HALFLIFE = 3.0
"""Observations at which confidence reaches 0.5. With k=3, one observation reads
0.25 confident and five read 0.63 — deliberately slow, because the failure this
guards against is a single correction being taken as settled fact."""

LR_DECAY_TAU = 5.0
"""Observations over which the learning rate halves. Chosen so an edge is still
meaningfully movable at ~5 observations (the realistic per-edge count for a few
hundred corrections spread across a fleet) and close to settled by ~25."""


class Evidence(str, Enum):
    """Where an observation came from. Drives the learning rate and is counted
    separately so the final graph can be decomposed by evidence source — which
    is itself a reportable result, not just bookkeeping."""

    SUPERVISOR = "supervisor"   # an operator explicitly chose this robot
    OUTCOME = "outcome"         # a segment completed within budget, uncorrected
    DISPLACED = "displaced"     # this robot was the one routed AWAY from


BASE_LR = {
    # A human deliberately choosing this robot is the strongest signal
    # available. Still below 1.0: at 1.0 the update degenerates to overwrite,
    # which is the behaviour this whole design exists to avoid.
    Evidence.SUPERVISOR: 0.50,
    # An uncorrected segment is weak, ambiguous evidence — it may mean the
    # routing was good, or only that nobody was watching. Deliberately small so
    # outcomes shade a graph that corrections shape.
    Evidence.OUTCOME: 0.15,
    # The robot an operator routed AWAY from. Weakest on purpose: rerouting to B
    # says the operator wanted B, NOT that A is incompetent. They may have known
    # B had a better demo ready, or simply that A had already spoken twice.
    # Treating a displacement as a judgement on A would let one preference for B
    # erode A across every topic B is good at.
    Evidence.DISPLACED: 0.08,
}


def _parse_ts(value) -> Optional[datetime]:
    """Parse a timestamp from a database row, tolerantly.

    Postgres emits however many fractional-second digits it needs — 5 in
    practice, e.g. '2026-09-02T00:01:07.73877+00:00'. Python 3.10's
    datetime.fromisoformat accepts exactly 3 or 6 and raises on anything else,
    so the obvious one-liner blows up on real data while passing every test
    written with a hand-made timestamp. Pad to 6.

    A timestamp that still will not parse degrades to None rather than raising:
    last_updated is metadata, and losing it must not make an edge unreadable.
    """
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    match = re.match(r"^(.*\.)(\d{1,6})(.*)$", text)
    if match:
        head, frac, tail = match.groups()
        text = f"{head}{frac.ljust(6, '0')}{tail}"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


# ── Edge ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RobotTopicEdge:
    """
    One robot's learned competence at one topic.

    Frozen: `update` returns a new edge rather than mutating, so a caller cannot
    half-apply an update and leave n_obs disagreeing with the weight it
    justified — the pair has to move together or not at all.
    """

    robot_id: str
    topic_id: str
    weight: float = PRIOR
    n_supervisor: int = 0
    n_outcome: int = 0
    n_displaced: int = 0
    last_updated: Optional[datetime] = None

    @property
    def n_obs(self) -> int:
        """Total observations. Drives both the learning rate and confidence."""
        return self.n_supervisor + self.n_outcome + self.n_displaced

    @property
    def confidence(self) -> float:
        """How much this edge's weight should be believed, in [0,1].

        Saturating in the observation count: n/(n+k). Zero observations reads
        0.0, so a fresh edge clamps to exactly neutral and cannot influence
        anything until something has actually been seen.
        """
        n = self.n_obs
        return n / (n + CONFIDENCE_HALFLIFE) if n else 0.0

    @property
    def clamped(self) -> float:
        """The weight as evidence, pulled toward neutral by how little is known.

            0.5 + (weight - 0.5) * confidence

        Same formula as preference_model.clamp_from, with accumulated agreement
        substituted for an LLM's self-reported certainty.
        """
        return NEUTRAL + (self.weight - NEUTRAL) * self.confidence

    @property
    def human_share(self) -> float:
        """Fraction of this edge's evidence that came from a person.

        Counts displacements as human: an operator did act, they just acted by
        choosing someone else. Excluding them would understate how much of the
        graph a person shaped.
        """
        if not self.n_obs:
            return 0.0
        return (self.n_supervisor + self.n_displaced) / self.n_obs

    def update(
        self,
        target: float,
        kind: Evidence = Evidence.SUPERVISOR,
        now: Optional[datetime] = None,
    ) -> "RobotTopicEdge":
        """Fold one observation in. Returns a new edge.

        `target` is what this observation says the weight should be — 1.0 for
        "this robot should have taken that question", 0.0 for "it should not
        have". Out-of-range targets are clamped rather than rejected: a caller
        passing 1.2 means "strongly yes", and refusing the whole observation over
        it would discard a real label.
        """
        target = max(0.0, min(1.0, float(target)))
        lr = learning_rate(kind, self.n_obs)
        # Convex combination — self-capping in [0,1], no separate clamp needed.
        new_weight = self.weight + lr * (target - self.weight)
        return RobotTopicEdge(
            robot_id=self.robot_id,
            topic_id=self.topic_id,
            weight=new_weight,
            n_supervisor=self.n_supervisor + (1 if kind is Evidence.SUPERVISOR else 0),
            n_outcome=self.n_outcome + (1 if kind is Evidence.OUTCOME else 0),
            n_displaced=self.n_displaced + (1 if kind is Evidence.DISPLACED else 0),
            last_updated=now or datetime.now(timezone.utc),
        )

    def as_row(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "topic_id": self.topic_id,
            "weight": round(self.weight, 6),
            "n_supervisor": self.n_supervisor,
            "n_outcome": self.n_outcome,
            "n_displaced": self.n_displaced,
            "last_updated": (self.last_updated or datetime.now(timezone.utc)).isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict) -> "RobotTopicEdge":
        ts = _parse_ts(row.get("last_updated"))
        return cls(
            robot_id=row["robot_id"],
            topic_id=row["topic_id"],
            weight=float(row.get("weight", PRIOR)),
            n_supervisor=int(row.get("n_supervisor", 0) or 0),
            n_outcome=int(row.get("n_outcome", 0) or 0),
            n_displaced=int(row.get("n_displaced", 0) or 0),
            last_updated=ts,
        )


@dataclass(frozen=True)
class TopicEdge:
    """An undirected topic↔topic semantic link. Endpoints stored sorted so the
    pair has one canonical row, matching link_related_topic's convention."""

    topic_a: str
    topic_b: str
    weight: float
    source: str = ""

    def __post_init__(self):
        # Capture both before writing either. Assigning topic_a first and then
        # reading self.topic_a for topic_b collapses both endpoints onto the
        # same value — a self-loop with the right weight, which looks plausible
        # in a listing and is silently wrong.
        if self.topic_a > self.topic_b:
            a, b = self.topic_a, self.topic_b
            object.__setattr__(self, "topic_a", b)
            object.__setattr__(self, "topic_b", a)

    def other(self, topic_id: str) -> Optional[str]:
        if topic_id == self.topic_a:
            return self.topic_b
        if topic_id == self.topic_b:
            return self.topic_a
        return None

    def as_row(self) -> dict:
        return {"topic_a": self.topic_a, "topic_b": self.topic_b,
                "weight": round(self.weight, 6), "source": self.source}


# ── Arithmetic ────────────────────────────────────────────────────────────────

def learning_rate(kind: Evidence, n_obs: int) -> float:
    """How far one observation moves an edge that already has `n_obs` behind it.

        lr = base(kind) / (1 + n_obs / tau)

    Decaying rather than fixed so early evidence shapes an edge and later
    evidence refines it. A fixed rate would leave a fifty-observation edge as
    jumpy as a fresh one, which is the oscillation problem in slower motion.

    Always in (0, 1], so the caller's convex combination stays valid.
    """
    base = BASE_LR.get(kind, BASE_LR[Evidence.OUTCOME])
    return base / (1.0 + max(0, n_obs) / LR_DECAY_TAU)


def clamp_from(weight: float, confidence: float) -> float:
    """Confidence-weighted pull toward neutral.

    Kept as a free function with this exact name and signature so it is
    recognisably the same operation as preference_model.clamp_from — the two
    should stay in step if either is retuned.
    """
    return NEUTRAL + (float(weight) - NEUTRAL) * float(confidence)


def blend(edges: list) -> float:
    """Evidence-weighted mean of several edges' clamped weights.

    Used when more than one edge speaks to the same question (a robot related to
    a topic both directly and through a neighbour). Weighting by confidence
    rather than averaging raw weights stops a single-observation edge from
    dragging a well-evidenced one around.
    """
    if not edges:
        return PRIOR
    total_conf = sum(e.confidence for e in edges)
    if total_conf <= 0:
        return PRIOR
    return sum(e.clamped * e.confidence for e in edges) / total_conf
