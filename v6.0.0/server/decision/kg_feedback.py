"""
decision/kg_feedback.py
=======================
Turning what an operator did into what the competence graph should learn.

This is the seam that closes the loop. Before it, an override landed in
demo_correction_log and stopped there: the graph could be read but never
learned, so no amount of rollout campaign changed a single weight.

Pure. Takes a correction, returns a list of observations to apply. It writes
nothing itself, so the translation can be unit-tested, replayed offline over a
stored correction log, and swapped without touching persistence.

THE ASYMMETRY THAT MATTERS
A reroute from A to B is one action making two claims of very different
strength:

    B is right      strong. A person deliberately picked them.
    A is wrong      weak.   They wanted B. That is not the same as judging A
                            incompetent — B may simply have had a better answer
                            ready, or A may already have spoken twice.

Recording both at the supervisor rate would let a single preference for B erode
A across every topic the two share. So the displaced robot is written at
Evidence.DISPLACED, roughly a sixth of the rate, and counted separately so the
graph can still be decomposed by evidence source afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from decision.kg import Evidence

# What a positive and a negative observation aim the weight at. Not 1.0/0.0 by
# accident: these are the targets the update moves PARTWAY toward, so they are
# the ceiling a robot can reach on unanimous evidence.
TARGET_CHOSEN = 1.0
TARGET_DISPLACED = 0.0


@dataclass(frozen=True)
class Observation:
    """One (robot, topic) update to apply. Deliberately not the edge itself —
    this module never reads the store, so it cannot know current weights."""

    robot_id: str
    topic_id: str
    target: float
    evidence: Evidence
    note: str = ""

    def as_row(self) -> dict:
        return {"robot_id": self.robot_id, "topic_id": self.topic_id,
                "target": self.target, "evidence": self.evidence.value,
                "note": self.note}


def from_reroute(
    topic_id: Optional[str],
    chosen_robot_id: str,
    displaced_robot_id: Optional[str] = None,
) -> list:
    """
    An operator sent a question to `chosen_robot_id` instead of whoever had it.

    Returns [] when the topic could not be resolved. That is the common case for
    a question that mentioned no subject, and it is correct: an observation
    filed against the wrong topic is worse than no observation, because it is
    indistinguishable from a real one afterwards.
    """
    if not topic_id or not chosen_robot_id:
        return []

    out = [Observation(chosen_robot_id, topic_id, TARGET_CHOSEN,
                       Evidence.SUPERVISOR, "operator chose this robot")]

    # Self-reroute is a no-op on the negative side: an operator re-confirming the
    # robot that already had the question is evidence FOR it and nothing else.
    if displaced_robot_id and displaced_robot_id != chosen_robot_id:
        out.append(Observation(displaced_robot_id, topic_id, TARGET_DISPLACED,
                               Evidence.DISPLACED, "operator routed away"))
    return out


@dataclass
class Segment:
    """
    What actually happened during one Q&A window.

    Exists to make the SILENCE RULE structural rather than a caller's
    responsibility. A Q&A window where nobody asked anything closes exactly like
    one where three questions were answered well, and "no correction fired" is
    true of both. Emitting an outcome for the silent one records that a robot
    handled questions during a period when it handled zero.

    The damage is not the weight — it is n_obs. That count drives the confidence
    clamp, the mechanism that stops a single correction from teaching a whole
    cluster. Feed it observations manufactured from silence and the clamp starts
    granting confidence to numbers with nothing behind them: the graph reports
    0.85 confidently where it should report a hedged 0.6. Months of hollow rows
    are very hard to unpick afterwards, because they are indistinguishable from
    real ones.

    So this class can only accumulate questions that were genuinely routed.
    `observations()` returns [] when nothing was, and there is no argument that
    overrides it.
    """

    routed: list = field(default_factory=list)   # (robot_id, topic_id) pairs
    corrected: bool = False

    def note_routed(self, robot_id: Optional[str], topic_id: Optional[str]) -> None:
        """A question was answered by `robot_id` and resolved to `topic_id`.

        A turn whose topic never resolved is NOT recorded. It was a real
        question, but there is no edge it belongs to, and attributing it to a
        guess would be worse than dropping it.
        """
        if robot_id and topic_id:
            self.routed.append((robot_id, topic_id))

    def note_correction(self) -> None:
        """An operator intervened during this segment."""
        self.corrected = True

    def reset(self) -> None:
        self.routed.clear()
        self.corrected = False

    @property
    def answered_anything(self) -> bool:
        return bool(self.routed)

    def observations(self) -> list:
        """Weak positive evidence for each robot/topic actually handled here.

        Empty when the segment was silent (the silence rule) or corrected (the
        correction already describes that event; recording both double-counts
        one thing and inflates n_obs the same way).
        """
        if self.corrected or not self.routed:
            return []
        seen, out = set(), []
        for robot_id, topic_id in self.routed:
            if (robot_id, topic_id) in seen:
                continue          # one segment is one observation per edge
            seen.add((robot_id, topic_id))
            out.append(Observation(robot_id, topic_id, TARGET_CHOSEN,
                                   Evidence.OUTCOME, "segment completed uncorrected"))
        return out


def from_segment_outcome(
    topic_ids: Iterable[str],
    robot_id: str,
    corrected: bool,
) -> list:
    """
    Convenience wrapper over Segment for callers that already have the pairs.

    Prefer Segment: passing a topic list here is exactly where a caller can
    invent topics nobody asked about. This exists for tests and for replaying a
    stored log, where the routed set is known to be real.
    """
    segment = Segment(corrected=corrected)
    for tid in topic_ids:
        segment.note_routed(robot_id, tid)
    return segment.observations()


def apply(observations: Iterable[Observation], repo) -> list:
    """
    Persist a batch. `repo` supplies apply_observation(robot, topic, target, kind).

    Injected rather than imported so this module keeps no dependency on data/,
    and so a rollout harness can pass an in-memory store. Failures are per
    observation: one unwritable edge must not discard the rest of a correction.
    """
    applied = []
    for o in observations:
        edge = repo.apply_observation(o.robot_id, o.topic_id, o.target, o.evidence)
        if edge is not None:
            applied.append(edge)
    return applied
