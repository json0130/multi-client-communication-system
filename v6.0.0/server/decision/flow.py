"""
decision/flow.py
================
The tour as a structure: blocks, roles, durations, and what may be cut.

Mostly this makes explicit what build_script() already knows. Every step already
carries `block_robot_id` (which project it belongs to) and `role` (what it is
for), and the rules about what survives an edit are already written down — as
prose, in comments. "PROJECT and QA are deliberately absent from COMPRESSIBLE."
"StepRole.CLOSING survives a DROP_REMAINING." Those are graph constraints
sitting in docstrings where nothing can check them.

Lifting them into structure buys three things:

  * the planner can ask what a cut would COST before making it
  * duration estimates attach to blocks rather than to a flat step list
  * the constraints become testable instead of conventional

WHAT THIS IS NOT
Not the competence graph. That one is a weighted similarity structure that
learns from corrections and answers "who should take this question". This is an
ordering structure with constraints that answers "what should the rest of the
tour look like". They stay separate: a correction against one means something
different from a correction against the other.

IMPORTANCE IS NOT STORED HERE
A block's importance depends on the visitor in front of you — someone who came
for the HRI work values a different block from someone who came for navigation.
So importance is computed per run and passed in, never written onto the graph.
The graph holds structure and duration, both of which are properties of the tour
itself. Same read-time discipline as competence propagation.

Q&A IS ALLOCATED, NOT PREDICTED
estimate() takes a qa_budget and refuses to guess one. Scripted step length is a
property of the content and averages usefully across runs; Q&A length is a
property of the operator and the group and does not. A caller wanting a total
says how much Q&A time it intends to grant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from decision.models import PlanOp, PlanOpKind, StepRole

# Used when a step has no measured duration yet. Deliberately generous: an
# underestimate makes the planner cut too little, and a tour that overruns is
# worse than one that trimmed a step it did not need to.
DEFAULT_STEP_SEC = 12.0

MIN_RUNS_TO_TRUST = 3
"""How many timings a step needs before its mean is used instead of
DEFAULT_STEP_SEC.

The point is NOT variance reduction. Measured against 49 real step rows, the
median relative standard error of the trusted set barely moves between a
threshold of 2 and 6 (~12% throughout) — scripted speech genuinely varies
that much, and no amount of averaging makes a 19s talk predictable to the
second.

The point is that at n=1 there is no spread to measure at all. A single run
that stalled on an ACK, or caught a robot mid-reconnect, IS the estimate, and
nothing in the data distinguishes it from a clean one. 23 of those 49 rows
are n=1. Feeding them to the planner means a step's whole contribution to the
budget can hinge on one bad recording, and — because the planner's rungs are
threshold-based — that is exactly the input that flips "compress" into "skip
a project" for reasons no one can see afterwards. tools/plan_replay_harness.py
was built to demonstrate that mechanism; this is the mitigation.

3 rather than 2 because it matches decision.kg.CONFIDENCE_HALFLIFE, which
answers the same question ("when is accumulated evidence worth believing?")
for the competence graph. Two systems, one answer, is worth more than either
being individually tuned.

Steps below the threshold are omitted from the durations map entirely rather
than down-weighted, so FlowGraph.measured_coverage() keeps meaning exactly
what it says: the fraction of the estimate backed by data worth trusting.
Coverage DROPS when this lands — that is the honest number, not a regression.
"""

DEFAULT_QA_BUDGET_SEC = 90.0
"""What a Q&A window is assumed to take when no budget is stated. Not a
prediction — a stated default, so a caller that forgets to allocate gets a
visible number rather than zero."""


@dataclass(frozen=True)
class StepRef:
    """One step, reduced to what the planner needs."""

    step_id: str
    robot_id: str
    role: str
    qa_window: bool = False
    block_robot_id: Optional[str] = None

    @property
    def compressible(self) -> bool:
        """Can this step be dropped without losing research content?

        The constraint that was previously only a comment on
        StepRole.COMPRESSIBLE. PROJECT and QA are never compressible: the talk
        and the chance to ask about it are what the tour is for.
        """
        return self.role in StepRole.COMPRESSIBLE

    @property
    def essential(self) -> bool:
        return self.role in (StepRole.PROJECT, StepRole.QA)


@dataclass(frozen=True)
class Block:
    """One project's stretch of the tour."""

    robot_id: str
    steps: tuple = ()

    @property
    def qa_steps(self) -> tuple:
        return tuple(s for s in self.steps if s.role == StepRole.QA)

    @property
    def compressible_steps(self) -> tuple:
        return tuple(s for s in self.steps if s.compressible)

    def scripted_seconds(self, durations: Optional[dict] = None,
                         compressed: bool = False) -> float:
        """Seconds of SCRIPTED speech in this block. Q&A is excluded entirely —
        it is allocated by the caller, not predicted here."""
        durations = durations or {}
        total = 0.0
        for s in self.steps:
            if s.role == StepRole.QA:
                continue
            if compressed and s.compressible:
                continue
            total += float(durations.get(s.step_id, DEFAULT_STEP_SEC))
        return total

    def seconds(self, durations: Optional[dict] = None,
                qa_budget: float = DEFAULT_QA_BUDGET_SEC,
                compressed: bool = False) -> float:
        """Total block time = scripted content + the Q&A you choose to grant."""
        return (self.scripted_seconds(durations, compressed)
                + qa_budget * len(self.qa_steps))

    def compression_saving(self, durations: Optional[dict] = None) -> float:
        """Seconds recovered by compressing, without touching Q&A."""
        return (self.scripted_seconds(durations)
                - self.scripted_seconds(durations, compressed=True))


@dataclass(frozen=True)
class FlowGraph:
    """The whole tour: an opening, an ordered list of blocks, and a closing."""

    opening: tuple = ()
    blocks: tuple = ()
    closing: tuple = ()

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_script(cls, steps: Iterable) -> "FlowGraph":
        """Build from a DemoStep list. Reads the tags build_script already sets.

        A step with no block_robot_id is opening or closing, decided by its
        role — which is why an untagged hand-written script produces a graph
        with no blocks rather than a wrong one. Better to plan nothing than to
        plan against a structure that was guessed.
        """
        opening, closing = [], []
        order: list = []
        by_block: dict = {}
        for s in steps:
            ref = StepRef(step_id=s.step_id, robot_id=s.robot_id,
                          role=getattr(s, "role", "") or "",
                          qa_window=bool(getattr(s, "qa_window", False)),
                          block_robot_id=getattr(s, "block_robot_id", None))
            if ref.block_robot_id:
                if ref.block_robot_id not in by_block:
                    by_block[ref.block_robot_id] = []
                    order.append(ref.block_robot_id)
                by_block[ref.block_robot_id].append(ref)
            elif ref.role == StepRole.CLOSING:
                closing.append(ref)
            else:
                opening.append(ref)
        return cls(
            opening=tuple(opening),
            blocks=tuple(Block(robot_id=r, steps=tuple(by_block[r])) for r in order),
            closing=tuple(closing),
        )

    # ── Reads ─────────────────────────────────────────────────────────────────

    def block(self, robot_id: str) -> Optional[Block]:
        return next((b for b in self.blocks if b.robot_id == robot_id), None)

    def fixed_seconds(self, durations: Optional[dict] = None) -> float:
        """Opening plus closing — the part no plan revision may remove.

        DROP_REMAINING keeps the closing precisely so a tour cut for time still
        ends rather than stopping mid-sentence, which makes this a floor on any
        achievable plan.
        """
        durations = durations or {}
        return sum(float(durations.get(s.step_id, DEFAULT_STEP_SEC))
                   for s in self.opening + self.closing)

    def estimate(self, durations: Optional[dict] = None,
                 qa_budget: float = DEFAULT_QA_BUDGET_SEC,
                 compressed: Sequence = (), skipped: Sequence = ()) -> dict:
        """Predicted tour length under a given set of cuts.

        `qa_budget` is required rather than inferred — see the module docstring.
        Returns the breakdown, not just a number, because a planner deciding
        what to cut needs to see where the time actually is.
        """
        durations = durations or {}
        compressed, skipped = set(compressed), set(skipped)
        fixed = self.fixed_seconds(durations)
        per_block, qa_total, scripted_total = {}, 0.0, 0.0
        for b in self.blocks:
            if b.robot_id in skipped:
                per_block[b.robot_id] = 0.0
                continue
            scripted = b.scripted_seconds(durations, b.robot_id in compressed)
            qa = qa_budget * len(b.qa_steps)
            per_block[b.robot_id] = scripted + qa
            scripted_total += scripted
            qa_total += qa
        return {
            "total_sec": round(fixed + scripted_total + qa_total, 1),
            "fixed_sec": round(fixed, 1),
            "scripted_sec": round(scripted_total, 1),
            "qa_sec": round(qa_total, 1),
            "per_block": {k: round(v, 1) for k, v in per_block.items()},
            "blocks_remaining": len([b for b in self.blocks
                                     if b.robot_id not in skipped]),
        }

    def measured_coverage(self, durations: Optional[dict] = None) -> float:
        """Fraction of steps with a real measurement behind them.

        An estimate built entirely from DEFAULT_STEP_SEC is arithmetic, not a
        prediction. Callers should surface this rather than quoting a total as
        though it were measured.
        """
        durations = durations or {}
        all_steps = [s for s in self.opening + self.closing
                     if s.role != StepRole.QA]
        for b in self.blocks:
            all_steps += [s for s in b.steps if s.role != StepRole.QA]
        if not all_steps:
            return 0.0
        return sum(1 for s in all_steps if s.step_id in durations) / len(all_steps)
