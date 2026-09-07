"""
tools/eval_oracle.py
====================
An independent implementation of the documented rules, used to derive the
correct answer for generated evaluation scenarios.

WHY THIS EXISTS
Generating 50 scenarios is only worth doing if their expected outputs come
from somewhere other than the system under test. Reading expectations off
what the planner produced would give 50 tests that prove the system agrees
with itself — every one of them passing on the day the ladder is reordered
wrongly, because the expectations would reorder with it.

So the ladder in decision/planner.py and the presence/absence policy in
decision/kg_policy.py are re-derived here from their written specifications:
the rung order in planner.py's module docstring, the 0.35/0.65 importance
blend in block_importance, ABSENT_ROBOT_POLICY in kg_policy.py. Different
data structures (plain dicts, no FlowGraph, no PlanOp), different control
flow, same stated rules.

WHAT THIS CATCHES AND WHAT IT DOES NOT
Honest about its limits: a reimplementation from the same specification
catches transcription errors, wiring mistakes, and regressions — a rung
silently reordered, importance stopping at the planner boundary, an absent
robot becoming reachable. It does NOT catch a misunderstanding shared
between the spec and both implementations, because it inherits the spec.

That is exactly why the four hand-derived scenarios in eval_scenarios.py
stay: they were reasoned out from first principles about what a visitor
should experience, so they check the ORACLE as much as the system. If the
oracle drifts from the rules, those four fail. Generated scenarios check
breadth; the hand-derived four check that the thing generating them is
itself right.

CONSTANTS ARE INPUTS, NOT RULES
DEFAULT_STEP_SEC, the Q&A default and floor are imported rather than
duplicated. They are parameters of the documented rule, and a deliberate
change to one should move the expected answers with it. The LADDER is what
is reimplemented, not its tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from decision.flow import DEFAULT_QA_BUDGET_SEC, DEFAULT_STEP_SEC
from decision.models import StepRole
from decision.planner import DEFAULT_IMPORTANCE, QA_FLOOR_SEC


@dataclass(frozen=True)
class BlockFacts:
    """One project's cost, reduced to what the ladder actually reasons about."""

    robot_id: str
    scripted_sec: float        # all non-Q&A steps, uncompressed
    compressible_sec: float    # the part COMPRESS would remove
    qa_windows: int
    importance: float


@dataclass(frozen=True)
class PlanOutcome:
    """What a visitor would experience. Same shape eval_scenarios scores on."""

    surviving_blocks: tuple
    compressed_blocks: frozenset
    qa_budget_sec: float
    feasible: bool
    total_sec: float


def block_facts_from_script(
    steps: Sequence,
    durations: dict,
    importance: dict,
) -> tuple:
    """
    (fixed_sec, [BlockFacts]) from a script, costed independently.

    Reads the same role tags build_script writes, but does its own summing —
    it does not call FlowGraph, which is the thing being checked.
    """
    fixed = 0.0
    order: list = []
    per_block: dict = {}

    for s in steps:
        block = getattr(s, "block_robot_id", None)
        role = getattr(s, "role", "") or ""
        cost = float(durations.get(s.step_id, DEFAULT_STEP_SEC))

        if not block:
            if role != StepRole.QA:
                fixed += cost          # opening and closing
            continue

        if block not in per_block:
            per_block[block] = {"scripted": 0.0, "compressible": 0.0, "qa": 0}
            order.append(block)

        if role == StepRole.QA:
            per_block[block]["qa"] += 1
            continue                    # Q&A is allocated, never costed here
        per_block[block]["scripted"] += cost
        if role in StepRole.COMPRESSIBLE:
            per_block[block]["compressible"] += cost

    blocks = tuple(
        BlockFacts(
            robot_id=b,
            scripted_sec=per_block[b]["scripted"],
            compressible_sec=per_block[b]["compressible"],
            qa_windows=per_block[b]["qa"],
            importance=float(importance.get(b, DEFAULT_IMPORTANCE)),
        )
        for b in order
    )
    return fixed, blocks


def derive_plan(
    budget_sec: float,
    fixed_sec: float,
    blocks: Sequence[BlockFacts],
    qa_default: float = DEFAULT_QA_BUDGET_SEC,
    qa_floor: float = QA_FLOOR_SEC,
) -> PlanOutcome:
    """
    The compression ladder, re-derived from decision/planner.py's docstring:

        1. tighten Q&A      largest share of tour time, least noticed
        2. compress blocks  drop scaffolding, keep the research talk
        3. skip a block     least important first, never the last one standing
        4. drop remaining   the budget cannot be met; the closing still runs

    Each rung is tried to exhaustion before the next, and every rung
    re-estimates rather than assuming its own saving.
    """
    qa = float(qa_default)
    compressed: set = set()
    skipped: set = set()

    def estimate() -> float:
        total = fixed_sec
        for b in blocks:
            if b.robot_id in skipped:
                continue
            total += b.scripted_sec - (b.compressible_sec
                                       if b.robot_id in compressed else 0.0)
            total += qa * b.qa_windows
        # Rounded to 0.1s because that is what "the estimate" IS —
        # FlowGraph.estimate specifies its total to one decimal, and every
        # rung's decision is a comparison against it. Without this the oracle
        # compares at float epsilon and disagrees on exact-fit budgets purely
        # from summation order: accumulating per-block gives
        # 197.0 + 203.00000000000003, one ULP over a 400s budget, and the
        # ladder takes a rung it should not. A tour-length estimate carries no
        # meaning below a decisecond, so rounding is the rule, not a fudge.
        return round(total, 1)

    def outcome(feasible: bool) -> PlanOutcome:
        return PlanOutcome(
            surviving_blocks=tuple(b.robot_id for b in blocks
                                   if b.robot_id not in skipped),
            compressed_blocks=frozenset(compressed),
            qa_budget_sec=qa,
            feasible=feasible,
            total_sec=round(estimate(), 1),
        )

    if estimate() <= budget_sec:
        return outcome(True)

    # Rung 1 — tighten every remaining Q&A window, down to the floor.
    if qa > qa_floor:
        needed = estimate() - budget_sec
        windows = sum(b.qa_windows for b in blocks if b.robot_id not in skipped)
        if windows:
            # Whole seconds: the op carries an integer and the orchestrator
            # enforces exactly that, so a budget is only real at the precision
            # it can actually be commanded at. See planner.py's rung 1.
            qa = max(qa_floor, float(round(qa - needed / windows)))
    if estimate() <= budget_sec:
        return outcome(True)

    # Least important first, ties broken by robot_id — the ordering both
    # rungs below share.
    by_importance = sorted(blocks, key=lambda b: (b.importance, b.robot_id))

    # Rung 2 — compress.
    for b in by_importance:
        if b.robot_id in skipped or b.robot_id in compressed:
            continue
        compressed.add(b.robot_id)
        if estimate() <= budget_sec:
            return outcome(True)

    # Rung 3 — skip, never the last block standing.
    for b in by_importance:
        if b.robot_id in skipped:
            continue
        if len(blocks) - len(skipped) <= 1:
            break
        skipped.add(b.robot_id)
        if estimate() <= budget_sec:
            return outcome(True)

    # Rung 4 — nothing left to protect; drop every remaining project.
    if estimate() > budget_sec and len(blocks) > len(skipped):
        for b in blocks:
            skipped.add(b.robot_id)
        return outcome(estimate() <= budget_sec)

    return outcome(False)


def derive_importance(
    defaults: dict,
    coverage: Optional[dict] = None,
) -> dict:
    """
    block_importance's blend, re-derived: a stated interest outweighs a
    hand-set default without erasing it.

        base = 0.35 * default + 0.65 * coverage      (when coverage is known)

    `coverage` is the per-robot topic coverage the competence graph reports.
    Passed in rather than recomputed: propagation over the graph is a
    separate mechanism with its own tests (test_kg_infer.py), and
    reimplementing it here would be testing that, not the ladder.
    """
    out = {}
    for robot_id, default in defaults.items():
        base = float(default)
        if coverage and robot_id in coverage:
            base = 0.35 * base + 0.65 * float(coverage[robot_id])
        out[robot_id] = round(max(0.0, min(1.0, base)), 4)
    return out


# ── Routing ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RouteOutcome:
    answering_robot: Optional[str]
    deferred_to: Optional[str]
    observations_written: int


def derive_route(
    topic_owner: Optional[str],
    absent: Iterable[str],
    remaining_blocks: Iterable[str],
    guide_robot_id: str,
    absent_policy: str = "defer",
    ineligible: Iterable[str] = (),
    learned_winner: Optional[str] = None,
) -> RouteOutcome:
    """
    Who answers, re-derived from decision/kg_policy.py's ABSENT_ROBOT_POLICY.

    `topic_owner` is the robot the graph would pick with everyone present —
    None when the utterance resolved to no topic. Resolution itself is word
    overlap with its own tests; this is about what happens AFTER a subject is
    identified.

    Deferring writes no observation: nobody answered, and the absent robot
    was never judged.
    """
    absent, remaining = set(absent), set(remaining_blocks)
    ineligible = set(ineligible)

    if topic_owner is None:
        # UNDECLARED topic — nobody's project covers it, so scope is silent
        # and the LEARNED weight is the only thing that could decide. It only
        # gets to when the candidates are actually distinguishable;
        # otherwise the graph is ordering robots on propagated fractions of
        # one distant observation, which the live graph did at a 0.003
        # margin. `learned_winner` is the robot with real evidence, or None
        # when nothing separates them — in which case the receiver answers or
        # delegation takes it, and nothing is recorded either way.
        if learned_winner and learned_winner not in set(ineligible) | set(absent):
            return RouteOutcome(learned_winner, None, 1)
        return RouteOutcome(None, None, 0)

    # DECLARED scope is exclusive: an owner that cannot answer does not hand
    # the topic to a robot whose project it is not.
    if topic_owner in ineligible:
        return RouteOutcome(None, None, 0)

    if topic_owner not in absent:
        return RouteOutcome(topic_owner, None, 1)

    if absent_policy == "defer" and topic_owner in remaining:
        return RouteOutcome(None, topic_owner, 0)

    # The policy is guide_answers, or the block was already cut and there is
    # no station left to defer to. Under EXCLUSIVE scope the guide is the
    # only permitted answerer here — never a peer.
    return RouteOutcome(guide_robot_id, None, 1)
