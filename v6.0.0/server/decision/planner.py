"""
decision/planner.py
===================
"The visitor has 15 minutes and the tour takes 25." What gets cut?

Two pieces: how important each block is to THIS visitor, and the ladder that
turns a shortfall into a list of PlanOps.

IMPORTANCE, IN THREE LAYERS
  1. a fixed default per project, hand-set   the fallback, and the common case —
                                             most visitors state no preference
  2. visitor-derived                         a stated interest resolves to
                                             topics, and a block owning those
                                             topics matters more
  3. learned from corrections                "you should not have cut that one".
                                             NOT BUILT. The hook is the
                                             `learned` argument, which is
                                             blended in when it exists.

Layers 1 and 2 are here. Layer 3 is deliberately absent rather than stubbed: a
placeholder that silently contributes zero is indistinguishable from a working
one that has no data.

Importance is computed PER RUN and never written onto the flow graph. It depends
on who is standing there, and the graph holds only what is true of the tour
itself. Same read-time discipline as competence propagation.

THE LADDER
Ordered by what a visitor actually notices, not by what recovers the most time:

  1. tighten Q&A      largest share of tour time and the least noticed. Nobody
                      remembers a question round that ended early; everybody
                      remembers a robot that never spoke.
  2. compress blocks  drop intro/handoff/greeting/prompt, keep the research talk
  3. skip a block     least important first, and only when 1 and 2 are not enough
  4. drop remaining   the tour cannot fit at all; the closing still runs

Each rung is tried to exhaustion before the next, and every rung re-estimates
rather than assuming its own saving.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from decision.flow import DEFAULT_QA_BUDGET_SEC, FlowGraph
from decision.kg import PRIOR
from decision.models import PlanOp, PlanOpKind

QA_FLOOR_SEC = 45.0
"""How short a Q&A window may be squeezed. Below this the window stops being a
question round and becomes a gesture at one — a visitor who gets 20 seconds to
ask about a project would be better served by the block being skipped honestly."""

DEFAULT_IMPORTANCE = 0.5
"""A block nobody has ranked. Neutral, so an unranked block is neither protected
nor first to go."""


# ── Importance ────────────────────────────────────────────────────────────────

def resolve_emphasis(
    utterance_topics: Optional[Sequence[str]],
    profile_topics: Optional[Sequence[str]],
) -> tuple:
    """
    Explicit trigger-type ordering for WHICH topics drive importance this turn.

    This makes a sequence explicit that used to be implicit and split across
    two files: whether a live message stated an interest, then whether the
    pre-demo visitor profile did, then falling back to no stated preference at
    all (layer-1 hand-set defaults, applied unconditionally by block_importance
    regardless of what this returns).

        1. a freshly stated interest THIS TURN       — most specific, most recent
        2. the pre-demo visitor profile's interest    — the standing baseline
        3. neither                                    — []; layer 1 alone applies

    A fresh utterance ALWAYS wins over the standing profile rather than being
    merged with it: a visitor who stated "I'm mainly interested in navigation"
    at the start and then asks specifically about emotion recognition mid-tour
    is refining, not adding to, their stated interest — merging the two would
    keep inflating the importance of a topic they have since moved past.

    Returns (topics, source) — source is "utterance" | "profile" | "none",
    which is what lets a later analysis separate "the visitor said this just
    now" from "this was the assumed interest all along".
    """
    if utterance_topics:
        return tuple(utterance_topics), "utterance"
    if profile_topics:
        return tuple(profile_topics), "profile"
    return (), "none"


DECLARED_COVERAGE = 0.9
"""Visitor-interest coverage credited to a robot whose DECLARED area covers
the topic, regardless of how little has been observed.

High rather than 1.0: declaring a topic is the clearest signal that a block
is what the visitor came for, but the hand-set default still gets its 35%
say, so a project the lab considers unmissable is not automatically demoted
below one the visitor mentioned in passing.

The value has to clear the defaults' own spread to do anything. With defaults
0.3/0.5/0.7 and the 0.35/0.65 blend, 0.9 lifts a declared block by 0.26 —
comfortably more than the 0.2 between adjacent defaults, and far more than
the 0.012 that unobserved declared edges produced before this existed."""


def block_importance(
    graph: FlowGraph,
    defaults: Optional[dict] = None,
    visitor_topics: Optional[Sequence[str]] = None,
    kg_edges: Optional[Iterable] = None,
    kg_links: Optional[Iterable] = None,
    learned: Optional[dict] = None,
) -> dict:
    """
    {robot_id: importance in [0,1]} for this run.

    `visitor_topics` are topic ids a visitor's stated interest resolved to. A
    block's visitor score is how well ITS robot covers those topics, read from
    the competence graph — which is the reuse that makes the two graphs worth
    keeping separate: competence answers "who knows this", and the planner asks
    it "so how much would cutting them cost this visitor".

    With no stated interest the defaults stand alone, which is the common case
    and must not degrade into everything scoring the same.
    """
    defaults = defaults or {}
    learned = learned or {}
    out = {}

    coverage = {}
    if visitor_topics and kg_edges is not None:
        from decision.kg_infer import infer
        edges, links = list(kg_edges), list(kg_links or [])
        wanted = set(visitor_topics)
        # Whose DECLARED area covers what the visitor asked about. This has to
        # be read separately from infer(), which only counts edges carrying
        # observations: on the live graph every declared edge sits at the
        # prior with nothing behind it, so coverage came back 0.5188 for the
        # robot that owns the topic against 0.5 for everyone else. Blended,
        # that is a 0.012 difference against hand-set defaults spread 0.2
        # apart — sixteen times too small to protect anything. A visitor could
        # state an interest and watch that exact project get cut.
        #
        # Declaring the topic is the strongest possible statement that a block
        # is the relevant one, and it is configuration, so it needs no
        # evidence to count. Learned coverage still applies on top, which is
        # what keeps an undeclared-but-demonstrated robot from scoring zero.
        declared = {b.robot_id for b in graph.blocks
                    if any(getattr(e, "specialised", False)
                           and e.robot_id == b.robot_id and e.topic_id in wanted
                           for e in edges)}
        for b in graph.blocks:
            posterior = infer(edges, links, b.robot_id, visitor_topics)
            scores = [posterior.get(t, PRIOR) for t in visitor_topics]
            learned_cov = sum(scores) / len(scores) if scores else PRIOR
            coverage[b.robot_id] = (max(DECLARED_COVERAGE, learned_cov)
                                    if b.robot_id in declared else learned_cov)

    for b in graph.blocks:
        base = float(defaults.get(b.robot_id, DEFAULT_IMPORTANCE))
        if b.robot_id in coverage:
            # A visitor's stated interest outweighs a hand-set default, but does
            # not erase it — a project the lab considers unmissable stays hard to
            # cut even for a visitor who did not ask for it.
            base = 0.35 * base + 0.65 * coverage[b.robot_id]
        if b.robot_id in learned:
            base = 0.5 * base + 0.5 * float(learned[b.robot_id])
        out[b.robot_id] = round(max(0.0, min(1.0, base)), 4)
    return out


# ── The ladder ────────────────────────────────────────────────────────────────

def plan_for_budget(
    graph: FlowGraph,
    budget_sec: float,
    durations: Optional[dict] = None,
    importance: Optional[dict] = None,
    qa_budget: float = DEFAULT_QA_BUDGET_SEC,
    qa_floor: float = QA_FLOOR_SEC,
) -> dict:
    """
    Fit the remaining tour into `budget_sec`. Returns ops plus the reasoning.

    Reports `feasible` rather than silently doing its best: a 5-minute budget
    for a tour whose opening and closing alone take 8 cannot be met, and a
    planner that returns a plan anyway has told the operator nothing.
    """
    durations = durations or {}
    importance = importance or {}
    ops: list = []
    trace: list = []

    current_qa = qa_budget
    compressed: set = set()
    skipped: set = set()

    def estimate():
        return graph.estimate(durations, current_qa, compressed, skipped)

    start = estimate()
    if start["total_sec"] <= budget_sec:
        return {"ops": [], "feasible": True, "fits_already": True,
                "estimate": start, "trace": ["already fits"],
                "measured_coverage": round(graph.measured_coverage(durations), 3)}

    # ── Rung 1: tighten Q&A ──────────────────────────────────────────────────
    if current_qa > qa_floor:
        needed = start["total_sec"] - budget_sec
        windows = sum(len(b.qa_steps) for b in graph.blocks if b.robot_id not in skipped)
        if windows:
            # Rounded to whole seconds BEFORE it is used, because that is the
            # value the op carries and the orchestrator enforces. Reasoning
            # with 67.667 while emitting 68 made the planner report a tour it
            # was not going to produce: three windows applied at 68s cost 1s
            # more than the estimate that declared the budget met, so
            # feasible=True described a plan that overran. Deciding in the
            # units the command actually uses makes the estimate and the plan
            # the same object by construction, and errs toward cutting
            # slightly more rather than promising a fit that is only
            # reachable at fractional precision.
            reduced = max(qa_floor, float(round(current_qa - needed / windows)))
            if reduced < current_qa:
                current_qa = reduced
                for b in graph.blocks:
                    if b.robot_id not in skipped and b.qa_steps:
                        ops.append(PlanOp(PlanOpKind.SET_QA_BUDGET,
                                          robot_id=b.robot_id, seconds=round(current_qa)))
                trace.append(f"Q&A tightened to {current_qa:.0f}s "
                             f"({estimate()['total_sec']:.0f}s)")
    if estimate()["total_sec"] <= budget_sec:
        return _done(ops, True, estimate(), trace, graph, durations)

    # ── Rung 2: compress, least important first ──────────────────────────────
    # Ordered so that if compressing two of three blocks is enough, the ones
    # that keep their full introduction are the ones this visitor came for.
    for b in sorted(graph.blocks,
                    key=lambda x: (importance.get(x.robot_id, DEFAULT_IMPORTANCE),
                                   x.robot_id)):
        if b.robot_id in skipped or b.robot_id in compressed:
            continue
        compressed.add(b.robot_id)
        ops.append(PlanOp(PlanOpKind.COMPRESS, robot_id=b.robot_id))
        trace.append(f"compressed {b.robot_id} ({estimate()['total_sec']:.0f}s)")
        if estimate()["total_sec"] <= budget_sec:
            return _done(ops, True, estimate(), trace, graph, durations)

    # ── Rung 3: skip, least important first ──────────────────────────────────
    for b in sorted(graph.blocks,
                    key=lambda x: (importance.get(x.robot_id, DEFAULT_IMPORTANCE),
                                   x.robot_id)):
        if b.robot_id in skipped:
            continue
        # Never skip the last one standing. A tour with no projects is not a
        # short tour, it is a different event, and the operator should be told
        # the budget is impossible rather than handed that.
        if len(graph.blocks) - len(skipped) <= 1:
            break
        skipped.add(b.robot_id)
        ops.append(PlanOp(PlanOpKind.SKIP, robot_id=b.robot_id))
        trace.append(f"skipped {b.robot_id} "
                     f"(importance {importance.get(b.robot_id, DEFAULT_IMPORTANCE):.2f}, "
                     f"{estimate()['total_sec']:.0f}s)")
        if estimate()["total_sec"] <= budget_sec:
            return _done(ops, True, estimate(), trace, graph, durations)

    # ── Rung 4: drop everything remaining ────────────────────────────────────
    # Reached only when even the LAST project, fully compressed, does not fit.
    # There is nothing left to protect at that point, so the choice is not
    # "which block to sacrifice" — it is "keep it and overrun, or close cleanly
    # without it". This is what the docstring's fourth rung promises and what
    # the old ad hoc severe-overrun rule used to do unconditionally; here it is
    # reached only after 1-3 have been exhausted.
    final = estimate()
    if final["total_sec"] > budget_sec and len(graph.blocks) > len(skipped):
        ops.append(PlanOp(PlanOpKind.DROP_REMAINING))
        dropped = graph.estimate(durations, current_qa, compressed,
                                 {b.robot_id for b in graph.blocks})
        feasible = dropped["total_sec"] <= budget_sec
        trace.append(f"dropped every remaining project ({dropped['total_sec']:.0f}s)")
        if not feasible:
            trace.append(f"still {dropped['total_sec'] - budget_sec:.0f}s over "
                         f"with nothing left to cut — budget cannot be met")
        return _done(ops, feasible, dropped, trace, graph, durations)

    trace.append(f"still {final['total_sec'] - budget_sec:.0f}s over with one "
                 f"project left — budget cannot be met")
    return _done(ops, False, final, trace, graph, durations)


def _done(ops, feasible, estimate, trace, graph, durations) -> dict:
    return {
        "ops": ops,
        "feasible": feasible,
        "fits_already": False,
        "estimate": estimate,
        "trace": trace,
        # An estimate built entirely from defaults is arithmetic, not a
        # prediction. Surfaced so a caller can say so rather than quoting the
        # total as measured.
        "measured_coverage": round(graph.measured_coverage(durations), 3),
    }
