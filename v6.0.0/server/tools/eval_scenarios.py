"""
tools/eval_scenarios.py
=======================
The objective half of the evaluation: scenarios whose correct answer is
defined UP FRONT, scored automatically, no participants needed.

    python3 tools/eval_scenarios.py            # score every scenario
    python3 tools/eval_scenarios.py --verbose  # show ops and traces too

WHY THE EXPECTATIONS ARE WRITTEN OUT BY HAND
Every `expected` below was derived by reasoning about the documented rules —
the compression ladder's rung order in decision/planner.py, the importance
blend in block_importance, ABSENT_ROBOT_POLICY in decision/kg_policy.py —
and written down BEFORE running anything. That ordering is the whole point.
An expectation read off the implementation's current output tests only that
the code is self-consistent; it cannot fail, and a metric that cannot fail
measures nothing. The derivation is recorded in each scenario's `rationale`
so a reader can check the reasoning without re-deriving it, and so a
disagreement between expectation and output is a real finding either way —
either the reasoning is wrong or the implementation is.

WHY SCORING IS ON OUTCOME, NOT ON THE OP LIST
Two different op sequences can produce an identical tour: tightening Q&A to
45s and then compressing a block reaches the same place as several other
orderings, and SET_QA_BUDGET emitted per-block versus once means nothing to
a visitor. Scoring exact op sequences would mark correct plans wrong. So the
assertions are about what a visitor would actually experience — which blocks
survive, in what order, how long the Q&A windows are, whether the budget was
met — and never about how the planner got there.

DURATIONS ARE FIXED, NOT LIVE
Deliberately not read from Supabase. tools/plan_replay_harness.py showed the
planner's rung can flip purely because accumulated duration data drifted, so
a scenario scored against live durations would silently change its own
correct answer between runs. These are plausible round numbers in the range
of the real measurements, held constant so the scenarios stay comparable
across conditions and across months.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.flow import FlowGraph                              # noqa: E402
from decision.kg import Evidence, RobotTopicEdge                 # noqa: E402
from decision.kg_feedback import Segment                         # noqa: E402
from decision.kg_policy import KGRouter                          # noqa: E402
from decision.planner import block_importance, plan_for_budget   # noqa: E402
from demo.demo_script import build_script                        # noqa: E402

GUIDE = "pepper_01"
CHATBOX, NAVEL, SILBOT = "chatbox_01", "navel_01", "silbot_01"
PROJECTS = [CHATBOX, NAVEL, SILBOT]

RAG = "topic:retrieval-augmented-generation"
NAV = "topic:social-robot-navigation"
NVI = "topic:non-verbal-interaction"

TOPICS = [
    {"id": RAG, "label": "retrieval augmented generation"},
    {"id": NAV, "label": "social robot navigation"},
    {"id": NVI, "label": "non verbal interaction"},
]

# Plausible round numbers in the range of the 49 real measurements taken so
# far. Held constant — see the module docstring.
FIXED_DURATIONS = {
    "greeting": 8, "lab_intro": 12, "overview": 8,
    "intro_project_a": 6, "introduce_chatbox_01": 6, "chatbox_01_greeting": 8,
    "chatbox_01_prompt": 5, "chatbox_01_project": 19, "transition_to_navel_01": 6,
    "intro_project_b": 6, "introduce_navel_01": 6, "navel_01_greeting": 8,
    "navel_01_prompt": 5, "navel_01_project": 19, "transition_to_silbot_01": 6,
    "intro_project_c": 6, "introduce_silbot_01": 6, "silbot_01_greeting": 8,
    "silbot_01_prompt": 5, "silbot_01_project": 19,
    "wrap_up": 15, "open_floor": 10,
}

# The arithmetic every expectation below is derived from, stated once:
#
#   fixed (opening 28 + closing 25)                        = 53s
#   scripted per block, uncompressed  chatbox 50, navel 50, silbot 44   = 144s
#   compressible per block (intro 6 + greeting 8 + prompt 5)            =  19s
#   scripted per block, compressed    chatbox 31, navel 31, silbot 25   =  87s
#   Q&A at the 90s default × 3 windows                                  = 270s
#   Q&A at the 45s floor  × 3 windows                                   = 135s
#
#   uncompressed, default Q&A   53 + 144 + 270 = 467s
#   Q&A at floor only           53 + 144 + 135 = 332s
#   Q&A at floor + all compressed 53 + 87 + 135 = 275s

QA_FLOOR = 45.0


def _edge(robot: str, topic: str, n: int = 8, target: float = 1.0) -> RobotTopicEdge:
    e = RobotTopicEdge(robot_id=robot, topic_id=topic)
    for _ in range(n):
        e = e.update(target, Evidence.SUPERVISOR)
    return e


# The competence graph the routing scenarios run against: each robot clearly
# owns one topic. Deliberately unambiguous — these scenarios test whether
# routing HAPPENS, not whether the graph can resolve a hard case.
EDGES = [_edge(CHATBOX, RAG), _edge(NAVEL, NVI), _edge(SILBOT, NAV)]


# ── Expectations ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExpectedPlan:
    """What a visitor would experience. Never an op list — see the docstring."""

    surviving_blocks: tuple      # in presentation order
    feasible: bool = True
    max_qa_budget_sec: Optional[float] = None   # every surviving window at most this
    min_qa_budget_sec: Optional[float] = None   # ...and at least this
    compressed_blocks: Optional[frozenset] = None   # None = don't assert


@dataclass(frozen=True)
class ExpectedRoute:
    answering_robot: Optional[str]     # None when the question is deferred
    deferred_to: Optional[str] = None
    observations_written: int = 0


@dataclass
class Scenario:
    id: str
    description: str
    rationale: str          # the hand-derivation; read this before trusting a pass
    run: Callable
    expected: object


# ── Scenario runners ──────────────────────────────────────────────────────────

def _plan(budget_sec: float, visitor_topics=None) -> dict:
    """The real planner over the real script, with fixed durations."""
    graph = FlowGraph.from_script(build_script(GUIDE, PROJECTS))
    importance = block_importance(
        graph, defaults={}, visitor_topics=visitor_topics,
        kg_edges=EDGES, kg_links=[],
    )
    result = plan_for_budget(graph, budget_sec, durations=FIXED_DURATIONS,
                             importance=importance)
    result["importance"] = importance
    return result


def _outcome(plan: dict) -> dict:
    """Reduce a plan to what a visitor would experience."""
    from decision.models import PlanOpKind
    skipped, compressed, qa_budgets = set(), set(), []
    dropped_all = False
    for op in plan["ops"]:
        if op.kind is PlanOpKind.SKIP:
            skipped.add(op.robot_id)
        elif op.kind is PlanOpKind.COMPRESS:
            compressed.add(op.robot_id)
        elif op.kind is PlanOpKind.SET_QA_BUDGET:
            qa_budgets.append(op.seconds)
        elif op.kind is PlanOpKind.DROP_REMAINING:
            dropped_all = True
    surviving = () if dropped_all else tuple(
        r for r in PROJECTS if r not in skipped)
    return {
        "surviving_blocks": surviving,
        "compressed_blocks": frozenset(compressed),
        "qa_budgets": qa_budgets,
        "feasible": plan["feasible"],
        "total_sec": plan["estimate"]["total_sec"],
    }


def _route(utterance: str, absent=(), remaining_blocks=None) -> dict:
    """The real router, plus what the Segment would record."""
    router = KGRouter(EDGES, [], TOPICS, explore=False,
                      absent_robot_ids=absent)
    decision = router.decide(
        utterance, PROJECTS,
        remaining_block_ids=(PROJECTS if remaining_blocks is None
                             else remaining_blocks),
        guide_robot_id=GUIDE,
    )
    segment = Segment()
    if decision is not None and not decision.is_deferred:
        segment.note_routed(decision.robot_id, decision.topic_id)
    return {
        "answering_robot": None if decision is None or decision.is_deferred
                           else decision.robot_id,
        "deferred_to": decision.deferred_to if decision else None,
        "observations_written": len(segment.observations()),
        "reason": decision.reason if decision else "(no opinion)",
    }


# ── The four scenarios ────────────────────────────────────────────────────────

SCENARIOS = [
    Scenario(
        id="S1-rushed-general",
        description=(
            "A general-audience group, no stated interest, 5m30s for a tour "
            "that would take 7m47s at full length. Every robot present."
        ),
        rationale=(
            "467s of tour into a 330s budget. The ladder's first rung tightens "
            "Q&A, and Q&A alone can absorb it: three windows dropping from 90s "
            "to the 45s floor recovers 135s, landing at 332s. That is 2s over, "
            "so rung 2 must also compress ONE block (the least important; all "
            "are equal here so the tie breaks alphabetically to chatbox_01), "
            "reaching 313s. No robot is skipped. This is the case the ladder's "
            "ordering exists for: a visitor notices a missing robot and does "
            "not notice a shorter question round, so all three still present."
        ),
        run=lambda: _outcome(_plan(330)),
        expected=ExpectedPlan(
            surviving_blocks=(CHATBOX, NAVEL, SILBOT),
            feasible=True,
            max_qa_budget_sec=QA_FLOOR,
            min_qa_budget_sec=QA_FLOOR,
            compressed_blocks=frozenset({CHATBOX}),
        ),
    ),

    Scenario(
        id="S2-stated-interest-protects",
        description=(
            "Same 4m10s budget, tight enough that one project must be cut. "
            "Visitor stated an interest in retrieval-augmented generation, "
            "which chatbox_01 owns."
        ),
        rationale=(
            "250s budget. Q&A to the floor gives 332s, compressing all three "
            "gives 275s, still over — so rung 3 must skip exactly one block, "
            "and 199s afterwards is comfortably inside. WHICH block is the "
            "whole point: block_importance blends a 0.5 default with the "
            "visitor's topic coverage (0.35/0.65), so chatbox_01 rises to "
            "~0.73 while navel_01 and silbot_01 stay at 0.5. Least-important "
            "goes first and the 0.5 tie breaks alphabetically, so navel_01 is "
            "cut and the project the visitor actually asked about survives. "
            "Contrast with S1's alphabetical tie-break: the stated interest is "
            "the ONLY difference, so a run that cuts chatbox_01 here means "
            "importance never reached the planner."
        ),
        run=lambda: _outcome(_plan(250, visitor_topics=[RAG])),
        expected=ExpectedPlan(
            surviving_blocks=(CHATBOX, SILBOT),
            feasible=True,
            max_qa_budget_sec=QA_FLOOR,
            compressed_blocks=frozenset({CHATBOX, NAVEL, SILBOT}),
        ),
    ),

    Scenario(
        id="S3-reroute-to-owner",
        description=(
            "The group is at ChatBox's station. A visitor asks ChatBox a "
            "question about social robot navigation — silbot_01's subject. "
            "Every robot is present and every block is still ahead."
        ),
        rationale=(
            "The utterance resolves to topic:social-robot-navigation by word "
            "overlap, and silbot_01 is the only robot with observed competence "
            "there (weight 1.0 over 8 supervisor observations, clamped ~0.86 "
            "against the others' 0.5 prior). Nothing blocks it: silbot_01 is "
            "eligible and present. So the question must leave the robot that "
            "received it and go to the one that owns the subject — the "
            "behaviour QA_ROUTE was recording but not acting on until "
            "route_question landed. One observation is written, because a real "
            "robot really answered a real question on a resolved topic."
        ),
        run=lambda: _route("how does social robot navigation work"),
        expected=ExpectedRoute(answering_robot=SILBOT, observations_written=1),
    ),

    Scenario(
        id="S4-absent-owner-defers",
        description=(
            "Identical to S3, except silbot_01 has stepped away from the group "
            "and its block is still ahead on the itinerary."
        ),
        rationale=(
            "Presence is filtered at the same point as eligibility, before "
            "ranking, so silbot_01 cannot be picked however well it rates. "
            "Under the default ABSENT_ROBOT_POLICY ('defer') and with "
            "silbot_01's block still in the remaining plan, the honest answer "
            "is that the topic gets covered properly at silbot_01's own "
            "station rather than improvised now by whoever happens to be "
            "standing there. Nobody answers this turn, so ZERO observations: "
            "no robot handled the question and the absent one was never "
            "judged, and recording either would inflate n_obs from an event "
            "that did not happen."
        ),
        run=lambda: _route("how does social robot navigation work",
                           absent=(SILBOT,)),
        expected=ExpectedRoute(answering_robot=None, deferred_to=SILBOT,
                               observations_written=0),
    ),
]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(scenario: Scenario, actual: dict) -> list:
    """[(assertion_name, passed, detail)] — outcome-level only."""
    e, out = scenario.expected, []

    if isinstance(e, ExpectedPlan):
        out.append(("surviving_blocks",
                    actual["surviving_blocks"] == e.surviving_blocks,
                    f"{actual['surviving_blocks']} vs {e.surviving_blocks}"))
        out.append(("feasible", actual["feasible"] == e.feasible,
                    f"{actual['feasible']} vs {e.feasible}"))
        if e.max_qa_budget_sec is not None:
            ok = all(b <= e.max_qa_budget_sec for b in actual["qa_budgets"])
            out.append((f"qa <= {e.max_qa_budget_sec:.0f}s", ok,
                        str(actual["qa_budgets"])))
        if e.min_qa_budget_sec is not None:
            ok = all(b >= e.min_qa_budget_sec for b in actual["qa_budgets"])
            out.append((f"qa >= {e.min_qa_budget_sec:.0f}s", ok,
                        str(actual["qa_budgets"])))
        if e.compressed_blocks is not None:
            out.append(("compressed_blocks",
                        actual["compressed_blocks"] == e.compressed_blocks,
                        f"{set(actual['compressed_blocks'])} vs {set(e.compressed_blocks)}"))

    elif isinstance(e, ExpectedRoute):
        out.append(("answering_robot",
                    actual["answering_robot"] == e.answering_robot,
                    f"{actual['answering_robot']} vs {e.answering_robot}"))
        out.append(("deferred_to", actual["deferred_to"] == e.deferred_to,
                    f"{actual['deferred_to']} vs {e.deferred_to}"))
        out.append(("observations_written",
                    actual["observations_written"] == e.observations_written,
                    f"{actual['observations_written']} vs {e.observations_written}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true",
                    help="also print the rationale and the raw outcome")
    args = ap.parse_args()

    total = passed = 0
    print("=" * 78)
    print("  Objective evaluation scenarios — expectations pre-registered")
    print("=" * 78)
    for sc in SCENARIOS:
        actual = sc.run()
        results = score(sc, actual)
        ok = all(p for _, p, _ in results)
        total += len(results)
        passed += sum(1 for _, p, _ in results if p)
        print(f"\n{'PASS' if ok else 'FAIL'}  {sc.id}")
        print(f"      {sc.description}")
        if args.verbose:
            print(f"      rationale: {sc.rationale}")
            print(f"      actual:    {actual}")
        for name, p, detail in results:
            print(f"      {'ok ' if p else 'XX '} {name:<22} {detail}")

    print("\n" + "=" * 78)
    print(f"  {passed}/{total} assertions passed across {len(SCENARIOS)} scenarios")
    print("=" * 78)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
