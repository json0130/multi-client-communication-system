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

from decision.flow import DEFAULT_QA_BUDGET_SEC, FlowGraph                              # noqa: E402
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
    {"id": "topic:text-to-speech", "label": "text to speech"},
    {"id": "topic:robot-hardware", "label": "robot hardware"},
]

# Plausible round numbers in the range of the 49 real measurements taken so
# far. Held constant — see the module docstring.
def _block_durations(robot: str) -> dict:
    """One block's steps, in the shape build_script now emits: a single
    merged intro+handoff, greeting, prompt, and the project talk split
    across PROJECT_CHECKLIST's three points."""
    return {
        f"introduce_{robot}": 10,        # teaser + hand-off, one utterance
        f"{robot}_greeting": 8,
        f"{robot}_prompt": 5,
        f"{robot}_project_problem": 7,
        f"{robot}_project_approach": 7,
        f"{robot}_project_impact": 5,
    }


FIXED_DURATIONS = {
    "greeting": 8, "lab_intro": 12, "overview": 8,
    **_block_durations(CHATBOX),
    **_block_durations(NAVEL),
    **_block_durations(SILBOT),
    "transition_to_navel_01": 6, "transition_to_silbot_01": 6,
    "wrap_up": 15, "open_floor": 10,
}

# The arithmetic every expectation below is derived from, stated once:
#
#   fixed (opening 28 + closing 25)                        = 53s
#   scripted per block, uncompressed  chatbox 48, navel 48, silbot 42   = 138s
#   compressible per block (greeting 8 + prompt 5)                      =  13s
#   scripted per block, compressed    chatbox 35, navel 35, silbot 29   =  99s
#   Q&A at the 90s default × 3 windows                                  = 270s
#   Q&A at the 45s floor  × 3 windows                                   = 135s
#
#   uncompressed, default Q&A   53 + 138 + 270 = 461s
#   Q&A at floor only           53 + 138 + 135 = 326s
#   Q&A at floor + all compressed 53 +  99 + 135 = 287s
#
# Compression saves LESS than it used to (13s a block, not 19s). The teaser
# and the hand-off used to be two steps and the teaser was compressible;
# merging them into one utterance means the whole thing carries the hand-off
# and so has to survive, since a robot must never start talking with nobody
# having introduced it. The ladder reaches for a skip sooner as a result —
# a real consequence of the merge, not an accident of these numbers.

QA_FLOOR = 45.0

# The lab's hand-set per-project priorities — importance layer 1, the value
# a deployment configures when no visitor has said anything.
#
# The SPREAD matters, not just the values. With all three equal, the
# planner's (importance, robot_id) sort falls through to the alphabet, and a
# scenario whose expected answer is the alphabetically-first block passes
# whether importance worked or was ignored entirely. Distinct defaults mean
# robot_id is never consulted, so the block that gets cut is evidence about
# importance and nothing else.
#
# chatbox_01 is deliberately LOWEST. A stated interest only ever raises the
# robot that owns the topic, so if the interest target already ranked above
# someone, the bump changes nothing about which block is cheapest to cut.
# Starting it at the bottom is what makes S2's contrast possible at all.
DEFAULTS = {CHATBOX: 0.3, NAVEL: 0.5, SILBOT: 0.7}


def _edge(robot: str, topic: str, n: int = 8, target: float = 1.0) -> RobotTopicEdge:
    e = RobotTopicEdge(robot_id=robot, topic_id=topic)
    for _ in range(n):
        e = e.update(target, Evidence.SUPERVISOR)
    return e


# An UNOWNED topic — in the live vocabulary, nothing declares text-to-speech,
# speech-recognition or robot-hardware. Routing must decline these rather
# than order robots on propagated noise.
ORPHAN = "topic:text-to-speech"

# The competence graph the routing scenarios run against.
#
# SHAPED LIKE PRODUCTION, which it was not. These edges used to be observed
# ones at weight 0.979 with specialised=False, so every routing scenario
# measured the LEARNED-WEIGHT path — while the live graph is 14 declared
# edges at the 0.5 prior with a single observation between them, and decides
# almost everything by declared scope. "Routing accuracy 100%" was scoring a
# configuration the system no longer runs in.
#
# So scope is declared here, as it is live, and the one observed edge is kept
# on an UNDECLARED topic, which is the only place a learned weight can still
# decide anything once scope is exclusive.
EDGES = [
    RobotTopicEdge(robot_id=CHATBOX, topic_id=RAG, specialised=True),
    RobotTopicEdge(robot_id=NAVEL, topic_id=NVI, specialised=True),
    RobotTopicEdge(robot_id=SILBOT, topic_id=NAV, specialised=True),
    # Undeclared, and observed strongly enough to clear MIN_EVIDENCE_MARGIN —
    # the learned path, still reachable exactly where scope is silent.
    _edge(CHATBOX, ORPHAN),
]


# ── Expectations ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExpectedPlan:
    """What a visitor would experience. Never an op list — see the docstring."""

    surviving_blocks: tuple      # in presentation order
    feasible: bool = True
    max_qa_budget_sec: Optional[float] = None   # every surviving window at most this
    min_qa_budget_sec: Optional[float] = None   # ...and at least this
    compressed_blocks: Optional[frozenset] = None   # None = don't assert
    compressed_count: Optional[int] = None      # when WHICH block is incidental
    # Surviving set the same scenario must produce WITHOUT the stated
    # interest. Asserting the two differ is what proves the interest changed
    # the outcome, rather than the outcome merely happening to look right.
    differs_without_interest: Optional[tuple] = None


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

def _plan(budget_sec: float, visitor_topics=None, projects=None,
          defaults=None) -> dict:
    """The real planner over the real script, with fixed durations."""
    projects = projects or PROJECTS
    graph = FlowGraph.from_script(build_script(GUIDE, projects))
    importance = block_importance(
        graph, defaults=DEFAULTS if defaults is None else defaults,
        visitor_topics=visitor_topics, kg_edges=EDGES, kg_links=[],
    )
    result = plan_for_budget(graph, budget_sec, durations=FIXED_DURATIONS,
                             importance=importance)
    result["importance"] = importance
    return result


def _outcome(plan: dict, projects=None) -> dict:
    """Reduce a plan to what a visitor would experience.

    `projects` must be the robot set this plan was built for — reading the
    module-level PROJECTS instead reports blocks that were never in the tour
    whenever a scenario uses a smaller fleet."""
    from decision.models import PlanOpKind
    projects = projects or PROJECTS
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
        r for r in projects if r not in skipped)
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
        id="S1-rung-order",
        description=(
            "A general-audience group, no stated interest, 5m20s for a tour "
            "that would take 7m41s at full length. Every robot present."
        ),
        rationale=(
            "461s of tour into a 320s budget. This scenario is about the "
            "ladder's ORDER, not about importance. Rung 1 tightens Q&A, and "
            "Q&A alone nearly absorbs it: three windows dropping from 90s to "
            "the 45s floor recovers 135s, landing at 326s — 6s over. So rung 2 "
            "must compress exactly ONE block to reach 313s, and rung 3 is "
            "never entered: no robot is skipped. That is the property worth "
            "pinning, because it is the reason the rungs are in this order at "
            "all — a visitor notices a missing robot and does not notice a "
            "shorter question round. WHICH block gets compressed is incidental "
            "here and deliberately not asserted; S2 is the scenario that tests "
            "importance, and asserting it here too would just duplicate it."
        ),
        run=lambda: _outcome(_plan(320)),
        expected=ExpectedPlan(
            surviving_blocks=(CHATBOX, NAVEL, SILBOT),
            feasible=True,
            max_qa_budget_sec=QA_FLOOR,
            min_qa_budget_sec=QA_FLOOR,
            compressed_count=1,
        ),
    ),

    Scenario(
        id="S2-stated-interest-protects",
        description=(
            "A 4m10s budget, tight enough that one project must be cut "
            "outright. Visitor stated an interest in retrieval-augmented "
            "generation, which chatbox_01 owns — and chatbox_01 is the block "
            "the lab's own priorities rank LOWEST."
        ),
        rationale=(
            "250s budget. Q&A to the floor gives 326s; compressing all three "
            "gives 287s, still over — so rung 3 must skip exactly one block, "
            "and 207s afterwards is comfortably inside. WHICH block is the "
            "entire point.\n"
            "  Hand-set defaults are chatbox 0.3, navel 0.5, silbot 0.7. With "
            "no stated interest the cheapest block to cut is chatbox_01, and "
            "the tour keeps navel and silbot.\n"
            "  A stated RAG interest resolves to chatbox_01's topic, whose "
            "graph coverage is 0.848; block_importance blends 0.35*0.3 + "
            "0.65*0.848 = 0.656. navel and silbot have no coverage for that "
            "topic so they sit at 0.35*default + 0.65*0.5, giving 0.500 and "
            "0.570. The ordering INVERTS: navel_01 is now cheapest and gets "
            "cut, and the project the visitor asked about survives.\n"
            "  Two properties make this a real test rather than a coincidence. "
            "All three importances are distinct (0.656/0.500/0.570), so the "
            "planner's (importance, robot_id) sort never reaches the "
            "tie-break and the alphabet plays no part. And the alphabet would "
            "give a DIFFERENT answer — chatbox_01 is alphabetically first, so "
            "an implementation that ignored importance entirely would cut it "
            "and produce (navel, silbot), which is exactly the surviving set "
            "asserted as the no-interest control below."
        ),
        run=lambda: {
            **_outcome(_plan(250, visitor_topics=[RAG])),
            "without_interest": _outcome(_plan(250))["surviving_blocks"],
        },
        expected=ExpectedPlan(
            surviving_blocks=(CHATBOX, SILBOT),
            feasible=True,
            max_qa_budget_sec=QA_FLOOR,
            compressed_blocks=frozenset({CHATBOX, NAVEL, SILBOT}),
            differs_without_interest=(NAVEL, SILBOT),
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
        if e.compressed_count is not None:
            out.append(("compressed_count",
                        len(actual["compressed_blocks"]) == e.compressed_count,
                        f"{len(actual['compressed_blocks'])} vs {e.compressed_count}"))
        if e.differs_without_interest is not None:
            got = actual.get("without_interest")
            out.append(("no-interest control",
                        got == e.differs_without_interest,
                        f"{got} vs {e.differs_without_interest}"))
            # The discrimination itself: if the stated interest changed
            # nothing, importance never reached the planner.
            out.append(("interest changed the cut",
                        got != actual["surviving_blocks"],
                        f"with={actual['surviving_blocks']} without={got}"))

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


# ── Generated scenarios ───────────────────────────────────────────────────────
# Inputs generated automatically; expected outputs derived from the documented
# rules by tools/eval_oracle.py, never read off what the planner produced.
#
# Importance here comes from hand-set defaults ALONE — no visitor interest.
# That keeps the oracle's importance trivially checkable (importance ==
# defaults) rather than requiring it to reimplement competence propagation,
# which has its own tests in test_kg_infer.py and would be a different thing
# to measure. The visitor-interest path is covered by hand-derived S2, where
# the blend is worked out explicitly.

ROBOT_SETS = [
    [CHATBOX, NAVEL],
    [CHATBOX, NAVEL, SILBOT],
]

# Budgets spanning every rung: comfortably fits, Q&A-only, Q&A+compress,
# one skip, several skips, and impossible.
#
# 40s is below the 53s opening-plus-closing floor, so it is the only budget
# here that reaches rung 4 AND comes back infeasible — dropping every project
# still leaves a tour that cannot fit. Without it the grid never exercised
# the "the budget cannot be met" branch at all, which
# test_the_generated_grid_is_actually_broad caught.
BUDGET_GRID = [520, 400, 340, 300, 270, 240, 190, 120, 40]

# Distinct orderings, so the planner's (importance, robot_id) sort never
# falls through to the alphabet in a generated case either.
DEFAULT_PROFILES = {
    "ascending":  {CHATBOX: 0.3, NAVEL: 0.5, SILBOT: 0.7},
    "descending": {CHATBOX: 0.8, NAVEL: 0.55, SILBOT: 0.2},
    "middle-out": {CHATBOX: 0.45, NAVEL: 0.9, SILBOT: 0.6},
}

TOPIC_OWNER = {RAG: CHATBOX, NVI: NAVEL, NAV: SILBOT}
TOPIC_UTTERANCE = {
    RAG: "how does retrieval augmented generation work",
    NVI: "tell me about non verbal interaction",
    NAV: "how does social robot navigation work",
}


def _generated_plan_cases():
    from tools.eval_oracle import block_facts_from_script, derive_plan

    for projects in ROBOT_SETS:
        steps = build_script(GUIDE, projects)
        for profile_name, profile in DEFAULT_PROFILES.items():
            defaults = {r: profile[r] for r in projects}
            fixed, blocks = block_facts_from_script(steps, FIXED_DURATIONS, defaults)
            for budget in BUDGET_GRID:
                expected = derive_plan(budget, fixed, blocks)
                yield {
                    "id": f"G-plan-{len(projects)}r-{profile_name}-{budget}s",
                    "kind": "plan",
                    "projects": tuple(projects),
                    "defaults": defaults,
                    "budget": budget,
                    "expected": expected,
                }


def _generated_route_cases():
    from tools.eval_oracle import derive_route

    for topic, owner in TOPIC_OWNER.items():
        for label, absent, remaining in (
            ("present", (), tuple(PROJECTS)),
            ("absent-block-ahead", (owner,), tuple(PROJECTS)),
            ("absent-block-cut", (owner,),
             tuple(r for r in PROJECTS if r != owner)),
        ):
            yield {
                "id": f"G-route-{owner}-{label}",
                "kind": "route",
                "utterance": TOPIC_UTTERANCE[topic],
                "absent": absent,
                "remaining": remaining,
                "expected": derive_route(owner, absent, remaining, GUIDE),
            }


def generated_cases() -> list:
    return list(_generated_plan_cases()) + list(_generated_route_cases())


def run_generated(case: dict) -> list:
    """[(assertion, passed, detail)] for one generated case."""
    e = case["expected"]
    out = []

    if case["kind"] == "plan":
        actual = _outcome(_plan(case["budget"], projects=list(case["projects"]),
                                defaults=case["defaults"]),
                          projects=list(case["projects"]))
        out.append(("surviving_blocks",
                    actual["surviving_blocks"] == e.surviving_blocks,
                    f"{actual['surviving_blocks']} vs {e.surviving_blocks}"))
        out.append(("compressed_blocks",
                    actual["compressed_blocks"] == e.compressed_blocks,
                    f"{set(actual['compressed_blocks'])} vs {set(e.compressed_blocks)}"))
        out.append(("feasible", actual["feasible"] == e.feasible,
                    f"{actual['feasible']} vs {e.feasible}"))
        # The planner emits a rounded integer per window; no ops at all means
        # Q&A was never tightened and stays at the default.
        got_qa = (actual["qa_budgets"][0] if actual["qa_budgets"]
                  else DEFAULT_QA_BUDGET_SEC)
        out.append(("qa_budget", got_qa == round(e.qa_budget_sec),
                    f"{got_qa} vs {round(e.qa_budget_sec)}"))
    else:
        actual = _route(case["utterance"], absent=case["absent"],
                        remaining_blocks=case["remaining"])
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
    ap.add_argument("--generated", action="store_true",
                    help="also run the rule-derived generated scenarios")
    args = ap.parse_args()

    total = passed = 0
    print("=" * 78)
    print("  Hand-derived scenarios — expectations reasoned out, not read off")
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
            print(f"      {'ok ' if p else 'XX '} {name:<24} {detail}")

    print("\n" + "=" * 78)
    print(f"  {passed}/{total} assertions passed across {len(SCENARIOS)} "
          f"hand-derived scenarios")
    print("=" * 78)

    if args.generated:
        cases = generated_cases()
        g_total = g_passed = 0
        failures = []
        for case in cases:
            results = run_generated(case)
            g_total += len(results)
            g_passed += sum(1 for _, p, _ in results if p)
            bad = [f"{n}: {d}" for n, p, d in results if not p]
            if bad:
                failures.append((case["id"], bad))
        print()
        print("=" * 78)
        print("  Generated scenarios — inputs generated, answers derived by rule")
        print("=" * 78)
        print(f"  {len(cases)} cases "
              f"({sum(1 for c in cases if c['kind'] == 'plan')} plan, "
              f"{sum(1 for c in cases if c['kind'] == 'route')} routing)")
        for cid, bad in failures:
            print(f"  FAIL {cid}")
            for b in bad:
                print(f"        {b}")
        print(f"  {g_passed}/{g_total} assertions passed")
        print("=" * 78)
        total += g_total
        passed += g_passed

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
