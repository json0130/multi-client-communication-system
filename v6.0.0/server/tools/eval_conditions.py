"""
tools/eval_conditions.py
========================
Scores the generated scenarios under each ABLATION CONDITION, so the
scenarios produce a comparison rather than a single self-report.

    python3 tools/eval_conditions.py            # the condition x family matrix
    python3 tools/eval_conditions.py --detail   # plus per-case failures

WHY THIS EXISTS
tools/eval_scenarios.py scores exactly one configuration: everything
switched on. That answers "does the system agree with the rules" and
nothing about whether any of the machinery earns its place. A number with
no baseline beside it is not a result.

THE CONDITIONS ARE REAL CODE PATHS, NOT SIMULATIONS OF THEM
Every condition here is reached by withholding a dependency the production
wiring injects, never by a flag that changes behaviour for the benchmark's
benefit:

  planner off   HeuristicPolicy(flow_planner=None) falls back to the ad hoc
                ladder that predates decision/planner.py — a fixed 60s Q&A
                constant, compress everything remaining, no importance
                ordering, no Q&A floor, no feasibility check. That code is
                still in _decide_revise for exactly this reason.
  routing off   No KGRouter, so QA_ROUTE degrades to its documented baseline:
                whoever received the question answers it.

So "baseline" is not a strawman written to lose. It is what this system
actually did before the graph and the planner existed, still reachable,
still exercised by its own tests.

WHAT IS DELIBERATELY ABSENT
No LLM condition. An LLM baseline needs a prompt carrying the op vocabulary
and stays nondeterministic even at temperature 0, so it would import
sampling noise into a comparison that is otherwise exact. It is a separate
piece of work with its own controls; mixing it in here would make every
number in this table a distribution rather than a value.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.flow import DEFAULT_QA_BUDGET_SEC, FlowGraph      # noqa: E402
from decision.models import (                                    # noqa: E402
    ActionKind, DecisionPoint, Observation, PlanOpKind, StepRole,
)
from decision.kg_policy import KGRouter                          # noqa: E402
from decision.policy import HeuristicPolicy                      # noqa: E402
from decision.planner import block_importance, plan_for_budget   # noqa: E402
from decision.flow import StepRef                                # noqa: E402
from demo.demo_script import build_script                        # noqa: E402

from tools import eval_scenarios as ev                           # noqa: E402
from tools.eval_oracle import block_facts_from_script            # noqa: E402


# ── Conditions ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    use_planner: bool
    use_kg_routing: bool


CONDITIONS = [
    Condition("full", "flow planner + competence-graph routing", True, True),
    Condition("no-planner", "ad hoc ladder + competence-graph routing", False, True),
    Condition("no-routing", "flow planner + receiver answers", True, False),
    Condition("baseline", "ad hoc ladder + receiver answers", False, False),
]


# ── Building the observation the policy actually reads ────────────────────────

def _observation(projects, budget_sec, utterance) -> Observation:
    """
    The state a live PLAN_REVISE would see at the top of a tour.

    engagement_by_robot is empty on purpose: the ad hoc ladder derives
    "which projects are still to come" from engagement rather than from the
    step list, so an empty map is what makes all of them count as remaining.
    That difference in where the two mechanisms get their facts is part of
    what is being compared.
    """
    steps = build_script(ev.GUIDE, list(projects))
    remaining = tuple(
        StepRef(step_id=s.step_id, robot_id=s.robot_id, role=s.role or "",
                qa_window=bool(s.qa_window), block_robot_id=s.block_robot_id)
        for s in steps
    )
    peers = tuple({"client_id": r, "robot_name": r, "robot_role": "",
                   "access_level": "local"}
                  for r in [ev.GUIDE] + list(projects))
    return Observation(
        step_id=steps[0].step_id, step_idx=0, total_steps=len(steps),
        steps_remaining=len(steps), demo_state="qa_window",
        last_speaker_id="visitor", user_utterance=utterance,
        elapsed_sec=0.0, time_budget_sec=budget_sec,
        # A stated time problem, so PLAN_REVISE fires on the utterance rather
        # than needing the clock to have run late first.
        projected_overrun_sec=None,
        engagement_by_robot={}, connected_peers=peers,
        guide_robot_id=ev.GUIDE, presenting_robot_id=projects[0],
        remaining_steps=remaining,
        decider_robot_id=projects[0], decider_access_level="local",
        scenario_id="eval", session_id="eval",
    )


def _flow_planner_for(defaults: dict) -> Callable:
    """app.build_flow_plan's shape, but over fixed durations and without the
    Supabase reads — the planner itself is identical."""
    def planner(obs):
        graph = FlowGraph.from_script(obs.remaining_steps)
        if not graph.blocks:
            return None
        importance = block_importance(graph, defaults=defaults,
                                      visitor_topics=None,
                                      kg_edges=ev.EDGES, kg_links=[])
        return plan_for_budget(
            graph, max(0.0, obs.time_budget_sec - obs.elapsed_sec),
            durations=ev.FIXED_DURATIONS, importance=importance)
    return planner


# ── Applying a plan, and costing the result ───────────────────────────────────

def outcome_from_ops(ops, projects, defaults, budget_sec) -> dict:
    """
    What the tour looks like after `ops`, costed with the same fixed
    durations the expectations were derived from.

    Written to be indifferent to WHICH mechanism produced the ops — the ad
    hoc ladder and the flow planner emit the same op vocabulary, and the
    whole point is to compare what a visitor ends up with, not how each
    arrived there. `fits` is recomputed here rather than taken from a
    planner's own `feasible` flag, because the ad hoc ladder does not report
    one and trusting a self-assessment from one arm and not the other would
    not be a fair comparison.
    """
    steps = build_script(ev.GUIDE, list(projects))
    fixed, blocks = block_facts_from_script(steps, ev.FIXED_DURATIONS, defaults)

    skipped, compressed, qa = set(), set(), DEFAULT_QA_BUDGET_SEC
    dropped_all = False
    for op in ops:
        if op.kind is PlanOpKind.SKIP:
            skipped.add(op.robot_id)
        elif op.kind is PlanOpKind.COMPRESS:
            compressed.add(op.robot_id)
        elif op.kind is PlanOpKind.SET_QA_BUDGET and op.seconds is not None:
            qa = float(op.seconds)
        elif op.kind is PlanOpKind.DROP_REMAINING:
            dropped_all = True

    if dropped_all:
        skipped = {b.robot_id for b in blocks}

    total = fixed
    for b in blocks:
        if b.robot_id in skipped:
            continue
        total += b.scripted_sec - (b.compressible_sec
                                   if b.robot_id in compressed else 0.0)
        total += qa * b.qa_windows
    total = round(total, 1)

    return {
        "surviving_blocks": tuple(b.robot_id for b in blocks
                                  if b.robot_id not in skipped),
        "compressed_blocks": frozenset(compressed),
        "qa_budget_sec": qa,
        "total_sec": total,
        "fits": total <= budget_sec,
    }


# ── Running one case under one condition ──────────────────────────────────────

def run_plan_case(case: dict, cond: Condition) -> list:
    obs = _observation(case["projects"], case["budget"],
                       "we are running out of time")
    policy = HeuristicPolicy(
        flow_planner=_flow_planner_for(case["defaults"]) if cond.use_planner else None
    )
    result = policy.decide(DecisionPoint.PLAN_REVISE, obs)
    ops = result.action.ops if result.action.kind is ActionKind.REVISE else ()
    actual = outcome_from_ops(ops, case["projects"], case["defaults"],
                              case["budget"])
    e = case["expected"]
    return [
        ("surviving_blocks", actual["surviving_blocks"] == e.surviving_blocks,
         f"{actual['surviving_blocks']} vs {e.surviving_blocks}"),
        ("compressed_blocks", actual["compressed_blocks"] == e.compressed_blocks,
         f"{set(actual['compressed_blocks'])} vs {set(e.compressed_blocks)}"),
        ("qa_budget", round(actual["qa_budget_sec"]) == round(e.qa_budget_sec),
         f"{round(actual['qa_budget_sec'])} vs {round(e.qa_budget_sec)}"),
        ("fits_budget", actual["fits"] == e.feasible,
         f"{actual['fits']} vs {e.feasible}"),
    ]


def run_route_case(case: dict, cond: Condition) -> list:
    e = case["expected"]
    receiver = ev.CHATBOX          # the group is at ChatBox's station

    if not cond.use_kg_routing:
        # The documented baseline: whoever heard the question answers it.
        # It can never defer, and it records one observation for the turn.
        actual = {"answering_robot": receiver, "deferred_to": None,
                  "observations_written": 1}
    else:
        router = KGRouter(ev.EDGES, [], ev.TOPICS, explore=False,
                          absent_robot_ids=case["absent"])
        d = router.decide(case["utterance"], ev.PROJECTS,
                          remaining_block_ids=case["remaining"],
                          guide_robot_id=ev.GUIDE)
        if d is None:
            actual = {"answering_robot": receiver, "deferred_to": None,
                      "observations_written": 1}
        else:
            actual = {
                "answering_robot": None if d.is_deferred else d.robot_id,
                "deferred_to": d.deferred_to,
                "observations_written": 0 if d.is_deferred else 1,
            }

    return [
        ("answering_robot", actual["answering_robot"] == e.answering_robot,
         f"{actual['answering_robot']} vs {e.answering_robot}"),
        ("deferred_to", actual["deferred_to"] == e.deferred_to,
         f"{actual['deferred_to']} vs {e.deferred_to}"),
        ("observations_written",
         actual["observations_written"] == e.observations_written,
         f"{actual['observations_written']} vs {e.observations_written}"),
    ]


def run_case(case: dict, cond: Condition) -> list:
    return (run_plan_case(case, cond) if case["kind"] == "plan"
            else run_route_case(case, cond))


def score_all() -> dict:
    """{condition_name: {family: (passed, total), 'failures': [...]}}"""
    cases = ev.generated_cases()
    out = {}
    for cond in CONDITIONS:
        tallies = {"plan": [0, 0], "route": [0, 0]}
        failures = []
        for case in cases:
            results = run_case(case, cond)
            fam = case["kind"]
            tallies[fam][0] += sum(1 for _, p, _ in results if p)
            tallies[fam][1] += len(results)
            bad = [f"{n}: {d}" for n, p, d in results if not p]
            if bad:
                failures.append((case["id"], bad))
        out[cond.name] = {"plan": tuple(tallies["plan"]),
                          "route": tuple(tallies["route"]),
                          "failures": failures}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", action="store_true",
                    help="list the failing cases per condition")
    args = ap.parse_args()

    cases = ev.generated_cases()
    scores = score_all()

    print("=" * 78)
    print("  Ablation over the generated scenarios")
    print(f"  {len(cases)} cases "
          f"({sum(1 for c in cases if c['kind']=='plan')} plan, "
          f"{sum(1 for c in cases if c['kind']=='route')} routing)")
    print("=" * 78)
    print(f"  {'condition':<12} {'plan correctness':>18} {'routing accuracy':>18}   what is on")
    for cond in CONDITIONS:
        s = scores[cond.name]
        pp, pt = s["plan"]
        rp, rt = s["route"]
        print(f"  {cond.name:<12} {f'{pp}/{pt}  {pp/pt:5.0%}':>18} "
              f"{f'{rp}/{rt}  {rp/rt:5.0%}':>18}   {cond.description}")

    if args.detail:
        for cond in CONDITIONS:
            fails = scores[cond.name]["failures"]
            if not fails:
                continue
            print(f"\n  ── {cond.name}: {len(fails)} failing case(s) ──")
            for cid, bad in fails[:12]:
                print(f"     {cid}")
                for b in bad:
                    print(f"        {b}")
            if len(fails) > 12:
                print(f"     ... and {len(fails)-12} more")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
