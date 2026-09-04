"""
tools/plan_replay_harness.py
=============================
Deterministic replay for PLAN_REVISE — diagnose "sometimes skips a project,
sometimes just shortens it" without needing a live demo.

decision.planner.plan_for_budget is pure: same graph + same budget + same
durations + same importance -> same ops, every time, no exceptions. Every
function on the real path — KGRouter.resolve_topic (word overlap, no LLM),
resolve_emphasis, block_importance, plan_for_budget — is pure Python with no
I/O and no sampling. app.py's build_flow_plan wraps that pure path with
exactly two LIVE, time-varying inputs: _step_durations() and _kg_snapshot(),
both read from Supabase, both accumulating as more demos run and more
corrections land — and both cache-invalidated the moment a new sample lands,
so even two Q&A windows in the SAME run can plan against different duration
data if a step ACK landed in between.

So there are exactly two explanations for "the same scenario produced two
different plans", and this harness is built to tell them apart:

  1. A bug — the pure path is not actually deterministic under identical
     inputs. Rule this out first: run_scenario() with FROZEN durations and kg
     data, N times, diff the results. This should never fail. If it does,
     that is the bug, and it is in plan_for_budget/block_importance, not in
     "randomness" — there isn't any on this path.

  2. Drift — durations or the kg graph moved between the two real occasions.
     Confirm by re-running the identical scenario against two duration
     snapshots (the observed one, and one nudged the way more accumulated
     samples would nudge it) and showing the rung actually changes.

Usage:
    python3 tools/plan_replay_harness.py
    python3 tools/plan_replay_harness.py --budget 600 --at chatbox_01:qa --interest emotion
    python3 tools/plan_replay_harness.py --drift 0.15
    python3 tools/plan_replay_harness.py --live       # pull real durations/kg from Supabase

NOTHING ON THE PATH IS MOCKED except the two Supabase reads, which become
explicit arguments here — same discipline as tools/demo_harness.py and
tools/demo_sim.py. Without --live, durations/kg default to empty (an
arithmetic-only estimate, measured_coverage 0.0) so this runs with no network
calls at all — pytest speed, not live-run speed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.flow import FlowGraph
from decision.kg_policy import KGRouter
from decision.planner import block_importance, plan_for_budget, resolve_emphasis
from demo.demo_script import build_script

DEFAULT_GUIDE = "pepper_01"
DEFAULT_PROJECTS = ["chatbox_01", "navel_01", "silbot_01"]


def run_scenario(
    *,
    guide_id: str = DEFAULT_GUIDE,
    project_ids: Sequence[str] = tuple(DEFAULT_PROJECTS),
    budget_sec: float,
    elapsed_sec: float = 0.0,
    at_robot: Optional[str] = None,
    at_role: str = "",
    utterance: str = "",
    profile_topics: Optional[Sequence[str]] = None,
    durations: Optional[dict] = None,
    kg_edges: Optional[list] = None,
    kg_links: Optional[list] = None,
    topics: Optional[list] = None,
) -> dict:
    """
    Run the exact path app.py::build_flow_plan runs, minus its two Supabase
    reads — durations/kg_edges/kg_links/topics become explicit arguments
    instead of live lookups, which is what makes this deterministic and
    network-free.

    `at_robot`/`at_role` position the play head — the "where in the tour did
    this fire" input that a real run never holds constant between two
    occasions unless you make it. Both default to unset, which plans from the
    very top of the tour. Pass both together to position elsewhere, e.g.
    at_robot="chatbox_01", at_role="qa" for a visitor turn during ChatBox's
    own Q&A window.
    """
    steps = build_script(guide_id, list(project_ids))

    if at_robot is None and not at_role:
        idx = 0
    else:
        idx = next(
            i for i, s in enumerate(steps)
            if s.block_robot_id == at_robot and s.role == at_role
        )
    remaining = steps[idx:]

    graph = FlowGraph.from_script(remaining)
    if not graph.blocks:
        return {"ops": [], "feasible": True, "fits_already": True,
                "estimate": {"total_sec": 0.0}, "trace": ["no blocks remaining"],
                "measured_coverage": 0.0, "emphasis_source": "none", "visitor_topics": ()}

    topics = topics or []
    utterance_topics = None
    if topics and utterance:
        tid = KGRouter([], [], topics).resolve_topic(utterance)
        if tid:
            utterance_topics = [tid]

    visitor_topics, emphasis_source = resolve_emphasis(utterance_topics, profile_topics)

    importance = block_importance(
        graph, defaults={}, visitor_topics=visitor_topics or None,
        kg_edges=kg_edges or [], kg_links=kg_links or [],
    )

    remaining_budget = max(0.0, budget_sec - elapsed_sec)
    result = plan_for_budget(graph, remaining_budget, durations=durations or {},
                              importance=importance)
    result["emphasis_source"] = emphasis_source
    result["visitor_topics"] = visitor_topics
    return result


def _canon(result: dict) -> tuple:
    """A JSON/print-safe view of a run_scenario() result: PlanOps reduced to
    their payload dicts. Also what "the same plan" means for comparison,
    since payload dicts (not PlanOp identity) are what determinism and drift
    checks diff against."""
    return {
        "ops": [op.payload() for op in result["ops"]],
        "feasible": result["feasible"],
        "fits_already": result.get("fits_already", False),
        "estimate": result["estimate"],
        "trace": result["trace"],
        "measured_coverage": result["measured_coverage"],
    }


def _op_kinds(view: dict) -> list:
    """The STRUCTURAL decision — which ops, targeting which robots, in what
    order — with numeric fields (seconds) dropped. Two plans that both
    tighten Q&A and compress the same two robots are "the same rung" even if
    the exact seconds allocated differ slightly, which is exactly what a
    small duration drift does to an otherwise-unchanged decision."""
    return [(op["kind"], op.get("robot_id")) for op in view["ops"]]


def _exactly_equal(a: dict, b: dict) -> bool:
    """Byte-for-byte agreement — ops AND the numbers behind them. What
    'deterministic' has to mean: identical inputs producing identical output,
    with nothing left to explain away as expected drift."""
    return (
        [tuple(sorted(op.items())) for op in a["ops"]]
        == [tuple(sorted(op.items())) for op in b["ops"]]
        and a["feasible"] == b["feasible"]
        and a["estimate"]["total_sec"] == b["estimate"]["total_sec"]
    )


def check_determinism(scenario: dict, n: int = 5) -> dict:
    """
    Run the same scenario n times. plan_for_budget has no I/O and no
    sampling, so this must always agree with itself EXACTLY — a mismatch
    here is a real bug in the pure path, not "the scenario is unstable".
    """
    views = [_canon(run_scenario(**scenario)) for _ in range(n)]
    mismatched = [i for i in range(1, n) if not _exactly_equal(views[i], views[0])]
    return {**views[0], "deterministic": not mismatched, "n": n, "mismatched_runs": mismatched}


def diff_under_duration_drift(scenario: dict, drift_pct: float) -> dict:
    """
    Same scenario, two duration snapshots: the one given, and one nudged by
    drift_pct — what accumulating a few more measured samples plausibly does
    to the mean. Shows whether that alone is enough to flip which rung fires,
    for THIS scenario, isolating the drift explanation from everything else.

    `rung_changed` compares the STRUCTURAL decision (_op_kinds), not exact
    total_sec — that number moves by roughly drift_pct on almost any
    scenario, by construction, whether or not the actual decision changed.
    Comparing it exactly would make every drift experiment "positive" and
    say nothing.
    """
    before_durations = dict(scenario.get("durations") or {})
    after_durations = {k: v * (1.0 + drift_pct) for k, v in before_durations.items()}

    before = _canon(run_scenario(**{**scenario, "durations": before_durations}))
    after = _canon(run_scenario(**{**scenario, "durations": after_durations}))

    return {"drift_pct": drift_pct, "before": before, "after": after,
            "rung_changed": _op_kinds(before) != _op_kinds(after)}


def _print_result(label: str, r: dict) -> None:
    print(f"\n{label}")
    for op in r["ops"]:
        print(f"  {op}")
    if not r["ops"]:
        print("  (no ops — fits already)" if r.get("fits_already") else "  (no ops)")
    print(f"  feasible={r.get('feasible')} total_sec={r['estimate'].get('total_sec')} "
          f"measured_coverage={r.get('measured_coverage')}")
    for line in r.get("trace", []):
        print(f"    · {line}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--guide", default=DEFAULT_GUIDE)
    p.add_argument("--projects", default=",".join(DEFAULT_PROJECTS),
                    help="comma-separated robot ids, presentation order")
    p.add_argument("--budget", type=float, default=600.0, help="total tour budget, seconds")
    p.add_argument("--elapsed", type=float, default=0.0, help="seconds already elapsed")
    p.add_argument("--at", default=None,
                    help="play-head position as robot_id:role, e.g. chatbox_01:qa. "
                         "Defaults to the first project's qa step.")
    p.add_argument("--utterance", default="", help="what the visitor said this turn")
    p.add_argument("--interest", default=None,
                    help="standing pre-demo visitor-profile topic id (layer 2)")
    p.add_argument("--runs", type=int, default=5, help="repetitions for the determinism check")
    p.add_argument("--drift", type=float, default=None,
                    help="also run the duration-drift experiment at this fraction, e.g. 0.15")
    p.add_argument("--live", action="store_true",
                    help="pull real durations/kg snapshot from Supabase instead of running empty")
    args = p.parse_args()

    project_ids = [s.strip() for s in args.projects.split(",") if s.strip()]
    at_robot, at_role = (None, "")
    if args.at:
        at_robot, at_role = args.at.split(":", 1)

    durations, kg_edges, kg_links, topics = {}, [], [], []
    if args.live:
        from app import _kg_snapshot, _step_durations
        durations = _step_durations()
        topics, kg_edges, kg_links = _kg_snapshot()
        print(f"[harness] live snapshot: {len(durations)} measured steps, "
              f"{len(kg_edges)} kg edges, {len(topics)} topics")

    scenario = dict(
        guide_id=args.guide, project_ids=project_ids,
        budget_sec=args.budget, elapsed_sec=args.elapsed,
        at_robot=at_robot, at_role=at_role,
        utterance=args.utterance,
        profile_topics=[args.interest] if args.interest else None,
        durations=durations, kg_edges=kg_edges, kg_links=kg_links, topics=topics,
    )

    det = check_determinism(scenario, n=args.runs)
    _print_result(f"Determinism check ({args.runs} runs, identical inputs)", det)
    print(f"  deterministic={det['deterministic']}"
          + ("" if det["deterministic"] else f"  MISMATCHED RUNS: {det['mismatched_runs']}"))

    if args.drift is not None:
        if not durations:
            print(f"\n[harness] --drift needs non-empty durations — pass --live, "
                  f"or this experiment has nothing to nudge.")
        else:
            drift = diff_under_duration_drift(scenario, args.drift)
            _print_result("Before drift", drift["before"])
            _print_result("After drift", drift["after"])
            print(f"\n  rung_changed={drift['rung_changed']} at drift_pct={args.drift}")


if __name__ == "__main__":
    main()
