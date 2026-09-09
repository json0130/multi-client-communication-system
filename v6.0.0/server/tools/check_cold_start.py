"""
tools/check_cold_start.py
=========================
Which decisions are actually decidable on a graph with no observations.

    python3 tools/check_cold_start.py            # audit the cold-start case
    python3 tools/check_cold_start.py --live     # audit the real graph too

WHY THIS EXISTS
Four separate mechanisms shipped silently inert, each found by accident:
the 009 role seed, QA routing, the planner's visitor-interest layer, and the
planner's hand-set priorities. All four failed the same way, and none of
them errored.

    a feature reads a number derived from OBSERVATION COUNTS
    the graph starts with none
    the number comes back technically correct and practically meaningless
    nothing errors, nothing warns, the feature quietly does nothing

The signature is that the value is real but too small to cross whatever
threshold sits downstream — or that there is no threshold at all, so an
argmax over identical values falls through to a tie-break and the decision
is made by alphabetical order.

So the test is never "does this return a number". It is:

    what does this produce with zero observations, and is that
    ENOUGH TO CHANGE THE DECISION BELOW IT?

Each check states the cold-start value, the margin it has to clear, and what
happens if it does not. INERT is a finding, not a failure of the check.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GUIDE = "pepper_01"
PROJECTS = ["chatbox_01", "navel_01", "silbot_01"]
TOPIC = "topic:retrieval-augmented-generation"
ORPHAN = "topic:text-to-speech"

OK, INERT, DECLINES = "OK", "INERT", "DECLINES"


def _row(site, value, threshold, verdict, note):
    return {"site": site, "value": value, "threshold": threshold,
            "verdict": verdict, "note": note}


def audit(edges, links, label: str) -> list:
    """Every place an observation-derived number reaches a decision."""
    from decision.flow import FlowGraph
    from decision.kg_infer import MIN_EVIDENCE_MARGIN, rank_robots, route
    from decision.planner import DEFAULT_IMPORTANCE, block_importance
    from decision.style_fit import REINFORCE_BELOW, StyleFit
    from demo.demo_script import build_script

    graph = FlowGraph.from_script(build_script(GUIDE, PROJECTS))
    out = []

    # 1 — routing on a DECLARED topic. Scope is configuration, so this must
    #     work with no evidence at all; that is the whole point of declaring.
    picked, reason = route(edges, links, TOPIC, PROJECTS, explore=False)
    out.append(_row(
        "route / declared topic", picked or "none", "scope, not evidence",
        OK if picked else INERT,
        "declared scope decides; needs no observations"))

    # 2 — routing on an UNDECLARED topic. Nothing declares it, so only a
    #     learned weight could decide, and it must be big enough to mean
    #     something.
    ranked = rank_robots(edges, links, ORPHAN, PROJECTS)
    spread = (ranked[0][1] - ranked[-1][1]) if len(ranked) > 1 else 0.0
    picked, reason = route(edges, links, ORPHAN, PROJECTS, explore=False)
    out.append(_row(
        "route / undeclared topic", f"spread {spread:.4f}",
        f">= {MIN_EVIDENCE_MARGIN}",
        DECLINES if picked is None else OK,
        "declines rather than ordering on noise" if picked is None
        else f"routes to {picked}"))

    # 3 — visitor interest. The layer that protects the project a visitor
    #     asked about; it has to beat the hand-set defaults' own spread.
    defaults = {"chatbox_01": 0.3, "navel_01": 0.5, "silbot_01": 0.7}
    with_i = block_importance(graph, defaults=defaults, visitor_topics=[TOPIC],
                              kg_edges=edges, kg_links=links)
    without = block_importance(graph, defaults=defaults, kg_edges=edges,
                               kg_links=links)
    moved = max(abs(with_i[r] - without[r]) for r in PROJECTS)
    spread_defaults = max(defaults.values()) - min(defaults.values())
    out.append(_row(
        "importance / visitor interest", f"moves {moved:.3f}",
        f"> {spread_defaults / len(PROJECTS):.3f} (adjacent defaults)",
        OK if moved > spread_defaults / len(PROJECTS) else INERT,
        "a stated interest can reorder the blocks" if moved > 0.05
        else "stated interest cannot change which block is cut"))

    # 4 — hand-set priorities as PRODUCTION supplies them, read from the
    #     scenario profile exactly as app.build_flow_plan does. Not a graph
    #     question, but the same shape: a layer whose values are all equal
    #     produces identical scores, and the skip order falls to the
    #     tie-break. Reading the real profile is the point — auditing a
    #     hardcoded {} here would test this file rather than the deployment.
    try:
        from core.profiles import ProfileRegistry
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prod_defaults = ProfileRegistry.from_directory(
            os.path.join(here, "profiles")).importance_defaults()
    except Exception as e:
        print(f"  (scenario profile unreadable: {e})")
        prod_defaults = {}
    live_imp = block_importance(graph, defaults=prod_defaults,
                                kg_edges=edges, kg_links=links)
    distinct = len(set(live_imp.values()))
    out.append(_row(
        "importance / production defaults",
        f"{distinct} distinct value(s) across {len(PROJECTS)} blocks",
        "> 1", OK if distinct > 1 else INERT,
        "which block is cut is decided by importance" if distinct > 1
        else ("all blocks tie; the skip order falls to the alphabet. Set "
              "`importance:` per robot in profiles/lab_demo.yaml")))

    # 5 — style fit. Correctly does nothing until rated; the check is that
    #     the threshold is reachable, not that it fires cold.
    cold = StyleFit(robot_id="silbot_01", style="technical")
    rated = cold
    for _ in range(2):
        rated = rated.record(0.0)
    out.append(_row(
        "style fit / reinforcement", f"cold {cold.clamped:.3f}",
        f"< {REINFORCE_BELOW}",
        OK if (not cold.needs_reinforcement and rated.needs_reinforcement)
        else INERT,
        "silent cold, reachable in 2 ratings"))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true",
                    help="also audit the real graph, not just a cold one")
    args = ap.parse_args()

    from decision.kg import RobotTopicEdge

    # A cold graph shaped like a fresh deployment: scope declared, nothing
    # ever observed. This is what every mechanism must be judged against.
    cold_edges = [
        RobotTopicEdge(robot_id="chatbox_01", topic_id=TOPIC, specialised=True),
        RobotTopicEdge(robot_id="navel_01",
                       topic_id="topic:emotion-recognition", specialised=True),
        RobotTopicEdge(robot_id="silbot_01",
                       topic_id="topic:social-robot-navigation", specialised=True),
    ]

    sets = [("COLD (declared scope, zero observations)", cold_edges, [])]
    if args.live:
        try:
            from data import demo_kg_repo as repo
            edges = [RobotTopicEdge.from_row(r) for r in repo.graph()]
            links = [(l["topic_a"], l["topic_b"], float(l["weight"]))
                     for l in repo.all_links()]
            sets.append(("LIVE graph", edges, links))
        except Exception as e:
            print(f"  (live graph unavailable: {e})")

    inert = 0
    for label, edges, links in sets:
        print("=" * 78)
        print(f"  {label}")
        print("=" * 78)
        print(f"  {'decision site':<34} {'cold value':<26} {'verdict'}")
        print("  " + "-" * 74)
        for r in audit(edges, links, label):
            print(f"  {r['site']:<34} {str(r['value']):<26} {r['verdict']}")
            print(f"  {'':<34} needs {r['threshold']}")
            print(f"  {'':<34} {r['note']}")
            if r["verdict"] == INERT:
                inert += 1
        print()

    print("=" * 78)
    if inert:
        print(f"  {inert} mechanism(s) INERT — they produce a number that cannot")
        print("  change any decision below them. That is the failure signature.")
    else:
        print("  No inert mechanisms. Every observation-derived number that")
        print("  reaches a decision can actually change it.")
    print("=" * 78)
    return 1 if inert else 0


if __name__ == "__main__":
    sys.exit(main())
