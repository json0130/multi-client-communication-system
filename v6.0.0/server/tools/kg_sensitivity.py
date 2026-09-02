"""
tools/kg_sensitivity.py
=======================
Does the propagation result depend on getting the topology right?

The topic↔topic links are the one part of this graph nobody learned. They are
either hand-authored (an author choosing the edges that make the result work) or
embedding-derived (a similarity function that, measured here, misses
`retrieval augmented generation ~ large language models` while linking
`human robot trust ~ robot hardware`). Neither is trustworthy on its own.

A propagation result that survives a WRONG topology is a stronger claim than one
measured on the right one. This perturbs the link weights and asks whether the
conclusions move:

    python3 tools/kg_sensitivity.py

Three things are reported:
  * reach and shift under perturbation, versus the unperturbed baseline
  * whether the ROUTING DECISION (argmax robot for a topic) ever flips
  * the same, under the authored and the derived topology in turn

The routing flip rate is the one that matters. Weights wobbling is expected;
the decision changing because of that wobble is not.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.kg import Evidence, PRIOR, RobotTopicEdge   # noqa: E402
from decision.kg_infer import infer, rank_robots           # noqa: E402

PERTURB = 0.15
TRIALS = 40
MOVED = 0.01
ROBOTS = ["robot_a", "robot_b", "robot_c"]


def perturb(links, delta, rng):
    """Jitter every link weight, keeping it a valid [0,1] similarity."""
    return [(a, b, max(0.05, min(1.0, w + rng.uniform(-delta, delta))))
            for a, b, w in links]


def scenario(topic_ids, rng):
    """A plausible sparse evidence state: a few robots, a few observed topics."""
    edges = []
    for r in ROBOTS:
        for tid in rng.sample(topic_ids, k=min(3, len(topic_ids))):
            e = RobotTopicEdge(robot_id=r, topic_id=tid)
            for _ in range(rng.randint(1, 3)):
                e = e.update(rng.choice([0.0, 1.0]), Evidence.SUPERVISOR)
            edges.append(e)
    return edges


def measure(links, topic_ids, rng, delta):
    """(mean reach, mean shift, routing flip rate) against the unperturbed graph."""
    reach, shift, flips, decisions = [], [], 0, 0
    for _ in range(TRIALS):
        edges = scenario(topic_ids, rng)
        jittered = perturb(links, delta, rng) if delta else links

        for tid in topic_ids:
            base = rank_robots(edges, links, tid, ROBOTS)
            test = rank_robots(edges, jittered, tid, ROBOTS)
            decisions += 1
            if base[0][0] != test[0][0]:
                flips += 1

        probe = ROBOTS[0]
        p_base = infer(edges, links, probe, topic_ids)
        p_test = infer(edges, jittered, probe, topic_ids)
        moved = [abs(p_test[t] - PRIOR) for t in topic_ids
                 if abs(p_test[t] - PRIOR) >= MOVED]
        reach.append(len(moved))
        shift.extend(abs(p_test[t] - p_base[t]) for t in topic_ids)

    return (sum(reach) / len(reach),
            sum(shift) / len(shift) if shift else 0.0,
            flips / decisions if decisions else 0.0)


def main() -> int:
    from tools.seed_kg import collect_topics, curated_links, embed_links
    from data.demo_kg_repo import all_links

    topics, _ = collect_topics()
    topic_ids = [t["id"] for t in topics]

    topologies = [("stored (in database)",
                   [(l["topic_a"], l["topic_b"], float(l["weight"])) for l in all_links()]),
                  ("authored",
                   [(l.topic_a, l.topic_b, l.weight) for l in curated_links(topics)])]
    try:
        derived, _m = embed_links(topics)
        topologies.append(("derived (embedding)",
                           [(l.topic_a, l.topic_b, l.weight) for l in derived]))
    except Exception as e:
        print(f"  ! embeddings unavailable, skipping derived topology: {str(e)[:50]}")

    print("=" * 72)
    print(f"  Topology sensitivity — {TRIALS} random evidence states, "
          f"weights jittered +/-{PERTURB}")
    print("=" * 72)
    print(f"  {'topology':24} {'links':>6} {'reach':>7} {'drift':>8} {'route flips':>12}")

    for name, links in topologies:
        rng = random.Random(7)          # same evidence states across topologies
        r0, _s0, _f0 = measure(links, topic_ids, rng, delta=0.0)
        rng = random.Random(7)
        r1, s1, f1 = measure(links, topic_ids, rng, delta=PERTURB)
        print(f"  {name:24} {len(links):>6} "
              f"{r0:>4.1f}→{r1:<4.1f} {s1:>8.3f} {f1:>11.1%}")

    print()
    print("  READ THIS AS:")
    print("    reach  topics moved per correction, unperturbed → perturbed.")
    print("    drift  mean absolute change in a posterior from the jitter.")
    print("    route flips  how often the argmax robot for a topic CHANGES when")
    print("           the topology is wrong by up to +/-0.15. This is the number")
    print("           that decides whether the result depends on the topology.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
