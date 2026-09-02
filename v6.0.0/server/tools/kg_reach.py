"""
tools/kg_reach.py
=================
Does a correction generalise, or only teach the exact topic it was made on?

This measures the single empirical claim the robot→topic design rests on. The
argument for choosing robot→topic over robot→robot or context→action was that
it is the only one where a correction can spread — along topic↔topic links — to
subjects nobody has corrected yet. If the link set is too sparse, that spread is
zero and the argument loses its strongest leg.

    python3 tools/kg_reach.py

For each topic in turn: pretend one supervisor correction landed there, run the
propagation read, and count how many OTHER topics moved and by how much.

No writes. Reads the seeded vocabulary and links from the database.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.kg import Evidence, PRIOR, RobotTopicEdge   # noqa: E402
from decision.kg_infer import infer                       # noqa: E402

MOVED = 0.01   # a posterior shift smaller than this is not worth calling movement
ROBOT = "_probe_robot"


def main() -> int:
    from data.demo_kg_repo import all_links, all_topics

    topics = all_topics()
    raw_links = all_links()
    if not topics:
        print("No vocabulary. Run tools/seed_kg.py first.")
        return 1

    topic_ids = [t["id"] for t in topics]
    label = {t["id"]: t["label"] for t in topics}
    links = [(l["topic_a"], l["topic_b"], float(l["weight"])) for l in raw_links]

    print("=" * 70)
    print("  Correction reach — how far one correction spreads")
    print("=" * 70)
    print(f"  {len(topics)} topics, {len(links)} links "
          f"({len(links) * 2 / len(topics):.2f} per topic)")
    print()

    for direction, target in (("positive", 1.0), ("negative", 0.0)):
        print(f"  ── A {direction} correction (target={target}) " + "─" * 28)
        print(f"      {'corrected topic':38} {'reached':>8} {'mean shift':>11}")
        reached_counts, shifts = [], []

        for tid in topic_ids:
            # One supervisor correction on this topic, nothing else known.
            edge = RobotTopicEdge(robot_id=ROBOT, topic_id=tid).update(
                target, Evidence.SUPERVISOR)
            posterior = infer([edge], links, ROBOT, topic_ids)

            moved = {t: abs(posterior.get(t, PRIOR) - PRIOR)
                     for t in topic_ids if t != tid}
            hit = {t: d for t, d in moved.items() if d >= MOVED}
            reached_counts.append(len(hit))
            shifts.extend(hit.values())

            mean = (sum(hit.values()) / len(hit)) if hit else 0.0
            print(f"      {label[tid][:36]:38} {len(hit):>8} {mean:>11.3f}")

        n = len(reached_counts)
        print()
        print(f"      topics reached per correction : "
              f"mean {sum(reached_counts)/n:.2f}, max {max(reached_counts)}, "
              f"zero-reach {sum(1 for c in reached_counts if c == 0)}/{n}")
        if shifts:
            print(f"      shift when reached            : "
                  f"mean {sum(shifts)/len(shifts):.3f}, max {max(shifts):.3f}")
        print()

    # ── The regime that will actually hold ───────────────────────────────────
    # A saturated edge (12+ corrections) is not the regime a mocked-tour campaign
    # produces. The sim yields ~2-3 corrections per run; spread over 14 topics
    # and a handful of robots, most edges will carry ONE to THREE. Leading with
    # the saturated number would be reporting a band the experiment never enters.
    print("  ── Reach by evidence, one robot, strongest link " + "─" * 20)
    print(f"      {'corrections':>12} {'own clamped':>12} {'best neighbour':>15} {'reached':>8}")
    best_src, best_dst, best_w = max(
        ((a, b, w) for a, b, w in links), key=lambda x: x[2], default=(None, None, 0))
    if best_src:
        e = RobotTopicEdge(robot_id=ROBOT, topic_id=best_src)
        for n in (1, 2, 3, 5, 12):
            while e.n_obs < n:
                e = e.update(1.0, Evidence.SUPERVISOR)
            p = infer([e], links, ROBOT, topic_ids)
            reached = sum(1 for t in topic_ids
                          if t != best_src and abs(p.get(t, PRIOR) - PRIOR) >= MOVED)
            marker = "  <- realistic" if n <= 3 else ""
            print(f"      {n:>12} {e.clamped:>12.3f} {p[best_dst]:>15.3f} "
                  f"{reached:>8}{marker}")
        print()
        print("      Report the 1-3 rows. The 12-correction row has the source")
        print("      edge at 0.99, which no realistic campaign will reach on more")
        print("      than a handful of edges.")
    print()

    print("  READ THIS AS:")
    print("    'reached' counts OTHER topics whose posterior moved by >= 0.01.")
    print("    Zero-reach topics are isolated — a correction there teaches")
    print("    nothing beyond itself, which is what robot→robot would do for")
    print("    every topic. A mean shift far below the correction's own move")
    print("    means propagation is technically present but weak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
