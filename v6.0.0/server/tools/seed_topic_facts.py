"""
tools/seed_topic_facts.py
=========================
A DRAFT of what each robot may say about its own topics.

    python3 tools/seed_topic_facts.py --dry-run    # print, write nothing
    python3 tools/seed_topic_facts.py              # insert
    python3 tools/seed_topic_facts.py --report     # what is still missing

EVERY ROW HERE IS UNVERIFIED AND MOST ARE DELIBERATELY VAGUE.
The only defensible source in the repository is robots.robot_role — three
sentences per robot — plus the topic vocabulary itself. Nothing states a
model, a dataset, a number or a paper, so nothing here does either. That is
the point: a live run had the LLM fill exactly this gap with "Extended
Kalman Filters" and "BM25", neither of which appears anywhere in this
system. A vague true row beats a specific invented one, and every row is
marked UNVERIFIED in the prompt until a researcher confirms it.

The `limitation` rows are the ones that make a demo honest, and they are the
ones only a researcher can write. They are seeded as explicit placeholders so
the gap is visible in tools/check_facts.py rather than silent.

WHAT TO DO WITH THIS
Correct it. Replace the vague rows with what is actually true, add the model
and dataset rows that only you know, delete anything wrong, and mark what
survives as verified. Until then the robots will describe their approach in
general terms and decline to quote specifics, which is the intended
behaviour for unreviewed content.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCE = "drafted from robots.robot_role"

# (topic_id, robot_id | None, kind, fact)
# Claims are traceable to robot_role or to the topic label. Where robot_role
# says nothing, the row says nothing specific either.
DRAFT = [
    # ── Silbot: human-aware navigation ───────────────────────────────────────
    ("topic:social-robot-navigation", "silbot_01", "method",
     "Plans paths through crowded spaces that respect personal space and "
     "social norms, rather than treating people as ordinary obstacles."),
    ("topic:social-robot-navigation", None, "limitation",
     "TODO — what does the navigation NOT handle yet? (crowd density, "
     "unmapped areas, running people?)"),
    ("topic:mapping-and-localisation", "silbot_01", "method",
     "Builds and maintains a map of the space and tracks the robot's own "
     "position within it while people move around."),
    ("topic:mapping-and-localisation", None, "limitation",
     "TODO — which SLAM approach, on what sensors, and where does it fail?"),
    ("topic:multi-robot-coordination", "silbot_01", "method",
     "Part of Silbot's declared area; coordination between robots sharing "
     "the same space."),

    # ── ChatBox: conversational AI ───────────────────────────────────────────
    ("topic:retrieval-augmented-generation", "chatbox_01", "method",
     "Combines a language model with retrieval over a knowledge base so "
     "answers are grounded in stored material rather than generated from "
     "the model's parameters alone."),
    ("topic:retrieval-augmented-generation", None, "limitation",
     "TODO — what is in the knowledge base, and what happens when retrieval "
     "returns nothing relevant?"),
    ("topic:long-term-interaction", "chatbox_01", "method",
     "Researches long-context memory so a robot can hold a coherent "
     "conversation across many turns."),
    ("topic:conversational-memory", "chatbox_01", "method",
     "Retains what a visitor said earlier in a conversation and uses it in "
     "later answers."),
    ("topic:large-language-models", "chatbox_01", "model",
     "TODO — which model actually runs on ChatBox, at what size?"),
    ("topic:knowledge-graphs", "chatbox_01", "method",
     "Structured knowledge the conversational system can retrieve from."),

    # ── Navel: emotion-aware interaction ─────────────────────────────────────
    ("topic:emotion-recognition", "navel_01", "method",
     "Reads facial expression and tone of voice to infer how a visitor is "
     "responding, and adapts its communication style to match."),
    ("topic:emotion-recognition", None, "limitation",
     "TODO — which emotions are distinguished, and how does accuracy hold up "
     "on real visitors rather than a benchmark?"),
    ("topic:facial-expression-analysis", "navel_01", "method",
     "Detects facial expressions as one of the two signals behind Navel's "
     "emotion estimate."),
    ("topic:facial-expression-analysis", "navel_01", "model",
     "TODO — which detector and which expression taxonomy?"),
    ("topic:non-verbal-interaction", "navel_01", "method",
     "Communication that is not carried by the words themselves — Navel's "
     "declared area alongside emotion recognition."),
    ("topic:social-signals", "navel_01", "method",
     "Cues a person gives off during interaction, used to adapt the robot's "
     "behaviour."),
    ("topic:human-robot-trust", "navel_01", "method",
     "How a visitor's confidence in the robot develops over an interaction."),

    # ── Unowned topics: background only, explicitly not anyone's result ──────
    ("topic:speech-recognition", None, "limitation",
     "No project in this lab claims speech recognition as its research area; "
     "it is infrastructure the demo uses, not a result being presented."),
    ("topic:text-to-speech", None, "limitation",
     "No project claims text-to-speech as its research area; it is how the "
     "robots speak, not what they study."),
    ("topic:robot-hardware", None, "limitation",
     "The platforms are off-the-shelf; the contribution is in the software "
     "each robot runs, not the hardware."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    from data import demo_facts_repo as repo

    if args.report:
        rows = repo.coverage()
        print(f"  {'topic':<34} {'facts':>6} {'verified':>9}  gap")
        print("  " + "-" * 68)
        for r in rows:
            gap = "" if r["facts"] else "NO GROUNDING — answers will be general only"
            if r["facts"] and not r["verified_facts"]:
                gap = "nothing reviewed yet"
            print(f"  {r['label'][:33]:<34} {r['facts']:>6} "
                  f"{r['verified_facts']:>9}  {gap}")
        print(f"\n  {repo.unverified_count()} unverified fact(s) in total.")
        return 0

    todo = sum(1 for _t, _r, _k, f in DRAFT if f.startswith("TODO"))
    print(f"  {len(DRAFT)} draft rows, of which {todo} are explicit TODO "
          f"placeholders for detail only a researcher can supply.")
    if args.dry_run:
        for t, r, k, f in DRAFT:
            print(f"    [{k:<10}] {t[6:]:<32} {r or '(general)':<12} {f[:60]}")
        print("\n  --dry-run: nothing written.")
        return 0

    written = 0
    for topic_id, robot_id, kind, fact in DRAFT:
        if repo.add(topic_id, kind, fact, robot_id=robot_id,
                    verified=False, source=SOURCE) is not None:
            written += 1
    print(f"  wrote {written}/{len(DRAFT)} rows, all verified=false.")
    print("  Review them with:  python3 tools/seed_topic_facts.py --report")
    return 0 if written == len(DRAFT) else 1


if __name__ == "__main__":
    sys.exit(main())
