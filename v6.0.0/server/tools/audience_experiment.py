"""
tools/audience_experiment.py
=============================
Does the visitor profile actually change how the robots talk?

Runs ONE fixed scenario — the same tour, the same visitor utterances, the
same interruption points — once per audience, changing nothing but
VisitorProfile.style. Everything else is held constant so any difference in
the transcripts is attributable to the style and to nothing else:

    python3 tools/audience_experiment.py                  # all three
    python3 tools/audience_experiment.py --styles technical
    python3 tools/audience_experiment.py --out runs/      # write transcripts

WHY A SEPARATE TOOL AND NOT demo_harness
demo_harness measures ROUTING — which robot took which question — and
deliberately stubs generation for campaign volume. This measures the TEXT,
so generation has to be real, and the visitor script has to be identical
across conditions rather than sampled. Different question, different tool.

WHAT IS HELD CONSTANT
The script, the robot order, the subjects, the visitor's words, and where
they interrupt. The competence graph is read once and shared, so routing
cannot drift between conditions either. The ONLY difference is the style
directive appended to each generation.

WHAT THIS CANNOT TELL YOU
Whether the styling is GOOD — only whether it is different, and how. Judging
"is this pitched right for a high-school group" is what the ratings in
decision/style_fit.py are for, and needs a person. This produces the
transcripts that make that judgement possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── The scenario, identical for every condition ──────────────────────────────
# Written to mirror the shape of the real runs in the logs: a barge-in during
# a project talk, a technical follow-up, an acknowledgement, and a close.
# `after` names the step this visitor turn interrupts.
GUIDE = "pepper_01"
PROJECTS = ["silbot_01", "chatbox_01"]

VISITOR_SCRIPT = [
    {"after": "silbot_01_project_problem",
     "say": "so which technique do you use"},
    {"after": "silbot_01_project_approach",
     "say": "and how does it know where a person is going"},
    {"after": "silbot_01_project_impact",
     "say": "okay thank you"},
    {"after": "chatbox_01_project_problem",
     "say": "what models do you use for that"},
    {"after": "chatbox_01_project_approach",
     "say": "okay that sounds cool"},
]

AUDIENCES = {
    "interactive": "High-school students on a school visit",
    "business":    "Industry visitors evaluating the lab's work",
    "technical":   "Visiting robotics researchers",
}


def _fmt(text: str, width: int = 76, indent: str = "      ") -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width,
                                   initial_indent=indent,
                                   subsequent_indent=indent)) or (indent + "(empty)")


def run_condition(style: str, registry, kg, out) -> dict:
    """One full tour under one audience style. Returns word/timing stats."""
    from decision.style_fit import framing_for
    from decision.visitor_profile import VisitorProfile
    from demo.demo_script import build_script

    profile = VisitorProfile(style=style)
    topics, edges, _links = kg
    labels = {t["id"]: t.get("label", t["id"]) for t in topics}
    subjects = {
        r: ", ".join(sorted(labels.get(e.topic_id, e.topic_id) for e in edges
                            if e.robot_id == r and e.specialised))
        for r in PROJECTS
    }

    script = build_script(GUIDE, PROJECTS, subjects=subjects)
    by_step = {s["after"]: s["say"] for s in VISITOR_SCRIPT}

    framing = framing_for(style, None)          # no ratings yet — plain directive
    say = out.write

    say(f"\n{'=' * 78}\n")
    say(f"  AUDIENCE: {AUDIENCES[style]}\n")
    say(f"  style={style!r}\n")
    say("  framing appended to every generation:\n")
    say(_fmt(framing.strip() or "(none — general audience)", indent="      ") + "\n")
    say(f"{'=' * 78}\n\n")

    started = time.time()
    words = 0

    for step in script:
        instance = registry.get(step.robot_id)
        if instance is None:
            continue

        if step.generate:
            spoken = instance.generate_demo_speech(step.text + framing).clean_text
        else:
            spoken = step.text
        words += len(spoken.split())
        say(f"  [{step.robot_id}] {step.step_id}\n")
        say(_fmt(spoken) + "\n\n")

        # A visitor interrupts here, in every condition, with the same words.
        utterance = by_step.get(step.step_id)
        if not utterance:
            continue

        say(f"  >>> VISITOR: \"{utterance}\"\n")

        # Does this close the window? The real system decides that BEFORE
        # generating anything, and an ADVANCE short-circuits the reply — so a
        # harness that always generates one overstates how much the robots
        # actually say. Uses the real phrase rules, minus the LLM classifier,
        # which needs a live window this offline replay does not have.
        if _closes_the_window(utterance):
            say("      (advance — window closes, demo continues; no reply generated)\n\n")
            continue

        answerer = _who_answers(utterance, step, registry, kg)
        replies: list = []
        registry.get(answerer).process_chat_stream(
            utterance, lambda t, _tag: replies.append(t),
            style_framing=framing,
        )
        reply = " ".join(replies).strip()
        words += len(reply.split())
        say(f"  [{answerer}] (Q&A)\n")
        say(_fmt(reply) + "\n\n")

    elapsed = time.time() - started
    say(f"  --- {words} words spoken, {elapsed:.0f}s wall clock ---\n")
    return {"style": style, "words": words, "sec": round(elapsed, 1)}


def _closes_the_window(utterance: str) -> bool:
    """True when the visitor's turn ends the Q&A rather than asking anything.

    The deterministic half of HeuristicPolicy._decide_advance — advance
    phrases, bare affirmations and stated time pressure. The LLM classifier
    is deliberately not consulted: it is nondeterministic, and a controlled
    comparison must not have the conditions diverge on a coin flip.
    """
    from decision.policy import (QA_ADVANCE_PHRASES, TIME_PRESSURE_PHRASES,
                                 _is_bare_affirmation, _matches)
    return bool(_matches(utterance, QA_ADVANCE_PHRASES)
                or _is_bare_affirmation(utterance)
                or _matches(utterance, TIME_PRESSURE_PHRASES))


def _who_answers(utterance: str, step, registry, kg) -> str:
    """The real routing decision, so the transcript reflects the real system."""
    from decision.kg_policy import KGRouter
    from decision.observation import looks_like_question
    topics, edges, links = kg
    presenter = step.block_robot_id or step.robot_id

    router = KGRouter(edges, links, topics, explore=False)
    d = router.decide(utterance, PROJECTS,
                      remaining_block_ids=set(PROJECTS),
                      guide_robot_id=GUIDE,
                      context_robot_id=presenter)
    if d is not None and d.robot_id and registry.get(d.robot_id) is not None:
        return d.robot_id
    # Unresolved: the presenting robot takes its own question, but only if it
    # IS a question — matching gateway/websocket_gateway.py::_kg_route.
    if looks_like_question(utterance) and registry.get(presenter) is not None:
        return presenter
    return GUIDE


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styles", nargs="*", default=list(AUDIENCES),
                    choices=list(AUDIENCES))
    ap.add_argument("--out", default="", help="directory to write transcripts to")
    args = ap.parse_args()

    from data import demo_kg_repo as repo
    from decision.kg import RobotTopicEdge
    from robot.robot_registry import RobotRegistry
    from core.rbac import GrantStore, RBACFilter
    from core.profiles import ProfileRegistry

    topics = repo.all_topics()
    edges = [RobotTopicEdge.from_row(r) for r in repo.graph()]
    links = [(l["topic_a"], l["topic_b"], float(l["weight"])) for l in repo.all_links()]
    kg = (topics, edges, links)

    profiles = ProfileRegistry.from_directory(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "profiles"))
    registry = RobotRegistry(profiles=profiles, rbac=RBACFilter(), grants=GrantStore())
    for rid in [GUIDE] + PROJECTS:
        if registry.connect(rid) is None:
            print(f"Could not build an instance for {rid} — is it in the robots table?")
            return 1

    stats = []
    for style in args.styles:
        path = os.path.join(args.out, f"audience_{style}.txt") if args.out else None
        if path:
            os.makedirs(args.out, exist_ok=True)
        with (open(path, "w") if path else sys.stdout) as fh:
            if path:
                sys.stdout.write(f"[audience] running {style} -> {path}\n")
                sys.stdout.flush()
            stats.append(run_condition(style, registry, kg, fh))

    print("\n" + "=" * 78)
    print("  Same scenario, same words, same interruptions — style only")
    print("=" * 78)
    print(f"  {'audience':<14} {'words':>7} {'seconds':>9}")
    for s in stats:
        print(f"  {s['style']:<14} {s['words']:>7} {s['sec']:>9}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
