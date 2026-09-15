"""
tools/audience_experiment.py
=============================
Does the visitor profile change how the robots talk — and does a hand-off to
a specialist actually fire?

Runs ONE fixed scenario — the same tour, the same visitor utterances, the
same interruption points — once per audience, changing nothing but
VisitorProfile.style. Any difference in the transcripts is attributable to
the style and to nothing else.

    python3 tools/audience_experiment.py                  # all three
    python3 tools/audience_experiment.py --styles technical
    python3 tools/audience_experiment.py --out runs/audience

IT DRIVES THE REAL GATEWAY
Every visitor turn goes through WebSocketGateway.route_question ->
process_chat_stream -> DelegationHandler, wired exactly as app.create_app()
wires it, with send_to_robot recorded instead of dialled. An earlier version
reimplemented the routing decision in the harness, which meant it could not
show the hand-off at all: whether Pepper announces a specialist and whether
that specialist then speaks are properties of the gateway, so the gateway is
what has to run.

Two different mechanisms can produce "Pepper names a robot, that robot
answers", and the transcript labels which one fired:

  KG REROUTE      the competence graph resolved the question to a topic
                  another robot declares. Deterministic. The receiver speaks
                  a one-line hand-off, the named robot answers in its own
                  voice.
  LLM DELEGATION  no confident graph read, so the answering robot's own LLM
                  decided to hand over and emitted a JSON block, which
                  DelegationHandler executes against the target.

Routing is tried first on purpose — it is deterministic and auditable, and
delegation is the fallback for what it cannot resolve.

WHAT THIS CANNOT TELL YOU
Whether the styling is GOOD — only whether it differs, and how. Judging "is
this pitched right for a school group" needs a person; that is what the
ratings in decision/style_fit.py are for. This produces the transcripts that
make the judgement possible.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GUIDE = "pepper_01"
PROJECTS = ["silbot_01", "chatbox_01"]

# Every visitor turn is addressed to PEPPER — the guide holds the microphone,
# which is the situation being tested: the visitor interrupts the guide while
# a specialist is mid-talk. `after` names the step the interruption lands on.
VISITOR_SCRIPT = [
    # Silbot is talking. Question about Silbot's own subject: the presenting
    # robot should take it, with no announcement — it is already the owner.
    {"after": "silbot_01_project_problem",
     "say": "so which technique do you use"},

    # Silbot is still talking, but this is CHATBOX's declared subject. This is
    # the hand-off case: Pepper should name ChatBox and ChatBox should answer.
    {"after": "silbot_01_project_approach",
     "say": "what about retrieval augmented generation, how does that work"},

    # An acknowledgement. Must NOT be routed anywhere.
    {"after": "silbot_01_project_impact",
     "say": "okay thank you"},

    # ChatBox is talking; this is SILBOT's subject. Hand-off the other way.
    {"after": "chatbox_01_project_problem",
     "say": "and how does social robot navigation avoid people"},

    # ChatBox's own subject while ChatBox presents.
    {"after": "chatbox_01_project_approach",
     "say": "what models do you use for that"},

    # An ORPHAN topic — no robot declares text-to-speech. Routing cannot
    # resolve it to a specialist, so this is the turn that gives the LLM
    # delegation path something to do. Without it the graph answers
    # everything and DelegationHandler is never exercised at all.
    {"after": "chatbox_01_project_impact",
     "say": "and which text to speech engine makes the voices"},

    # A question whose answer is NOT in demo_topic_facts. The honesty rule
    # should produce "I would have to check" rather than a plausible
    # invention — this is the turn that proves grounding changed behaviour
    # rather than just adding text to a prompt.
    {"after": "silbot_01_project_impact",
     "say": "what frame rate does the mapping run at"},
]

AUDIENCES = {
    "interactive": "High-school students on a school visit",
    "business":    "Industry visitors evaluating the lab's work",
    "technical":   "Visiting robotics researchers",
}


def _wrap(text: str, indent: str = "      ") -> str:
    return "\n".join(textwrap.wrap(text, 76, initial_indent=indent,
                                   subsequent_indent=indent)) or indent + "(empty)"


def _closes_the_window(utterance: str) -> bool:
    """True when the turn ends the Q&A instead of asking anything.

    The deterministic half of HeuristicPolicy._decide_advance. The LLM
    classifier is deliberately not consulted: it is nondeterministic, and a
    controlled comparison must not have its conditions diverge on a coin flip.
    """
    from decision.policy import (QA_ADVANCE_PHRASES, TIME_PRESSURE_PHRASES,
                                 _is_bare_affirmation, _matches)
    return bool(_matches(utterance, QA_ADVANCE_PHRASES)
                or _is_bare_affirmation(utterance)
                or _matches(utterance, TIME_PRESSURE_PHRASES))


def build_system(style: str):
    """A real gateway + orchestrator, wired as app.create_app() wires them."""
    import app as server_app
    from core.profiles import ProfileRegistry
    from core.rbac import GrantStore, RBACFilter
    from decision import DecisionRecorder
    from decision.visitor_profile import VisitorProfile
    from demo.demo_orchestrator import DemoOrchestrator
    from demo.demo_script import build_script
    from gateway.websocket_gateway import WebSocketGateway
    from robot.robot_registry import RobotRegistry

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    profiles = ProfileRegistry.from_directory(os.path.join(here, "profiles"))
    registry = RobotRegistry(rbac=RBACFilter(), grants=GrantStore(),
                             profiles=profiles)
    for rid in [GUIDE] + PROJECTS:
        if registry.connect(rid) is None:
            raise SystemExit(f"Could not build an instance for {rid}.")

    # NO kg_observer. This is a measurement tool, and wiring one trains the
    # graph on the very runs used to measure it — the same fault as an
    # unattended harness run, which demo_harness._observe already refuses.
    # It was wired, and twelve outcome observations landed on an undeclared
    # silbot_01 -> text-to-speech edge, taking its weight to 0.82 and
    # changing what the next run routed. Measurement must not move what it
    # measures.
    gw = WebSocketGateway(registry, recorder=DecisionRecorder(),
                          kg_router_factory=server_app.build_kg_router,
                          kg_observer=None)
    orch = DemoOrchestrator(gw, session_context=gw.session_context,
                            subject_lookup=server_app.lookup_subject)
    subjects = {r: server_app.lookup_subject(r) for r in PROJECTS}
    orch.load_script(build_script(GUIDE, PROJECTS, subjects=subjects))
    gw.set_demo_orchestrator(orch)
    orch._visitor_profile = VisitorProfile(style=style)
    return gw, orch, registry


def run_condition(style: str, out) -> dict:
    from decision.observation import looks_like_question
    from demo.demo_orchestrator import DemoState

    gw, orch, registry = build_system(style)
    say = out.write

    # Record what each robot is told to say instead of dialling a socket.
    spoken: list = []
    gw.send_to_robot = lambda cid, d: spoken.append((cid, d))

    say(f"\n{'=' * 78}\n  AUDIENCE: {AUDIENCES[style]}   (style={style!r})\n")
    say(f"{'=' * 78}\n")
    say("  Style directive appended to every generation:\n")
    say(_wrap(orch.framing_for_robot(GUIDE).strip() or "(none — general audience)") + "\n\n")

    by_step = {s["after"]: s["say"] for s in VISITOR_SCRIPT}
    started, words = time.time(), 0
    handoffs, delegations, presenter_takes = 0, 0, 0

    for idx, step in enumerate(orch._script):
        orch._idx = idx
        orch._state = DemoState.RUNNING
        before = len(spoken)
        orch._send_step(step)                      # real generation + framing
        for _cid, d in spoken[before:]:
            text = d.get("text") or ""
            if text:
                words += len(text.split())
                say(f"  [{step.robot_id}] {step.step_id}\n")
                say(_wrap(text) + "\n\n")

        utterance = by_step.get(step.step_id)
        if not utterance:
            continue

        say(f'  >>> VISITOR (to Pepper): "{utterance}"\n')
        if _closes_the_window(utterance):
            say("      -> advance: window closes, demo continues. No reply.\n\n")
            continue

        from decision.observation import is_acknowledgement
        if is_acknowledgement(utterance):
            say("      -> acknowledgement: fixed reply, no generation "
                "(deterministic, see is_acknowledgement)\n\n")
            continue

        # A Q&A window is what an interruption opens; routing only runs inside one.
        orch._state = DemoState.QA_WINDOW
        gw.on_qa_window_open()

        mark = len(spoken)
        pepper = registry.get(GUIDE)
        target_inst, target_id, handoff = gw.route_question(pepper, utterance)

        if handoff:
            handoffs += 1
            words += len(handoff.split())
            say("      -- KG REROUTE: the graph resolved this to another robot's subject\n")
            say(f"  [{GUIDE}] hand-off\n")
            say(_wrap(handoff) + "\n\n")
        elif target_id != GUIDE:
            presenter_takes += 1
            say(f"      -- presenter fallback: {target_id} takes its own question "
                f"(silent, no announcement)\n")

        if target_inst is None:
            say("      -- deferred: covered at that robot's own station. No reply.\n\n")
            gw.on_qa_window_close()
            continue

        grounding = gw.grounding_for(target_id, utterance)
        if grounding:
            say(f"      -- grounded on {len(grounding)} stored fact(s)"
                + (" [includes UNVERIFIED]\n"
                   if any("UNVERIFIED" in g for g in grounding) else "\n"))
        else:
            say("      -- NO stored facts for this topic: the honesty rule "
                "applies, specifics must be declined\n")

        replies: list = []
        result = target_inst.process_chat_stream(
            utterance, lambda t, _tag: replies.append(t),
            style_framing=gw.style_framing_for(target_id),
            grounded_facts=grounding)
        reply = " ".join(replies).strip()
        if reply:
            words += len(reply.split())
            say(f"  [{target_id}] answers\n")
            say(_wrap(reply) + "\n\n")

        # The LLM's own hand-off, if the graph had no opinion and it chose one.
        if result.is_delegation and result.delegation_target:
            from gateway.delegation_handler import DelegationHandler
            delegations += 1
            say("      -- LLM DELEGATION: the answering robot handed over itself\n")
            mark2 = len(spoken)
            DelegationHandler(registry, gw).execute_sync(
                target_id, result.delegation_target,
                f"Answer the visitor: {utterance}")
            for cid, d in spoken[mark2:]:
                t = d.get("text") or d.get("clean_text") or ""
                if t and "?" not in d.get("step_id", ""):
                    words += len(t.split())
                    say(f"  [{cid}] delegated answer\n")
                    say(_wrap(t) + "\n\n")

        gw.on_qa_window_close()

    elapsed = time.time() - started
    say(f"  --- {words} words | {handoffs} KG reroute(s) | {delegations} LLM "
        f"delegation(s) | {presenter_takes} presenter take(s) | {elapsed:.0f}s ---\n")
    return {"style": style, "words": words, "handoffs": handoffs,
            "delegations": delegations, "presenter": presenter_takes,
            "sec": round(elapsed, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--styles", nargs="*", default=list(AUDIENCES),
                    choices=list(AUDIENCES))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    stats = []
    for style in args.styles:
        path = os.path.join(args.out, f"audience_{style}.txt") if args.out else None
        # NEVER `with sys.stdout as fh:` — a with-block calls __exit__ on
        # whatever it wraps when the body finishes, and TextIOWrapper.__exit__
        # closes the underlying stream. With no --out, `fh` was real stdout on
        # every iteration, so the first style's block closed process stdout
        # for good; the second style's very first `say(...)` call then raised
        # "ValueError: I/O operation on closed file" and the run silently
        # produced only the first style (interactive, first in AUDIENCES).
        # Only a file THIS LOOP opened may ever be closed here.
        if path:
            os.makedirs(args.out, exist_ok=True)
            sys.stdout.write(f"[audience] {style} -> {path}\n")
            sys.stdout.flush()
            fh = open(path, "w")
        else:
            fh = sys.stdout
        try:
            stats.append(run_condition(style, fh))
        finally:
            if path:
                fh.close()

    print("\n" + "=" * 78)
    print("  Same tour, same visitor words, same interruptions — style only")
    print("=" * 78)
    print(f"  {'audience':<13} {'words':>6} {'KG reroute':>11} {'LLM deleg':>10} "
          f"{'presenter':>10} {'sec':>6}")
    for s in stats:
        print(f"  {s['style']:<13} {s['words']:>6} {s['handoffs']:>11} "
              f"{s['delegations']:>10} {s['presenter']:>10} {s['sec']:>6}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
