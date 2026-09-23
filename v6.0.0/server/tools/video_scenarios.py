#!/usr/bin/env python3
"""
tools/video_scenarios.py
=========================
Drive the LIVE demo (real Gazebo robots, real TTS) through one of three
scripted visitor scenarios, for recording. Talks to the already-running
server over plain HTTP — nothing here touches Gazebo directly.

Every Q&A window in demo_script.py is qa_timeout=0 (manual "Move On" only),
so even the no-questions scenario has to actively close each window —
there is no such thing as "just let it run" here.

    clean     — no visitor interaction. Every Q&A window closes immediately,
                as if nobody had a question.
    qa        — one topic question asked in each Q&A window, then closed.
                Never speaks while a robot is mid-presentation.
    interrupt — same as `qa`, PLUS one scripted interruption fired mid-
                presentation at the first project robot that's speaking,
                to show the barge-in path (tts_stop -> qa_interrupt ->
                resume of the remaining text).

Start the full demo stack first (Gazebo, gazebo_client bridges, app.py),
start screen recording, then run one of:

    python3 tools/video_scenarios.py clean
    python3 tools/video_scenarios.py qa
    python3 tools/video_scenarios.py interrupt
    python3 tools/video_scenarios.py qa --answer-pause 12
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import requests

GUIDE = "pepper_01"
PROJECT_ROBOTS = ["chatbox_01", "navel_01", "silbot_01"]

QUESTIONS = {
    "chatbox_01": "How does retrieval-augmented generation actually help you answer questions?",
    "navel_01": "How do you detect someone's emotion from their face or voice?",
    "silbot_01": "How do you navigate around people without bumping into them?",
    "pepper_01": "What's your favourite part of working in this lab?",
}
INTERRUPT_QUESTION = "Sorry to interrupt — what does your robot actually do?"

# Distinct from all four robot voices (Pepper=en-GB-Sonia, ChatBox=en-US-Guy,
# Navel=en-AU-Natasha, Silbot=en-GB-Ryan — see gazebo_client/configs/*.json)
# so a visitor line is never mistaken for a robot's own voice.
VISITOR_VOICE = "en-US-AriaNeural"

POLL_SEC = 0.5
DEFAULT_ANSWER_PAUSE_SEC = 10.0  # time to let TTS finish before closing a window
DEFAULT_INTERRUPT_AT_SEC = 4.0   # how far into a presentation to barge in
NO_VOICE_DELAY_SEC = 3.0         # --no-voice: silent placeholder instead of a spoken line


def _status(server: str) -> dict:
    return requests.get(f"{server}/demo/status", timeout=5).json()


def _speak_visitor(text: str) -> None:
    """Synthesize `text` in a voice distinct from every robot and play it
    through the default audio sink — the same sink the recording captures
    (its .monitor source), so this lands in the video with no extra mixing
    step, overlapping any robot speech already in progress exactly like a
    live interruption would."""
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        subprocess.run(["edge-tts", "--voice", VISITOR_VOICE, "--text", text,
                         "--write-media", path],
                        check=True, capture_output=True, timeout=20)
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                        check=False, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    [voice] synthesis/playback failed, continuing without audio: {e}")
    finally:
        os.unlink(path)


def _ask(server: str, client_id: str, text: str, voice: bool = True) -> None:
    print(f"    visitor -> {client_id}: {text!r}")
    if voice:
        _speak_visitor(text)
    else:
        time.sleep(NO_VOICE_DELAY_SEC)
    requests.post(f"{server}/robots/{client_id}/chat",
                   json={"message": text, "source": "visitor"}, timeout=30)


def _block_robot(s: dict) -> str | None:
    """The project a Q&A window belongs to. The window itself is hosted by
    the guide (robot_id), so picking the question by robot_id asked the
    guide's own question in every window."""
    steps, i = s.get("steps") or [], s.get("step_idx", 0)
    return (steps[i].get("block_robot_id") if i < len(steps) else None) or s["robot_id"]


def _next(server: str, reason: str = "operator move-on") -> None:
    requests.post(f"{server}/demo/next", json={"reason": reason}, timeout=5)


def _start(server: str, robot_ids: list[str] | None, interest: str = "") -> None:
    body = {"robot_ids": robot_ids} if robot_ids else {}
    if interest:
        body["visitor_interest"] = interest
    resp = requests.post(f"{server}/demo/start", json=body, timeout=10)
    resp.raise_for_status()
    print("Demo started.")


def run_clean(server: str, **_ignored) -> None:
    """No visitor interaction. Every Q&A window closes immediately."""
    handled: set[str | None] = set()
    while True:
        s = _status(server)
        state = s["state"]
        if state in ("completed", "idle", "error"):
            print(f"Done ({state}).")
            return
        if state == "qa_window" and s["step_id"] not in handled:
            handled.add(s["step_id"])
            print(f"  [qa_window @ {s['robot_id']}] no questions — moving on")
            _next(server)
        time.sleep(POLL_SEC)


def run_qa(server: str, answer_pause: float = DEFAULT_ANSWER_PAUSE_SEC,
           voice: bool = True, **_ignored) -> None:
    """One topic question per Q&A window, then close it."""
    handled: set[str | None] = set()
    while True:
        s = _status(server)
        state = s["state"]
        if state in ("completed", "idle", "error"):
            print(f"Done ({state}).")
            return
        if state == "qa_window" and s["step_id"] not in handled:
            handled.add(s["step_id"])
            robot = s["robot_id"] or GUIDE
            question = QUESTIONS.get(_block_robot(s), "Can you tell me more about your work?")
            _ask(server, robot, question, voice=voice)
            time.sleep(answer_pause)
            print(f"  [qa_window @ {robot}] closing")
            _next(server)
        time.sleep(POLL_SEC)


def run_interrupt(server: str, answer_pause: float = DEFAULT_ANSWER_PAUSE_SEC,
                   interrupt_at: float = DEFAULT_INTERRUPT_AT_SEC,
                   voice: bool = True) -> None:
    """Same as `qa`, plus one scripted interruption during the first
    project robot's presentation (mid-speech, not a Q&A window)."""
    handled: set[str | None] = set()
    running_since: dict[str | None, float] = {}
    interrupted = False
    while True:
        s = _status(server)
        state, step_id, robot = s["state"], s["step_id"], s["robot_id"]
        if state in ("completed", "idle", "error"):
            print(f"Done ({state}).")
            return

        if state == "waiting_ack" and robot in PROJECT_ROBOTS:
            running_since.setdefault(step_id, time.time())
            if not interrupted and time.time() - running_since[step_id] >= interrupt_at:
                interrupted = True
                print(f"  [interrupt @ {robot}] barging in mid-presentation")
                _ask(server, robot, INTERRUPT_QUESTION, voice=voice)
                time.sleep(answer_pause)
                handled.add(step_id)  # the interrupt opened a qa_window; already closing it
                print(f"  [qa_window @ {robot}] closing (post-interrupt)")
                _next(server)

        if state == "qa_window" and step_id not in handled:
            handled.add(step_id)
            question = QUESTIONS.get(_block_robot(s), "Can you tell me more about your work?")
            _ask(server, robot or GUIDE, question, voice=voice)
            time.sleep(answer_pause)
            print(f"  [qa_window @ {robot}] closing")
            _next(server)

        time.sleep(POLL_SEC)


TIME_PRESSURE_LINE = "Sorry, we're running out of time. Can we keep it short?"


def run_time(server: str, voice: bool = True, **_ignored) -> None:
    """Visitor states time pressure in the first project's Q&A window; every
    later window closes with no question. The time-pressure line closes its
    own window (and revises the rest of the tour), so no _next for it."""
    handled: set[str | None] = set()
    said = False
    while True:
        s = _status(server)
        state = s["state"]
        if state in ("completed", "idle", "error"):
            print(f"Done ({state}).")
            return
        if state == "qa_window" and s["step_id"] not in handled:
            handled.add(s["step_id"])
            if not said and _block_robot(s) in PROJECT_ROBOTS:
                said = True
                _ask(server, s["robot_id"] or GUIDE, TIME_PRESSURE_LINE, voice=voice)
                print(f"  [qa_window @ {_block_robot(s)}] time pressure stated; revisions: {s.get('revisions')}")
            else:
                print(f"  [qa_window @ {_block_robot(s)}] no questions — moving on")
                _next(server)
        time.sleep(POLL_SEC)


SCENARIOS = {"clean": run_clean, "qa": run_qa, "interrupt": run_interrupt, "time": run_time}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Drive the live demo through a scripted visitor scenario for video.")
    ap.add_argument("scenario", choices=SCENARIOS)
    ap.add_argument("--server", default="http://127.0.0.1:5000")
    ap.add_argument("--no-start", action="store_true",
                     help="demo was already started elsewhere — just drive it")
    ap.add_argument("--robots", nargs="*", default=None,
                     help="robot_ids for /demo/start (default: the server's own script order)")
    ap.add_argument("--answer-pause", type=float, default=DEFAULT_ANSWER_PAUSE_SEC,
                     help="seconds to let a spoken answer finish before closing its window")
    ap.add_argument("--interrupt-at", type=float, default=DEFAULT_INTERRUPT_AT_SEC,
                     help="[interrupt only] seconds into a presentation before barging in")
    ap.add_argument("--interest", default="",
                     help="visitor_interest for /demo/start, e.g. 'robot navigation'")
    ap.add_argument("--no-voice", action="store_true",
                     help="skip synthesizing a visitor voice; wait "
                          f"{NO_VOICE_DELAY_SEC:.0f}s instead (dub it in yourself later)")
    args = ap.parse_args()

    try:
        requests.get(f"{args.server}/health", timeout=3)
    except requests.RequestException as e:
        print(f"Cannot reach the server at {args.server} — is app.py running?\n  {e}")
        return 1

    if not args.no_start:
        _start(args.server, args.robots, args.interest)
        time.sleep(1.0)

    print(f"Running scenario '{args.scenario}' against {args.server} ...")
    SCENARIOS[args.scenario](args.server, answer_pause=args.answer_pause,
                              interrupt_at=args.interrupt_at, voice=not args.no_voice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
