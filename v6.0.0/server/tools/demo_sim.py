"""
tools/demo_sim.py
=================
Run a whole lab demo headlessly and see what the decision layer records.

No robots, no LLM, no hardware. The orchestrator, the gateway, HeuristicPolicy
and the recorder are all REAL — only the transport, the robot instances and the
visitors are simulated. That is the point: this exercises the same code path a
live demo takes, so a decision logged here would have been logged there.

    python3 tools/demo_sim.py                 # in-memory, prints a summary
    python3 tools/demo_sim.py --persist       # also write to Supabase
    python3 tools/demo_sim.py --robots 4 --budget 300 --seed 7

What it produces, per run:
  * a QA_ADVANCE decision for every visitor turn in a Q&A window
  * a QA_ROUTE decision alongside each
  * a PLAN_REVISE decision when a scripted visitor asks for one
  * supervisor CORRECTIONS, injected at a configurable rate

The correction rate is the knob that matters. Real operators do not override
uniformly, so a simulated run tells you the pipeline works — it does NOT tell you
what a real correction distribution looks like. Do not train on this output.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rbac import AccessLevel, RobotIdentity          # noqa: E402
from decision import DecisionRecorder, MemoryDecisionSink  # noqa: E402
from demo.demo_orchestrator import DemoOrchestrator, DemoState  # noqa: E402
from gateway.websocket_gateway import WebSocketGateway     # noqa: E402

SCENARIO = "lab_demo_sim"

# Visitor turns fed into each Q&A window, in order. Chosen to hit every branch of
# HeuristicPolicy: the question heuristic, the advance-phrase list, the intent
# classifier, and each PLAN_REVISE trigger.
VISITOR_SCRIPT = [
    "What is retrieval augmented generation?",       # question_heuristic  -> stay
    "how does the emotion model work?",              # question_heuristic  -> stay
    "hmm, alright then",                             # llm_classifier      -> stay/advance
    "can we hear more about {a}",                    # interest_request    -> extend_qa
    "that's fascinating",                            # llm_classifier
    "we are running out of time",                    # time_pressure       -> compress
    "okay let's move on",                            # advance_phrase      -> advance
    "can we skip the {b} part",                      # skip_request        -> skip
    "no more questions",                             # advance_phrase      -> advance
]

OPERATOR_REASONS = [
    "dragging on", "visitors lost interest", "wrong robot answered",
    "running late", "visitors had more to ask",
]


# ── Simulated robots ──────────────────────────────────────────────────────────

class SimInstance:
    """The slice of RobotInstance the decision layer actually touches."""

    def __init__(self, client_id, name, role, level=AccessLevel.LOCAL, rng=None):
        self.client_id = client_id
        self.robot_name = name
        self.access_level = level
        self._role = role
        self._rng = rng or random.Random(0)
        self.classify_calls = 0

    @property
    def identity(self) -> RobotIdentity:
        return RobotIdentity(
            robot_id=self.client_id, scenario_id=SCENARIO,
            session_id=f"sim-{self.client_id}",
            access_level=self.access_level, role=self._role,
        )

    def classify_qa_intent(self, message: str) -> str:
        """Stand-in for the LLM classifier. Biased toward 'continue', matching
        the real one's safe default — a simulated demo should not end early."""
        self.classify_calls += 1
        return "done" if self._rng.random() < 0.25 else "continue"


class SimRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}

    def get(self, client_id):
        return self._by_id.get(client_id)

    def get_all(self, exclude_id=None):
        return [i for i in self._by_id.values() if i.client_id != exclude_id]


class SimGateway(WebSocketGateway):
    """
    Real gateway, fake wire.

    Subclassed rather than mocked so every decision still runs through
    WebSocketGateway._decide and HeuristicPolicy — if this file passed and the
    live path differed, the exercise would be worthless.
    """

    def __init__(self, registry, recorder, verbose=False):
        super().__init__(registry, recorder=recorder)
        self.sent = []
        self._verbose = verbose

    def send_to_robot(self, client_id, data):
        self.sent.append((client_id, data))
        if self._verbose and data.get("event") == "demo_step":
            text = (data.get("text") or "")[:70]
            print(f"    → {client_id}: {text}")
        # Auto-ACK: the robots we are pretending to have always finish speaking.
        #
        # Deferred onto a timer, NOT called inline. _send_step sends before the
        # orchestrator enters WAITING_ACK and clears _ack_event, so an inline ACK
        # is wiped by that clear and the run loop then waits out the full step
        # timeout. The delay only has to outlast that window.
        if data.get("event") == "demo_step" and data.get("require_ack"):
            step_id = data.get("step_id")
            if step_id and self._demo_orchestrator:
                threading.Timer(
                    0.02, self._demo_orchestrator.receive_ack, args=(step_id,),
                ).start()

    def generate_demo_step(self, robot_id, instruction):
        # The real path calls an LLM here. Return a short stand-in rather than the
        # instruction verbatim: the orchestrator treats "generated == instruction"
        # as a generation FAILURE and logs a warning per step, which buried the
        # warnings that actually mattered under 20 lines of noise.
        first = instruction.split(".")[0][:60]
        return f"[DEFAULT] ({robot_id} says) {first}."


# ── The run ───────────────────────────────────────────────────────────────────

def run(n_robots=3, budget=None, seed=1, correction_rate=0.3,
        persist=False, verbose=False) -> dict:
    rng = random.Random(seed)

    guide = "pepper_01"
    projects = [f"robot_{chr(97 + i)}" for i in range(n_robots)]
    names = {p: p.replace("robot_", "Robot ").title() for p in projects}

    registry = SimRegistry(
        [SimInstance(guide, "Pepper", "Lab guide", AccessLevel.GLOBAL, rng)]
        + [SimInstance(p, names[p], f"{p} research", AccessLevel.LOCAL, rng)
           for p in projects]
    )

    sink = MemoryDecisionSink()
    if persist:
        from decision import BatchingDecisionSink
        from data.demo_decision_repo import write_corrections, write_events
        sink = BatchingDecisionSink(decision_writer=write_events,
                                    correction_writer=write_corrections)

    recorder = DecisionRecorder(sink)
    gw = SimGateway(registry, recorder, verbose=verbose)
    orch = DemoOrchestrator(gw, transition_delay=0.0, recorder=recorder,
                            session_context=gw.session_context)
    gw.set_demo_orchestrator(orch)

    stats = Counter()
    turns_fed = [0]

    def visitor_thread():
        """Feed visitor turns whenever a Q&A window is open, and sometimes
        override the outcome as a supervisor would."""
        script = list(VISITOR_SCRIPT)
        while True:
            status = orch.get_status()
            state = status.get("state")
            if state in ("idle", "completed", "error"):
                return
            if state != "qa_window":
                time.sleep(0.01)
                continue

            if not script:
                # Out of scripted turns — close the window so the tour advances.
                orch.manual_next(source="operator", reason="end of scripted visitors")
                stats["correction_manual_next"] += 1
                time.sleep(0.02)
                continue

            text = script.pop(0).format(a=names[projects[0]],
                                        b=names[projects[-1]])
            speaker = rng.choice(projects)
            turns_fed[0] += 1

            if verbose:
                print(f"    visitor → {speaker}: {text!r}")

            result = gw.check_qa_advance_from_user(registry.get(speaker), text)
            if result is not None:
                stats[f"advance:{result.mechanism}:{result.action.kind.value}"] += 1
            gw.check_plan_revision(registry.get(speaker), text)
            gw._decide(  # QA_ROUTE — logged, outcome unchanged by the baseline
                __import__("decision").DecisionPoint.QA_ROUTE,
                registry.get(speaker), text)

            # Supervisor override. Real operators are not this uniform; this only
            # proves corrections reach the log.
            if rng.random() < correction_rate:
                if rng.random() < 0.7:
                    orch.manual_next(source="operator",
                                     reason=rng.choice(OPERATOR_REASONS))
                    stats["correction_manual_next"] += 1
                else:
                    orch.qa_interrupt(source="operator",
                                      reason=rng.choice(OPERATOR_REASONS))
                    stats["correction_qa_interrupt"] += 1
            time.sleep(0.02)

    def _counts() -> dict:
        """Row counts for the persisted tables, or -1 when unreadable."""
        if not persist:
            return {}
        from data.connection import get_client
        out = {}
        for t in ("demo_decision_log", "demo_correction_log"):
            try:
                out[t] = get_client().table(t).select(
                    "*", count="exact").limit(1).execute().count or 0
            except Exception:
                out[t] = -1
        return out

    before = _counts()
    t0 = time.time()
    orch.start(robot_ids=[guide] + projects, time_budget_sec=budget)
    driver = threading.Thread(target=visitor_thread, daemon=True, name="sim-visitors")
    driver.start()

    # Wait for the tour to finish, with a hard ceiling so a stuck run cannot hang.
    deadline = time.time() + 60
    while time.time() < deadline:
        if orch.get_status()["state"] in ("completed", "idle", "error"):
            break
        time.sleep(0.05)
    else:
        orch.stop()
        print("  ! simulation hit the 60s ceiling — stopped")

    driver.join(timeout=2)
    recorder.flush()
    elapsed = time.time() - t0

    if persist:
        # The batching sink writes on a daemon thread; give the final flush a
        # moment to land before counting, or a healthy run looks empty.
        time.sleep(0.5)

    return {
        "sink": sink, "orch": orch, "gw": gw, "stats": stats,
        "elapsed": elapsed, "turns": turns_fed[0], "persist": persist,
        "final_state": orch.get_status()["state"],
        "row_counts": (before, _counts()) if persist else ({}, {}),
    }


def report(res: dict) -> None:
    sink, orch, stats = res["sink"], res["orch"], res["stats"]
    status = orch.get_status()

    print()
    print("=" * 62)
    print("  Simulated demo — result")
    print("=" * 62)
    print(f"  final state     : {res['final_state']}")
    print(f"  steps in script : {status['total']} (after revisions)")
    print(f"  visitor turns   : {res['turns']}")
    print(f"  wall clock      : {res['elapsed']:.1f}s")

    revisions = status.get("revisions") or []
    if revisions:
        print(f"  plan revisions  : {len(revisions)}")
        for r in revisions:
            ops = ", ".join(f"{o['kind']}({o.get('robot_id') or ''})".replace("()", "")
                            for o in r["ops"])
            print(f"      at {r['at_step']}: {ops}  [{r['source']}] {r['reason'][:40]}")
    else:
        print("  plan revisions  : none")

    if res["persist"]:
        # Count the rows rather than claiming success. The sinks degrade a write
        # failure to a warning by design, so "the run finished" says nothing
        # about whether anything landed — this printed a cheerful lie once
        # already, while RLS was rejecting every insert.
        print()
        before, after = res["row_counts"]
        for table in ("demo_decision_log", "demo_correction_log"):
            delta = after[table] - before[table]
            if after[table] < 0:
                print(f"  {table:22} could not be counted — check the DB")
            elif delta > 0:
                print(f"  {table:22} +{delta} rows  (now {after[table]})")
            else:
                print(f"  {table:22} NO ROWS WRITTEN — check the warnings above")
        if all(after[t] - before[t] <= 0 for t in
               ("demo_decision_log", "demo_correction_log")):
            print()
            print("  Nothing persisted. Run tools/check_migrations.py — it probes")
            print("  an actual insert and will name the reason.")
            return
        print()
        print("  Inspect with:")
        print("    python3 -c \"from data.demo_decision_repo import *; "
              "print(corrections_by_mechanism())\"")
        return

    print()
    print("  Decisions by mechanism")
    by_mech = sink.decisions_by_mechanism()
    if not by_mech:
        print("      (none — the tour never opened a Q&A window)")
    for mech, n in sorted(by_mech.items(), key=lambda kv: -kv[1]):
        print(f"      {mech:22} {n:4}")

    print()
    print("  Decisions by point")
    by_point = Counter(d.decision_point for d in sink.decisions)
    for point, n in sorted(by_point.items(), key=lambda kv: -kv[1]):
        print(f"      {point:22} {n:4}")

    print()
    print(f"  Corrections           : {len(sink.corrections)}")
    attached = sum(1 for c in sink.corrections if c.decision_id)
    print(f"      attached to a decision : {attached}")
    print(f"      orphan (no decision)   : {len(sink.corrections) - attached}")

    # The headline number the analysis depends on.
    if sink.decisions:
        rate = len(sink.corrections) / len(sink.decisions)
        print(f"      correction rate        : {rate:.1%}")

    print()
    print("  Correction rate per mechanism  (what demo_corrections_by_mechanism computes)")
    corrected = Counter()
    by_id = {d.decision_id: d for d in sink.decisions}
    for c in sink.corrections:
        d = by_id.get(c.decision_id)
        if d:
            corrected[d.mechanism] += 1
    for mech, n in sorted(by_mech.items(), key=lambda kv: -kv[1]):
        print(f"      {mech:22} {corrected[mech]:3}/{n:<4} "
              f"{corrected[mech] / n:6.1%}")

    # Sanity checks a human would otherwise have to eyeball.
    print()
    problems = []
    if not sink.decisions:
        problems.append("no decisions logged")
    if any(not d.mechanism for d in sink.decisions):
        problems.append("a decision has a null mechanism")
    if any(not d.observation.get("connected_peers") for d in sink.decisions):
        problems.append("a decision logged no peers")
    if any(d.session_id is None for d in sink.decisions):
        problems.append("a decision has no session_id (would not join rbac_audit_log)")
    print("  CHECKS:", "; ".join(problems) if problems else "all passed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--robots", type=int, default=3, help="project robots (default 3)")
    ap.add_argument("--budget", type=float, default=None,
                    help="time budget in seconds; omit to disable clock-driven revision")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--correction-rate", type=float, default=0.3,
                    help="probability an operator overrides a turn (default 0.3)")
    ap.add_argument("--persist", action="store_true",
                    help="write to Supabase instead of memory (needs migration 004)")
    ap.add_argument("--verbose", action="store_true", help="print every step and turn")
    args = ap.parse_args()

    if args.persist:
        # Fail early and clearly rather than dropping every batch into a warning.
        from data.connection import get_client
        try:
            get_client().table("demo_decision_log").select("*").limit(1).execute()
        except Exception as e:
            print("Cannot write: demo_decision_log is not reachable.")
            print(f"  {str(e)[:100]}")
            print("Apply data/migrations/apply_all.sql first, then re-run.")
            return 1

    res = run(n_robots=args.robots, budget=args.budget, seed=args.seed,
              correction_rate=args.correction_rate,
              persist=args.persist, verbose=args.verbose)
    report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
