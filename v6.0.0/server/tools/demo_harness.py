"""
tools/demo_harness.py
=====================
Run the lab demo without robots, with a human acting as supervisor.

Campaign infrastructure, not a side tool: a live demo yields 2-3 corrections and
happens rarely, so this is the only way to accumulate correction data at the
volume the decision model needs.

    python3 tools/demo_harness.py --seed 1 --questions 8
    python3 tools/demo_harness.py --mode eval          # participant-facing view
    python3 tools/demo_harness.py --auto accept        # unattended, no operator
    python3 tools/demo_harness.py --reset              # empty graph, keep vocabulary
    python3 tools/demo_harness.py --metrics            # read the campaign so far

NOTHING ON THE DECISION PATH IS MOCKED
The real build_script, the real DemoOrchestrator, the real KGRouter and route(),
and the real persistence — corrections go through kg_feedback.from_reroute and
outcomes through Segment, the same calls /kg/reroute and the window-close hook
make. If any of that were stubbed the collected data would be in a schema that
does not match production, and the campaign would have to be re-run.

Only the edges are stubbed, and only these:
  * robot clients   — no WebSocket, no TTS, no waiting on physical speech
  * speech input    — visitor questions arrive as text
  * timing          — steps advance immediately unless the operator is asked

THE PAUSE IS THE POINT
At each routing decision the operator sees the question, the resolved topic, the
ranked candidates with clamped weight and n_obs, and which mechanism fired.
Accepting is not a no-op: it is what lets the segment close clean and emit its
OUTCOME signal, which is the only way the graph ever hears about routing that
went right.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rbac import AccessLevel, RobotIdentity          # noqa: E402
from decision import DecisionRecorder, MemoryDecisionSink  # noqa: E402
from decision.kg import RobotTopicEdge                     # noqa: E402
from decision.kg_feedback import Segment, apply, from_reroute   # noqa: E402
from decision.kg_infer import rank_robots                  # noqa: E402
from decision.kg_policy import KGRouter                    # noqa: E402
from demo.demo_orchestrator import DemoOrchestrator, DemoState  # noqa: E402
from gateway.websocket_gateway import WebSocketGateway     # noqa: E402
from tools.fixtures.questions import CORPUS_VERSION, QUESTIONS  # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
SCENARIO = "lab_demo_harness"


# ── Stubs, strictly at the edges ──────────────────────────────────────────────

class HarnessInstance:
    """A robot with no body. Everything the decision path reads, nothing else.

    Gains a real LLM when generation is on, because in evaluation mode the
    speech IS the thing being evaluated — a participant listening to instruction
    text ("Explain your research project to a non-expert audience. 3-4
    sentences.") is not experiencing the demo, they are reading its stage
    directions.
    """

    def __init__(self, client_id, name, role, level=AccessLevel.LOCAL, llm=None):
        self.client_id, self.robot_name = client_id, name
        self.access_level, self._role = level, role
        self.llm = llm
        self._history: list = []

    def generate_demo_speech(self, instruction: str):
        """Real generation, with the same system prompt shape robot_instance uses.

        History is carried across steps so a tour builds context rather than
        producing four disconnected paragraphs — which is one of the things a
        study participant would notice.
        """
        from types import SimpleNamespace
        if not self.llm or not self.llm.is_available():
            return SimpleNamespace(response=instruction, clean_text=instruction,
                                   emotion_tag="", is_delegation=False)
        system = (
            f"You are {self.robot_name}. Your role: {self._role}\n\n"
            "DEMO SPEECH RULES:\n"
            "1. The VERY FIRST character of your response MUST be '['.\n"
            "2. Use EXACTLY ONE emotion tag from: [DEFAULT], [WAVE], [HAPPY], [POINT]\n"
            "3. Speak naturally as yourself. The instruction tells you what to "
            "say and how long to speak.\n"
            "4. Never read the instruction aloud."
        )
        try:
            resp = self.llm.generate_with_history(system, self._history[-8:], instruction)
            text = (resp.text or "").strip()
            self._history.append({"role": "user", "content": instruction})
            self._history.append({"role": "assistant", "content": text})
            clean = text.split("]", 1)[-1].strip() if text.startswith("[") else text
            return SimpleNamespace(response=text, clean_text=clean,
                                   emotion_tag="", is_delegation=False)
        except Exception as e:
            print(f"    ! generation failed for {self.robot_name}: {str(e)[:60]}")
            return SimpleNamespace(response=instruction, clean_text=instruction,
                                   emotion_tag="", is_delegation=False)

    @property
    def identity(self):
        return RobotIdentity(robot_id=self.client_id, scenario_id=SCENARIO,
                             session_id=f"harness-{self.client_id}",
                             access_level=self.access_level, role=self._role)

    def classify_qa_intent(self, message):
        return "continue"      # the harness decides when a window closes


class HarnessRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}

    def get(self, cid):
        return self._by_id.get(cid)

    def get_all(self, exclude_id=None):
        return [i for i in self._by_id.values() if i.client_id != exclude_id]


# Words per second of spoken delivery. Deliberately slower than a person reads
# aloud, because TTS at demo pace is slower still and the point is that a
# participant experiences the tour's real rhythm.
SPEAKING_RATE = 2.4
MAX_STEP_SEC = 25.0


class HarnessGateway(WebSocketGateway):
    """Real gateway; the wire is a print statement, optionally a slow one."""

    def __init__(self, registry, recorder, generate=False, quiet=False,
                 observer=None, pace="instant"):
        super().__init__(registry, recorder=recorder, kg_observer=observer)
        self._generate, self._quiet = generate, quiet
        self._pace = pace

    def _speech_seconds(self, text: str) -> float:
        """How long this line would take to say.

        Instant advance is right for a campaign — nobody is listening, and a
        thousand rollouts should not take a week. It is wrong for a study: steps
        flashing past is not a demo, and a participant asked to rate pacing
        cannot rate pacing that does not exist.
        """
        if self._pace != "realistic":
            return 0.0
        words = len((text or "").split())
        return min(MAX_STEP_SEC, max(1.2, words / SPEAKING_RATE))

    def send_to_robot(self, client_id, data):
        if data.get("event") == "demo_step":
            instance = self._registry.get(client_id)
            label = instance.robot_name if instance else client_id
            raw = (data.get("text") or "").replace("\n", " ")
            if not self._quiet:
                shown = raw if self._pace == "realistic" else raw[:96]
                print(f"    {label}: {shown}")
            delay = self._speech_seconds(raw)
            if data.get("require_ack") and self._demo_orchestrator:
                # Deferred: _send_step sends before the orchestrator enters
                # WAITING_ACK and clears the event, so an inline ack is wiped.
                # The delay doubles as speech duration when pacing is realistic.
                threading.Timer(max(0.02, delay),
                                self._demo_orchestrator.receive_ack,
                                args=(data.get("step_id"),)).start()

    def generate_demo_step(self, robot_id, instruction):
        if self._generate:
            return super().generate_demo_step(robot_id, instruction)
        # Generation costs tokens and seconds, and the routing decision does not
        # depend on what the robot actually said. Off by default for campaign
        # volume; --generate for a realistic dry run before a real demo.
        first = instruction.split(".")[0][:80]
        return f"[DEFAULT] {first}."


# ── Graph snapshot / reset ────────────────────────────────────────────────────

def snapshot_graph() -> list:
    from data import demo_kg_repo as repo
    return repo.graph()


def reset_graph() -> int:
    """Delete every learned edge. Vocabulary and links are left alone."""
    from data.connection import get_client
    from data import demo_kg_repo as repo
    rows = repo.graph()
    client = get_client()
    for r in rows:
        client.table("demo_robot_topic").delete() \
            .eq("robot_id", r["robot_id"]).eq("topic_id", r["topic_id"]).execute()
    return len(rows)


# ── The run ───────────────────────────────────────────────────────────────────

class Harness:

    def __init__(self, args):
        self.args = args
        self.run_id = uuid.uuid4().hex[:12]
        self.decisions: list = []
        self.rng = random.Random(args.seed)
        self.eval_mode = args.mode == "eval"

        from data import demo_kg_repo as repo
        self.repo = repo
        self.topics = repo.all_topics()
        if not self.topics:
            raise SystemExit("No topic vocabulary. Run: python3 tools/seed_kg.py")
        self.label_of = {t["id"]: t["label"] for t in self.topics}

        guide = "pepper_01"
        projects = ["chatbox_jetson_001", "navel_001", "silbot_01"]
        self.guide, self.projects = guide, projects
        names = {"chatbox_jetson_001": "ChatBox", "navel_001": "Navel",
                 "silbot_01": "Silbot"}
        roles = {
            "chatbox_jetson_001": "You research retrieval augmented generation "
                                  "and conversational memory for robots.",
            "navel_001": "You research emotion recognition and social signals "
                         "in human-robot interaction.",
            "silbot_01": "You research social navigation and mapping for robots "
                         "moving among people.",
        }
        llm = self._build_llm() if args.generate else None
        registry = HarnessRegistry(
            [HarnessInstance(guide, "Pepper",
                             "You are the guide hosting a tour of the CARES "
                             "robotics lab.", AccessLevel.GLOBAL, llm=llm)]
            + [HarnessInstance(p, names[p], roles[p], llm=llm) for p in projects])
        self.names = dict(names, **{guide: "Pepper"})

        self.sink = MemoryDecisionSink()
        recorder = DecisionRecorder(self.sink)
        self.gw = HarnessGateway(registry, recorder,
                                 generate=args.generate,
                                 quiet=args.mode == "collect" and args.quiet,
                                 observer=self._observe,
                                 pace=args.pace)
        # A realistic run also keeps the orchestrator's settle delay between
        # steps. At 0 the next robot starts the instant the previous one stops,
        # which reads as an interruption rather than a handover.
        self.orch = DemoOrchestrator(self.gw,
                                     transition_delay=1.2 if args.pace == "realistic" else 0.0,
                                     recorder=recorder,
                                     session_context=self.gw.session_context)
        self.gw.set_demo_orchestrator(self.orch)
        self.registry = registry
        self.before = snapshot_graph()

    @staticmethod
    def _build_llm():
        """The real LLM module, or None. Generation is best-effort: an
        unreachable model degrades to instruction text with a warning rather
        than aborting a session a participant is sitting through."""
        try:
            from modules.llm.llm_module import LLMModule
            mod = LLMModule()
            if mod.initialize() and mod.is_available():
                print(f"  generation: {mod.provider_name}")
                return mod
            print("  ! LLM unavailable — steps will show instruction text")
        except Exception as e:
            print(f"  ! LLM init failed ({str(e)[:50]}) — instruction text only")
        return None

    # ── Graph access, refetched per decision so learning is visible mid-run ──
    def _router(self) -> KGRouter:
        edges = [RobotTopicEdge.from_row(r) for r in self.repo.graph()]
        links = [(l["topic_a"], l["topic_b"], float(l["weight"]))
                 for l in self.repo.all_links()]
        self._edges, self._links = edges, links
        return KGRouter(edges, links, self.topics, explore=self.args.explore)

    def _observe(self, observations) -> None:
        """Persist outcome observations — unless this is an evaluation session.

        A study participant is not a supervisor. Writing OUTCOME rows from an
        evaluation would train the graph on the very session being used to
        measure it, so the reported result would partly reflect learning that
        happened during measurement. --eval-writes opts in deliberately; the
        default keeps evaluation read-only.
        """
        if self.eval_mode and not self.args.eval_writes:
            return
        apply(observations, self.repo)
        for o in observations:
            self._last_outcomes.append(o)

    # ── The pause ─────────────────────────────────────────────────────────────

    def _ask(self, question: dict, router: KGRouter) -> dict:
        """Show the decision, take the operator's answer, write the result."""
        topic_id = router.resolve_topic(question["text"])
        decision = router.decide(question["text"], self.projects)
        ranked = (rank_robots(self._edges, self._links, topic_id, self.projects)
                  if topic_id else [])
        obs_by_robot = {}
        for r in self.projects:
            e = next((x for x in self._edges
                      if x.robot_id == r and x.topic_id == topic_id), None)
            obs_by_robot[r] = e.n_obs if e else 0

        if decision is not None:
            system_pick, mechanism = decision.robot_id, (
                "kg_explore" if decision.reason.startswith("explore") else "kg_argmax")
        else:
            # Fail-open: unresolved or ambiguous. Whoever was asked answers.
            system_pick, mechanism = self.rng.choice(self.projects), "fallback_receiver"

        print()
        print(f'  Visitor: "{question["text"]}"')
        if not self.eval_mode:
            if topic_id:
                print(f"    topic     {self.label_of[topic_id]}")
            else:
                why = "ambiguous — names two subjects" if question["ambiguous"] else "no match"
                print(f"    topic     UNRESOLVED ({why}) — no observation will be written")
            print(f"    mechanism {mechanism}")
            if ranked:
                print("    candidates:")
                for rid, score in ranked:
                    mark = " <- system pick" if rid == system_pick else ""
                    print(f"      {self.names[rid]:9} {score:.3f}  "
                          f"n_obs={obs_by_robot[rid]}{mark}")
            else:
                print(f"    system pick {self.names[system_pick]} (nothing to rank)")
        else:
            print(f"    {self.names[system_pick]} answers.")

        action, final = self._operator_choice(system_pick)

        wrote = None
        if action == "reroute" and topic_id:
            apply(from_reroute(topic_id, final, system_pick), self.repo)
            self.gw.note_routing_correction()
            wrote = "supervisor+displaced"
        elif action == "accept" and topic_id:
            # Credited at window close, not here — see Segment.
            self.segment.note_routed(final, topic_id)
            wrote = "pending_outcome"

        row = {
            "run_id": self.run_id, "seed": self.args.seed,
            "corpus_version": CORPUS_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": question["text"],
            "intended_topic": question["intended"],
            "resolved_topic": topic_id,
            "from_participant": bool(question.get("from_participant")),
            # A live question has no intended topic, so it cannot be scored.
            # Counting it as a miss would make resolution accuracy fall simply
            # because a participant spoke.
            "resolution_ok": (None if question.get("from_participant")
                              else topic_id == question["intended"]),
            # The failure CLASS, not just pass/fail. Unresolved is the safe
            # failure: no observation is written, data is lost but nothing is
            # polluted. Mis-resolved writes a real observation against the wrong
            # topic, and afterwards it is indistinguishable from a correct one.
            # Collapsing them into one accuracy number hides the only one that
            # does lasting damage.
            "resolution_class": (
                "unscored" if question.get("from_participant")
                else "correct" if topic_id == question["intended"]
                else "unresolved" if question["intended"] and topic_id is None
                else "mis_resolved" if question["intended"] and topic_id
                else "spurious"),
            "ambiguous": question["ambiguous"],
            "mechanism": mechanism,
            "system_pick": system_pick,
            "operator_action": action,
            "final_robot": final,
            "wrote": wrote,
            # The graph as it was WHEN THE DECISION WAS MADE. A correction is
            # only interpretable against what the system knew at the time.
            "candidates": [{"robot_id": r, "score": round(s, 4),
                            "n_obs": obs_by_robot[r]} for r, s in ranked],
        }
        self.decisions.append(row)
        return row

    def _prompt_participant(self):
        """Take a question from the person in front of the robots.

        Returned in the corpus's shape with intended=None, because a live
        question has no ground-truth topic — resolution accuracy is only
        measurable against the fixture, and pretending otherwise would put
        unverifiable rows in the same column as verified ones.
        """
        # The orchestrator's runner thread prints step text from another
        # thread. Settling first, then printing the prompt on its own line and
        # reading with a bare input(), stops the two colliding mid-line — which
        # a participant sees as the robot talking over the question box.
        time.sleep(0.35)
        sys.stdout.write("\n  Your question (press enter to move on)\n  > ")
        sys.stdout.flush()
        try:
            text = input().strip()
        except EOFError:
            return None
        if not text:
            return None
        return {"text": text, "intended": None, "ambiguous": False,
                "from_participant": True}

    def _operator_choice(self, system_pick: str) -> tuple:
        if self.args.auto == "accept" or self.eval_mode:
            return "accept", system_pick
        if self.args.auto == "random":
            if self.rng.random() < 0.25:
                other = [r for r in self.projects if r != system_pick]
                return "reroute", self.rng.choice(other)
            return "accept", system_pick

        options = {str(i + 1): r for i, r in enumerate(self.projects)}
        prompt = ("    [enter] accept  |  "
                  + "  ".join(f"{k}={self.names[v]}" for k, v in options.items())
                  + "  |  q=quit\n    > ")
        while True:
            try:
                choice = input(prompt).strip().lower()
            except EOFError:
                return "accept", system_pick
            if choice in ("", "a"):
                return "accept", system_pick
            if choice == "q":
                raise KeyboardInterrupt
            if choice in options:
                picked = options[choice]
                if picked == system_pick:
                    print("    (that is already the pick — recorded as accept)")
                    return "accept", system_pick
                return "reroute", picked
            print("    ?")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        questions = list(QUESTIONS)
        self.rng.shuffle(questions)
        queue = questions[: self.args.questions]

        print("=" * 72)
        print(f"  Demo harness — run {self.run_id}  seed={self.args.seed}  "
              f"mode={self.args.mode}")
        print(f"  {len(queue)} questions from corpus v{CORPUS_VERSION}, "
              f"explore={'on' if self.args.explore else 'off'}, "
              f"generate={'on' if self.args.generate else 'off'}, "
              f"pace={self.args.pace}")
        if self.eval_mode:
            writes = "ON (confounds the measurement)" if self.args.eval_writes else "off"
            print(f"  evaluation session — internals hidden, graph writes {writes}")
        print("=" * 72)

        self.segment = Segment()
        self._last_outcomes = []
        self.orch.start(robot_ids=[self.guide] + self.projects,
                        time_budget_sec=self.args.budget)

        asked = 0
        deadline = time.time() + 900
        while time.time() < deadline:
            state = self.orch.get_status().get("state")
            if state in ("completed", "idle", "error"):
                break
            if state != "qa_window":
                time.sleep(0.01)
                continue

            self.gw.on_qa_window_open()
            self.segment = self.gw._segment
            per_window = min(self.args.per_window, len(queue) - asked)
            if per_window <= 0:
                self.orch.manual_next(source="harness")
                continue

            router = self._router()
            for _ in range(per_window):
                question = (self._prompt_participant()
                            if self.args.questions_from == "participant"
                            else queue[asked])
                if question is None:        # participant had nothing to ask
                    break
                self._ask(question, router)
                asked += 1
                router = self._router()      # learning is visible within a window

            self._last_outcomes = []
            self.orch.qa_end(source="harness")
            # Wait for the close hook to actually run. qa_end only SETS an event;
            # the orchestrator's runner thread emits the outcome observations,
            # so reporting immediately reads an empty list and claims nothing was
            # credited when three edges were.
            closed = time.time() + 3.0
            while time.time() < closed:
                if self.orch.get_status().get("state") != "qa_window":
                    break
                time.sleep(0.01)
            time.sleep(0.05)
            if self._last_outcomes and not self.eval_mode:
                print(f"    -> segment closed clean: "
                      f"{len(self._last_outcomes)} edge(s) credited (outcome)")
            elif not self.eval_mode:
                print("    -> segment closed: nothing credited "
                      "(corrected, silent, or unresolved)")

        self.orch.stop()
        self._write_log()
        if not self.eval_mode:
            self._report()

    # ── Output ────────────────────────────────────────────────────────────────

    def _write_log(self) -> None:
        os.makedirs(RUNS_DIR, exist_ok=True)
        path = os.path.join(RUNS_DIR, f"{self.run_id}.jsonl")
        after = snapshot_graph()
        with open(path, "w") as fh:
            fh.write(json.dumps({"type": "run", "run_id": self.run_id,
                                 "seed": self.args.seed, "mode": self.args.mode,
                                 "auto": self.args.auto,
                                 "corpus_version": CORPUS_VERSION,
                                 "explore": self.args.explore,
                                 "graph_before": self.before,
                                 "graph_after": after}) + "\n")
            for row in self.decisions:
                fh.write(json.dumps({"type": "decision", **row}) + "\n")
        self.log_path = path

    def _report(self) -> None:
        d = self.decisions
        if not d:
            print("\n  No decisions recorded.")
            return
        corrected = [x for x in d if x["operator_action"] == "reroute"]
        resolved = [x for x in d if x["resolved_topic"]]
        by_mech = Counter(x["mechanism"] for x in d)
        corr_by_mech = Counter(x["mechanism"] for x in corrected)

        print()
        print("=" * 72)
        print(f"  Run {self.run_id} — {len(d)} decisions")
        print("=" * 72)
        print(f"  correction rate         {len(corrected)}/{len(d)} "
              f"= {len(corrected)/len(d):.0%}")
        cls = Counter(x["resolution_class"] for x in d)
        scored = len(d) - cls["unscored"]
        print(f"  topic resolution        {cls['correct']}/{scored} correct  "
              f"| {cls['unresolved']} unresolved (safe)  "
              f"| {cls['mis_resolved']} MIS-RESOLVED  "
              f"| {cls['spurious']} spurious")
        if cls["mis_resolved"] or cls["spurious"]:
            print("      ^ mis-resolved/spurious write observations against the")
            print("        WRONG topic and cannot be distinguished afterwards.")
        print()
        print(f"  {'mechanism':20} {'decisions':>10} {'corrected':>10} {'rate':>7}")
        for m, n in by_mech.most_common():
            print(f"  {m:20} {n:>10} {corr_by_mech[m]:>10} {corr_by_mech[m]/n:>6.0%}")

        s = self.repo.summary()
        print()
        print(f"  graph: {s['observed_edges']}/{s['edges']} edges observed, "
              f"{s['n_supervisor']} supervisor + {s['n_outcome']} outcome, "
              f"human share {s['human_share']:.0%}")
        print(f"  log:   {self.log_path}")


# ── Campaign metrics ──────────────────────────────────────────────────────────

def metrics() -> int:
    """Read every run in runs/ and report the campaign curve."""
    if not os.path.isdir(RUNS_DIR):
        print("No runs yet.")
        return 1
    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.endswith(".jsonl"):
            continue
        head, rows = None, []
        with open(os.path.join(RUNS_DIR, name)) as fh:
            for line in fh:
                obj = json.loads(line)
                (rows if obj["type"] == "decision" else [None]).append(obj) \
                    if obj["type"] == "decision" else None
                if obj["type"] == "run":
                    head = obj
                elif obj["type"] == "decision":
                    pass
        with open(os.path.join(RUNS_DIR, name)) as fh:
            rows = [json.loads(l) for l in fh if json.loads(l)["type"] == "decision"]
        if head and rows:
            runs.append((head, rows))
    if not runs:
        print("No completed runs.")
        return 1

    print("=" * 78)
    print(f"  Campaign — {len(runs)} runs")
    print("=" * 78)
    print(f"  {'run':14} {'seed':>5} {'n':>4} {'corrected':>10} {'rate':>7} "
          f"{'resolution':>11} {'explore rate':>13}")
    for head, rows in runs:
        corrected = [r for r in rows if r["operator_action"] == "reroute"]
        expl = [r for r in rows if r["mechanism"] == "kg_explore"]
        expl_corr = [r for r in expl if r["operator_action"] == "reroute"]
        res_ok = sum(1 for r in rows if r["resolution_ok"])
        print(f"  {head['run_id']:14} {head['seed']:>5} {len(rows):>4} "
              f"{len(corrected):>10} {len(corrected)/len(rows):>6.0%} "
              f"{res_ok/len(rows):>10.0%} "
              f"{(len(expl_corr)/len(expl) if expl else 0):>12.0%}")

    allrows = [r for _h, rs in runs for r in rs]
    by_mech = Counter(r["mechanism"] for r in allrows)
    corr = Counter(r["mechanism"] for r in allrows if r["operator_action"] == "reroute")
    cls = Counter(r.get("resolution_class", "?") for r in allrows)
    auto = any(h.get("auto", "off") != "off" for h, _ in runs)

    print()
    print("  Pooled resolution")
    print(f"    correct {cls['correct']}  |  unresolved {cls['unresolved']} (safe)  "
          f"|  MIS-RESOLVED {cls['mis_resolved']}  |  spurious {cls['spurious']}")

    print()
    print("  Pooled, by mechanism — does exploration pay for itself?")
    for m, n in by_mech.most_common():
        # A rate on a handful of decisions is not a rate. 3/5 has a confidence
        # interval running from roughly 15% to 95%, which is consistent with
        # exploration being catastrophic AND with it being fine.
        flag = "  (n too small to read)" if n < 30 else ""
        print(f"    {m:20} {n:>5} decisions  {corr[m]:>4} corrected  "
              f"{corr[m]/n:>5.0%}{flag}")

    print()
    print("  NOTE ON WHAT THESE NUMBERS MEAN")
    print("    Correction rate measures the OPERATOR, not routing quality. With")
    print("    a stub operator it measures the stub: a uniform-random stub")
    print("    should show the same rate for every mechanism, and a gap between")
    print("    them means either small-n noise or coupling in the stub itself.")
    print("    Only a human operator makes these comparable across mechanisms.")
    if auto:
        print("    >> This campaign contains STUB-OPERATOR runs. Do not report.")
    return 0


def _explicit(flag: str) -> bool:
    """Did the caller actually pass this flag? Mode defaults must not silently
    overwrite something a researcher set on purpose."""
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def main() -> int:
    ap = argparse.ArgumentParser(description="Lab demo harness.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--questions", type=int, default=8, help="questions this run")
    ap.add_argument("--per-window", type=int, default=2)
    ap.add_argument("--budget", type=float, default=None, help="tour budget, seconds")
    ap.add_argument("--mode", choices=["collect", "eval"], default="collect")
    ap.add_argument("--auto", choices=["off", "accept", "random"], default="off",
                    help="unattended operator, for smoke runs")
    ap.add_argument("--generate", action="store_true",
                    help="real LLM speech per step (slow; for a dry run)")
    ap.add_argument("--no-explore", dest="explore", action="store_false",
                    help="argmax only — a live demo must not explore")
    ap.add_argument("--questions-from", choices=["corpus", "participant"],
                    default="corpus", dest="questions_from",
                    help="corpus = the fixed fixture (comparable across runs); "
                         "participant = typed live (an actual study session)")
    ap.add_argument("--pace", choices=["instant", "realistic"], default="instant",
                    help="realistic = steps take as long as the speech would")
    ap.add_argument("--eval-writes", action="store_true",
                    help="let an evaluation session write to the graph "
                         "(off by default — it confounds the measurement)")
    ap.add_argument("--quiet", action="store_true", help="hide robot speech")
    ap.add_argument("--reset", action="store_true", help="clear learned edges and exit")
    ap.add_argument("--metrics", action="store_true", help="report the campaign and exit")
    ap.set_defaults(explore=True)
    args = ap.parse_args()

    # ── Mode contract ────────────────────────────────────────────────────────
    # Evaluation is a different EXPERIMENT, not a different view, so its
    # defaults differ in four ways and each is load-bearing:
    #
    #   generation on    the speech is what is being evaluated
    #   realistic pace   steps flashing past is not a demo, and a participant
    #                    cannot rate pacing that does not exist
    #   explore off      a participant must not be given a deliberately
    #                    uncertain robot to satisfy the training loop
    #   graph read-only  training on the session that measures you confounds
    #                    the result
    #
    # Each stays overridable, but you have to say so.
    if args.mode == "eval":
        if not _explicit("--generate"):
            args.generate = True
        if not _explicit("--pace"):
            args.pace = "realistic"
        if not _explicit("--no-explore"):
            args.explore = False
        if args.auto == "off":
            args.auto = "accept"      # no operator sits in an evaluation
        if not _explicit("--questions-from"):
            # A participant who only watches is not in a demo either. They are
            # the visitor, so they ask. --questions-from corpus forces the fixed
            # set instead, for a controlled condition where every participant
            # must hear the same tour.
            args.questions_from = "participant"

    if args.reset:
        print(f"Cleared {reset_graph()} learned edge(s). Vocabulary untouched.")
        return 0
    if args.metrics:
        return metrics()
    try:
        Harness(args).run()
    except KeyboardInterrupt:
        print("\n  Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
