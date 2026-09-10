"""
gateway/websocket_gateway.py
=============================
The server-side WebSocket client pool.

KEY DESIGN: The SERVER initiates connections TO robots (not the other way around).
Each robot runs a small WebSocket server on a known IP + port stored in Supabase.
This gateway dials out to them and keeps the connections alive.

Responsibilities:
  - Connect to a robot given its (ip, port) from the DB
  - Receive messages from robots (image frames, speech audio, chat text)
  - Push responses back to robots (chat_response, commands)
  - Reconnect automatically if a connection drops

Q&A decisions
-------------
During a Q&A window this gateway used to decide inline whether the window should
close and who should answer, via a chain of phrase lists, a prefix heuristic and
two LLM calls. That chain now lives in decision/policy.py::HeuristicPolicy, and
this module asks it instead:

    build_observation(...) -> policy.decide(point, obs) -> record -> execute

Behaviour is unchanged — the same rules in the same precedence order. What is new
is that each decision, and the rule that made it, is recorded. See decision/.

Requires: pip install websocket-client
"""

from __future__ import annotations
import json
import logging
import re
import threading
import time
from typing import Optional, Callable, TYPE_CHECKING

import websocket   # websocket-client library

from decision import (
    ActionKind,
    DecisionPoint,
    DecisionRecorder,
    DemoRunTracker,
    HeuristicPolicy,
    Mechanism,
    QA_ADVANCE_PHRASES,
    QA_CLOSING_PHRASES,
    build_decision,
    build_observation,
    guide_and_presenter,
    is_acknowledgement,
    looks_like_question,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from robot.robot_registry import RobotRegistry


# Kept as a module-level alias: the implementation moved to
# decision.observation.looks_like_question so the policy and the gateway agree
# on what counts as a question.
_looks_like_question = looks_like_question


# How long to wait before attempting a reconnect (seconds)
RECONNECT_DELAY = 5
MAX_RECONNECT_ATTEMPTS = 10

ADVANCE_ACK = {
    # What the guide says when a visitor's turn closes the Q&A window. Fixed
    # rather than generated: there is nothing to answer, and the mechanism
    # that closed the window is the only thing that varies.
    Mechanism.TIME_PRESSURE:
        "[DEFAULT] Of course — we'll keep it brief and move along.",
    Mechanism.LLM_CLASSIFIER:
        "[DEFAULT] Great! Let's continue with the demonstration then!",
    "default":
        "[DEFAULT] Of course, let's move on.",
}

# Used INSTEAD of the above when PLAN_REVISE changed the tour on the same
# turn, because the change is the substance and the pleasantry should not
# crowd it: "Of course — we'll keep it brief and move along — we'll skip
# Silbot and carry on from there" says the same thing twice before getting to
# the point.
ADVANCE_ACK_WITH_CHANGE = {
    Mechanism.TIME_PRESSURE: "[DEFAULT] Understood",
    "default":               "[DEFAULT] Of course",
}

SPEAKING_WORDS_PER_SEC = 2.4
"""Rough speaking rate, used only to wait out a sentence the server cannot
hear the end of. Matches tools/demo_harness.py's SPEAKING_RATE so the offline
harness and the live server model the same robot."""


def _spoken_text(text: str) -> str:
    """What a listener would actually hear: the text minus emotion tags.

    Empty when a chunk is nothing but a tag, which is how a blank turn
    reached the transcript under a robot's name.
    """
    return re.sub(r"\[[A-Z_]+\]", "", text or "").strip()


def _speaking_seconds(text: str) -> float:
    """How long `text` will take to say, near enough to wait for.

    Capped: a long generation should not park the tour indefinitely if the
    estimate is wrong, and 20s is already far beyond any sign-off.
    """
    words = len((text or "").split())
    return min(20.0, words / SPEAKING_WORDS_PER_SEC) if words else 0.0


QA_AUTO_CLOSE_SEC = 3.0
"""Silence after a robot invites further questions before the tour moves on
by itself.

Measured from the end of GENERATION, not the end of speech — the server gets
no end-of-TTS signal for streamed chat sentences — so the visitor's real
silence is this minus however long the sign-off takes to say. Was 5s, which
in a live run left a noticeable dead gap after the robot had audibly
finished."""


class RobotConnection:
    """Manages a single persistent WebSocket connection to one robot."""

    def __init__(
        self,
        client_id: str,
        ip: str,
        port: int,
        on_message: Callable,
        on_close: Callable,
    ):
        self.client_id = client_id
        self.ip = ip
        self.port = port
        self._on_message = on_message
        self._on_close = on_close

        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._stop = False

    @property
    def url(self) -> str:
        return f"ws://{self.ip}:{self.port}"

    def connect(self):
        """Start connection in a background thread."""
        self._stop = False
        self._start_thread()

    def disconnect(self):
        """Close the connection cleanly."""
        self._stop = True
        if self._ws:
            self._ws.close()
        self._connected = False

    def send(self, data: dict):
        """Send a JSON message to the robot."""
        if self._ws and self._connected:
            try:
                self._ws.send(json.dumps(data))
            except Exception as e:
                print(f"[WS] Send error to {self.client_id}: {e}")
        else:
            print(f"[WS] Cannot send to {self.client_id} — not connected.")

    def is_connected(self) -> bool:
        return self._connected

    # ── Internal ──────────────────────────────────────────────────────────────

    def _start_thread(self):
        self._ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._handle_message,
            on_error=self._on_error,
            on_close=self._handle_close,
        )
        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
        )
        self._thread.start()
        print(f"[WS] Connecting to {self.client_id} at {self.url}...")

    def _on_open(self, ws):
        self._connected = True
        self._reconnect_attempts = 0
        print(f"[WS] Connected to {self.client_id}")

    def _handle_message(self, ws, raw):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw}
        self._on_message(self.client_id, data)

    def _on_error(self, ws, error):
        print(f"[WS] Error from {self.client_id}: {error}")

    def _handle_close(self, ws, code, msg):
        self._connected = False
        print(f"[WS] Connection to {self.client_id} closed (code={code})")
        self._on_close(self.client_id)

        # Auto-reconnect unless we're stopping deliberately
        if not self._stop and self._reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            print(f"[WS] Reconnecting to {self.client_id} in {RECONNECT_DELAY}s "
                  f"(attempt {self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})...")
            time.sleep(RECONNECT_DELAY)
            self._start_thread()
        elif self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print(f"[WS] Giving up on {self.client_id} after "
                  f"{MAX_RECONNECT_ATTEMPTS} attempts.")


class WebSocketGateway:
    """
    Pool of RobotConnection objects.
    The HTTP gateway and delegation handler call send_to_robot().
    The registry calls connect_robot() / disconnect_robot().
    """

    def __init__(self, registry: "RobotRegistry", recorder=None, policy=None,
                 kg_router_factory=None, kg_observer=None, flow_planner=None):
        self._registry = registry
        self._connections: dict[str, RobotConnection] = {}
        self._lock = threading.Lock()
        self._demo_orchestrator = None   # set via set_demo_orchestrator()

        # ── Decision layer ────────────────────────────────────────────────────
        # Both are optional and default to something inert-but-working: without a
        # recorder decisions are made and discarded, which is exactly the old
        # behaviour. The policy defaults to the heuristic baseline, with the two
        # LLM calls injected as callables so decision/ never imports robot/.
        self._recorder = recorder if recorder is not None else DecisionRecorder()
        self._tracker = DemoRunTracker()
        self._policy = policy if policy is not None else HeuristicPolicy(
            intent_classifier=self._classify_intent,
            wrap_up_judge=self._judge_wrap_up,
            flow_planner=flow_planner,
        )
        # Optional KG-backed routing. A factory rather than a router, because the
        # graph changes as corrections land and a snapshot captured at boot would
        # go stale within one demo. None = keep the baseline (whoever heard the
        # question answers), which is what runs when the graph is unseeded.
        self._kg_router_factory = kg_router_factory

        # Where outcome observations go when a Q&A window closes cleanly.
        # A callable taking a list of kg_feedback.Observation, so the gateway
        # never imports data/ and a rollout harness can pass an in-memory store.
        self._kg_observer = kg_observer
        from decision.kg_feedback import Segment
        self._segment = Segment()

        # Where each robot is, and who is close enough to take a question.
        # Owned here rather than passed in because it is live runtime state a
        # ROS2 subscriber will write into continuously — see
        # decision/presence.py for why it is pose-shaped and not persisted.
        from decision.presence import PresenceTracker
        self._presence = PresenceTracker()

        # What the robots have actually been told to say, newest last. The
        # dashboard builds its transcript from /demo/status, which only knows
        # the CURRENT SCRIPTED STEP — so every Q&A exchange driven by speech
        # was invisible to the operator. During a voice-led tour that is the
        # entire conversation. Recorded here because send_to_robot is the one
        # choke point every utterance passes through.
        #
        # Bounded: a tour is minutes long and this is a debugging surface, not
        # storage. The decision log is the durable record.
        from collections import deque
        self._utterances = deque(maxlen=400)
        self._utterance_seq = 0
        self._utterance_lock = threading.Lock()

        # Who the last hand-off named, so the same line is not repeated in
        # front of every answer from the same robot — see route_question.
        self._handed_off_to = None

        # When the robots are expected to fall silent, as a wall clock. The
        # server never hears the end of a sentence: TTS happens on the robot
        # and only ACK-bearing demo steps report back. So every timer that
        # said "wait for the robot to finish" was really counting from the
        # moment GENERATION finished, with the whole utterance still ahead of
        # it. This is the one place that models what is still being said.
        self._speech_lock = threading.Lock()
        self._quiet_at = 0.0
        # Whether this window has already invited further questions.
        self._asked_more_questions = False

        # Pending auto-close of the Q&A window after a robot signs off.
        # Cancelled the moment a visitor speaks — see cancel_qa_auto_close.
        self._auto_close_timer = None
        self._auto_close_lock = threading.Lock()

        # Per-decision scratch space. Thread-local because every robot's
        # connection dispatches messages on its own reader thread, so two
        # visitors talking to two robots at once would otherwise interleave
        # here — one robot's LLM classifier answering for another's question.
        self._scratch = threading.local()

    # The two phrase lists now live in decision/policy.py, which owns the Q&A
    # rules. These aliases stay because they are the baseline being measured:
    # a phrase list edited in one place and read in another is how the five
    # mechanisms drifted apart to begin with.
    _QA_CLOSING_PHRASES = QA_CLOSING_PHRASES
    _QA_ADVANCE_PHRASES = QA_ADVANCE_PHRASES

    def set_demo_orchestrator(self, orchestrator):
        """Wire up the DemoOrchestrator so ACK packets are forwarded to it."""
        self._demo_orchestrator = orchestrator

    # ── Decision layer ────────────────────────────────────────────────────────

    @property
    def recorder(self) -> DecisionRecorder:
        return self._recorder

    @property
    def tracker(self) -> DemoRunTracker:
        return self._tracker

    def _record_utterance(self, client_id: str, data: dict) -> None:
        """Note something a robot was told to say, for the operator's view.

        Scripted demo steps are skipped — the dashboard already shows those
        from /demo/status, and recording them here would double every line.
        What it captures is exactly what was missing: streamed Q&A sentences
        and the synthetic `_qa_*` steps the policy generates.
        """
        event = data.get("event")
        # Judge emptiness on what a listener would HEAR: "[DEFAULT]" is a
        # directive, not a line, and recording it printed a blank turn.
        text = _spoken_text(data.get("text") or "")
        if not text:
            return
        step_id = data.get("step_id") or ""
        if event == "demo_step" and not step_id.startswith("_qa"):
            return
        if event not in ("demo_step", "chat_sentence"):
            return
        inst = self._registry.get(client_id)
        with self._utterance_lock:
            self._utterance_seq += 1
            self._utterances.append({
                "seq": self._utterance_seq,
                "robot_id": client_id,
                "robot_name": getattr(inst, "robot_name", None) or client_id,
                # Strip the emotion tag the robot reads as a directive, not
                # as speech — the dashboard shows what a visitor would hear.
                "text": text,
                "kind": step_id or event,
                "at": time.time(),
            })

    def utterances_since(self, seq: int = 0) -> dict:
        """Everything said after `seq`. The dashboard polls with its last seq."""
        with self._utterance_lock:
            rows = [u for u in self._utterances if u["seq"] > seq]
            latest = self._utterance_seq
        return {"utterances": rows, "seq": latest}

    @property
    def presence(self):
        """Robot locations and derived in_conversation. See decision/presence.py."""
        return self._presence

    def session_context(self) -> dict:
        """
        Identifiers for whatever the orchestrator is about to log.

        Taken from the guide robot's RBAC identity, because the guide is present
        for the whole run while project robots come and go. Using the same
        scenario_id / session_id as rbac_audit_log is what makes the two tables
        joinable — see data/migrations/004_demo_decisions.sql.
        """
        try:
            status = self._demo_orchestrator.get_status() if self._demo_orchestrator else {}
            steps = status.get("steps") or []
            guide_id = steps[0].get("robot_id") if steps else None
            instance = self._registry.get(guide_id) if guide_id else None
            if instance is None:
                return {}
            identity = instance.identity
            return {
                "scenario_id": identity.scenario_id,
                "session_id": identity.session_id,
            }
        except Exception as e:
            logger.warning(f"[WS Gateway] session_context failed: {e}")
            return {}

    def on_qa_window_open(self) -> None:
        """Called by DemoOrchestrator when a Q&A window opens."""
        self._tracker.open_window()
        self._segment.reset()
        self._asked_more_questions = False
        # A timer left over from the previous window would close this one
        # almost as soon as it opened.
        self.cancel_qa_auto_close("new window opened")

    def on_qa_window_close(self) -> None:
        """Called by DemoOrchestrator when a Q&A window closes.

        This is where the graph hears about routing that went RIGHT. Without it
        the only thing ever written is corrections, so every edge is built from
        failures and human_share is trivially 100% — a decomposition that says
        nothing.

        Nothing is emitted for a silent window. See kg_feedback.Segment: the
        rule is structural there, not a condition here, so a future caller
        cannot reintroduce hollow observations by taking a different path.
        """
        self._tracker.close_window()
        self._asked_more_questions = False
        self.cancel_qa_auto_close("window closed")
        try:
            observations = self._segment.observations()
            if observations and self._kg_observer is not None:
                self._kg_observer(observations)
                logger.info(f"[KG outcome] {len(observations)} edge(s) credited "
                            f"for an uncorrected segment")
        except Exception as e:
            logger.warning(f"[WS Gateway] outcome emission failed: {e}")
        finally:
            self._segment.reset()

    def note_routing_correction(self) -> None:
        """An operator overrode routing during this segment.

        Suppresses the segment's outcome observations — the correction already
        describes the event, and recording both would count one thing twice,
        inflating n_obs and with it the confidence the clamp grants.
        """
        self._segment.note_correction()

    def _decide(self, point: DecisionPoint, decider, user_utterance: str = ""):
        """
        Ask the policy, record the answer, hand back the PolicyResult.

        Every Q&A decision goes through here so that "what the system chose" and
        "what got logged" cannot drift apart — the failure mode this whole layer
        exists to prevent. A recording failure is swallowed: the result is still
        returned and the demo still runs.

        Callers get the whole result, not just the action, because the mechanism
        changes what happens next: the original code spoke a canned line when the
        LLM classifier closed a window but stayed silent when a phrase match did.
        """
        status = self._demo_orchestrator.get_status() if self._demo_orchestrator else {}
        obs = build_observation(
            status=status,
            registry=self._registry,
            tracker=self._tracker,
            decider=decider,
            user_utterance=user_utterance,
        )
        # The injected callables receive only what the Policy protocol passes
        # them, so the robot they should speak as travels out of band.
        self._scratch.decider_id = getattr(decider, "client_id", None)
        self._scratch.wrap_up_text = None
        self._scratch.plan_change = None
        # Cleared per decision, not just set on the defer path: _scratch is
        # thread-local and a Flask worker handles many turns, so a defer on
        # one turn would otherwise still be readable on the next one.
        self._scratch.deferred_to = None
        result = self._policy.decide(point, obs)

        # QA_ROUTE only: let the competence graph override the baseline, which
        # routes to whoever heard the question. Everything else stays with
        # HeuristicPolicy — QA_ADVANCE and PLAN_REVISE are contextual decisions
        # about timing, not questions about who knows what.
        if point is DecisionPoint.QA_ROUTE and self._kg_router_factory is not None:
            kg = self._kg_route(obs)
            if kg is not None:
                result = kg

        try:
            self._recorder.record(build_decision(
                point=point,
                action=result.action,
                mechanism=result.mechanism,
                observation=obs,
            ))
        except Exception as e:
            logger.warning(f"[WS Gateway] could not record decision: {e}")
        logger.info(
            f"[Decision] {point.value} → {result.action.describe()} "
            f"({result.mechanism})"
        )
        return result

    def _kg_route(self, obs):
        """Ask the competence graph who should answer. None = no opinion.

        Wrapped in a blanket except on purpose: this sits on the path a visitor's
        question takes, and a graph that is unseeded, unreachable or malformed
        must degrade to the baseline rather than drop the question.
        """
        from decision.policy import PolicyResult
        try:
            router = self._kg_router_factory()
            if router is None:
                return None
            peers = [p["client_id"] for p in obs.connected_peers
                     if p.get("client_id") and p["client_id"] != obs.guide_robot_id]
            # Blocks still ahead of the play head — what a defer is allowed to
            # promise. See KGRouter._handle_absent.
            remaining_blocks = {s.block_robot_id for s in obs.remaining_steps
                                if s.block_robot_id}
            # Who is too far from the group to take a question. The group's
            # position is the presenting robot's — see PresenceTracker.
            # Empty with no poses and no overrides, which is the fail-open
            # default: presence never removes a candidate until something
            # actually says where the robots are.
            absent = self._presence.absent(peers, obs.presenting_robot_id)
            decision = router.decide(
                obs.user_utterance, peers,
                remaining_block_ids=remaining_blocks,
                guide_robot_id=obs.guide_robot_id,
                absent_robot_ids=absent,
                # Whoever is presenting owns the subject on the floor, so an
                # ambiguous follow-up stays with them rather than being
                # carried off by one coincidental word.
                context_robot_id=obs.presenting_robot_id,
            )
            from decision.models import Action

            if decision is None:
                # No topic resolved. Inside a robot's own block the question
                # is almost certainly about THAT robot, so it answers rather
                # than whoever's microphone caught it — which during a Q&A
                # window is usually the guide, and the guide answering for a
                # specialist is how a live run got "Silbot specializes in
                # machine learning techniques", invented on the spot.
                #
                # Deliberately no observation: nothing resolved, so there is
                # no edge this turn belongs to. Recording it would be filing
                # evidence against a guessed topic, which Segment.note_routed
                # refuses for the same reason.
                # ONLY for something that is actually a question. "okay",
                # "okay thank you" and "wait" all resolve to no topic too, and
                # rerouting those produced a handoff line on every single turn
                # of a live run — "Silbot can tell you more about that" in
                # answer to "thank you". An acknowledgement is not a question
                # looking for an owner.
                presenter = obs.presenting_robot_id
                if (presenter and presenter != obs.decider_robot_id
                        and looks_like_question(obs.user_utterance)
                        and self._registry.get(presenter) is not None):
                    logger.info(f"[KG route] '{obs.user_utterance[:40]}' -> no topic "
                                f"-> presenter {presenter} answers")
                    return PolicyResult(Action.route_to(presenter),
                                        "presenter_fallback")
                return None
            logger.info(
                f"[KG route] '{obs.user_utterance[:40]}' -> {decision.topic_label} "
                f"-> {decision.robot_id} ({decision.reason}, {decision.score})")
            # The mechanism records WHICH rule fired, so an exploration pick is
            # distinguishable from a confident one in the correction-rate view.
            # The two presence outcomes are named separately for the same
            # reason: "the graph picked this robot" and "the graph's pick had
            # walked away" are different events and must not be averaged.
            if decision.is_deferred:
                mechanism = "kg_defer"
            elif decision.reason.startswith("guide answers"):
                mechanism = "kg_guide_answers"
            elif decision.reason.startswith("explore"):
                mechanism = "kg_explore"
            else:
                mechanism = "kg_argmax"

            if decision.is_deferred:
                # NOTHING is recorded. No robot answered, and the absent one
                # was never judged — writing an observation here would credit
                # or blame an edge for a turn nobody took. Same discipline as
                # the unresolved-topic path in Segment.note_routed, and the
                # silence rule it exists to protect (decision/kg_feedback.py).
                #
                # Which robot is being deferred TO travels out of band, the
                # same way decider_id and wrap_up_text do: Action carries a
                # robot_id for who acts, and here that is the guide, not the
                # robot the visitor is being promised.
                self._scratch.deferred_to = decision.deferred_to
                return PolicyResult(Action.guide_interject(obs.guide_robot_id),
                                    mechanism)

            # Only a question that actually resolved to a topic is recorded, so
            # the segment can never credit an edge for a turn it did not handle.
            #
            # had_alternatives is what stops declared scope laundering itself
            # into learned competence: when scope narrowed the field to one
            # specialist there was no choice to make, and a clean segment
            # there says only that scope fired. See Segment.note_routed.
            self._segment.note_routed(
                decision.robot_id, decision.topic_id,
                had_alternatives=decision.candidates_considered > 1,
            )
            return PolicyResult(Action.route_to(decision.robot_id), mechanism)
        except Exception as e:
            logger.warning(f"[WS Gateway] KG routing failed, using receiver: {e}")
            return None

    def _classify_intent(self, message: str) -> str:
        """
        HeuristicPolicy's LLM classifier, bound to the robot that is speaking.

        Injected rather than imported — decision/ must not reach into robot/.
        Falls back to 'continue', matching classify_qa_intent's own safe default,
        so a missing instance never skips a real question.
        """
        decider_id = getattr(self._scratch, "decider_id", None)
        instance = self._registry.get(decider_id) if decider_id else None
        if instance is None or not hasattr(instance, "classify_qa_intent"):
            return "continue"
        return instance.classify_qa_intent(message)

    def _judge_wrap_up(self, obs) -> bool:
        """
        HeuristicPolicy's LLM moderator: does the guide think this is a natural
        wrap-up point?

        Also stashes the transition sentence the guide generated, so that if the
        policy returns GUIDE_INTERJECT the caller can speak the exact text that
        justified the decision rather than generating a second one.
        """
        self._scratch.wrap_up_text = None
        guide_id = obs.guide_robot_id
        if not guide_id:
            return False
        guide = self._registry.get(guide_id)
        if not guide or not hasattr(guide, "process_chat"):
            return False

        prompt = (
            f"[Demo moderator context — Q&A step: {obs.step_id}] "
            f"A research robot just responded: \"{obs.last_robot_utterance[:200]}\". "
            f"As the demo moderator, decide: is this a natural wrap-up point where visitors "
            f"seem satisfied and we could transition to the next part of the demo? "
            f"If YES — write a single warm 1-sentence transition (e.g. 'Wonderful! "
            f"Shall we move on to the next part?'). "
            f"If NO — respond with exactly: NO"
        )
        result = guide.process_chat(prompt)
        reply = (result.clean_text or "").strip()
        if reply and reply.upper() != "NO" and len(reply) > 5:
            self._scratch.wrap_up_text = result.response
            return True
        return False

    def generate_demo_step(self, robot_id: str, instruction: str) -> str:
        """
        Generate speech text for a demo step server-side using the robot's
        LLM instance via generate_demo_speech() (demo-appropriate prompt,
        no delegation logic, correct length handling).
        Returns raw response (includes emotion tag) on success,
        or the original instruction as fallback.
        """
        instance = self._registry.get(robot_id)
        if not instance:
            logger.warning(f"[WS Gateway] generate_demo_step: no instance for '{robot_id}' "
                           f"— connected ids: {list(self._connections.keys())}")
            return instruction

        # Replace client_ids with robot names so the LLM speaks proper names
        for peer in self._registry.get_all():
            if peer.client_id and peer.robot_name and peer.client_id != peer.robot_name:
                instruction = instruction.replace(peer.client_id, peer.robot_name)

        logger.info(f"[WS Gateway] Generating demo speech for '{robot_id}'...")
        try:
            result = instance.generate_demo_speech(instruction)
            generated = result.response or instruction
            logger.info(f"[WS Gateway] Generated ({robot_id}): {generated[:100]}"
                        f"{'...' if len(generated) > 100 else ''}")
            return generated
        except Exception as e:
            logger.error(f"[WS Gateway] generate_demo_step failed for '{robot_id}': {e}",
                         exc_info=True)
            return instruction

    MAX_QUIET_WAIT_SEC = 25.0
    """Ceiling on waiting for the robots to stop talking.

    The estimate can be wrong — a robot may be muted, disconnected, or
    playing nothing at all — and a tour that parks itself forever on a bad
    guess is worse than one that occasionally speaks a little early."""

    def note_speech(self, client_id: str, data: dict) -> None:
        """Add whatever was just sent to the estimate of what is still being said.

        Modelled as a queue, not a single utterance: the answer to one
        question arrives as several `chat_sentence` events in quick
        succession and the robot speaks them one after another, so their
        durations add. Taking the max instead would have said a four-sentence
        answer lasts as long as its longest sentence.

        Best-effort by design — an over- or under-estimate costs a second of
        pacing, never correctness, and every caller caps how long it will
        act on this.
        """
        event = data.get("event")
        if event == "tts_stop":
            # Speech was cut off mid-utterance; whatever was queued is gone.
            with self._speech_lock:
                self._quiet_at = 0.0
            return
        if event == "chat_sentence":
            text = data.get("text", "")
        elif event == "demo_step" and not data.get("require_ack"):
            # An ACK-bearing step is already waited for properly — the robot
            # reports back when it has finished speaking it. Counting those
            # here would charge for the same speech twice.
            text = data.get("text", "")
        else:
            return
        seconds = _speaking_seconds(_spoken_text(text))
        if seconds <= 0:
            return
        with self._speech_lock:
            self._quiet_at = max(self._quiet_at, time.time()) + seconds

    def seconds_until_quiet(self) -> float:
        """How much longer the robots are expected to keep talking, in seconds.

        0.0 when nothing is in flight, which is the common case and makes
        this safe to add to any delay unconditionally.
        """
        with self._speech_lock:
            remaining = self._quiet_at - time.time()
        return max(0.0, min(self.MAX_QUIET_WAIT_SEC, remaining))

    def cancel_qa_auto_close(self, why: str = "") -> None:
        """Stop a pending auto-close. Idempotent."""
        with self._auto_close_lock:
            timer, self._auto_close_timer = self._auto_close_timer, None
        if timer is not None:
            timer.cancel()
            logger.info(f"[WS Gateway] Auto-close cancelled{(' — ' + why) if why else ''}.")

    def _schedule_qa_auto_close(self, delay: float) -> None:
        """
        Close the Q&A window in `delay` seconds unless something cancels it.

        A timer rather than an immediate close because the robot has just
        invited more questions — "let me know if you have any other
        questions" — and closing on that instant would cut off the visitor
        who was drawing breath to take it up. The delay is the pause a person
        leaves after asking; if nobody fills it, the tour moves on by itself
        instead of waiting for someone to say "move on" out loud.

        Note the delay starts when GENERATION finishes, not when the robot
        stops speaking — the server has no end-of-TTS signal for streamed
        chat sentences, only for demo steps that require an ACK. So the
        visitor's real silence is this minus however long the closing
        sentence takes to say. QA_AUTO_CLOSE_SEC is set with that in mind.
        """
        def fire():
            orch = self._demo_orchestrator
            if orch is None or orch.get_status().get("state") != "qa_window":
                return          # someone else already closed it
            logger.info(f"[WS Gateway] Closing Q&A — robot invited further "
                        f"questions and {delay:.0f}s passed in silence.")
            orch.qa_end(source="policy")

        self._schedule_qa_action(delay, fire)

    def _schedule_qa_action(self, delay: float, fn) -> None:
        """Run `fn` in `delay` seconds unless the window changes first.

        One timer slot, shared by every deferred reaction to a Q&A turn, so
        cancel_qa_auto_close cancels whichever is pending — a visitor speaking
        must call off a queued auto-close and a queued wrap-up alike. Two
        independent timers would have let a stale wrap-up fire on top of the
        answer to the question that made it stale.
        """
        self.cancel_qa_auto_close()

        def run():
            with self._auto_close_lock:
                self._auto_close_timer = None
            fn()

        timer = threading.Timer(delay, run)
        timer.daemon = True
        with self._auto_close_lock:
            self._auto_close_timer = timer
        timer.start()

    def claim_more_questions_prompt(self) -> bool:
        """True the FIRST time this Q&A window asks "any other questions?".

        Reset when a window opens or closes. The prompt exists so a visitor
        knows they can keep going; saying it after every answer tells them
        nothing new and starts to read as impatience.
        """
        with self._utterance_lock:
            if self._asked_more_questions:
                return False
            self._asked_more_questions = True
            return True

    def _presenting_robot_id(self):
        """Whose block the tour is currently in, or None outside a demo."""
        orch = self._demo_orchestrator
        if orch is None:
            return None
        try:
            return guide_and_presenter(orch.get_status())[1]
        except Exception:
            return None

    def style_framing_for(self, robot_id: str) -> str:
        """The visitor's style directive for this robot, or "" — best-effort.

        Q&A answers used to be the one generated text that ignored the visitor
        profile entirely. See RobotInstance.process_chat_stream.
        """
        orch = self._demo_orchestrator
        if orch is None:
            return ""
        try:
            return orch.framing_for_robot(robot_id) or ""
        except Exception as e:
            logger.warning(f"[WS Gateway] style framing lookup failed: {e}")
            return ""

    def grounding_for(self, robot_id: str, utterance: str) -> list:
        """Verified detail this robot may state about what was just asked.

        Resolves the utterance to a topic with the same resolver routing
        used, so the facts a robot is given are about the subject it was
        actually routed on. Empty on any failure, or when nothing has been
        recorded for the topic — which puts the robot back where it was
        before the facts table existed: able to talk generally, and
        instructed to decline specifics rather than invent them.
        """
        try:
            router = self._kg_router_factory() if self._kg_router_factory else None
            if router is None:
                return []
            from data import demo_facts_repo
            from decision.grounding import format_facts
            topic_id = router.resolve_topic(utterance)
            if not topic_id:
                # No topic, so no targeted facts — but handing the model
                # nothing is the condition under which it invents. A question
                # asked during a robot's own block is almost certainly about
                # its work, so ground on everything that robot may state.
                return format_facts(demo_facts_repo.facts_for_robot(robot_id))
            return format_facts(demo_facts_repo.facts_for(topic_id, robot_id))
        except Exception as e:
            logger.warning(f"[WS Gateway] grounding lookup failed: {e}")
            return []

    def check_qa_auto_close(self, responding_robot_id: str, clean_text: str):
        """
        Decide, after a robot responds, whether the Q&A window should close.

        Returns what it set in motion — "auto_close", "wrap_up", or None —
        because the caller has its own follow-up to suppress. A live run had
        the guide say "Wonderful! Shall we move on to the next part of the
        demo?" and the answering robot say "Do you have any other questions,
        or shall we continue the demonstration?" in the same second, in two
        voices, contradicting each other about what was being asked. Both
        paths are reasonable on their own; what was missing is that they are
        alternatives.

        WIRED as of the closing-phrase auto-advance change. It was dormant for
        a long time — the closing-phrase list and the guide's LLM wrap-up
        judgement were both written and never called, so of the five Q&A
        mechanisms only three ever ran. Turning it on was a deliberate
        behaviour change, not a refactor: Q&A windows now close on their own
        when a robot signs off, where before the visitor had to say "move on"
        even after the robot had audibly finished.

        An ADVANCE here SCHEDULES the close rather than doing it — see
        _schedule_qa_auto_close for why the pause matters.

        The guide never judges its own responses — the recursion guard the
        original had, preserved.
        """
        if not self._demo_orchestrator or not clean_text:
            return None
        status = self._demo_orchestrator.get_status()
        if status.get("state") != "qa_window":
            return None

        self._tracker.note_robot_turn(responding_robot_id, clean_text)

        # The GUIDE, not status["robot_id"] — that field is the current STEP's
        # robot, which during a project block IS the presenting robot. Using it
        # here meant the guard fired on exactly the robot most likely to sign
        # off: a live run had Silbot say "If you have more questions, feel free
        # to ask" and the window stayed open, because Silbot happened to own
        # the step. guide_and_presenter reads the real host off step 0.
        guide_id, _presenter = guide_and_presenter(status)
        is_guide = bool(guide_id and responding_robot_id == guide_id)

        decider = self._registry.get(responding_robot_id)
        result = self._decide(DecisionPoint.QA_ADVANCE, decider)
        action = result.action

        # The recursion guard applies to the LLM MODERATOR only. That path
        # asks the guide to judge, so letting it judge its own words could
        # loop. The closing-phrase path cannot loop — it is a string match —
        # and blocking it here meant the guide could say "Let's continue with
        # our tour" and nothing would close, which is exactly the robot most
        # likely to say it.
        if is_guide and result.mechanism != Mechanism.CLOSING_PHRASE:
            return None

        if action.kind is ActionKind.ADVANCE:
            # The clock starts when GENERATION finishes, not when the robot
            # stops talking — the server gets no end-of-TTS signal for
            # streamed sentences. So the visitor's real silence is this minus
            # however long the sign-off takes to say, and a robot that ends
            # by ASKING something ("Is there anything else you'd like to
            # know?") was being answered by the tour moving on before the
            # question had finished playing. Reported exactly that way.
            #
            # Estimating the speaking time and waiting for it first makes the
            # pause mean what it says: three seconds of silence AFTER the
            # robot finishes, not three seconds from the middle of a sentence.
            self._schedule_qa_auto_close(
                QA_AUTO_CLOSE_SEC + self.seconds_until_quiet())
            return "auto_close"

        if action.kind is ActionKind.GUIDE_INTERJECT:
            # Speak the sentence the guide already generated while judging.
            # Generating a second one would risk it contradicting the first.
            text = getattr(self._scratch, "wrap_up_text", None)
            target = action.robot_id or guide_id
            if not (text and target):
                return None

            # AFTER the answer has been said, not on top of it. The judgement
            # is made when GENERATION finishes, and the answering robot is
            # still working through its sentences at that point — a live run
            # had Pepper's "Shall we move on?" start while Navel was two
            # sentences into explaining the pipeline. Same reasoning as the
            # auto-close delay above; the guide waits out the answer it is
            # reacting to.
            def _say_wrap_up():
                orch = self._demo_orchestrator
                if orch is None or orch.get_status().get("state") != "qa_window":
                    return   # the window closed while we waited — nothing to wrap
                self.send_to_robot(target, {
                    "event":       "demo_step",
                    "step_id":     "_qa_wrap_up",
                    "text":        text,
                    "require_ack": False,
                })

            self._schedule_qa_action(self.seconds_until_quiet(), _say_wrap_up)
            return "wrap_up"

        return None

    def check_qa_advance_from_user(self, decider, user_text: str):
        """
        Decide what a visitor turn means during a Q&A window.

        Replaces the inline cascade that used to sit in _on_message: advance
        phrase, then question heuristic, then the LLM classifier. Same order,
        same outcomes — see decision/policy.py::HeuristicPolicy._decide_advance.

        Returns the PolicyResult, or None when no decision was due. The caller
        needs the mechanism, not just the action: a window closed by the LLM
        classifier ends the turn with a canned line, while one closed by a phrase
        match still lets the robot answer what was said.

        Recording the turn is this method's job, not the caller's. HeuristicPolicy
        branches on last_speaker_id to pick between the visitor chain and the
        robot-response chain, so a caller that forgot to update the tracker first
        would get a plausible answer from entirely the wrong set of rules. Three
        call sites had to remember; now none do.
        """
        if not self._demo_orchestrator or not user_text:
            return None

        # The visitor took the robot up on its offer, so the pause that was
        # counting down toward moving on is over. Cancelled before any
        # decision is made: whatever this turn means, it means the window
        # should not close because nobody said anything.
        self.cancel_qa_auto_close("visitor spoke")

        # Engagement is tracked against the BLOCK the visitor is actually
        # asking about, not whichever robot's mic happened to receive the
        # audio. Those two used to always be the same robot, before
        # route_question() could send a turn to a different robot than the
        # receiver — and even without that, the guide's own mic can be what
        # picks up a question about a project it isn't presenting. Getting
        # this wrong silently corrupts _remaining_projects and the interest
        # extension logic downstream, both of which read engagement_by_robot
        # to mean "has this project already drawn visitor interest".
        status = self._demo_orchestrator.get_status()
        _, presenter_id = guide_and_presenter(status)
        self._tracker.note_visitor_turn(
            presenter_id or getattr(decider, "client_id", None) or "unknown", user_text
        )

        if status.get("state") != "qa_window":
            return None

        result = self._decide(DecisionPoint.QA_ADVANCE, decider, user_text)
        if result.action.kind is not ActionKind.ADVANCE:
            return result

        print(f"[WS Gateway] Closing Q&A — advance intent from user: '{user_text[:50]}' "
              f"({result.mechanism})")
        self._demo_orchestrator.qa_end(source="policy")
        return result

    def check_plan_revision(self, decider, user_text: str) -> bool:
        """
        Decide whether a visitor turn should change the rest of the tour.

        New in this layer — "we're running out of time" or "skip that one"
        previously closed one window and left the remaining script untouched.
        Runs after the advance decision so an explicit "move on" is still just
        an advance, not a plan edit.

        Returns True if the script was changed. What CHANGED is stashed for
        the caller to announce — see _scratch.plan_change and
        describe_plan_change. A revision the visitor is not told about is
        indistinguishable from being ignored: a live run had a visitor ask to
        skip because they were short of time, the skip was applied, and all
        they heard was "Of course — we'll keep it brief", which does not say
        that a project had just been dropped from their tour.
        """
        self._scratch.plan_change = None
        if not self._demo_orchestrator or not user_text:
            return False

        action = self._decide(DecisionPoint.PLAN_REVISE, decider, user_text).action
        if action.kind is not ActionKind.REVISE or not action.ops:
            return False

        result = self._demo_orchestrator.revise_script(
            action.ops, source="policy", reason=user_text[:200]
        )
        applied = bool(result.get("applied"))
        if applied:
            self._scratch.plan_change = self.describe_plan_change(action.ops)
        return applied

    def advance_ack_text(self, mechanism: str) -> str:
        """What the guide says when a visitor's turn closes the window.

        If PLAN_REVISE changed the tour on this same turn, the change is named
        here. That is the only place the visitor learns about it: revise_script
        edits the script silently, so a dropped project would otherwise just
        not happen, and "Of course — we'll keep it brief" reads as agreement
        rather than as an itinerary change. A visitor who asked to skip
        something is entitled to hear what was skipped.
        """
        change = getattr(self._scratch, "plan_change", None)
        if not change:
            # A stated time problem is a REQUEST, and silence in answer to a
            # request is indistinguishable from being ignored. A live run had
            # a visitor say "I'm actually running out of time so can we skip?"
            # and hear nothing at all — so they said it again, and again.
            # Even when nothing about the plan can change, they get told the
            # tour heard them.
            if mechanism == Mechanism.TIME_PRESSURE:
                return ADVANCE_ACK[Mechanism.TIME_PRESSURE]

            # Otherwise say nothing. A visitor who says "no more questions"
            # then hears the guide say "Great! Let's continue with the
            # demonstration then!" is being told something the tour is about
            # to demonstrate by simply continuing — and it puts the guide in
            # front of a robot that was mid-presentation. The tour resuming
            # IS the acknowledgement.
            #
            # A plan CHANGE is different: the visitor cannot see that from the
            # tour resuming, because what changed is what is no longer coming.
            return ""
        lead = ADVANCE_ACK_WITH_CHANGE.get(mechanism,
                                           ADVANCE_ACK_WITH_CHANGE["default"])
        return f"{lead} — {change}."

    def describe_plan_change(self, ops) -> str:
        """One clause naming what the tour just lost, for the guide to say.

        Named for the visitor, not for the log: "skip" is an op kind, but what
        a visitor needs to hear is which project they will no longer see. Uses
        robot NAMES for the same reason.
        """
        from decision.models import PlanOpKind

        def name(rid):
            inst = self._registry.get(rid) if rid else None
            return (getattr(inst, "robot_name", None) or rid) if rid else ""

        dropped = [name(o.robot_id) for o in ops
                   if o.kind is PlanOpKind.SKIP and o.robot_id]
        if any(o.kind is PlanOpKind.DROP_REMAINING for o in ops):
            return "we'll head straight to the wrap-up"
        if dropped:
            joined = dropped[0] if len(dropped) == 1 else \
                ", ".join(dropped[:-1]) + " and " + dropped[-1]
            return f"we'll skip {joined} and carry on from there"
        if any(o.kind is PlanOpKind.COMPRESS for o in ops):
            return "I'll keep the introductions short from here"
        if any(o.kind is PlanOpKind.SET_QA_BUDGET for o in ops):
            return "we'll keep the question rounds shorter"
        return ""

    def route_question(self, instance, message: str):
        """
        Decide who should actually answer, and hand off if it is not the
        robot that heard the question.

        QA_ROUTE used to be recorded and nothing else — the receiver always
        answered regardless of what the competence graph thought, so a
        visitor asking a deep technical follow-up about ChatBox's research
        got that answer FROM Pepper, in Pepper's voice, purely because
        Pepper's mic happened to pick up the question. This is what makes
        the decision matter: when the graph is confident enough to name a
        different, connected robot, that robot gets the question and
        answers it in its own voice, and the receiving robot gets a one-line
        handoff instead of improvising an answer about someone else's
        research.

        Returns (target_instance, target_client_id, handoff_text). handoff_text
        is None whenever no reroute happened — the common case, and every
        case where the graph has not learned enough yet to be confident, or
        names a robot that is not actually connected.

        A DEFER (the best robot for this topic is not in the conversation and
        its station is still ahead) returns target_instance=None: nobody
        answers this turn, and the caller speaks `handoff_text` from the guide
        instead of generating a reply. See decision/kg_policy.py's
        ABSENT_ROBOT_POLICY.
        """
        receiver_id = getattr(instance, "client_id", None)
        result = self._decide(DecisionPoint.QA_ROUTE, instance, message)
        if result is None:
            return instance, receiver_id, None

        # Deferred: the guide takes the floor to say the topic is coming up,
        # and no robot generates an answer at all.
        if result.action.kind is ActionKind.GUIDE_INTERJECT:
            deferred_to = getattr(self._scratch, "deferred_to", None)
            target_name = deferred_to or "that robot"
            target_inst = self._registry.get(deferred_to) if deferred_to else None
            if target_inst is not None:
                target_name = getattr(target_inst, "robot_name", None) or deferred_to
            return None, result.action.robot_id or receiver_id, (
                f"That's {target_name}'s area — we'll cover it properly when we "
                f"get to their station."
            )

        target_id = result.action.robot_id

        # NOTE ON WHAT IS *NOT* HERE.
        #
        # This used to silence the hand-off whenever the target was the robot
        # currently presenting, on the reasoning that announcing a robot the
        # group is already standing in front of is redundant. That was an
        # over-correction. The real cause of the noise it was fixing was
        # acknowledgements being routed at all ("okay thank you" producing
        # "Silbot can tell you more about that"), and that is fixed upstream
        # now by is_acknowledgement.
        #
        # With that gone, every surviving reroute is a real change of speaker,
        # and the rule is simply: if the answer comes from a robot other than
        # the one addressed, say so. Silence there is what made the dashboard
        # confusing — a voice changing with no explanation. The
        # target_id == receiver_id check below already covers the case that
        # genuinely needs no announcement: you asked the robot that answers.
        #
        #   interrupt a project robot, ask about its work  -> it just answers
        #   interrupt the guide, ask a technical question  -> the guide hands
        #                                                     over out loud,
        #                                                     then the
        #                                                     specialist answers

        # Guide stepping in because the robot that owns this topic is away and
        # its block has already been cut — there is no station left to defer
        # to. Distinct phrasing from an ordinary reroute: "Pepper can tell you
        # more about that" frames the guide as the better source, which is
        # not what happened. It is a fallback, and saying so is both honest
        # and more natural than a handoff line that does not fit.
        if (result.mechanism == "kg_guide_answers"
                and target_id and target_id != receiver_id):
            guide = self._registry.get(target_id)
            if guide is not None:
                return guide, target_id, (
                    "Let me pick that one up — I can give you the short version now."
                )

        if (result.action.kind is not ActionKind.ROUTE_TO
                or not target_id or target_id == receiver_id):
            return instance, receiver_id, None

        target = self._registry.get(target_id)
        if target is None:
            return instance, receiver_id, None

        target_name = getattr(target, "robot_name", None) or target_id

        # ONCE PER CHANGE OF SPEAKER, not once per question and not once per
        # window. A live run put the identical line in front of three
        # consecutive answers from the same robot; resetting it per window
        # then brought it back at the top of every window, so a visitor who
        # asked something in each of three windows heard the same sentence
        # three times. What is worth announcing is the FLOOR MOVING to a
        # robot other than the last one that had it. Asking the same robot
        # again needs no announcement — the visitor just watched it answer.
        with self._utterance_lock:
            already = self._handed_off_to == target_id
            self._handed_off_to = target_id
        if already:
            return target, target_id, None

        return target, target_id, self._handoff_line(receiver_id, target_name, message)

    HANDOFF_FALLBACK = "{name} can tell you more about that — let's hear from them!"

    def _handoff_line(self, speaker_id, target_name: str, message: str) -> str:
        """What the addressed robot says as it passes the question on.

        Generated, because it was a single f-string and every hand-off in a
        tour came out word for word identical — "ChatBox can tell you more
        about that — let's hear from them!", then the same again for Silbot,
        then for Navel. Written down in a transcript that reads as a template,
        which is what it was.

        Generation sees the actual question, so the line can refer to what was
        asked rather than to "that". It is also the only LLM call between the
        visitor speaking and the answer starting, so the instruction asks for
        one short sentence — and any failure falls straight back to the fixed
        line rather than delaying the answer further.
        """
        instruction = (
            f"A visitor just asked you: \"{message[:200]}\". "
            f"{target_name} is the robot who should answer it, not you. "
            f"Say ONE short sentence handing the question to {target_name} by name — "
            f"warm and natural, under 15 words. Do not answer the question yourself, "
            f"do not greet anyone, and do not add anything after the hand-off."
        )
        fallback = self.HANDOFF_FALLBACK.format(name=target_name)
        if not speaker_id:
            return fallback
        try:
            generated = self.generate_demo_step(speaker_id, instruction)
        except Exception as e:
            logger.warning(f"[WS Gateway] hand-off generation failed: {e}")
            return fallback
        # generate_demo_step echoes the instruction back on failure — that is
        # its documented fallback, and speaking it aloud would tell the
        # visitor how the prompt was written.
        if not generated or generated == instruction:
            return fallback
        spoken = _spoken_text(generated)
        return spoken or fallback

    # ── Public API ────────────────────────────────────────────────────────────

    def connect_robot(self, client_id: str) -> bool:
        """
        Open a WebSocket connection to a robot.
        Looks up ip/port from the DB.
        Returns True if connection was initiated.
        """
        from data import robot_repo
        addr = robot_repo.get_robot_address(client_id)
        if not addr:
            print(f"[WS Gateway] No address for {client_id} — "
                  "set ip_address and ws_port in the web UI.")
            return False

        ip, port = addr
        with self._lock:
            if client_id in self._connections:
                print(f"[WS Gateway] Already connected to {client_id}")
                return True

            conn = RobotConnection(
                client_id=client_id,
                ip=ip,
                port=port,
                on_message=self._on_message,
                on_close=self._on_robot_close,
            )
            self._connections[client_id] = conn
            conn.connect()

            # Give it a moment to establish
            time.sleep(0.5)

            # Trigger registry to create the instance
            self._registry.connect(client_id)
            return True

    def disconnect_robot(self, client_id: str):
        """Close connection and remove from pool."""
        with self._lock:
            conn = self._connections.pop(client_id, None)
            if conn:
                conn.disconnect()
        self._registry.disconnect(client_id)

    def send_to_robot(self, client_id: str, data: dict):
        """Send a JSON payload to a specific robot."""
        event = data.get("event", data.get("type", "?"))
        # Build a readable summary for the terminal
        extra = ""
        if event == "chat_sentence":
            extra = f" | \"{data.get('text', '')[:60]}\""
            if data.get("emotion_tag"):
                extra += f" [{data['emotion_tag']}]"
        elif event == "chat_response":
            extra = f" | \"{data.get('clean_text', data.get('response', ''))[:60]}\""
        elif event == "demo_step":
            step_id = data.get("step_id", "")
            text_preview = data.get("text", "")[:50]
            extra = f" | step={step_id} \"{text_preview}\""
        elif event == "tts_stop":
            extra = " | (interrupt TTS)"
        elif event == "speech_response":
            extra = f" | transcription=\"{data.get('transcription', '')[:40]}\""
        print(f"[→ {client_id}] {event}{extra}")

        # Mirror it to the operator's transcript. Best-effort and never in
        # the way of actually sending: a bookkeeping failure must not stop a
        # robot speaking.
        try:
            self._record_utterance(client_id, data)
        except Exception as e:
            logger.warning(f"[WS Gateway] could not record utterance: {e}")

        # Every outbound utterance passes through here, which makes this the
        # one place that can know what is still being spoken.
        try:
            self.note_speech(client_id, data)
        except Exception as e:
            logger.warning(f"[WS Gateway] could not time utterance: {e}")

        with self._lock:
            conn = self._connections.get(client_id)
        if conn:
            conn.send(data)
        else:
            print(f"[WS Gateway] No connection for {client_id}")

    def get_connected_ids(self) -> list[str]:
        with self._lock:
            return list(self._connections.keys())

    def shutdown(self):
        """Close all connections."""
        with self._lock:
            for conn in self._connections.values():
                conn.disconnect()
            self._connections.clear()
        self._registry.shutdown()

    # ── Message routing ───────────────────────────────────────────────────────

    def _on_message(self, client_id: str, data: dict):
        """
        Route an incoming message from a robot to the right handler.

        Expected message types from the robot:
          - "chat"        : { "type": "chat", "message": "..." }
          - "speech"      : { "type": "speech", "audio": "<base64>" }
          - "image_frame" : { "type": "image_frame", "frame": "<base64>" }
        """
        msg_type = data.get("type")
        instance = self._registry.get(client_id)

        if not instance:
            print(f"[WS Gateway] Message from unregistered robot: {client_id}")
            return

        try:
            if msg_type == "chat":
                message = data.get("message", "")
                if message:
                    # Stop any in-progress TTS immediately — user talking = robot listens.
                    # Still a raw phrase check: barging in is a transport concern,
                    # not a decision, and it must happen before any LLM call.
                    if not any(p in message.lower() for p in self._QA_ADVANCE_PHRASES):
                        self.send_to_robot(client_id, {"event": "tts_stop"})
                        # Also pause the demo if it was running
                        if self._demo_orchestrator:
                            status = self._demo_orchestrator.get_status()
                            if status["state"] in ("running", "waiting_ack"):
                                self._demo_orchestrator.qa_interrupt(source="auto")

                    # One QA_ADVANCE decision now covers what used to be two
                    # separate passes (the advance-phrase check, then the
                    # question-heuristic/classifier chain). The precedence and
                    # the outcomes are identical — see HeuristicPolicy.
                    result = self.check_qa_advance_from_user(instance, message)

                    # PLAN_REVISE first, unconditionally. It used to sit after
                    # the classifier's early return below, which meant a message
                    # classified as "done" (closing THIS window) skipped plan
                    # revision entirely — so "we're running out of time" could
                    # get read as "no more questions" and the tour never
                    # shortened, because the request `return`ed before revision
                    # was ever asked for. The two decisions are independent and
                    # both must get a chance to fire from the same message.
                    self.check_plan_revision(instance, message)

                    if result is not None and result.action.kind is ActionKind.ADVANCE:
                        # EVERY advance mechanism ends the turn here, not just
                        # the classifier. A phrase match used to fall through
                        # and generate, on the theory that the visitor might
                        # have said something else worth answering. In
                        # practice the window is closing, so the answer
                        # arrives as the tour is already moving — and the
                        # reply is generated by whichever robot held the mic,
                        # which need not be the guide. A live run had a
                        # visitor tell Navel "I'm running out of time, can we
                        # skip", and Navel answered "Sure, let's skip it.
                        # What's next on your agenda?" — asking the visitor
                        # to run the tour, while the clock they had just
                        # complained about kept running.
                        #
                        # A fixed line from the GUIDE instead: it is the
                        # robot whose job this is, it costs no generation on
                        # a turn that has nothing to answer, and time
                        # pressure is the case where spending seconds is
                        # exactly wrong.
                        ack = self.advance_ack_text(result.mechanism)
                        if ack:
                            guide_id, _p = guide_and_presenter(
                                self._demo_orchestrator.get_status())
                            self.send_to_robot(guide_id or client_id, {
                                "event": "demo_step",
                                "step_id": "_qa_advance_ack",
                                "text": ack,
                                "require_ack": False,
                            })
                        return

                    # A nod, not a question. Answered from a fixed line rather
                    # than by the LLM: told not to narrate the tour, a 7B
                    # model still replied to "okay thank you" with "Great,
                    # let us move on to the next project!", announcing a
                    # transition the script had not reached. Deciding this in
                    # code is the same fix the question heuristic got —
                    # prompt compliance was not enough. Also saves a
                    # generation on a turn that needs none.
                    if is_acknowledgement(message):
                        self.send_to_robot(client_id, {
                            "event": "chat_sentence",
                            "text": "Of course.",
                            "emotion_tag": "DEFAULT",
                        })
                        return

                    # Who actually answers. A confident competence-graph read
                    # can name a DIFFERENT connected robot than the one that
                    # received this message — the receiver gets a one-line
                    # handoff and the named robot answers in its own voice,
                    # instead of the receiver improvising an answer about
                    # someone else's research just because its mic heard the
                    # question first.
                    target_instance, target_id, handoff = self.route_question(instance, message)
                    if handoff:
                        self.send_to_robot(client_id, {
                            "event": "chat_sentence",
                            "text": handoff,
                            "emotion_tag": "DEFAULT",
                        })

                    # Deferred: the guide has just promised the topic will be
                    # covered at that robot's station, so nobody answers now.
                    # Returning before process_chat_stream is the point —
                    # generating a reply here would be the improvised answer
                    # deferring exists to avoid.
                    if target_instance is None:
                        return

                    # Let the hand-off be HEARD before the answer starts —
                    # see the same guard in http_gateway. Two robots, two
                    # speakers, nothing else serialising them.
                    _wait = {"left": _speaking_seconds(handoff) if handoff else 0.0}

                    def _on_sentence(clean_text, emotion_tag):
                        if _wait["left"] > 0:
                            time.sleep(_wait["left"])
                            _wait["left"] = 0.0
                        if '```' in clean_text:  # Skip delegation JSON blocks — never speak raw JSON
                            return
                        # A chunk that is only an emotion tag is not speech.
                        # The model sometimes emits a trailing "[DEFAULT]" of
                        # its own, and it was sent as a sentence: the robot
                        # got a line with nothing to say and the transcript
                        # showed a blank turn under its name.
                        if not _spoken_text(clean_text):
                            return
                        self.send_to_robot(target_id, {
                            "event": "chat_sentence",
                            "text": clean_text,
                            "emotion_tag": emotion_tag,
                        })

                    result = target_instance.process_chat_stream(
                        message, _on_sentence,
                        style_framing=self.style_framing_for(target_id),
                        grounded_facts=self.grounding_for(target_id, message),
                    )
                    # Did the robot just sign off? If so the window closes on
                    # its own after a pause, instead of waiting for someone to
                    # say "move on" out loud.
                    self.check_qa_auto_close(target_id, result.clean_text or "")
                    # Handle delegation if needed
                    if result.is_delegation and result.delegation_target:
                        from gateway.delegation_handler import DelegationHandler
                        handler = DelegationHandler(self._registry, self)
                        handler.handle(target_id, result.response)

            elif msg_type == "speech":
                audio_b64 = data.get("audio", "")
                if audio_b64:
                    result = instance.process_speech(audio_b64)

                    _is_advance = result.transcription and any(
                        p in result.transcription.lower() for p in self._QA_ADVANCE_PHRASES
                    )

                    # Fast-path: advance phrase in active QA window — robot gives a brief
                    # acknowledgment, then Pepper's transition generates in parallel.
                    if _is_advance and (
                        self._demo_orchestrator and
                        self._demo_orchestrator.get_status()["state"] == "qa_window"
                    ):
                        self.send_to_robot(client_id, {"event": "tts_stop"})
                        # Use first sentence of the LLM response as a brief acknowledgment.
                        # Falls back to a fixed phrase if no LLM response was generated.
                        ack_text = "Of course, let's move on!"
                        ack_tag  = "DEFAULT"
                        if result.chat and result.chat.clean_text:
                            import re as _re
                            first = _re.split(r'(?<=[.!?])\s+', result.chat.clean_text.strip())[0]
                            if first:
                                ack_text = first
                                ack_tag  = result.chat.emotion_tag or "DEFAULT"
                        self.send_to_robot(client_id, {
                            "event": "speech_response",
                            "transcription": result.transcription,
                            "confidence": result.confidence,
                            "response":    f"[{ack_tag}] {ack_text}",
                            "emotion_tag": ack_tag,
                            "clean_text":  ack_text,
                        })
                        # This is the barge-in fast path for a raw ADVANCE_PHRASE
                        # match on the transcription — it can fire on a message
                        # that ALSO states time pressure ("let's move on, we're
                        # running out of time"), so plan revision gets the same
                        # chance here as everywhere else before the early return.
                        self.check_plan_revision(instance, result.transcription)
                        self.check_qa_advance_from_user(instance, result.transcription)
                        return

                    # Stop any in-progress TTS immediately — user talking = robot listens
                    if result.transcription and not _is_advance:
                        self.send_to_robot(client_id, {"event": "tts_stop"})
                        if self._demo_orchestrator:
                            status = self._demo_orchestrator.get_status()
                            if status["state"] in ("running", "waiting_ack"):
                                self._demo_orchestrator.qa_interrupt(source="auto")
                    response_data: dict = {
                        "event": "speech_response",
                        "transcription": result.transcription,
                        "confidence": result.confidence,
                    }
                    if result.chat:
                        response_data.update({
                            "response": result.chat.response,
                            "emotion_tag": result.chat.emotion_tag,
                            "clean_text": result.chat.clean_text,
                        })
                        if result.chat.is_delegation and result.chat.delegation_target:
                            from gateway.delegation_handler import DelegationHandler
                            handler = DelegationHandler(self._registry, self)
                            handler.handle(client_id, result.chat.response)
                    self.send_to_robot(client_id, response_data)
                    # Check advance intent — and, now, whether the visitor asked
                    # for the rest of the tour to change.
                    if result.transcription:
                        self.check_qa_advance_from_user(instance, result.transcription)
                        self.check_plan_revision(instance, result.transcription)

            elif msg_type == "image_frame":
                frame_b64 = data.get("frame", "")
                if frame_b64:
                    result = instance.process_frame(frame_b64)
                    self.send_to_robot(client_id, {
                        "event": "emotion_update",
                        **result,
                    })

            elif msg_type == "ack":
                # Demo step acknowledgement — forward to orchestrator
                step_id = data.get("step_id")
                if self._demo_orchestrator and step_id:
                    self._demo_orchestrator.receive_ack(step_id)
                else:
                    print(f"[WS Gateway] ACK from {client_id}: step_id='{step_id}' "
                          "(no orchestrator running)")

            elif msg_type == "tts_interrupted":
                # A barge-in cut a demo step off mid-speech — the client reports
                # back whatever it never got to say, so it can be resumed
                # instead of silently skipped once the interruption resolves.
                step_id = data.get("step_id")
                remaining = data.get("remaining_text", "")
                if self._demo_orchestrator and step_id and remaining:
                    self._demo_orchestrator.note_interrupted_step(
                        client_id, step_id, remaining
                    )

            else:
                print(f"[WS Gateway] Unknown message type '{msg_type}' "
                      f"from {client_id}")

        except Exception as e:
            print(f"[WS Gateway] Error handling '{msg_type}' "
                  f"from {client_id}: {e}")

    def _on_robot_close(self, client_id: str):
        """Called when a robot's connection drops unexpectedly."""
        print(f"[WS Gateway] {client_id} connection dropped.")
        self._registry.disconnect(client_id)
        with self._lock:
            self._connections.pop(client_id, None)