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
                 kg_router_factory=None, kg_observer=None):
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
            decision = router.decide(obs.user_utterance, peers)
            if decision is None:
                return None
            from decision.models import Action
            logger.info(
                f"[KG route] '{obs.user_utterance[:40]}' -> {decision.topic_label} "
                f"-> {decision.robot_id} ({decision.reason}, {decision.score})")
            # The mechanism records WHICH rule fired, so an exploration pick is
            # distinguishable from a confident one in the correction-rate view.
            mechanism = ("kg_explore" if decision.reason.startswith("explore")
                         else "kg_argmax")
            # Only a question that actually resolved to a topic is recorded, so
            # the segment can never credit an edge for a turn it did not handle.
            self._segment.note_routed(decision.robot_id, decision.topic_id)
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

    def check_qa_auto_close(self, responding_robot_id: str, clean_text: str):
        """
        Decide, after a robot responds, whether the Q&A window should close.

        DORMANT — nothing calls this, and nothing called its predecessor either.
        The closing-phrase list and the guide's LLM wrap-up judgement were both
        written and then never wired to a call site, so of the five Q&A
        mechanisms only three have ever run: the advance phrases, the question
        heuristic, and the intent classifier. It is left uncalled deliberately:
        activating it here would change live demo behaviour under cover of a
        refactor. Call it from the response path to turn it on, and expect the
        Q&A windows to start closing on their own.

        The guide never judges its own responses — the recursion guard the
        original had, preserved.
        """
        if not self._demo_orchestrator or not clean_text:
            return
        status = self._demo_orchestrator.get_status()
        if status.get("state") != "qa_window":
            return

        self._tracker.note_robot_turn(responding_robot_id, clean_text)

        guide_id = status.get("robot_id")
        if guide_id and responding_robot_id == guide_id:
            return   # don't recurse on the guide's own responses

        decider = self._registry.get(responding_robot_id)
        action = self._decide(DecisionPoint.QA_ADVANCE, decider).action

        if action.kind is ActionKind.ADVANCE:
            print("[WS Gateway] Closing Q&A — closing phrase in robot response.")
            self._demo_orchestrator.qa_end(source="policy")
            return

        if action.kind is ActionKind.GUIDE_INTERJECT:
            # Speak the sentence the guide already generated while judging.
            # Generating a second one would risk it contradicting the first.
            text = getattr(self._scratch, "wrap_up_text", None)
            target = action.robot_id or guide_id
            if text and target:
                self.send_to_robot(target, {
                    "event":       "demo_step",
                    "step_id":     "_qa_wrap_up",
                    "text":        text,
                    "require_ack": False,
                })

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

        self._tracker.note_visitor_turn(
            getattr(decider, "client_id", None) or "unknown", user_text
        )

        if self._demo_orchestrator.get_status().get("state") != "qa_window":
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

        Returns True if the script was changed.
        """
        if not self._demo_orchestrator or not user_text:
            return False

        action = self._decide(DecisionPoint.PLAN_REVISE, decider, user_text).action
        if action.kind is not ActionKind.REVISE or not action.ops:
            return False

        result = self._demo_orchestrator.revise_script(
            action.ops, source="policy", reason=user_text[:200]
        )
        return bool(result.get("applied"))

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
                    if result is not None and result.action.kind is ActionKind.ADVANCE:
                        if result.mechanism == Mechanism.LLM_CLASSIFIER:
                            # Classifier path: acknowledge and end the turn. A
                            # phrase match falls through instead, so the robot
                            # still answers whatever else the visitor said.
                            self.send_to_robot(client_id, {
                                "event": "demo_step",
                                "step_id": "_qa_classifier_done",
                                "text": "[DEFAULT] Great! Let's continue with the demonstration then!",
                                "require_ack": False,
                            })
                            return

                    # Should the rest of the tour change? Only acts on an explicit
                    # request, or on the clock when the run was started with a
                    # time budget — without one this is always a no-op.
                    self.check_plan_revision(instance, message)

                    # Who answers. The baseline routes to whoever heard the
                    # question, so this changes nothing yet; it exists so the
                    # choice is on the record and can be compared against.
                    self._decide(DecisionPoint.QA_ROUTE, instance, message)

                    def _on_sentence(clean_text, emotion_tag):
                        if '```' in clean_text:  # Skip delegation JSON blocks — never speak raw JSON
                            return
                        self.send_to_robot(client_id, {
                            "event": "chat_sentence",
                            "text": clean_text,
                            "emotion_tag": emotion_tag,
                        })

                    result = instance.process_chat_stream(message, _on_sentence)
                    # Handle delegation if needed
                    if result.is_delegation and result.delegation_target:
                        from gateway.delegation_handler import DelegationHandler
                        handler = DelegationHandler(self._registry, self)
                        handler.handle(client_id, result.response)

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