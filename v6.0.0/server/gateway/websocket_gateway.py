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

Requires: pip install websocket-client
"""

from __future__ import annotations
import json
import logging
import threading
import time
from typing import Optional, Callable, TYPE_CHECKING

import websocket   # websocket-client library

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from robot.robot_registry import RobotRegistry


def _looks_like_question(text: str) -> bool:
    """Heuristic pre-filter: skip the LLM classifier for obvious questions."""
    t = text.lower().strip()
    if "?" in t:
        return True
    return t.startswith((
        "what ", "how ", "why ", "where ", "when ", "who ", "which ",
        "can ", "could ", "tell me", "explain", "describe",
        "is there", "are there", "do you", "does it",
    ))


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

    def __init__(self, registry: "RobotRegistry"):
        self._registry = registry
        self._connections: dict[str, RobotConnection] = {}
        self._lock = threading.Lock()
        self._demo_orchestrator = None   # set via set_demo_orchestrator()

    # Phrases from a ROBOT RESPONSE that signal it is wrapping up the Q&A.
    _QA_CLOSING_PHRASES = [
        # Research robot wrapping up their explanation
        "let me know if you have any other questions",
        "any other questions",
        "feel free to ask",
        "hope that answers",
        "hope that helps",
        "is there anything else",
        "anything else i can help",
        "don't hesitate to ask",
        "please don't hesitate",
        "happy to answer more",
        "if you'd like to know more",
        # Pepper acknowledging a "move on" request from the visitor
        "let us move on",
        "let's move on",
        "moving on to",
        "moving on now",
        "let's proceed",
        "shall proceed",
        "proceed to the next",
        "sure thing",
        "great, moving",
    ]

    # Phrases in the USER'S INPUT that signal clear intent to advance the demo.
    _QA_ADVANCE_PHRASES = [
        "move on",
        "next project",
        "next robot",
        "next step",
        "proceed",
        "can we continue",
        "shall we continue",
        "ready to continue",
        "ready to move",
        "let's go",
        "let's continue",
        "let us continue",
        "no more questions",
        "no questions",
        "that's all",
        "no thank you",
        "continue the demo",
        "continue the demonstration",
        # Natural dismissals that the LLM classifier tends to misread
        "carry on",
        "move along",
        "all good",
        "we're good",
        "i'm good",
        "im good",
        "that's fine",
        "that's okay",
        "it's okay",
        "no worries",
        "never mind",
        "forget it",
        "done here",
        "we're done",
        "all done",
    ]

    def set_demo_orchestrator(self, orchestrator):
        """Wire up the DemoOrchestrator so ACK packets are forwarded to it."""
        self._demo_orchestrator = orchestrator

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

    def check_qa_auto_close(self, clean_text: str):
        """
        Called after a robot sends a chat response during the demo.
        If the demo is in QA_WINDOW and the response contains a closing/advance phrase,
        automatically end the Q&A window so the demo can advance.
        """
        if not self._demo_orchestrator or not clean_text:
            return
        if self._demo_orchestrator.get_status()["state"] != "qa_window":
            return
        text_lower = clean_text.lower()
        if any(phrase in text_lower for phrase in self._QA_CLOSING_PHRASES):
            print(f"[WS Gateway] Auto-closing Q&A — closing phrase in robot response.")
            self._demo_orchestrator.qa_end()

    def check_qa_advance_from_user(self, user_text: str):
        """
        Called with the raw user input message during the demo.
        If the demo is in QA_WINDOW and the user is clearly asking to advance,
        end the Q&A window immediately (before the robot even responds).
        """
        if not self._demo_orchestrator or not user_text:
            return
        if self._demo_orchestrator.get_status()["state"] != "qa_window":
            return
        text_lower = user_text.lower()
        if any(phrase in text_lower for phrase in self._QA_ADVANCE_PHRASES):
            print(f"[WS Gateway] Auto-closing Q&A — advance intent from user: '{user_text[:50]}'")
            self._demo_orchestrator.qa_end()

    def _check_qa_pepper_wrap_up(self, responding_robot_id: str, robot_clean_text: str):
        """
        After a research robot responds during QA_WINDOW, ask Pepper's LLM
        whether this is a natural wrap-up point. Pepper generates a brief
        transition sentence if YES, stays silent if NO.
        Run in a daemon thread — must not block the message handler.
        """
        if not self._demo_orchestrator or not robot_clean_text:
            return
        status = self._demo_orchestrator.get_status()
        if status["state"] != "qa_window":
            return

        # pepper_id is the robot_id on the current Q&A step
        pepper_id = status.get("robot_id")
        if not pepper_id or responding_robot_id == pepper_id:
            return   # don't recurse on Pepper's own responses

        pepper_instance = self._registry.get(pepper_id)
        if not pepper_instance or not hasattr(pepper_instance, "process_chat"):
            return

        prompt = (
            f"[Demo moderator context — Q&A step: {status.get('step_id')}] "
            f"A research robot just responded: \"{robot_clean_text[:200]}\". "
            f"As the demo moderator, decide: is this a natural wrap-up point where visitors "
            f"seem satisfied and we could transition to the next part of the demo? "
            f"If YES — write a single warm 1-sentence transition (e.g. 'Wonderful! "
            f"Shall we move on to the next part?'). "
            f"If NO — respond with exactly: NO"
        )
        try:
            result = pepper_instance.process_chat(prompt)
            reply = (result.clean_text or "").strip()
            if reply and reply.upper() != "NO" and len(reply) > 5:
                self.send_to_robot(pepper_id, {
                    "event":       "demo_step",
                    "step_id":     "_qa_wrap_up",
                    "text":        result.response,
                    "require_ack": False,
                })
        except Exception as e:
            print(f"[WS Gateway] Pepper wrap-up eval failed: {e}")

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
                    # Stop any in-progress TTS immediately — user talking = robot listens
                    if not any(p in message.lower() for p in self._QA_ADVANCE_PHRASES):
                        self.send_to_robot(client_id, {"event": "tts_stop"})
                        # Also pause the demo if it was running
                        if self._demo_orchestrator:
                            status = self._demo_orchestrator.get_status()
                            if status["state"] in ("running", "waiting_ack"):
                                self._demo_orchestrator.qa_interrupt()

                    self.check_qa_advance_from_user(message)

                    # Q&A intent classification — skip classifier for obvious questions
                    if self._demo_orchestrator:
                        if self._demo_orchestrator.get_status()["state"] == "qa_window":
                            if not any(p in message.lower() for p in self._QA_ADVANCE_PHRASES):
                                if _looks_like_question(message):
                                    print(f"[QA Router] '{message[:60]}' → question detected → full chat")
                                else:
                                    print(f"[QA Router] '{message[:60]}' → calling LLM classifier...")
                                    intent = instance.classify_qa_intent(message)
                                    if intent == "done":
                                        print(f"[QA Router] Classifier → 'done' → advancing demo")
                                        self._demo_orchestrator.qa_end()
                                        self.send_to_robot(client_id, {
                                            "event": "demo_step",
                                            "step_id": "_qa_classifier_done",
                                            "text": "[DEFAULT] Great! Let's continue with the demonstration then!",
                                            "require_ack": False,
                                        })
                                        return
                                    else:
                                        print(f"[QA Router] Classifier → 'continue' → full chat")

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
                        self.check_qa_advance_from_user(result.transcription)
                        return

                    # Stop any in-progress TTS immediately — user talking = robot listens
                    if result.transcription and not _is_advance:
                        self.send_to_robot(client_id, {"event": "tts_stop"})
                        if self._demo_orchestrator:
                            status = self._demo_orchestrator.get_status()
                            if status["state"] in ("running", "waiting_ack"):
                                self._demo_orchestrator.qa_interrupt()
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
                    # Check advance intent from visitor's transcription
                    if result.transcription:
                        self.check_qa_advance_from_user(result.transcription)

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