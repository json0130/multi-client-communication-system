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
import threading
import time
from typing import Optional, Callable, TYPE_CHECKING

import websocket   # websocket-client library

if TYPE_CHECKING:
    from robot.robot_registry import RobotRegistry


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
    ]

    def set_demo_orchestrator(self, orchestrator):
        """Wire up the DemoOrchestrator so ACK packets are forwarded to it."""
        self._demo_orchestrator = orchestrator

    def check_qa_auto_close(self, clean_text: str):
        """
        Called after any robot sends a chat response during the demo.
        If the demo is in QA_WINDOW and the response contains a closing phrase,
        automatically end the Q&A window so the demo can advance.
        """
        if not self._demo_orchestrator or not clean_text:
            return
        if self._demo_orchestrator.get_status()["state"] != "qa_window":
            return
        text_lower = clean_text.lower()
        if any(phrase in text_lower for phrase in self._QA_CLOSING_PHRASES):
            print(f"[WS Gateway] Auto-closing Q&A — closing phrase detected.")
            self._demo_orchestrator.qa_end()

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
                    result = instance.process_chat(message)
                    self.send_to_robot(client_id, {
                        "event": "chat_response",
                        "response": result.response,
                        "emotion_tag": result.emotion_tag,
                        "clean_text": result.clean_text,
                    })
                    self.check_qa_auto_close(result.clean_text)
                    # Handle delegation if needed
                    if result.is_delegation and result.delegation_target:
                        from gateway.delegation_handler import DelegationHandler
                        handler = DelegationHandler(self._registry, self)
                        handler.handle(client_id, result.response)

            elif msg_type == "speech":
                audio_b64 = data.get("audio", "")
                if audio_b64:
                    result = instance.process_speech(audio_b64)
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