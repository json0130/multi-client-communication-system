"""
client.py
=========
Base client for the ChatBox robot (non-Docker, v6 architecture).

Key change from v5:
  The robot NO LONGER dials out to the server.
  It runs a WebSocket SERVER and WAITS for the central server to connect.

All Input/Output module interfaces are unchanged:
  self.client.send_to_server(type, data)
  self.client.process_server_response(data, type)
  self.client.is_speaking          (threading.Event)
  self.client.tts_started_event    (threading.Event)
  self.client.running              (bool)
"""

import asyncio
import websockets
import json
import re
import time
import base64
import threading
import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Abstract base classes (identical to v5) ───────────────────────────────────

class BaseModule(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = False
        self.client = None

    @abstractmethod
    def initialize(self) -> bool: ...

    @abstractmethod
    def start(self) -> bool: ...

    @abstractmethod
    def stop(self): ...

    def set_client(self, client):
        self.client = client


class InputModule(BaseModule):
    @abstractmethod
    def get_data(self) -> Optional[Any]: ...


class OutputModule(BaseModule):
    @abstractmethod
    def process_output(self, data: Any) -> bool: ...


# ── ServerConnection — robot listens, central server connects ─────────────────

class ServerConnection:
    """
    Runs a WebSocket server on ws_port.
    The central server dials in when the operator clicks "Connect" in the web UI.

    Public interface used by robot.py:
        register_handler(event, callback)   ← replaces sio.on()
        send(message_dict)                  ← send JSON to server
        is_connected()                      ← bool
        register_with_server()              ← HTTP POST to tell server our address
        start_server()                      ← begin listening
    """

    def __init__(self, ws_port: int, server_url: str, client_config: Dict):
        self.ws_port = ws_port
        self.server_url = server_url.rstrip("/")
        self.client_config = client_config
        self.client_id = client_config.get("client_id", "robot_001")

        self._ws = None
        self._loop = asyncio.new_event_loop()
        self._connected = threading.Event()
        self._handlers: Dict[str, callable] = {}
        self._lock = threading.Lock()

    # ── Handler registration ──────────────────────────────────────────────────

    def register_handler(self, event: str, callback: callable):
        """
        Register a callback for a named event pushed by the server.

        Usage (in robot.py):
            self.server_connection.register_handler('chat_response', self.on_chat_response)

        Replaces the old: self.server_connection.sio.on('chat_response', ...)
        """
        with self._lock:
            self._handlers[event] = callback
        logger.debug(f"[WS] Handler registered for event: {event}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_server(self):
        """Start WebSocket server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True, name="ws-server")
        t.start()
        logger.info(f"[WS] Robot WebSocket server listening on 0.0.0.0:{self.ws_port}")
        logger.info(f"[WS] Waiting for central server to connect...")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(
            self._handle_connection,
            "0.0.0.0",
            self.ws_port,
            ping_interval=20,
            ping_timeout=10,
        ):
            await asyncio.Future()  # run forever

    async def _handle_connection(self, websocket):
        addr = websocket.remote_address
        logger.info(f"[WS] Central server connected from {addr[0]}:{addr[1]}")
        self._ws = websocket
        self._connected.set()

        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                    self._dispatch(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"[WS] Bad JSON: {e}")
        except websockets.ConnectionClosed:
            logger.warning("[WS] Central server disconnected.")
        except Exception as e:
            logger.error(f"[WS] Connection error: {e}")
        finally:
            self._ws = None
            self._connected.clear()
            logger.info("[WS] Waiting for central server to reconnect...")

    def _dispatch(self, data: dict):
        """Route an incoming server message to the right handler."""
        event = data.get("event")
        if not event:
            return

        with self._lock:
            handler = self._handlers.get(event)

        if handler:
            # Run in separate thread so asyncio loop is never blocked
            threading.Thread(target=handler, args=(data,), daemon=True).start()
        else:
            logger.debug(f"[WS] No handler for event '{event}'")

    # ── Sending ───────────────────────────────────────────────────────────────

    def send(self, message: dict) -> bool:
        """
        Send a JSON message to the central server.
        Thread-safe — call from any module thread.
        """
        if self._ws is None or not self._connected.is_set():
            logger.warning("[WS] Cannot send — server not connected.")
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps(message)),
                self._loop,
            )
            future.result(timeout=5)
            return True
        except Exception as e:
            logger.error(f"[WS] Send error: {e}")
            return False

    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ── Registration ──────────────────────────────────────────────────────────

    def register_with_server(self) -> bool:
        """
        HTTP POST /robots/register on the central server.
        Tells it: "I am at this IP and port — come connect to me."
        """
        url = f"{self.server_url}/robots/register"
        payload = {
            "client_id":   self.client_id,
            "robot_name":  self.client_config.get("robot_name", "Robot"),
            "robot_role":  self.client_config.get("robot_role", "You are a helpful robot."),
            "allowed_tags": self.client_config.get("allowed_tags", ["[DEFAULT]"]),
            "modules":     self.client_config.get("modules", ["gpt"]),
            "ip_address":  self.client_config.get("ip_address"),
            "ws_port":     self.ws_port,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"[Registration] OK — {resp.json().get('message', '')}")
                return True
            else:
                logger.warning(f"[Registration] Server returned {resp.status_code}")
                return False
        except requests.ConnectionError:
            logger.warning(
                f"[Registration] Could not reach {self.server_url}. "
                "Robot is still listening for connections."
            )
            return False
        except Exception as e:
            logger.warning(f"[Registration] Error: {e}")
            return False


# ── BasicClient ───────────────────────────────────────────────────────────────

class BasicClient:
    """
    Main client. Manages Input/Output modules and server communication.
    Public interface unchanged from v5 — all existing modules work as-is.
    """

    def __init__(self, config_file: str = "client_config.json"):
        self.config = self._load_config(config_file)
        if not self.config:
            raise RuntimeError(f"Failed to load config from {config_file}")

        ws_port    = self.config.get("ws_port", 8765)
        server_url = self.config.get("server_url", "http://localhost:5000")

        self.server_connection = ServerConnection(ws_port, server_url, self.config)

        # Default handlers — subclasses override by calling register_handler()
        self.server_connection.register_handler("chat_response",   self._default_chat_handler)
        self.server_connection.register_handler("speech_response", self._default_speech_handler)
        self.server_connection.register_handler("emotion_update",  self._default_emotion_handler)

        self.input_modules:  Dict[str, InputModule]  = {}
        self.output_modules: Dict[str, OutputModule] = {}
        self.arduino_module = None

        self.running = False

        # Shared events used by edge_tts_output and voice_input (unchanged)
        self.is_speaking       = threading.Event()
        self.tts_started_event = threading.Event()

        logger.info(f"[Client] {self.config.get('robot_name', 'Robot')}")
        logger.info(f"         ID     : {self.config.get('client_id')}")
        logger.info(f"         Server : {server_url}")
        logger.info(f"         WS port: {ws_port}")

    # ── Default event handlers ────────────────────────────────────────────────

    def _default_chat_handler(self, data: dict):
        self.process_server_response(data, "chat")

    def _default_speech_handler(self, data: dict):
        self.process_server_response(data, "speech")

    def _default_emotion_handler(self, data: dict):
        pass  # override in subclass if needed

    # ── Module registration ───────────────────────────────────────────────────

    def register_input_module(self, module: InputModule) -> bool:
        try:
            module.set_client(self)
            if module.initialize():
                self.input_modules[module.name] = module
                logger.info(f"[Modules] Input  '{module.name}' registered")
                return True
            logger.error(f"[Modules] Input  '{module.name}' failed to initialize")
            return False
        except Exception as e:
            logger.error(f"[Modules] Input  '{module.name}' error: {e}")
            return False

    def register_output_module(self, module: OutputModule) -> bool:
        try:
            module.set_client(self)
            if module.initialize():
                self.output_modules[module.name] = module
                logger.info(f"[Modules] Output '{module.name}' registered")
                return True
            logger.error(f"[Modules] Output '{module.name}' failed to initialize")
            return False
        except Exception as e:
            logger.error(f"[Modules] Output '{module.name}' error: {e}")
            return False

    # ── Sending (interface unchanged from v5) ─────────────────────────────────

    def send_to_server(self, data_type: str, data: Any) -> Optional[dict]:
        """
        Send data to the central server.
        Called by Input modules exactly as before — return value is now always None
        because responses arrive asynchronously via registered event handlers.

            self.client.send_to_server('chat',   message_string)
            self.client.send_to_server('speech', wav_bytes)
            self.client.send_to_server('frame',  base64_string or dict)
        """
        if data_type == "chat":
            self.server_connection.send({"type": "chat", "message": data})

        elif data_type == "speech":
            audio_b64 = base64.b64encode(data).decode("utf-8")
            self.server_connection.send({"type": "speech", "audio": audio_b64})

        elif data_type == "frame":
            # RealSense sends a dict with 'color' key; regular cameras send a string
            if isinstance(data, dict):
                frame_b64 = data.get("color", data.get("frame", ""))
            else:
                frame_b64 = data
            self.server_connection.send({"type": "image_frame", "frame": frame_b64})

        else:
            logger.warning(f"[Client] Unknown data_type: {data_type}")

        return None  # responses come via WS event handlers

    # ── Processing responses (interface unchanged from v5) ────────────────────

    def process_server_response(self, response_data: Optional[dict], response_type: str = "chat"):
        """
        Route a server response to all output modules.
        Called by default event handlers or directly by voice_input after STT.
        Handles None gracefully (voice_input passes the return value of send_to_server).
        """
        if not response_data:
            return

        response_text = response_data.get("response", "")
        if not response_text:
            transcription = response_data.get("transcription", "")
            if transcription:
                logger.info(f"[STT] Transcribed: '{transcription}'")
            return

        logger.info(f"[Server] {response_text[:100]}")

        # Extract emotion tag and dispatch to hardware hook
        match = re.search(r"\[(.*?)\]", response_text)
        if match:
            self.on_emotion_detected(match.group(1))

        self.tts_started_event.clear()

        for name, module in self.output_modules.items():
            try:
                module.process_output({
                    "text": response_text,
                    "type": response_type,
                    "full_response": response_data,
                })
            except Exception as e:
                logger.error(f"[Modules] Output '{name}' error: {e}")

        if response_type == "speech":
            transcription = response_data.get("transcription", "")
            if transcription:
                logger.info(f"[STT] Transcribed: '{transcription}'")

    def on_emotion_detected(self, emotion_tag: str):
        """Hook — override in subclasses to drive hardware."""
        pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            # Tell server where to find us
            self.server_connection.register_with_server()
            # Start listening for the server's incoming connection
            self.server_connection.start_server()

            self.running = True

            for name, module in self.input_modules.items():
                try:
                    module.start()
                    logger.info(f"[Modules] Started input  '{name}'")
                except Exception as e:
                    logger.error(f"[Modules] Error starting input '{name}': {e}")

            for name, module in self.output_modules.items():
                try:
                    module.start()
                    logger.info(f"[Modules] Started output '{name}'")
                except Exception as e:
                    logger.error(f"[Modules] Error starting output '{name}': {e}")

            return True
        except Exception as e:
            logger.error(f"[Client] Start error: {e}")
            return False

    def stop(self):
        logger.info("[Client] Stopping...")
        self.running = False
        for module in list(self.input_modules.values()) + list(self.output_modules.values()):
            try:
                module.stop()
            except Exception as e:
                logger.error(f"[Client] Error stopping '{module.name}': {e}")
        logger.info("[Client] Stopped.")

    def run(self):
        """Start and block until Ctrl+C."""
        try:
            if not self.start():
                return
            logger.info("[Client] Running — press Ctrl+C to stop")
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("[Client] Ctrl+C received")
        except Exception as e:
            logger.error(f"[Client] Runtime error: {e}")
        finally:
            self.stop()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_config(self, config_file: str) -> Optional[dict]:
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            logger.info(f"[Client] Config loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"[Client] Config file not found: {config_file}")
            return None
        except Exception as e:
            logger.error(f"[Client] Config load error: {e}")
            return None