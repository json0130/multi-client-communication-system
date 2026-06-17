"""
client.py — Base Client (v6.0.0)
=================================
Key change from v5.x:
  The robot no longer dials OUT to the server.
  Instead it runs a WebSocket SERVER and waits for the central
  server to connect to it.

Everything the Input/Output modules touch is unchanged:
  self.client.send_to_server(type, data)
  self.client.process_server_response(data, type)
  self.client.is_speaking          (threading.Event)
  self.client.tts_started_event    (threading.Event)
  self.client.running              (bool)

The only thing that changed in robot.py is one line:
  OLD: self.server_connection.sio.on('chat_response', handler)
  NEW: self.server_connection.register_handler('chat_response', handler)
"""

import asyncio
import websockets
import json
import re
import time
import threading
import requests
import base64
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Abstract module base classes (unchanged from v5.x) ────────────────────────

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


# ── ServerConnection — WebSocket SERVER (robot listens, server connects) ──────

class ServerConnection:
    """
    Runs an asyncio WebSocket server on ws_port.
    The central server connects TO this robot when the operator
    clicks "Connect" in the web management UI.

    Thread safety:
      The asyncio loop runs in a dedicated background thread.
      send() is safe to call from any thread.
    """

    def __init__(self, ws_port: int, server_url: str, client_config: Dict):
        self.ws_port = ws_port
        self.server_url = server_url.rstrip("/")
        self.client_config = client_config
        self.client_id = client_config.get("client_id", "robot_001")

        self._ws = None                          # active websocket (one server at a time)
        self._loop = asyncio.new_event_loop()    # dedicated asyncio loop
        self._connected = threading.Event()      # set when server is connected
        self._handlers: Dict[str, callable] = {} # event -> callback
        self._lock = threading.Lock()

    # ── Handler registration ───────────────────────────────────────────────────

    def register_handler(self, event: str, callback: callable):
        """
        Register a callback for a named event pushed by the server.
        Replaces any existing handler for that event.

        robot.py usage:
            self.server_connection.register_handler('chat_response', self.on_chat_response)
        """
        with self._lock:
            self._handlers[event] = callback

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_server(self):
        """Start the WebSocket server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True, name="ws-server")
        t.start()
        logger.info(f"[WS Server] Listening on 0.0.0.0:{self.ws_port}")
        logger.info(f"[WS Server] Waiting for central server to connect...")

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
            logger.info(f"[WS Server] Ready on port {self.ws_port}")
            await asyncio.Future()  # run forever

    async def _handle_connection(self, websocket):
        """Called by asyncio when the central server connects."""
        addr = websocket.remote_address
        logger.info(f"[WS Server] Central server connected from {addr}")
        self._ws = websocket
        self._connected.set()

        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                    self._dispatch(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"[WS Server] Bad JSON from server: {e}")
        except websockets.ConnectionClosed:
            logger.warning("[WS Server] Central server disconnected.")
        except Exception as e:
            logger.error(f"[WS Server] Connection error: {e}")
        finally:
            self._ws = None
            self._connected.clear()
            logger.info("[WS Server] Waiting for server to reconnect...")

    def _dispatch(self, data: dict):
        """Route an incoming server message to the registered handler."""
        event = data.get("event")
        if not event:
            logger.debug(f"[WS Server] Message with no event: {data}")
            return

        if event == "persona_update":
            threading.Thread(
                target=self._apply_persona_update, args=(data,), daemon=True
            ).start()
            return

        with self._lock:
            handler = self._handlers.get(event)

        if handler:
            threading.Thread(
                target=handler, args=(data,), daemon=True
            ).start()
        else:
            logger.debug(f"[WS Server] No handler registered for event '{event}'")

    def _apply_persona_update(self, data: dict):
        """
        Handle persona_update pushed by the central server.
        1. Updates in-memory config immediately
        2. Writes updated client_config.json to disk
        3. Calls any registered persona_update handler (so robot.py can update TTS)
        """
        import json as _json
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] Applying persona: '{persona_name}'")

        updatable = {
            "robot_role": "robot_role",
            "allowed_tags": "allowed_tags",
            "modules": "modules",
            "voice_config": "voice_config",
            "capabilities": "capabilities",
            "personality": "personality",
        }
        changed = {cfg_key: data[ws_key]
                   for ws_key, cfg_key in updatable.items() if ws_key in data}
        if not changed:
            return

        self.client_config.update(changed)

        config_path = self.client_config.get("_config_file", "client_config.json")
        try:
            with open(config_path, "r") as f:
                on_disk = _json.load(f)
            on_disk.update(changed)
            with open(config_path, "w") as f:
                _json.dump(on_disk, f, indent=4)
            logger.info(f"[Persona] Saved to {config_path}")
        except Exception as e:
            logger.error(f"[Persona] Failed to write config: {e}")

        with self._lock:
            handler = self._handlers.get("persona_update")
        if handler:
            handler(data)

    # ── Sending ───────────────────────────────────────────────────────────────

    def send(self, message: dict) -> bool:
        """
        Send a JSON message to the central server.
        Thread-safe — callable from any module thread.
        Returns True on success.
        """
        if self._ws is None or not self._connected.is_set():
            logger.warning("[WS Server] Cannot send — server not connected yet.")
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps(message)),
                self._loop,
            )
            future.result(timeout=5)
            return True
        except Exception as e:
            logger.error(f"[WS Server] Send error: {e}")
            return False

    # ── Status ────────────────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def wait_for_server(self, timeout: float = None) -> bool:
        """Block until the central server connects (or timeout). Returns True if connected."""
        return self._connected.wait(timeout=timeout)

    # ── Registration with central server ─────────────────────────────────────

    def register_with_server(self, ip_address: str = None) -> bool:
        """
        HTTP POST to the central server's /robots/register endpoint.
        Tells the server: "I exist at this IP/port, come connect to me."
        """
        url = f"{self.server_url}/robots/register"
        payload = {
            "client_id": self.client_id,
            "robot_name": self.client_config.get("robot_name", "Robot"),
            "robot_role": self.client_config.get("robot_role", "You are a helpful robot."),
            "allowed_tags": self.client_config.get("allowed_tags", ["[DEFAULT]"]),
            "modules": self.client_config.get("modules", ["gpt"]),
            "ip_address": ip_address or self.client_config.get("ip_address"),
            "ws_port": self.ws_port,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"[Registration] Registered with server: {resp.json().get('message', 'OK')}")
                return True
            else:
                logger.warning(f"[Registration] Server returned {resp.status_code} — {resp.text}")
                return False
        except requests.ConnectionError:
            logger.warning(
                f"[Registration] Could not reach server at {self.server_url}. "
                "Will retry on next start. Robot is still listening for connections."
            )
            return False
        except Exception as e:
            logger.warning(f"[Registration] Unexpected error: {e}")
            return False


# ── BasicClient — identical public interface to v5.x ─────────────────────────

class BasicClient:
    """
    Main client class. Manages modules and server communication.

    Public interface unchanged from v5.x so all Input/Output modules
    work without modification.
    """

    def __init__(self, config_file: str = "client_config.json"):
        self.config = self._load_config(config_file)
        if not self.config:
            raise RuntimeError(f"Failed to load config from {config_file}")

        ws_port = self.config.get("ws_port", 8765)
        server_url = self.config.get("server_url", "http://localhost:5000")

        self.server_connection = ServerConnection(ws_port, server_url, self.config)

        # Register default handlers — subclasses can override by calling
        # self.server_connection.register_handler() with their own callbacks
        self.server_connection.register_handler("chat_response",    self._default_chat_handler)
        self.server_connection.register_handler("chat_sentence",    self._on_chat_sentence)
        self.server_connection.register_handler("speech_response",  self._default_speech_handler)
        self.server_connection.register_handler("emotion_update",   self._default_emotion_handler)
        self.server_connection.register_handler("demo_step",        self._on_demo_step)
        self.server_connection.register_handler("tts_stop",         self._on_tts_stop)

        self.input_modules:  Dict[str, InputModule]  = {}
        self.output_modules: Dict[str, OutputModule] = {}

        self.running = False

        # Events used by edge_tts_output and voice_input (interface unchanged)
        self.is_speaking       = threading.Event()
        self.tts_started_event = threading.Event()

        logger.info(f"[Client] {self.config.get('robot_name', 'Robot')} initialising")
        logger.info(f"         ID      : {self.config.get('client_id')}")
        logger.info(f"         Server  : {server_url}")
        logger.info(f"         WS port : {ws_port}")
        logger.info(f"         Modules : {', '.join(self.config.get('modules', []))}")

    # ── Default server event handlers ─────────────────────────────────────────

    def _default_chat_handler(self, data: dict):
        """Called when server pushes a chat_response event."""
        self.process_server_response(data, "chat")

    def _default_speech_handler(self, data: dict):
        """Called when server pushes a speech_response event."""
        self.process_server_response(data, "speech")

    def _default_emotion_handler(self, data: dict):
        """Called when server pushes an emotion_update event. Override if needed."""
        pass

    def _on_demo_step(self, data: dict):
        """
        Handle a demo_step event pushed by the DemoOrchestrator.

        When generate=False (default): text goes straight to TTS (verbatim).
        When generate=True: text is treated as a prompt — the LLM generates
        the actual speech, building up natural conversation history across steps.
        This method BLOCKS until TTS completes, then sends ACK.
        """
        step_id  = data.get("step_id", "")
        text     = data.get("text", "")
        need_ack = data.get("require_ack", True)

        if not text:
            if need_ack:
                self.send_ack(step_id)
            return

        logger.info(f"[Demo] Step '{step_id}': {text[:60]}{'...' if len(text) > 60 else ''}")

        # Extract emotion tag for hardware
        match = re.search(r"\[(.*?)\]", text)
        if match:
            self.on_emotion_detected(match.group(1))

        # Clear any leftover chat_sentence items so this demo step plays immediately.
        # Only for require_ack steps — fire-and-forget steps can follow chat items.
        if need_ack:
            for module in self.output_modules.values():
                if hasattr(module, 'clear_non_callback_items'):
                    try:
                        module.clear_non_callback_items()
                    except Exception as e:
                        logger.warning(f"[Demo] clear_non_callback_items error: {e}")

        # Per-step completion event — set inside _tts_worker.finally after playback
        tts_done = threading.Event()

        def _on_tts_done():
            logger.info(f"[Demo] TTS completed for '{step_id}'")
            tts_done.set()

        # Route to the FIRST TTS output module only — break after the first speak
        # to prevent any second module from speaking the same text again.
        spoken = False
        for name, module in self.output_modules.items():
            try:
                if hasattr(module, "speak_with_callback"):
                    module.speak_with_callback(
                        text,
                        callback=_on_tts_done if need_ack else None,
                    )
                    spoken = True
                    break   # one robot, one voice — stop after first TTS module
            except Exception as e:
                logger.error(f"[Demo] TTS module '{name}' error: {e}")

        if not need_ack:
            return

        if not spoken:
            logger.warning(f"[Demo] No TTS module for step '{step_id}' — ACK now")
            self.send_ack(step_id)
            return

        # Block this handler thread until _on_tts_done fires (or hard timeout)
        completed = tts_done.wait(timeout=120)
        if not completed:
            logger.warning(f"[Demo] TTS wait timed out for '{step_id}' — ACK anyway")

        self.send_ack(step_id)

    def _on_chat_sentence(self, data: dict):
        """Handle a streamed sentence — dispatch emotion and feed to TTS immediately."""
        clean = data.get("text", "").strip()
        emotion = data.get("emotion_tag", "")
        if emotion:
            self.on_emotion_detected(emotion)
        if clean:
            for module in self.output_modules.values():
                try:
                    module.process_output({"text": clean})
                except Exception as e:
                    logger.error(f"[Modules] chat_sentence output error: {e}")

    def _on_tts_stop(self, _data: dict):
        """Stop in-progress TTS immediately — called by server during QA interrupt."""
        for module in self.output_modules.values():
            if hasattr(module, "interrupt"):
                module.interrupt()

    def send_ack(self, step_id: str):
        """Send an ACK packet to the server for the given demo step."""
        sent = self.server_connection.send({"type": "ack", "step_id": step_id})
        if sent:
            logger.info(f"[Demo] ACK sent for step '{step_id}'.")
        else:
            logger.warning(f"[Demo] Could not send ACK for '{step_id}' — not connected.")

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

    # ── Sending data to server (interface unchanged) ──────────────────────────

    def send_to_server(self, data_type: str, data: Any) -> Optional[dict]:
        """
        Send data to the central server over the persistent WebSocket.

        Called by Input modules:
            self.client.send_to_server('chat',   message_string)
            self.client.send_to_server('speech', wav_bytes)
            self.client.send_to_server('frame',  base64_string_or_dict)

        Returns None — responses arrive asynchronously via registered handlers.
        Modules that called process_server_response(response) will get None
        passed in, which returns early gracefully.
        """
        if data_type == "chat":
            self.server_connection.send({
                "type": "chat",
                "message": data,
            })

        elif data_type == "speech":
            audio_b64 = base64.b64encode(data).decode("utf-8")
            self.server_connection.send({
                "type": "speech",
                "audio": audio_b64,
            })

        elif data_type == "frame":
            # frame_data may be a base64 string or a dict with 'color' key (RealSense)
            if isinstance(data, dict):
                frame_b64 = data.get("color", data.get("frame", ""))
            else:
                frame_b64 = data
            self.server_connection.send({
                "type": "image_frame",
                "frame": frame_b64,
            })

        else:
            logger.warning(f"[Client] Unknown data_type: {data_type}")

        # Responses come via WS event handlers, not as return values
        return None

    # ── Processing server responses (interface unchanged) ─────────────────────

    def process_server_response(self, response_data: Optional[dict], response_type: str = "chat"):
        """
        Process a response dict from the server and route to output modules.
        Called either by Input modules (with None) or by WS event handlers (with data).
        """
        if not response_data:
            return

        response_text = response_data.get("response", "")
        if not response_text:
            # speech_response may only have transcription (no LLM response)
            transcription = response_data.get("transcription")
            if transcription:
                logger.info(f"[STT] Transcribed: '{transcription}'")
            return

        logger.info(f"[Server] {response_text[:80]}{'...' if len(response_text) > 80 else ''}")

        # Extract and dispatch emotion tag
        match = re.search(r"\[(.*?)\]", response_text)
        if match:
            self.on_emotion_detected(match.group(1))

        self.tts_started_event.clear()

        # Route to all output modules
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
        """
        Hook called when an emotion tag is found in a server response.
        Override in subclasses to send commands to hardware (e.g. Arduino).
        """
        pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Start the client:
          1. Register with server via HTTP (tell it our IP + WS port)
          2. Start the WebSocket server (listen for incoming connection)
          3. Start all registered modules
        """
        try:
            # 1. Tell the server where to find us
            self.server_connection.register_with_server()

            # 2. Start WS server — server will connect when operator clicks "Connect"
            self.server_connection.start_server()

            self.running = True

            # 3. Start input modules
            for name, module in self.input_modules.items():
                try:
                    module.start()
                    logger.info(f"[Modules] Started input '{name}'")
                except Exception as e:
                    logger.error(f"[Modules] Error starting input '{name}': {e}")

            # 4. Start output modules
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
        all_modules = (
            list(self.input_modules.values()) +
            list(self.output_modules.values())
        )
        for module in all_modules:
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
            config["_config_file"] = config_file  # store path for persona updates
            logger.info(f"[Client] Config loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"[Client] Config file not found: {config_file}")
            return None
        except Exception as e:
            logger.error(f"[Client] Config load error: {e}")
            return None