"""
chatbox.py — ChatBox client for v6.0.0 multi-robot lab demo server
===================================================================
Self-contained client: runs a WebSocket server that the central server
dials out to (v6.0.0 inverted architecture).

Run:
    python3 chatbox.py

Architecture:
    Central Server  ──connects to──►  ChatBox WS server (this process)
    ChatBox WS server  ──registers via HTTP──►  Central Server /robots/register
"""

import os
import re
import sys
import logging
import subprocess
import time
import threading
from typing import Optional

# Self-contained: import the local client.py + modules bundled in this folder
# (this folder is deployed standalone to the ChatBox robot — no ../client dependency).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule
from OutputModules.arduino_output import ArduinoOutputModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Maps LLM emotion tags to ESP32 command strings
EMOTION_MAP = {
    "GREETING":    "greeting",
    "WAVE":        "wave",
    "POINT":       "point",
    "CONFUSED":    "confused",
    "SHRUG":       "shrug",
    "ANGRY":       "angry",
    "SAD":         "sad",
    "SLEEP":       "sleep",
    "DEFAULT":     "default",
    "POSE":        "pose",
    "HAPPY":       "greeting",
    "FEAR":        "sad",
    "SURPRISE":    "confused",
    "NEUTRAL":     "default",
    "HANDS_CLAP":  "hands_clap",
    "EARS_WIGGLE": "ears_wiggle",
}


class ChatBoxClient(BasicClient):
    """
    ChatBox client for the v6.0.0 multi-robot lab demo server.

    Extends BasicClient with:
      - Arduino TCP/Serial output (gesture control via ESP32)
      - Emotion tag → Arduino command mapping
      - Sentence-by-sentence TTS for natural conversational flow
      - Dynamic USB speaker detection
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        self.arduino_module: Optional[ArduinoOutputModule] = None
        self._setup_modules()
        self._register_event_handlers()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _register_event_handlers(self):
        self.server_connection.register_handler("chat_response",   self._on_chat_response)
        self.server_connection.register_handler("speech_response", self._on_speech_response)
        self.server_connection.register_handler("persona_update",  self._on_persona_update)

    def _on_chat_response(self, data: dict):
        response_text = data.get("response", "")
        if not response_text:
            return

        match = re.search(r"\[(.*?)\]", response_text)
        emotion = match.group(1) if match else None
        clean_text = re.sub(r"\[.*?\]", "", response_text).strip()

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(response_text)

        tts = self.output_modules.get("edge_tts_output")
        if not tts:
            if emotion:
                self.on_emotion_detected(emotion)
            return

        sentences = self._split_sentences(clean_text)
        if not sentences:
            return

        # Fire emotion at the moment the first sentence starts playing
        start_cb = (lambda e: lambda: self.on_emotion_detected(e))(emotion) if emotion else None
        tts.process_output_synced(sentences[0], start_callback=start_cb)

        for sentence in sentences[1:]:
            tts.process_output(sentence)

    def _on_speech_response(self, data: dict):
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] '{transcription}'")

        if data.get("response"):
            self._on_chat_response(data)

    def _on_persona_update(self, data: dict):
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] Switching to: '{persona_name}'")

        voice_config = data.get("voice_config", {})
        if voice_config:
            tts = self.output_modules.get("edge_tts_output")
            if tts and hasattr(tts, "update_voice_config"):
                tts.update_voice_config(voice_config)

        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(f"Persona updated to {persona_name}.")

    # ── Emotion → Arduino ─────────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        tag = emotion_tag.strip().upper()
        logger.info(f"[Emotion] {tag}")
        command = EMOTION_MAP.get(tag, "default")
        if self.arduino_module and self.arduino_module.is_connected():
            self.arduino_module.send_command(command)
        else:
            logger.debug(f"[Arduino] Skipped '{command}' — not connected")

    # ── Module setup ──────────────────────────────────────────────────────────

    def _detect_usb_speaker(self) -> Optional[str]:
        """Find the USB speaker's ALSA card ID. Returns 'plughw:N,0' or None."""
        try:
            output = subprocess.check_output(['aplay', '-l'], text=True)
            for line in output.split('\n'):
                if line.startswith("card") and ("UACDemoV10" in line or "USB Audio" in line):
                    match = re.search(r"card (\d+):", line)
                    if match:
                        card_num = match.group(1)
                        logger.info(f"[Setup] Auto-detected USB speaker on card {card_num}")
                        return f"plughw:{card_num},0"
        except Exception as e:
            logger.warning(f"[Setup] Speaker detection failed: {e}")
        return None

    def _setup_modules(self):
        # Auto-detect USB speaker and inject into edge_tts_config
        speaker_device = self._detect_usb_speaker()
        if speaker_device:
            if "edge_tts_config" not in self.config:
                self.config["edge_tts_config"] = {}
            self.config["edge_tts_config"]["audio_cmd"] = ["aplay", "-D", speaker_device]

        # ── Voice input ───────────────────────────────────────────────────────
        if "speech" in self.config.get("modules", []):
            voice_cfg = self.config.get("voice_config", {})
            voice = VoiceInputModule("voice_input", voice_cfg)
            self.register_input_module(voice)

        # ── Console output ────────────────────────────────────────────────────
        console = ConsoleOutputModule("console_output", self.config.get("console_config", {}))
        self.register_output_module(console)

        # ── TTS output ────────────────────────────────────────────────────────
        edge_cfg = self.config.get("edge_tts_config", {})
        edge = EdgeTTSOutputModule("edge_tts_output", edge_cfg)
        if not self.register_output_module(edge):
            logger.warning("[Setup] EdgeTTS failed to register")

        # ── Arduino output ────────────────────────────────────────────────────
        if self.config.get("features", {}).get("arduino_integration", True):
            arduino_cfg = self.config.get("arduino_output", {})
            self.arduino_module = ArduinoOutputModule("arduino_output", arduino_cfg)
            self.arduino_module.on_connected        = lambda: logger.info("[Arduino] Connected")
            self.arduino_module.on_disconnected     = lambda: logger.warning("[Arduino] Disconnected")
            self.arduino_module.on_connection_error = lambda e: logger.error(f"[Arduino] Error: {e}")
            if not self.register_output_module(self.arduino_module):
                logger.warning("[Setup] Arduino failed to register")
                self.arduino_module = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in parts if len(s.strip()) > 2]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _print_startup_info(self):
        print("\n" + "=" * 60)
        print(f"  {self.config.get('robot_name', 'ChatBox')} — Online")
        print("=" * 60)
        print(f"  ID       : {self.config.get('client_id')}")
        print(f"  Server   : {self.config.get('server_url')}")
        print(f"  WS port  : {self.config.get('ws_port')}")
        print(f"  Modules  : {', '.join(self.config.get('modules', []))}")
        print()
        print("  Input modules :")
        for n in self.input_modules:
            print(f"    {n}")
        print("  Output modules:")
        for n in self.output_modules:
            print(f"    {n}")
        if self.arduino_module:
            host = self.arduino_module.config.get("host", "?")
            port = self.arduino_module.config.get("port", 8888)
            status = "connected" if self.arduino_module.is_connected() else "connecting..."
            print(f"  ESP32/Arduino : {host}:{port} ({status})")
        tts_cfg = self.config.get("edge_tts_config", {})
        if "audio_cmd" in tts_cfg:
            print(f"  Audio output  : {' '.join(tts_cfg['audio_cmd'])}")
        print("=" * 60 + "\n")

    def run(self):
        try:
            if not self.start():
                return

            logger.info("[Startup] Waiting for central server connection (15s)...")
            if self.server_connection.wait_for_server(timeout=15):
                logger.info("[Startup] Server connected.")
            else:
                logger.warning("[Startup] Server not connected yet — will retry in background.")

            if self.arduino_module:
                logger.info("[Startup] Waiting for ESP32 connection (15s)...")
                for _ in range(15):
                    if self.arduino_module.is_connected():
                        break
                    time.sleep(1)
                else:
                    logger.warning("[Startup] ESP32 not connected yet — will retry in background.")

            self._print_startup_info()

            logger.info("[ChatBox] Running — press Ctrl+C to stop")
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("[ChatBox] Ctrl+C received")
        except Exception as e:
            logger.error(f"[ChatBox] Runtime error: {e}", exc_info=True)
        finally:
            self.stop()


def main():
    try:
        client = ChatBoxClient("client_config.json")
        client.run()
        return 0
    except FileNotFoundError:
        print("Error: client_config.json not found in chatbox_client/")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
