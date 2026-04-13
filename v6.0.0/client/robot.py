# robot.py — ChatBox robot client (v6, Phase 3)
import sys
import os
import re
import json
import logging
from typing import Optional
import serial.tools.list_ports

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from InputModules.text_input import TextInputModule
from InputModules.realsense_input import RealSenseInputModule

from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule
from OutputModules.tts_output import PyttsxTTSOutputModule
from OutputModules.arduino_output import ArduinoOutputModule

logger = logging.getLogger(__name__)


class SimpleConcurrentClient(BasicClient):
    """
    ChatBox robot client (v6).
    Phase 3 addition: handles persona_update event from server
    → updates TTS voice config + Arduino emotion map at runtime.
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)

        self.emotion_map = {
            "GREETING": "greeting", "WAVE": "wave",   "POINT": "point",
            "CONFUSED": "confused", "SHRUG": "shrug", "ANGRY": "angry",
            "SAD": "sad",           "SLEEP": "sleep", "DEFAULT": "default",
            "POSE": "pose",         "IDLE": "idle",
        }

        self.setup_all_modules()
        self._register_custom_event_handlers()

    # ── Hardware emotion hook ─────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        """Overrides BasicClient hook — sends gesture command to Arduino."""
        valid = list(self.emotion_map.keys())
        clean = emotion_tag.strip().upper()
        if clean not in valid:
            logger.warning(f"[Arduino] Unknown tag '{clean}' — using DEFAULT")
            clean = "DEFAULT"
        command = self.emotion_map[clean]
        logger.info(f"[Arduino] Sending: {command}")
        if self.arduino_module:
            self.arduino_module.send_command(command)

    # ── WebSocket event handlers ──────────────────────────────────────────────

    def _register_custom_event_handlers(self):
        self.server_connection.register_handler("chat_response",   self.on_chat_response)
        self.server_connection.register_handler("speech_response", self.on_speech_response)
        self.server_connection.register_handler("persona_update",  self.on_persona_update)
        logger.info("[Client] Event handlers registered")

    def on_chat_response(self, data: dict):
        response_text = data.get("response", "")
        if not response_text:
            return

        match = re.search(r"\[(.*?)\]", response_text)
        if match:
            self.on_emotion_detected(match.group(1))

        clean_text = re.sub(r"\[.*?\]", "", response_text).strip()

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(response_text)

        if "edge_tts_output" in self.output_modules:
            self.output_modules["edge_tts_output"].process_output(clean_text)
        elif "pyttsx_tts" in self.output_modules:
            self.output_modules["pyttsx_tts"].process_output(clean_text)

    def on_speech_response(self, data: dict):
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] '{transcription}'")
        if data.get("response"):
            self.on_chat_response(data)

    def on_persona_update(self, data: dict):
        """
        Called when the server assigns a new persona to this robot.
        Updates:
          1. TTS voice config (language, gender, rate) — live, no restart
          2. Prints confirmation to console
        
        In-memory config + disk write is handled by client.py _apply_persona_update.
        This method handles the hardware-level side effects.
        """
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] ── Switching to: '{persona_name}' ──")

        # 1. Update TTS voice if voice_config provided
        voice_config = data.get("voice_config", {})
        if voice_config:
            tts = self.output_modules.get("edge_tts_output")
            if tts and hasattr(tts, "update_voice_config"):
                tts.update_voice_config(voice_config)
                logger.info(f"[Persona] TTS voice updated: {voice_config}")
            else:
                logger.warning("[Persona] Edge TTS not found — voice config not applied")

        # 2. Log capabilities for awareness
        capabilities = data.get("capabilities", {})
        if capabilities:
            active = [k for k, v in capabilities.items() if v]
            if active:
                logger.info(f"[Persona] Active capabilities: {', '.join(active)}")

        # 3. Announce the switch via TTS
        announcement = f"Persona updated to {persona_name}."
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(announcement)

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(
                f"[PERSONA] Switched to: {persona_name}"
            )

    # ── Module setup ──────────────────────────────────────────────────────────

    def setup_all_modules(self):

        # ── INPUT: Voice ──────────────────────────────────────────────────────
        if "speech" in self.config.get("modules", []):
            logger.info("[Setup] Voice input...")
            voice_config = self.config.get("voice_config", {
                "sample_rate": 48000, "channels": 1,
                "input_device_index": 11, "max_record_time": 30,
            })
            voice = VoiceInputModule("voice_input", voice_config)
            self.register_input_module(voice)
            voice.start()

        # ── INPUT: Camera (RealSense) ─────────────────────────────────────────
        if "emotion" in self.config.get("modules", []):
            logger.info("[Setup] RealSense camera...")
            cam_config = self.config.get("camera_config", {
                "width": 1280, "height": 720,
                "fps": 15, "send_fps": 5, "jpeg_quality": 85,
            })
            cam = RealSenseInputModule("camera_input", cam_config)
            if not self.register_input_module(cam):
                logger.warning("[Setup] RealSense failed — emotion disabled")

        # ── OUTPUT: Console ───────────────────────────────────────────────────
        logger.info("[Setup] Console output...")
        console = ConsoleOutputModule("console_output", self.config.get("console_config", {}))
        self.register_output_module(console)
        console.start()

        # ── OUTPUT: TTS ───────────────────────────────────────────────────────
        logger.info("[Setup] Edge TTS...")
        edge_cfg = self.config.get("edge_tts_config", {
            "voice": "en-US-AriaNeural", "rate": "+0%",
            "pitch": "+0Hz", "remove_emotion_tags": True,
        })
        edge = EdgeTTSOutputModule("edge_tts_output", edge_cfg)
        if self.register_output_module(edge):
            edge.start()
        else:
            logger.warning("[Setup] Edge TTS failed — check gtts/ffmpeg")

        # ── OUTPUT: Arduino ───────────────────────────────────────────────────
        if self.config.get("features", {}).get("arduino_integration", True):
            logger.info("[Setup] Arduino...")
            arduino_cfg = self.config.get("arduino_output", {})
            detected = self._detect_arduino_port()
            if detected:
                arduino_cfg["arduino_port"] = detected
                logger.info(f"[Setup] Arduino auto-detected on {detected}")
            else:
                logger.warning(f"[Setup] Using config port: {arduino_cfg.get('arduino_port')}")

            arduino_cfg.setdefault("arduino_baud", 115200)
            arduino_cfg.setdefault("auto_connect", True)

            self.arduino_module = ArduinoOutputModule("arduino_output", arduino_cfg)
            self.arduino_module.on_connected       = self._on_arduino_connected
            self.arduino_module.on_disconnected    = self._on_arduino_disconnected
            self.arduino_module.on_connection_error = self._on_arduino_error

            if self.register_output_module(self.arduino_module):
                logger.info("[Setup] Arduino registered")
            else:
                logger.warning("[Setup] Arduino failed to register")

    # ── Arduino helpers ───────────────────────────────────────────────────────

    def _detect_arduino_port(self) -> Optional[str]:
        known = ["CP210x", "CH340", "USB Serial", "Arduino"]
        for port in serial.tools.list_ports.comports():
            for k in known:
                if (port.description and k in port.description) or \
                   (port.manufacturer and k in port.manufacturer):
                    return port.device
        return None

    def _on_arduino_connected(self):    logger.info("[Arduino] Connected")
    def _on_arduino_disconnected(self): logger.warning("[Arduino] Disconnected")
    def _on_arduino_error(self, msg):   logger.error(f"[Arduino] Error: {msg}")

    # ── Startup info ──────────────────────────────────────────────────────────

    def print_startup_info(self):
        print("\n" + "=" * 60)
        print("  CHATBOX — waiting for central server to connect")
        print("=" * 60)
        print(f"  Robot    : {self.config.get('robot_name', 'Unknown')}")
        print(f"  ID       : {self.config.get('client_id', 'Unknown')}")
        print(f"  Server   : {self.config.get('server_url', 'Unknown')}")
        print(f"  WS port  : {self.config.get('ws_port', 8765)}")
        print(f"  IP       : {self.config.get('ip_address', 'not set')}")
        print(f"  Modules  : {', '.join(self.config.get('modules', []))}")
        print()
        print("  Input modules :")
        for n in self.input_modules:   print(f"    {n}")
        print("  Output modules:")
        for n in self.output_modules:  print(f"    {n}")
        print()
        print("  To connect from the server:")
        print(f"    curl -X POST {self.config.get('server_url')}/robots/{self.config.get('client_id')}/connect")
        print("=" * 60 + "\n")


def main():
    try:
        client = SimpleConcurrentClient("client_config.json")
        client.print_startup_info()
        client.run()
        return 0
    except FileNotFoundError:
        print("Error: client_config.json not found")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())