"""
robots/chatbox — ChatBox (Jetson): USB speaker, microphone, ESP32 gestures
==========================================================================
    python3 run.py chatbox

On top of the shared client:
  - tag → ESP32 gesture command (EMOTION_MAP), sent over TCP/serial by ArduinoOutputModule
  - a whole chat_response is spoken sentence by sentence, the gesture firing as the first one starts
  - the USB speaker's ALSA card is found at startup
"""

import logging
import re
import subprocess
from typing import Optional

from client import BasicClient
from InputModules.voice_input import VoiceInputModule
from OutputModules.arduino_output import ArduinoOutputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule

logger = logging.getLogger(__name__)


# Maps LLM emotion tags to ESP32 command strings
EMOTION_MAP = {
    "GREETING":    "greeting",
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


class Robot(BasicClient):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        self.arduino_module: Optional[ArduinoOutputModule] = None
        self._setup_modules()
        self.server_connection.register_handler("chat_response",   self._on_chat_response)
        self.server_connection.register_handler("speech_response", self._on_speech_response)

    # ── Event handlers ────────────────────────────────────────────────────────

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

    def on_persona_update(self, data: dict):
        super().on_persona_update(data)
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(f"Persona updated to {data.get('persona_name', 'Unknown')}.")

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
            self.config.setdefault("edge_tts_config", {})["audio_cmd"] = ["aplay", "-D", speaker_device]

        if "speech" in self.config.get("modules", []):
            self.register_input_module(VoiceInputModule("voice_input", self.config.get("voice_config", {})))

        self.register_output_module(ConsoleOutputModule("console_output", self.config.get("console_config", {})))

        # ponytail: the raw block, not edge_tts_config(): it would turn this block's "voice" into tts_voice and
        # switch ChatBox from the gTTS voice it has always had to en-US-AriaNeural. Change on purpose, live-tested.
        if not self.register_output_module(EdgeTTSOutputModule("edge_tts_output", self.config.get("edge_tts_config", {}))):
            logger.warning("[Setup] EdgeTTS failed to register")

        if self.config.get("features", {}).get("arduino_integration", True):
            self.arduino_module = ArduinoOutputModule("arduino_output", self.config.get("arduino_output", {}))
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
