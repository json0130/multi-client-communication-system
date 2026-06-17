"""
irobi_client/irobi_client.py
============================
iRobi robot client for the v6.0.0 multi-client communication system.

Inputs  : text (keyboard) and/or voice (microphone via VAD)
Outputs : console display + Google TTS (gTTS + ffmpeg)

Run from this directory:
    python irobi_client.py

Requires client_config.json in the same directory.
"""

import sys
import os
import re
import logging

# ── Path setup ────────────────────────────────────────────────────────────────
# Insert _HERE first so local InputModules/ and OutputModules/ shadow the ones
# in ../client. Then add ../client so BasicClient (client.py) is findable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'client'))
sys.path.insert(0, _HERE)   # takes priority — local modules shadow client/

from client import BasicClient                              # noqa: E402
from InputModules.voice_input import VoiceInputModule      # noqa: E402
from InputModules.text_input import TextInputModule        # noqa: E402
from OutputModules.console_output import ConsoleOutputModule  # noqa: E402
from OutputModules.edge_tts_output import EdgeTTSOutputModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class iRobiClient(BasicClient):
    """
    iRobi robot client.

    Input  : VoiceInputModule (VAD + manual record + inline text)
             Falls back to TextInputModule when PyAudio is unavailable.
    Output : ConsoleOutputModule + EdgeTTSOutputModule
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        self._setup_modules()
        self._register_handlers()

    # ── Emotion hook ──────────────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        """Log emotion tags — extend here to trigger iRobi animations."""
        logger.info(f"[Emotion] {emotion_tag.strip().upper()}")

    # ── Event handlers ────────────────────────────────────────────────────────

    def _register_handlers(self):
        self.server_connection.register_handler("chat_response",   self._on_chat_response)
        self.server_connection.register_handler("speech_response", self._on_speech_response)
        self.server_connection.register_handler("persona_update",  self._on_persona_update)
        logger.info("[iRobi] Event handlers registered")

    def _on_chat_response(self, data: dict):
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

    def _on_speech_response(self, data: dict):
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] '{transcription}'")
        if data.get("response"):
            self._on_chat_response(data)

    def _on_persona_update(self, data: dict):
        """Apply new persona: update TTS voice config live — no restart needed."""
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] Switching to: '{persona_name}'")

        voice_config = data.get("voice_config", {})
        if voice_config:
            tts = self.output_modules.get("edge_tts_output")
            if tts and hasattr(tts, "update_voice_config"):
                tts.update_voice_config(voice_config)
                logger.info(f"[Persona] TTS voice updated: {voice_config}")

        if "console_output" in self.output_modules:
            self.output_modules["console_output"].process_output(
                f"[PERSONA] Switched to: {persona_name}"
            )

    # ── Module setup ──────────────────────────────────────────────────────────

    def _setup_modules(self):
        modules_enabled = self.config.get("modules", [])

        # ── INPUT: Voice (VAD + inline text) ─────────────────────────────────
        if "speech" in modules_enabled:
            logger.info("[Setup] Voice input (VAD + keyboard)...")
            voice_cfg = self.config.get("voice_config", {
                "sample_rate": 48000, "channels": 1,
                "input_device_index": None, "max_record_time": 30,
            })
            voice = VoiceInputModule("voice_input", voice_cfg)
            if voice.initialize():
                self.register_input_module(voice)
                voice.start()
                logger.info("[Setup] Voice input active")
            else:
                # PyAudio not available — fall back to text-only input
                logger.warning("[Setup] Voice input unavailable — falling back to text input")
                self._start_text_input()
        else:
            # Speech module not enabled — use text-only input
            self._start_text_input()

        # ── OUTPUT: Console ───────────────────────────────────────────────────
        logger.info("[Setup] Console output...")
        console = ConsoleOutputModule("console_output", self.config.get("console_config", {}))
        self.register_output_module(console)
        console.start()

        # ── OUTPUT: TTS ───────────────────────────────────────────────────────
        logger.info("[Setup] Edge TTS...")
        tts_cfg = self.config.get("edge_tts_config", {
            "language": "en", "gender": "female",
            "rate": "+0%", "remove_emotion_tags": True,
        })
        tts = EdgeTTSOutputModule("edge_tts_output", tts_cfg)
        if self.register_output_module(tts):
            tts.start()
        else:
            logger.warning("[Setup] Edge TTS failed — check gtts/ffmpeg installation")

    def _start_text_input(self):
        logger.info("[Setup] Text input...")
        text = TextInputModule("text_input", {})
        self.register_input_module(text)
        text.start()

    # ── Startup banner ────────────────────────────────────────────────────────

    def print_startup_info(self):
        cfg = self.config
        print("\n" + "=" * 60)
        print(f"  iRobi Client — waiting for server to connect")
        print("=" * 60)
        print(f"  Robot    : {cfg.get('robot_name', 'iRobi')}")
        print(f"  ID       : {cfg.get('client_id', 'irobi_001')}")
        print(f"  Server   : {cfg.get('server_url', 'not set')}")
        print(f"  WS port  : {cfg.get('ws_port', 8766)}")
        print(f"  IP       : {cfg.get('ip_address', 'not set')}")
        print(f"  Modules  : {', '.join(cfg.get('modules', []))}")
        print()
        print("  Input modules :")
        for n in self.input_modules:
            print(f"    - {n}")
        print("  Output modules:")
        for n in self.output_modules:
            print(f"    - {n}")
        print()
        print("  To connect from the server dashboard or:")
        print(f"    curl -X POST {cfg.get('server_url')}/robots/{cfg.get('client_id')}/connect")
        print("=" * 60 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    try:
        client = iRobiClient("client_config.json")
        client.print_startup_info()
        client.run()
        return 0
    except FileNotFoundError:
        print("Error: client_config.json not found in irobi_client/")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
