# robot.py — Robot client (v6)
import sys
import re
import logging

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule

logger = logging.getLogger(__name__)


class SimpleConcurrentClient(BasicClient):
    """
    Robot client (v6) — verbal interaction only (no Arduino hardware).
    Handles persona_update event from server → updates TTS voice config at runtime.
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        self.setup_all_modules()
        self._register_custom_event_handlers()

    # ── Emotion hook ──────────────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        """Log emotion tag — extend this if you add hardware later."""
        logger.info(f"[Emotion] {emotion_tag.strip().upper()}")

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

    def on_speech_response(self, data: dict):
        transcription = data.get("transcription", "")
        if transcription:
            logger.info(f"[STT] '{transcription}'")
        if data.get("response"):
            self.on_chat_response(data)

    def on_persona_update(self, data: dict):
        """
        Called when the server assigns a new persona to this robot.
        Updates TTS voice config live — no restart needed.
        """
        persona_name = data.get("persona_name", "Unknown")
        logger.info(f"[Persona] Switching to: '{persona_name}'")

        voice_config = data.get("voice_config", {})
        if voice_config:
            tts = self.output_modules.get("edge_tts_output")
            if tts and hasattr(tts, "update_voice_config"):
                tts.update_voice_config(voice_config)
                logger.info(f"[Persona] TTS voice updated: {voice_config}")

        capabilities = data.get("capabilities", {})
        if capabilities:
            active = [k for k, v in capabilities.items() if v]
            if active:
                logger.info(f"[Persona] Active capabilities: {', '.join(active)}")

        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(f"Persona updated to {persona_name}.")

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

        # ── OUTPUT: Console ───────────────────────────────────────────────────
        logger.info("[Setup] Console output...")
        console = ConsoleOutputModule("console_output", self.config.get("console_config", {}))
        self.register_output_module(console)
        console.start()

        # ── OUTPUT: TTS ───────────────────────────────────────────────────────
        logger.info("[Setup] Edge TTS...")
        # Built from the ROBOT's own config, not from an "edge_tts_config"
        # block no config file has ever contained. That .get() defaulted every
        # robot to the same literal dict, so the per-robot `tts_voice` sitting
        # at the top level of pepper_01.json / chatbox_01.json / navel_01.json
        # / silbot_01.json never reached the TTS module at all: it read no
        # voice, fell through to gTTS — which has no voice selection — and all
        # four robots spoke in one voice. Reported twice.
        edge_cfg = dict(self.config.get("edge_tts_config") or {})
        edge_cfg.setdefault("rate", "+0%")
        edge_cfg.setdefault("pitch", "+0Hz")
        edge_cfg.setdefault("remove_emotion_tags", True)
        # Top-level keys win nothing they have not been given: each is copied
        # only if the robot actually set it, and only if edge_tts_config did
        # not already say something about it.
        for key in ("tts_voice", "audio_device", "audio_cmd", "sim_speed"):
            if self.config.get(key) is not None:
                edge_cfg.setdefault(key, self.config[key])
        edge_cfg.setdefault("tts_voice", edge_cfg.get("voice", ""))
        logger.info(f"[Setup] TTS voice: {edge_cfg.get('tts_voice') or 'gTTS default'}")
        edge = EdgeTTSOutputModule("edge_tts_output", edge_cfg)
        if self.register_output_module(edge):
            edge.start()
        else:
            logger.warning("[Setup] Edge TTS failed — check gtts/ffmpeg")

    # ── Startup info ──────────────────────────────────────────────────────────

    def print_startup_info(self):
        print("\n" + "=" * 60)
        print(f"  {self.config.get('robot_name', 'Robot')} — waiting for server to connect")
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
