"""
run.py — Navel client for v6.0.0 multi-robot lab demo server
=============================================================
Self-contained client: runs a WebSocket server that the central server
dials out to (v6.0.0 inverted architecture), and speaks through the Navel
robot's onboard SDK.

NOTE: this entrypoint is intentionally NOT named `navel.py` — the Navel SDK
package is named `navel`, and a `navel.py` in this folder would shadow it
(causing `import navel` to fail with a circular import).

Run (on the Navel robot):
    python3 run.py

Architecture:
    Central Server  ──connects to──►  Navel WS server (this process)
    Navel WS server ──registers via HTTP──►  Central Server /robots/register

Participates in the CARES lab demo exactly like ChatBox: it receives demo_step
events from the DemoOrchestrator, speaks them via NavelTTSOutputModule, and ACKs
when done. Q&A windows route visitor speech (STT) → LLM → TTS automatically.
"""

import os
import re
import sys
import time
import logging
from typing import Optional

# Self-contained: import the local client.py + modules bundled in this folder
# (this folder is deployed standalone to the Navel robot — no ../client dependency).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from client import BasicClient

from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.navel_tts_output import NavelTTSOutputModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NavelClient(BasicClient):
    """
    Navel client for the v6.0.0 multi-robot lab demo server.

    Extends BasicClient with:
      - Navel SDK TTS (robot.say) via NavelTTSOutputModule
      - persona_update → live voice-config update
    Inherits the base demo_step → TTS → ACK flow unchanged (do NOT override it),
    so Navel stays in sync with the DemoOrchestrator just like ChatBox.
    """

    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        # NOTE: IP auto-detection ("power-on and ready") lives in BasicClient —
        # register_with_server() fills in a real LAN IP when the config has a
        # placeholder, so no per-robot handling is needed here.
        self._setup_modules()
        self._register_event_handlers()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _register_event_handlers(self):
        # Note: demo_step stays bound to the inherited BasicClient._on_demo_step.
        self.server_connection.register_handler("chat_response",   self._on_chat_response)
        self.server_connection.register_handler("speech_response", self._on_speech_response)
        self.server_connection.register_handler("persona_update",  self._on_persona_update)
        logger.info("[Navel] Event handlers registered")

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

        tts = self.output_modules.get("tts_output")
        if tts and clean_text:
            tts.process_output(clean_text)

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
            tts = self.output_modules.get("tts_output")
            if tts and hasattr(tts, "update_voice_config"):
                tts.update_voice_config(voice_config)

        capabilities = data.get("capabilities", {})
        if capabilities:
            active = [k for k, v in capabilities.items() if v]
            if active:
                logger.info(f"[Persona] Active capabilities: {', '.join(active)}")

    # ── Emotion hook ──────────────────────────────────────────────────────────

    def on_emotion_detected(self, emotion_tag: str):
        """Navel has no Arduino hardware — log only.
        TODO: map emotion tags → Navel facial expressions via the SDK."""
        logger.info(f"[Emotion] {emotion_tag.strip().upper()}")

    # ── Module setup ──────────────────────────────────────────────────────────

    def _setup_modules(self):
        # ── Voice input ───────────────────────────────────────────────────────
        if "speech" in self.config.get("modules", []):
            voice_cfg = self.config.get("voice_config", {})
            voice = VoiceInputModule("voice_input", voice_cfg)
            self.register_input_module(voice)

        # ── Console output ────────────────────────────────────────────────────
        console = ConsoleOutputModule("console_output", self.config.get("console_config", {}))
        self.register_output_module(console)

        # ── TTS output (Navel SDK) ────────────────────────────────────────────
        tts_cfg = self.config.get("tts_config", {})
        navel_tts = NavelTTSOutputModule("tts_output", tts_cfg)
        if not self.register_output_module(navel_tts):
            logger.warning("[Setup] Navel TTS failed to register — check the navel SDK")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _print_startup_info(self):
        print("\n" + "=" * 60)
        print(f"  {self.config.get('robot_name', 'Navel')} — Online")
        print("=" * 60)
        print(f"  ID       : {self.config.get('client_id')}")
        print(f"  Server   : {self.config.get('server_url')}")
        print(f"  WS port  : {self.config.get('ws_port')}")
        print(f"  IP       : {self.config.get('ip_address')}")
        print(f"  Modules  : {', '.join(self.config.get('modules', []))}")
        print()
        print("  Input modules :")
        for n in self.input_modules:
            print(f"    {n}")
        print("  Output modules:")
        for n in self.output_modules:
            print(f"    {n}")
        print()
        print("  To connect from the server:")
        print(f"    curl -X POST {self.config.get('server_url')}/robots/{self.config.get('client_id')}/connect")
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

            self._print_startup_info()

            logger.info("[Navel] Running — press Ctrl+C to stop")
            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("[Navel] Ctrl+C received")
        except Exception as e:
            logger.error(f"[Navel] Runtime error: {e}", exc_info=True)
        finally:
            self.stop()


def main():
    try:
        client = NavelClient("client_config.json")
        client.run()
        return 0
    except FileNotFoundError:
        print("Error: client_config.json not found in navel_client/")
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
        return 0
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
