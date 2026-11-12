"""
robots/generic.py — a robot that talks and listens through the machine it runs on
==================================================================================
Microphone with the "speech" module (typed input when there is no PyAudio),
console, and edge-tts/gTTS speech. No body to move: tags are logged.

run.py uses this for any config that has no robot module of its own (iRobi,
SilBot, a laptop):  python3 run.py irobi
"""

import logging

from client import BasicClient
from InputModules.text_input import TextInputModule
from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule, edge_tts_config

logger = logging.getLogger(__name__)


class Robot(BasicClient):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        # No "speech" module = no input at all (run_lab_test.py runs four of these on one terminal).
        if "speech" in self.config.get("modules", []) and not self.register_input_module(
                VoiceInputModule("voice_input", self.config.get("voice_config", {}))):
            logger.info("[Setup] No microphone (PyAudio) — typed input instead")
            self.register_input_module(TextInputModule("text_input", {}))
        self.register_output_module(ConsoleOutputModule("console_output", self.config.get("console_config", {})))
        tts_cfg = edge_tts_config(self.config)
        logger.info(f"[Setup] TTS voice: {tts_cfg.get('tts_voice') or 'gTTS default'}")
        if not self.register_output_module(EdgeTTSOutputModule("edge_tts_output", tts_cfg)):
            logger.warning("[Setup] Edge TTS failed — check gtts/ffmpeg")

    def on_persona_update(self, data: dict):
        super().on_persona_update(data)
        tts = self.output_modules.get("edge_tts_output")
        if tts:
            tts.process_output(f"Persona updated to {data.get('persona_name', 'Unknown')}.")
