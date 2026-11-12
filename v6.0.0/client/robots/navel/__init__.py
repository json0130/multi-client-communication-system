"""
robots/navel — Navel: speaks through the onboard SDK, listens on its ODAS mic array
===================================================================================
    python3 run.py navel          (on the robot; see navel-demo.service)

Everything else, including demo_step → TTS → ACK, is the shared client's, so
Navel stays in step with the DemoOrchestrator like every other robot.
Each sentence's gesture tag (picked server-side, Method B) is only logged here:
mapping it to a Navel SDK motion is still to do (on_emotion_detected).
"""

import logging

from client import BasicClient
from OutputModules.console_output import ConsoleOutputModule

from .tts_output import NavelTTSOutputModule
from .voice_input import VoiceInputModule

logger = logging.getLogger(__name__)


class Robot(BasicClient):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        if "speech" in self.config.get("modules", []):
            self.register_input_module(VoiceInputModule("voice_input", self.config.get("voice_config", {})))
        self.register_output_module(ConsoleOutputModule("console_output", self.config.get("console_config", {})))
        if not self.register_output_module(NavelTTSOutputModule("tts_output", self.config.get("tts_config", {}))):
            logger.warning("[Setup] Navel TTS failed to register — check the navel SDK")
