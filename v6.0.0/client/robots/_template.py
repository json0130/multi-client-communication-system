"""
robots/_template.py — start here for a new robot platform
=========================================================
1. Copy this file to robots/<name>.py (or robots/<name>/__init__.py when the
   robot needs more files, like robots/navel/).
2. Fill in say() and on_emotion_detected() below; delete what you don't need.
3. Write configs/<name>.json: copy configs/generic.json and set client_id,
   robot_name, server_url, robot_role and allowed_tags (the tags this body can
   perform, e.g. "[WAVE]"; the server only asks for those).
4. python3 run.py <name>

The shared client (client.py) already does all of the protocol: registering,
the WebSocket (server dials in, or "connect_to_server": true to dial out),
streamed sentences in order, demo steps and their ACKs, barge-in (tts_stop),
tts_state, and persona_update (saved to the config file, voice retuned through
update_voice_config()). Tags are stripped before anything is spoken.

Need nothing robot-specific at all (speakers + mic on the machine)? Don't write
a module: a config is enough, robots/generic.py runs it (like iRobi).
"""

import logging

from client import BasicClient
from InputModules.voice_input import VoiceInputModule
from OutputModules.console_output import ConsoleOutputModule
from OutputModules.sim_tts_output import SimTTSOutputModule

logger = logging.getLogger(__name__)


class Speech(SimTTSOutputModule):
    """The robot's voice. Everything but say() is inherited: the queue, demo-step
    ACK callbacks, interrupt() with the unspoken remainder, tts_state."""

    def initialize(self) -> bool:
        # Connect to the robot's speech SDK here; return False if it isn't there.
        return True

    def say(self, sentence: str):
        # Speak one sentence and return once it has been said.
        # Blocking SDK:      sdk.say(sentence)
        # Non-blocking SDK:  sdk.say(sentence); self._interrupt_event.wait(self.estimated_duration(sentence))
        super().say(sentence)          # placeholder: logs it and waits as long as saying it would take

    def update_voice_config(self, voice_config: dict):
        # persona_update: voice, rate, pitch... as the robot's SDK takes them.
        logger.info(f"[Speech] voice config: {voice_config}")


class Robot(BasicClient):

    def __init__(self, config_file: str):
        super().__init__(config_file)
        if "speech" in self.config.get("modules", []):
            # The machine's own mic. A robot with its own mic array: an InputModule that
            # calls self.client.send_to_server("speech", wav_bytes) (see robots/navel/voice_input.py).
            self.register_input_module(VoiceInputModule("voice_input", self.config.get("voice_config", {})))
        self.register_output_module(ConsoleOutputModule("console_output", self.config.get("console_config", {})))
        self.register_output_module(Speech("speech", self.config.get("speech_config", {})))

    def on_emotion_detected(self, emotion_tag: str):
        # A tag the server sent, e.g. "WAVE": perform it. Unknown tags: ignore.
        logger.info(f"[Gesture] {emotion_tag.strip().upper()}")
