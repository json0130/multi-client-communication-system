# OutputModules/navel_tts_output.py — Navel SDK text-to-speech
#
# Speaks via the Navel robot's onboard SDK (`robot.say()`), which is NON-BLOCKING.
# Because robot.say() returns immediately, we estimate the speech duration from the
# word count so the demo ACK (fired via the _callback) lands after Navel has actually
# finished speaking — keeping the orchestrator in sync. update_voice_config() lets the
# server retune voice live via persona_update.
#
# NOTE on `import navel`:
#   The Navel SDK needs the NAVEL_SHM_LIB env var pointing at its shared-memory .so
#   *before* it is imported. We set a sensible default here (overridable by exporting
#   NAVEL_SHM_LIB yourself, or via the tts_config "navel_shm_lib" key) so the deployed
#   folder works without remembering to export it manually.
import os
import glob
import asyncio
import threading
import logging
import re

logger = logging.getLogger(__name__)


def _resolve_navel_shm_lib() -> str:
    """Return a path to libnavel_shm*.so for the NAVEL_SHM_LIB env var."""
    # 1. Respect an already-exported value.
    existing = os.environ.get("NAVEL_SHM_LIB")
    if existing:
        return existing

    # 2. Known default location on the Navel robot (aarch64 / cpython-3.11).
    default = "/usr/lib/python3/dist-packages/navel/libnavel_shm.cpython-311-aarch64-linux-gnu.so"
    if os.path.exists(default):
        return default

    # 3. Fallback: glob for any libnavel_shm*.so inside the installed navel package
    #    (covers other Python versions / architectures).
    for base in ("/usr/lib/python3/dist-packages/navel",
                 "/usr/local/lib/python3*/dist-packages/navel"):
        matches = glob.glob(os.path.join(base, "libnavel_shm*.so"))
        if matches:
            return matches[0]

    # 4. Nothing found — return the default so the import error is explicit.
    return default


# Must be set BEFORE `import navel`.
os.environ.setdefault("NAVEL_SHM_LIB", _resolve_navel_shm_lib())
logger.info(f"[NavelTTS] NAVEL_SHM_LIB = {os.environ.get('NAVEL_SHM_LIB')}")

import navel  # noqa: E402  (must follow the NAVEL_SHM_LIB setup above)
from client import OutputModule  # noqa: E402

class NavelTTSOutputModule(OutputModule):
    def __init__(self, name="tts_output", config=None):
        super().__init__(name, config)
        self._text_queue = None
        self._loop = None

        # Default Persona Parameters (Based on Navel SDK Tags)
        self.volume = 60      # Lowered default volume (Range: 20 to 400)
        self.pitch = 100      # 100% default pitch
        self.speed = 100      # 100% default speed
        self.language = "en1" # Default English

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        threading.Thread(target=self._run_navel_loop, daemon=True, name="navel-tts").start()
        return True

    def stop(self):
        if self._loop and self._text_queue:
            asyncio.run_coroutine_threadsafe(self._text_queue.put(None), self._loop)

    def _run_navel_loop(self):
        try:
            navel.run(self._navel_coroutine)
        except Exception as e:
            logger.error(f"[NavelTTS] Fatal error: {e}")

    async def _navel_coroutine(self, robot: navel.Robot):
        self._loop = asyncio.get_running_loop()
        self._text_queue = asyncio.Queue()
        logger.info("[NavelTTS] Ready and waiting for text.")

        while True:
            data = await self._text_queue.get()
            if data is None: break

            text = data.get("text", "")

            # Apply dynamic Persona parameters
            # If the server sent new parameters in this payload, update our state
            if "volume" in data: self.volume = data["volume"]
            if "pitch" in data:  self.pitch  = data["pitch"]
            if "speed" in data:  self.speed  = data["speed"]
            if "lang" in data:   self.language = data["lang"]

            # Construct the Navel TTS Tag string
            tag_prefix = f"<vol,{self.volume}><rpit,{self.pitch}><rspd,{self.speed}><lang,{self.language}>"
            tagged_text = tag_prefix + text

            logger.info(f"[NavelTTS] Speaking: '{text}' (Vol: {self.volume}, Pitch: {self.pitch}%)")

            if self.client:
                self.client.is_speaking.set()

            # Estimate how long the robot will speak (robot.say() is non-blocking)
            word_count = len(text.split())
            words_per_sec = 2.5 * (self.speed / 100.0)
            speech_duration = max(0.5, word_count / words_per_sec)

            callback = data.get("_callback")
            try:
                robot.say(tagged_text)
                await asyncio.sleep(speech_duration)
            except Exception as e:
                logger.error(f"[NavelTTS] Speech error: {e}")
            finally:
                if self.client:
                    self.client.is_speaking.clear()
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"[NavelTTS] Callback error: {e}")

    def update_voice_config(self, voice_config: dict):
        """Apply a persona_update voice config (volume / pitch / speed / language)."""
        if not voice_config:
            return
        if "volume" in voice_config: self.volume = voice_config["volume"]
        if "pitch"  in voice_config: self.pitch  = voice_config["pitch"]
        if "speed"  in voice_config: self.speed  = voice_config["speed"]
        if "rate"   in voice_config: self.speed  = voice_config["rate"]
        if "lang"     in voice_config: self.language = voice_config["lang"]
        if "language" in voice_config: self.language = voice_config["language"]
        logger.info(f"[NavelTTS] Voice updated → vol={self.volume} pitch={self.pitch} speed={self.speed} lang={self.language}")

    def interrupt(self):
        """Drain pending TTS queue. Current utterance will finish (Navel SDK limitation)."""
        if self._text_queue and self._loop:
            async def _drain():
                while not self._text_queue.empty():
                    try:
                        self._text_queue.get_nowait()
                    except Exception:
                        break
            asyncio.run_coroutine_threadsafe(_drain(), self._loop)

    def clear_non_callback_items(self):
        """Remove pending chat_sentence items (no callback) from asyncio queue.
        Items with _callback (demo steps) are kept. Blocks until drain completes."""
        if not self._text_queue or not self._loop:
            return
        async def _drain_non_callback():
            keep = []
            while not self._text_queue.empty():
                try:
                    item = self._text_queue.get_nowait()
                    if item and item.get("_callback"):
                        keep.append(item)
                except Exception:
                    break
            for item in keep:
                await self._text_queue.put(item)
        future = asyncio.run_coroutine_threadsafe(_drain_non_callback(), self._loop)
        try:
            future.result(timeout=1)
        except Exception:
            pass

    def process_output(self, data) -> bool:
        if not self._text_queue or not self._loop:
            return False

        if isinstance(data, str):
            data = {"text": data}

        # strip any leading emotion tag so it is not spoken aloud
        text = data.get("text", "")
        if text:
            data = dict(data)
            data["text"] = re.sub(r'\[.*?\]', '', text).strip()

        if data.get("text"):
            asyncio.run_coroutine_threadsafe(self._text_queue.put(data), self._loop)
            return True
        return False

    def speak_with_callback(self, text: str, callback=None) -> bool:
        """
        Queue text for TTS and fire callback() after playback finishes.
        Used by BasicClient._on_demo_step() to send ACK after speech.
        """
        if not self._text_queue or not self._loop:
            if callback:
                callback()
            return False

        clean = re.sub(r'\[.*?\]', '', text).strip()
        if not clean:
            if callback:
                callback()
            return False

        asyncio.run_coroutine_threadsafe(
            self._text_queue.put({"text": clean, "_callback": callback}),
            self._loop,
        )
        return True
