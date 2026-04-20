# modules/output/navel_tts_output.py
import asyncio
import threading
import logging
import re
import navel
from client import OutputModule

logger = logging.getLogger(__name__)

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

            callback = data.get("_callback")
            try:
                robot.say(tagged_text)
                await asyncio.sleep(0.5)
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

    def process_output(self, data: dict) -> bool:
        if not self._text_queue or not self._loop:
            return False

        if "text" in data:
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