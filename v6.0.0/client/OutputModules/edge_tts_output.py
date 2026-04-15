# OutputModules/edge_tts_output.py
import subprocess
import re
import threading
import queue
import logging
import os
import tempfile
from typing import Dict, Any
from client import OutputModule
from gtts import gTTS

logger = logging.getLogger(__name__)


class EdgeTTSOutputModule(OutputModule):
    """
    Google TTS output with runtime voice config update support.
    Call update_voice_config(dict) at any time — takes effect on the next utterance.
    """

    def __init__(self, name: str = "edge_tts_output", config: Dict = None):
        super().__init__(name, config)
        self.max_length   = self.config.get('max_length', 500)
        self.talking_speed = "1.25"

        # Voice settings — readable/writable at runtime
        self._voice_lock = threading.Lock()
        self._language   = self.config.get('language', 'en')
        self._gender     = self.config.get('gender', 'female')   # stored, informs future providers
        self._rate       = self.config.get('rate', '+0%')

        # Speaker output (hardcoded to USB speaker; override in config if needed)
        self._audio_cmd = self.config.get('audio_cmd', ['aplay', '-D', 'plughw:2,0'])

        self.tts_queue  = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()

    # ── BaseModule interface ───────────────────────────────────────────────────

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            return True
        return False

    def stop(self):
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self.tts_queue.put(None)
            if self.tts_thread:
                self.tts_thread.join(timeout=2)

    def process_output(self, data: Any) -> bool:
        if not self.enabled:
            return False
        try:
            text = data.get('text', '') if isinstance(data, dict) else str(data)
            text = self._prepare_text(text)
            if text and len(text.strip()) > 2:
                self.tts_queue.put((text, None))
                return True
            return False
        except Exception as e:
            logger.error(f"[TTS] Processing error: {e}")
            return False

    def speak_with_callback(self, text: str, callback=None) -> bool:
        """
        Queue text for TTS and fire callback() after playback finishes.
        Used by BasicClient._on_demo_step() to send ACK after speech.
        """
        if not self.enabled:
            if callback:
                callback()   # fire immediately so ACK isn't lost
            return False
        text = self._prepare_text(text)
        if text and len(text.strip()) > 2:
            self.tts_queue.put((text, callback))
            return True
        # Nothing to speak — fire callback right away
        if callback:
            callback()
        return False

    # ── Runtime voice update (called by robot.py on persona_update) ───────────

    def update_voice_config(self, voice_config: dict):
        """
        Update voice settings at runtime — takes effect on the next utterance.
        Safe to call from any thread.
        
        Accepted keys:
          language  : str  e.g. 'en', 'es', 'fr', 'ja'
          gender    : str  'female' | 'male'  (stored for future providers)
          rate      : str  e.g. '+0%', '+10%'
        """
        with self._voice_lock:
            if 'language' in voice_config:
                self._language = voice_config['language']
                logger.info(f"[TTS] Language → {self._language}")
            if 'gender' in voice_config:
                self._gender = voice_config['gender']
                logger.info(f"[TTS] Gender   → {self._gender}")
            if 'rate' in voice_config:
                self._rate = voice_config['rate']
                logger.info(f"[TTS] Rate     → {self._rate}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _prepare_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text)     # strip emotion tags
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[{}"]', '', text)
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0] + '...'
        return text

    def _tts_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.tts_queue.get(timeout=1)
                if item is None:
                    break
                # Items are always (text, callback) tuples
                text, callback = item if isinstance(item, tuple) else (item, None)
                self._speak_text(text)
                self.tts_queue.task_done()
                # Fire ACK callback AFTER playback completes
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"[TTS] Callback error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[TTS] Worker error: {e}")

    def _speak_text(self, text: str):
        # Snapshot voice settings for this utterance
        with self._voice_lock:
            language = self._language

        tmp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name

        try:
            tts = gTTS(text=text, lang=language)
            tts.save(tmp_mp3)

            subprocess.run([
                'ffmpeg', '-i', tmp_mp3,
                '-filter:a', f'atempo={self.talking_speed}',
                '-ar', '22050', '-ac', '1', '-sample_fmt', 's16', '-y', tmp_wav,
            ], capture_output=True, check=True)

            # Signal microphone mute
            if self.client:
                if not hasattr(self.client, 'is_speaking'):
                    self.client.is_speaking = threading.Event()
                self.client.is_speaking.set()
                if hasattr(self.client, 'tts_started_event'):
                    self.client.tts_started_event.set()

            subprocess.run(self._audio_cmd + [tmp_wav], capture_output=True)

        except Exception as e:
            logger.error(f"[TTS] Playback error: {e}")
        finally:
            if self.client and hasattr(self.client, 'is_speaking'):
                self.client.is_speaking.clear()
            for f in [tmp_mp3, tmp_wav]:
                if os.path.exists(f):
                    os.unlink(f)