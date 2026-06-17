# OutputModules/edge_tts_output.py
import subprocess
import re
import threading
import queue
import logging
import os
import tempfile
import time
import concurrent.futures
from typing import Dict, Any, Optional
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

        self._interrupt_event = threading.Event()
        self._aplay_proc: Optional[subprocess.Popen] = None
        self._aplay_lock  = threading.Lock()
        self._sim_speed   = self.config.get('sim_speed', 1.0)

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

    def interrupt(self):
        """Drain the queue and stop after current sentence finishes — no mid-word cutoff."""
        self._interrupt_event.set()
        # Don't kill aplay — let the current sentence finish naturally, matching Navel behaviour.
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def clear_non_callback_items(self):
        """Remove pending chat_sentence items (no callback) from queue.
        Items with callbacks (demo steps) are kept. Does not stop current playback."""
        keep = []
        while True:
            try:
                item = self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                if isinstance(item, tuple) and item[1] is not None:
                    keep.append(item)
            except queue.Empty:
                break
        for item in keep:
            self.tts_queue.put(item)

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
            except queue.Empty:
                continue

            # Items are always (text, callback) tuples
            text, callback = item if isinstance(item, tuple) else (item, None)
            try:
                self._speak_text(text)
            except Exception as e:
                logger.error(f"[TTS] Playback error: {e}")
            finally:
                self.tts_queue.task_done()
                # Fire ACK callback AFTER playback — always, even on error
                if callback:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"[TTS] Callback error: {e}")

    def _speak_text(self, text: str):
        self._interrupt_event.clear()
        with self._voice_lock:
            language = self._language

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if not sentences:
            return

        if self.client:
            if not hasattr(self.client, 'is_speaking'):
                self.client.is_speaking = threading.Event()
            self.client.is_speaking.set()
            if hasattr(self.client, 'tts_started_event'):
                self.client.tts_started_event.set()

        audio_paths = []
        try:
            # Generate all sentence audio files in parallel — cuts gTTS overhead from N×2s to ~2s
            max_workers = min(len(sentences), 4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(self._generate_audio, s, language) for s in sentences]
            audio_paths = [f.result() for f in futures]

            for i, (sentence, (mp3, wav)) in enumerate(zip(sentences, audio_paths)):
                if self._interrupt_event.is_set():
                    break
                self._play_audio(sentence, mp3, wav)
                audio_paths[i] = (None, None)  # consumed by _play_audio
                if not self._interrupt_event.is_set() and i < len(sentences) - 1:
                    if self.client and hasattr(self.client, 'is_speaking'):
                        self.client.is_speaking.clear()
                    time.sleep(0.2)
                    if self.client and hasattr(self.client, 'is_speaking'):
                        self.client.is_speaking.set()
        finally:
            if self.client and hasattr(self.client, 'is_speaking'):
                self.client.is_speaking.clear()
            logger.debug("[TTS] is_speaking cleared")
            for mp3, wav in audio_paths:
                for f in [mp3, wav]:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except Exception:
                            pass

    def _generate_audio(self, text: str, language: str) -> tuple:
        """Generate mp3+wav for one sentence. Returns (mp3, wav) paths or (None, None) on failure."""
        if self._interrupt_event.is_set():
            return None, None
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            tts = gTTS(text=text, lang=language)
            tts.save(tmp_mp3)
            result = subprocess.run([
                'ffmpeg', '-i', tmp_mp3,
                '-filter:a', f'atempo={self.talking_speed}',
                '-ar', '22050', '-ac', '1', '-sample_fmt', 's16', '-y', tmp_wav,
            ], capture_output=True)
            if result.returncode == 0:
                return tmp_mp3, tmp_wav
            for f in [tmp_mp3, tmp_wav]:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass
            return None, None
        except Exception as e:
            logger.error(f"[TTS] Audio generation error: {e}")
            for f in [tmp_mp3, tmp_wav]:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass
            return None, None

    def _play_audio(self, text: str, mp3: Optional[str], wav: Optional[str]):
        """Play pre-generated audio, or simulate duration if unavailable."""
        try:
            if wav and os.path.exists(wav):
                logger.info(f"[TTS] Sentence: {text[:60]}{'...' if len(text) > 60 else ''}")
                with self._aplay_lock:
                    if self._interrupt_event.is_set():
                        return
                    self._aplay_proc = subprocess.Popen(
                        self._audio_cmd + [wav],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                self._aplay_proc.wait()

                if self._aplay_proc.returncode != 0 and not self._interrupt_event.is_set():
                    with self._aplay_lock:
                        self._aplay_proc = subprocess.Popen(
                            ['aplay', wav],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    self._aplay_proc.wait()
                    if self._aplay_proc.returncode != 0:
                        self._sim_sleep(text)
            else:
                self._sim_sleep(text)
        finally:
            for f in [mp3, wav]:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass

    def _sim_sleep(self, text: str):
        """Sleep to simulate playback duration. Set sim_speed=0 in config to skip (test mode)."""
        if self._sim_speed <= 0:
            return
        duration = max(0.5, len(text.split()) / 2.5) * self._sim_speed
        self._interrupt_event.wait(timeout=duration)