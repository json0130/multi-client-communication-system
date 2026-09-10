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
        # A NAMED neural voice, so four robots do not sound like one.
        # edge-tts is installed and this module is named after it, but every
        # utterance was going through gTTS — which has no voice selection at
        # all, so Pepper, ChatBox, Navel and Silbot shared a single voice and
        # a visitor could not tell by ear who was speaking. Set `tts_voice`
        # per robot in its client config; gTTS remains the fallback when
        # edge-tts is unavailable or fails.
        self._voice      = self.config.get('tts_voice', '')

        # Speaker output (hardcoded to USB speaker; override in config if needed)
        # ALSA's 'default', not a card number. This was pinned to
        # plughw:2,0 — the built-in analog output on the machine it was
        # written for — so on any other machine, or after plugging in a
        # headset, the robots kept talking to the PC speakers while the
        # listener's actual output device stayed silent. Observed exactly
        # that: system default sink was the USB headset on card 3, audio went
        # to card 2.
        #
        # 'default' follows whatever PipeWire/PulseAudio is set to, which is
        # what a person means by "my headphones". A real robot with a fixed
        # sound card should override it — set `audio_device` (or the whole
        # `audio_cmd`) in that robot's client config.
        self._audio_cmd = self.config.get('audio_cmd') or self._default_audio_cmd()

        self.tts_queue  = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()

        self._interrupt_event = threading.Event()
        self._aplay_proc: Optional[subprocess.Popen] = None
        self._warned_playback = False   # so a failing player says why, once
        self._warned_voice = False      # and a missing voice, once
        self._aplay_lock  = threading.Lock()
        self._sim_speed   = self.config.get('sim_speed', 1.0)

        # Sentence-level progress for the utterance currently playing, so
        # interrupt() can report exactly what was never said — not just that
        # something was cut off. _current_has_callback distinguishes a demo
        # step (worth resuming) from a plain chat_sentence (not — a casual
        # reply losing its tail mid-conversation is normal).
        self._progress_lock = threading.Lock()
        self._current_sentences: list = []
        self._current_idx = -1
        self._current_has_callback = False

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

    def _default_audio_cmd(self) -> list:
        """How to play a wav, preferring whatever follows the system's own
        output setting.

        `paplay` is a PulseAudio/PipeWire client, so it always plays to the
        CURRENT default sink — which is what a person means by "my
        headphones". `aplay -D default` normally reaches the same place
        through ALSA's pulse bridge, and bare `aplay` is the last resort.

        A robot with a fixed sound card should not rely on any of this: set
        `audio_cmd` in its client config and this is skipped entirely.
        """
        import shutil
        device = self.config.get('audio_device')
        if device:
            return ['aplay', '-D', device]
        if shutil.which('paplay'):
            return ['paplay']
        if shutil.which('pw-play'):
            return ['pw-play']
        return ['aplay', '-D', 'default']

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

    def interrupt(self) -> str:
        """
        Drain the queue and stop after current sentence finishes — no
        mid-word cutoff. Returns whatever sentences after the one in
        progress were never spoken (joined back into text), or "" if nothing
        was left, if the queue was empty, or if what was interrupted was a
        plain chat_sentence rather than a demo step worth resuming.
        """
        self._interrupt_event.set()
        with self._progress_lock:
            if self._current_has_callback and self._current_sentences:
                remainder = ' '.join(self._current_sentences[self._current_idx + 1:])
            else:
                remainder = ''
        # Don't kill aplay — let the current sentence finish naturally, matching Navel behaviour.
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break
        return remainder

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
            if 'tts_voice' in voice_config:
                self._voice = voice_config['tts_voice']
                logger.info(f"[TTS] Voice → {self._voice}")
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
                self._speak_text(text, has_callback=callback is not None)
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

    def _speak_text(self, text: str, has_callback: bool = False):
        self._interrupt_event.clear()
        with self._voice_lock:
            language = self._language

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        if not sentences:
            return

        with self._progress_lock:
            self._current_sentences = sentences
            self._current_idx = -1
            self._current_has_callback = has_callback

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
                with self._progress_lock:
                    self._current_idx = i
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
            with self._progress_lock:
                self._current_sentences = []
                self._current_idx = -1
                self._current_has_callback = False
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
            with self._voice_lock:
                voice = self._voice
            made = False
            if voice:
                try:
                    import edge_tts
                    edge_tts.Communicate(text, voice, rate=self._rate).save_sync(tmp_mp3)
                    made = os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 0
                except Exception as e:
                    if not self._warned_voice:
                        self._warned_voice = True
                        logger.warning(f"[TTS] voice {voice!r} unavailable ({e}); "
                                       f"falling back to gTTS for this robot.")
            if not made:
                gTTS(text=text, lang=language).save(tmp_mp3)
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
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    )
                self._aplay_proc.wait()

                if self._aplay_proc.returncode != 0 and not self._interrupt_event.is_set():
                    # Say WHY once. A silent fallback is how the missing
                    # ffmpeg went unnoticed for a whole run of tours: every
                    # sentence failed and the module still logged "TTS
                    # completed", so the only symptom was a quiet room.
                    if not self._warned_playback:
                        self._warned_playback = True
                        detail = ""
                        try:
                            err = (self._aplay_proc.stderr.read() or b"").decode(
                                "utf-8", "replace").strip()
                            detail = f" — {err.splitlines()[0]}" if err else ""
                        except Exception:
                            pass
                        logger.warning(
                            f"[TTS] {' '.join(self._audio_cmd)} failed"
                            f"{detail}. Falling back to plain aplay; audio may "
                            f"be going to the wrong device.")
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