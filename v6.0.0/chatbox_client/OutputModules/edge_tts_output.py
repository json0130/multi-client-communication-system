"""
OutputModules/edge_tts_output.py
==================================
TTS output for ChatBox: gTTS (online) → Piper (offline neural) fallback.

Audio is played via aplay. Speaker device is auto-detected by ChatBoxClient
and injected into config['audio_cmd'] before this module is constructed.

Key methods for callers:
  process_output(data)                    — queue text for TTS
  process_output_synced(data, start_cb)  — queue text; fire start_cb at playback start
  speak_with_callback(text, callback)    — queue text; fire callback after playback ends
  interrupt()                             — clear queue (stops future items)
  clear_non_callback_items()             — remove queued items that have no post-callback
  update_voice_config(voice_config)      — change language/gender/rate at runtime
"""

import json as _json
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional

from client import OutputModule

logger = logging.getLogger(__name__)

# (text, post_callback, start_callback, item_type)
_Item = tuple


class EdgeTTSOutputModule(OutputModule):

    def __init__(self, name: str = "edge_tts_output", config: Dict = None):
        super().__init__(name, config)

        self._max_length   = self.config.get('max_length', 500)
        self._audio_cmd    = self.config.get('audio_cmd', ['aplay', '-D', 'plughw:3,0'])
        self._talking_speed = "1.25"

        # Voice settings — updated at runtime via update_voice_config()
        self._voice_lock = threading.Lock()
        self._language   = self.config.get('language', 'en')
        self._gender     = self.config.get('gender', 'female')
        self._rate       = self.config.get('rate', '+0%')

        self._tts_queue  = queue.Queue()
        self._tts_thread = None
        self._stop_event = threading.Event()

        # Persistent piper subprocess — model loaded once, reused per utterance
        self._piper_proc    = None
        self._piper_out_dir = tempfile.mkdtemp(prefix='piper_out_')
        self._piper_seq     = 0

    # ── BaseModule interface ──────────────────────────────────────────────────

    def initialize(self) -> bool:
        return True

    def start(self) -> bool:
        if self.enabled:
            return False
        self.enabled = True
        self._stop_event.clear()
        self._tts_thread = threading.Thread(
            target=self._tts_worker, daemon=True, name="tts-worker"
        )
        self._tts_thread.start()
        threading.Thread(target=self._start_piper_process, daemon=True).start()
        return True

    def stop(self):
        if not self.enabled:
            return
        self.enabled = False
        self._stop_event.set()
        self._tts_queue.put(None)
        if self._tts_thread:
            self._tts_thread.join(timeout=2)
        if self._piper_proc:
            try:
                self._piper_proc.stdin.close()
            except Exception:
                pass
            self._piper_proc.terminate()
            self._piper_proc = None

    # ── Public API ────────────────────────────────────────────────────────────

    def process_output(self, data: Any) -> bool:
        if not self.enabled:
            return False
        text = self._extract_text(data)
        if text:
            self._tts_queue.put((text, None, None, 'text'))
            return True
        return False

    def process_output_synced(self, data: Any, start_callback: Optional[Callable] = None) -> bool:
        """Queue text; fire start_callback at the moment audio playback begins."""
        if not self.enabled:
            if start_callback:
                start_callback()
            return False
        text = self._extract_text(data)
        if text:
            self._tts_queue.put((text, None, start_callback, 'text'))
            return True
        if start_callback:
            start_callback()
        return False

    def speak_with_callback(self, text: str, callback: Optional[Callable] = None) -> bool:
        """Queue text; fire callback after playback ends (used by demo_step ACK)."""
        if not self.enabled:
            if callback:
                callback()
            return False
        text = self._prepare_text(text)
        if text:
            self._tts_queue.put((text, callback, None, 'text'))
            return True
        if callback:
            callback()
        return False

    def interrupt(self):
        """Clear the queue so no further items are spoken after the current one."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break

    def clear_non_callback_items(self):
        """Remove queued items that have no post-callback (regular chat items).
        Keeps demo_step items (which carry a post-callback for ACK)."""
        remaining = []
        while not self._tts_queue.empty():
            try:
                item = self._tts_queue.get_nowait()
                if item is None:
                    continue
                _, post_cb, _, _ = item
                if post_cb is not None:
                    remaining.append(item)
            except queue.Empty:
                break
        for item in remaining:
            self._tts_queue.put(item)

    def update_voice_config(self, voice_config: dict):
        with self._voice_lock:
            if 'language' in voice_config:
                self._language = voice_config['language']
            if 'gender' in voice_config:
                self._gender = voice_config['gender']
            if 'rate' in voice_config:
                self._rate = voice_config['rate']
        logger.info(f"[TTS] Voice config updated: {voice_config}")

    # ── TTS worker ────────────────────────────────────────────────────────────

    def _tts_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=1)
                if item is None:
                    break
            except queue.Empty:
                continue

            text, post_cb, start_cb, item_type = item
            try:
                if item_type == 'text':
                    self._speak_text(text, start_cb)
            except Exception as e:
                logger.error(f"[TTS] Playback error: {e}")
            finally:
                self._tts_queue.task_done()
                if post_cb:
                    try:
                        post_cb()
                    except Exception as e:
                        logger.error(f"[TTS] Post-callback error: {e}")

    def _speak_text(self, text: str, start_callback: Optional[Callable] = None):
        with self._voice_lock:
            language = self._language

        if self.client:
            if not hasattr(self.client, 'is_speaking'):
                self.client.is_speaking = threading.Event()
            self.client.is_speaking.set()
            if hasattr(self.client, 'tts_started_event'):
                self.client.tts_started_event.set()

        try:
            if self._speak_gtts(text, language, start_callback):
                return
            if self._speak_piper(text, start_callback):
                return
            logger.warning(f"[TTS] All methods failed: {text[:60]}")
            if start_callback:
                try:
                    start_callback()
                except Exception:
                    pass
            time.sleep(max(1.0, len(text.split()) / 2.5))
        except Exception as e:
            logger.error(f"[TTS] Speak error: {e}")
            time.sleep(max(1.0, len(text.split()) / 2.5))
        finally:
            if self.client and hasattr(self.client, 'is_speaking'):
                self.client.is_speaking.clear()

    # ── gTTS (online, primary) ────────────────────────────────────────────────

    def _speak_gtts(self, text: str, language: str, start_callback: Optional[Callable] = None) -> bool:
        try:
            from gtts import gTTS
        except ImportError:
            return False

        tmp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            gTTS(text=text, lang=language).save(tmp_mp3)
            subprocess.run([
                'ffmpeg', '-i', tmp_mp3,
                '-filter:a', f'atempo={self._talking_speed}',
                '-ar', '22050', '-ac', '1', '-sample_fmt', 's16', '-y', tmp_wav,
            ], capture_output=True, check=True)

            logger.info(f"[TTS] gTTS: {text[:60]}{'...' if len(text) > 60 else ''}")
            if start_callback:
                try:
                    start_callback()
                except Exception:
                    pass
            result = subprocess.run(self._audio_cmd + [tmp_wav], capture_output=True)
            if result.returncode != 0:
                subprocess.run(['aplay', tmp_wav], capture_output=True)
            return True
        except Exception as e:
            logger.warning(f"[TTS] gTTS failed: {e}")
            return False
        finally:
            for f in [tmp_mp3, tmp_wav]:
                if os.path.exists(f):
                    os.unlink(f)

    # ── Piper (offline, fallback) ─────────────────────────────────────────────

    def _start_piper_process(self) -> bool:
        model = self.config.get('piper_model', os.path.expanduser('~/piper-voices/en_US-amy-medium.onnx'))
        if not os.path.exists(model) or not os.path.exists(model + '.json'):
            logger.warning(f"[TTS] Piper model not found: {model}")
            return False
        try:
            self._piper_proc = subprocess.Popen(
                ['piper',
                 '--model',            model,
                 '--json-input',
                 '--length_scale',     str(self.config.get('piper_length_scale', 1.0)),
                 '--noise_scale',      str(self.config.get('piper_noise_scale', 0.667)),
                 '--sentence_silence', '0.2'],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._piper_seq = 0
            logger.info("[TTS] Piper process started")
            return True
        except Exception as e:
            logger.warning(f"[TTS] Cannot start piper: {e}")
            return False

    def _speak_piper(self, text: str, start_callback: Optional[Callable] = None) -> bool:
        model = self.config.get('piper_model', os.path.expanduser('~/piper-voices/en_US-amy-medium.onnx'))
        if not os.path.exists(model) or not os.path.exists(model + '.json'):
            return False

        if self._piper_proc and self._piper_proc.poll() is None:
            return self._speak_piper_warm(text, start_callback)

        if self._piper_proc is None:
            threading.Thread(target=self._start_piper_process, daemon=True).start()

        # One-shot fallback
        tmp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            proc = subprocess.run(
                ['piper',
                 '--model',            model,
                 '--length_scale',     str(self.config.get('piper_length_scale', 1.0)),
                 '--noise_scale',      str(self.config.get('piper_noise_scale', 0.667)),
                 '--sentence_silence', '0.2',
                 '--output_file',      tmp_wav],
                input=text.encode(),
                capture_output=True,
                timeout=20,
            )
            if proc.returncode != 0 or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
                return False
            logger.info(f"[TTS] Piper (oneshot): {text[:60]}")
            if start_callback:
                try:
                    start_callback()
                except Exception:
                    pass
            result = subprocess.run(self._audio_cmd + [tmp_wav], capture_output=True)
            if result.returncode != 0:
                subprocess.run(['aplay', tmp_wav], capture_output=True)
            return True
        except Exception as e:
            logger.warning(f"[TTS] Piper oneshot failed: {e}")
            return False
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)

    def _speak_piper_warm(self, text: str, start_callback: Optional[Callable] = None) -> bool:
        seq      = self._piper_seq
        self._piper_seq += 1
        out_file = os.path.join(self._piper_out_dir, f'{seq}.wav')
        if os.path.exists(out_file):
            os.unlink(out_file)

        try:
            self._piper_proc.stdin.write(
                (_json.dumps({"text": text, "output_file": out_file}) + '\n').encode()
            )
            self._piper_proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("[TTS] Piper pipe broken — restarting")
            self._piper_proc = None
            threading.Thread(target=self._start_piper_process, daemon=True).start()
            return False

        deadline = time.time() + 20
        while time.time() < deadline:
            if self._piper_proc.poll() is not None:
                logger.warning("[TTS] Piper process died — restarting")
                self._piper_proc = None
                threading.Thread(target=self._start_piper_process, daemon=True).start()
                return False
            if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                time.sleep(0.05)
                break
            time.sleep(0.05)
        else:
            logger.warning("[TTS] Piper warm timeout")
            return False

        logger.info(f"[TTS] Piper (warm): {text[:60]}")
        if start_callback:
            try:
                start_callback()
            except Exception:
                pass
        result = subprocess.run(self._audio_cmd + [out_file], capture_output=True)
        if result.returncode != 0:
            subprocess.run(['aplay', out_file], capture_output=True)
        if os.path.exists(out_file):
            os.unlink(out_file)
        return True

    # ── Text helpers ──────────────────────────────────────────────────────────

    def _extract_text(self, data: Any) -> Optional[str]:
        text = data.get('text', '') if isinstance(data, dict) else str(data)
        return self._prepare_text(text)

    def _prepare_text(self, text: str) -> Optional[str]:
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[{}"]', '', text)
        if self._max_length and len(text) > self._max_length:
            text = text[:self._max_length].rsplit(' ', 1)[0] + '...'
        return text if len(text) > 2 else None
