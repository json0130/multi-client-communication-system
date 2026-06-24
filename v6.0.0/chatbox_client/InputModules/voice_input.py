"""
InputModules/voice_input.py
============================
PyAudio-based voice input with VAD (volume-triggered) and keyboard fallback.

Audio captured at native device sample rate, downsampled to 16 kHz, then
forwarded to the server as a WAV file via send_to_server('speech', wav_bytes).

Keyboard fallback:
  - Type text + Enter  → sends as chat message
  - Press Enter alone  → starts/stops manual voice recording
  - Type 'exit'        → stops the client
"""

import audioop
import collections
import logging
import os
import tempfile
import threading
import time
import wave
from typing import Dict, Optional

from client import InputModule

logger = logging.getLogger(__name__)

try:
    import pyaudio
    import numpy as np
    _PYAUDIO_AVAILABLE = True
except ImportError:
    _PYAUDIO_AVAILABLE = False
    logger.warning("[Voice] PyAudio or numpy not available — voice input disabled")


class VoiceInputModule(InputModule):
    """
    Hybrid voice input: VAD auto-trigger + keyboard text/manual record fallback.

    Config keys (all optional):
      sample_rate        : int   target sample rate for server (default 16000)
      max_record_time    : float max seconds per auto-recording (default 30)
      vad_trigger_threshold : int  RMS level to start recording (default 1000)
      volume_threshold   : int   RMS floor to keep recording (default 500)
      silence_seconds    : float seconds of silence before sending (default 1.5)
    """

    def __init__(self, name: str = "voice_input", config: Dict = None):
        super().__init__(name, config)

        self._target_rate   = self.config.get("sample_rate", 16000)
        self._max_record    = self.config.get("max_record_time", 30)
        self._vad_trigger   = self.config.get("vad_trigger_threshold", 1000)
        self._vol_threshold = self.config.get("volume_threshold", 500)
        self._silence_sec   = self.config.get("silence_seconds", 1.5)

        self._chunk_duration  = 0.08   # 80 ms frames
        self._native_rate     = 48000
        self._native_channels = 1
        self._device_index: Optional[int] = None

        self._audio  = None
        self._stream = None
        self._resample_state = None

        # Pre-roll buffer: ~1s of audio captured before VAD triggers
        _pre_roll_chunks = int(1.0 / self._chunk_duration)
        self._pre_roll   = collections.deque(maxlen=_pre_roll_chunks)

        self._mode    = "listening"   # "listening" | "auto_record" | "manual_record"
        self._frames  = []
        self._silence_chunks = 0
        self._silence_limit  = int((self._target_rate / int(self._target_rate * self._chunk_duration)) * self._silence_sec)

        self._stop_event   = threading.Event()
        self._audio_thread = None
        self._kb_thread    = None

    # ── BaseModule interface ──────────────────────────────────────────────────

    def initialize(self) -> bool:
        if not _PYAUDIO_AVAILABLE:
            logger.warning("[Voice] PyAudio unavailable — voice input disabled")
            return False
        try:
            self._audio = pyaudio.PyAudio()
            self._find_microphone()
            chunk = int(self._native_rate * self._chunk_duration)
            logger.info(
                f"[Voice] Init — device {self._device_index}, "
                f"native {self._native_rate} Hz → target {self._target_rate} Hz, "
                f"chunk {chunk}"
            )
            return True
        except Exception as e:
            logger.error(f"[Voice] Init failed: {e}")
            return False

    def start(self) -> bool:
        if self.enabled:
            return False
        self.enabled = True
        self._stop_event.clear()
        self._resample_state = None

        if self._open_stream():
            self._audio_thread = threading.Thread(
                target=self._audio_loop, daemon=True, name="voice-audio"
            )
            self._audio_thread.start()

        self._kb_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True, name="voice-keyboard"
        )
        self._kb_thread.start()
        return True

    def stop(self):
        self.enabled = False
        self._stop_event.set()
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
        logger.info("[Voice] Stopped")

    def get_data(self):
        return None

    # ── Microphone detection ──────────────────────────────────────────────────

    def _find_microphone(self):
        if not self._audio:
            return
        for i in range(self._audio.get_device_count()):
            try:
                info = self._audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    name_lower = info['name'].lower()
                    usb_keywords = ['uacdemov1.0', 'usb audio', 'usb', 'microphone', 'mic']
                    if any(k in name_lower for k in usb_keywords) and 'tegra' not in name_lower:
                        self._device_index = i
                        self._native_rate  = int(info['defaultSampleRate'])
                        logger.info(f"[Voice] USB mic: {info['name']} (device {i}, {self._native_rate} Hz)")
                        return
            except Exception:
                continue
        logger.info("[Voice] No USB mic found — using default input device")

    def _open_stream(self) -> bool:
        try:
            chunk = int(self._native_rate * self._chunk_duration)
            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=self._native_channels,
                rate=self._native_rate,
                input=True,
                input_device_index=self._device_index,
                frames_per_buffer=chunk,
            )
            return True
        except Exception as e:
            logger.error(f"[Voice] Failed to open audio stream: {e}")
            return False

    # ── Audio capture loop ────────────────────────────────────────────────────

    def _audio_loop(self):
        chunk = int(self._native_rate * self._chunk_duration)
        while not self._stop_event.is_set():
            try:
                raw = self._stream.read(chunk, exception_on_overflow=False)

                # Downsample to target rate
                if self._native_rate != self._target_rate:
                    resampled, self._resample_state = audioop.ratecv(
                        raw, 2, 1, self._native_rate, self._target_rate, self._resample_state
                    )
                else:
                    resampled = raw

                # Pause while TTS is playing to avoid echo
                if self.client and hasattr(self.client, 'is_speaking') and self.client.is_speaking.is_set():
                    self._pre_roll.clear()
                    continue

                audio_np = np.frombuffer(resampled, dtype=np.int16)
                rms = float(np.sqrt(np.mean(np.square(audio_np.astype(np.float32)))))

                if self._mode == "listening":
                    self._pre_roll.append(resampled)
                    if rms > self._vad_trigger:
                        logger.info("[Voice] VAD triggered — recording")
                        self._mode   = "auto_record"
                        self._frames = list(self._pre_roll)
                        self._pre_roll.clear()
                        self._silence_chunks = 0

                elif self._mode == "auto_record":
                    self._frames.append(resampled)
                    if rms < self._vol_threshold:
                        self._silence_chunks += 1
                    else:
                        self._silence_chunks = 0
                    if self._silence_chunks > self._silence_limit:
                        logger.info("[Voice] Silence — sending audio")
                        self._send_audio()
                        self._mode = "listening"

                elif self._mode == "manual_record":
                    self._frames.append(resampled)

            except Exception as e:
                logger.error(f"[Voice] Audio loop error: {e}")
                time.sleep(0.1)

    # ── Keyboard fallback ─────────────────────────────────────────────────────

    def _keyboard_loop(self):
        logger.info("[Voice] Keyboard ready — type text or press Enter for manual record")
        while not self._stop_event.is_set() and self.enabled:
            try:
                user_input = input("\n[You] text or Enter for voice: ").strip()
                if user_input.lower() == 'exit':
                    if self.client:
                        self.client.running = False
                    break
                elif user_input:
                    threading.Thread(
                        target=self._send_text, args=(user_input,), daemon=True
                    ).start()
                else:
                    if self._mode != "manual_record":
                        self._mode   = "manual_record"
                        self._frames = []
                        logger.info("[Voice] Manual record started — press Enter to stop")
                        input()
                        logger.info("[Voice] Manual record stopped — sending")
                        self._send_audio()
                        self._mode = "listening"
            except KeyboardInterrupt:
                break
            except EOFError:
                break

    # ── Audio send helpers ────────────────────────────────────────────────────

    def _send_audio(self):
        if not self._frames or not self.client:
            self._frames = []
            return
        frames = self._frames
        self._frames = []
        threading.Thread(target=self._encode_and_send, args=(frames,), daemon=True).start()

    def _encode_and_send(self, frames: list):
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(self._target_rate)
                wf.writeframes(b''.join(frames))
            tmp.close()
            with open(tmp.name, 'rb') as f:
                wav_bytes = f.read()
            os.unlink(tmp.name)
            self.client.send_to_server('speech', wav_bytes)
        except Exception as e:
            logger.error(f"[Voice] Encode/send error: {e}")

    def _send_text(self, text: str):
        if self.client:
            self.client.send_to_server('chat', text)
