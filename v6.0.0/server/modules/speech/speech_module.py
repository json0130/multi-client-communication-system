"""
modules/speech/speech_module.py
================================
Speech-to-text module using Faster-Whisper.
Implements BaseModule so robot_instance.py can attach/detach it like any other module.

Requires: pip install faster-whisper
"""

from __future__ import annotations
import os
import base64
import tempfile
import wave
import threading
from dataclasses import dataclass
from typing import Optional

from modules.base import BaseModule
from core.config import cfg


@dataclass
class SpeechResult:
    success: bool
    transcription: str      # empty string on failure
    confidence: float       # 0.0 – 100.0
    language: str           # detected language code e.g. "en"
    error: str              # empty string on success


class SpeechModule(BaseModule):

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._available = False

        # Read from central config
        self._model_size = cfg.speech.whisper_model_size
        self._device = cfg.speech.whisper_device
        self._compute_type = cfg.speech.whisper_compute_type
        self._max_length_sec = cfg.speech.max_audio_length_sec
        self._sample_rate = cfg.speech.sample_rate

    # ── BaseModule interface ───────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Load the Whisper model. Falls back to CPU if GPU fails."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[SpeechModule] faster-whisper not installed. "
                  "Run: pip install faster-whisper")
            return False

        device, compute_type = self._resolve_device()

        try:
            self._model = WhisperModel(
                self._model_size,
                device=device,
                compute_type=compute_type,
            )
            self._available = True
            print(f"[SpeechModule] Loaded Whisper '{self._model_size}' "
                  f"on {device} ({compute_type})")
            return True

        except Exception as gpu_err:
            if device == "cuda":
                print(f"[SpeechModule] GPU load failed ({gpu_err}) — retrying on CPU")
                try:
                    self._model = WhisperModel(
                        self._model_size, device="cpu", compute_type="int8"
                    )
                    self._available = True
                    print(f"[SpeechModule] Loaded Whisper '{self._model_size}' on cpu (int8)")
                    return True
                except Exception as cpu_err:
                    print(f"[SpeechModule] CPU fallback also failed: {cpu_err}")
            else:
                print(f"[SpeechModule] Model load failed: {gpu_err}")

        return False

    def is_available(self) -> bool:
        return self._available and self._model is not None

    def get_status(self) -> dict:
        return {
            "module": "speech",
            "available": self._available,
            "model_size": self._model_size,
            "device": self._device,
            "max_audio_sec": self._max_length_sec,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe_b64(self, audio_b64: str) -> SpeechResult:
        """
        Decode base64 WAV audio and transcribe it.
        This is the main entry point called by robot_instance.py.
        """
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return SpeechResult(False, "", 0.0, "", f"Base64 decode failed: {e}")

        return self.transcribe_bytes(audio_bytes)

    def transcribe_bytes(self, audio_bytes: bytes) -> SpeechResult:
        """Transcribe raw WAV bytes."""
        if not self.is_available():
            return SpeechResult(False, "", 0.0, "", "Speech module not initialised")

        # Validate the WAV file
        valid, error = self._validate_wav(audio_bytes)
        if not valid:
            return SpeechResult(False, "", 0.0, "", error)

        with self._lock:
            tmp_path = None
            try:
                # Write to temp file — Whisper needs a file path
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                segments, info = self._model.transcribe(
                    tmp_path,
                    language=None,          # auto-detect
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )

                parts = []
                log_probs = []
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        parts.append(text)
                        log_probs.append(seg.avg_logprob)

                if not parts:
                    return SpeechResult(False, "", 0.0, info.language,
                                        "No speech detected")

                transcription = " ".join(parts).strip()
                avg_lp = sum(log_probs) / len(log_probs)
                confidence = max(0.0, min(100.0, (avg_lp + 1) * 100))

                print(f"[SpeechModule] Transcribed: '{transcription}' "
                      f"({confidence:.1f}% conf, lang={info.language})")
                return SpeechResult(True, transcription, confidence,
                                    info.language, "")

            except Exception as e:
                print(f"[SpeechModule] Transcription error: {e}")
                return SpeechResult(False, "", 0.0, "", str(e))
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_device(self) -> tuple[str, str]:
        """Return (device, compute_type) based on config and availability."""
        if self._device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda", self._compute_type
            except ImportError:
                pass
            return "cpu", "int8"

        if self._device == "cuda":
            return "cuda", self._compute_type

        return "cpu", "int8"

    def _validate_wav(self, audio_bytes: bytes) -> tuple[bool, str]:
        """Return (is_valid, error_message)."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            with wave.open(tmp_path, "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / rate

            if duration < 0.1:
                return False, f"Audio too short ({duration:.2f}s)"
            if duration > self._max_length_sec:
                return False, f"Audio too long ({duration:.1f}s, max {self._max_length_sec}s)"

            return True, ""

        except Exception as e:
            return False, f"Invalid WAV file: {e}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)