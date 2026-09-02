# InputModules/voice_input.py — Navel ODAS microphone capture
#
# Captures from the Navel robot's "odas_1" mic array at 48kHz, applies a simple
# volume-based VAD, downsamples to 16kHz, and sends WAV bytes to the server as
# a 'speech' message. Echo-suppressed via client.is_speaking while Navel talks.
import sys
import threading
import tempfile
import wave
import os
import logging
import audioop
import numpy as np
import pyaudio
from client import InputModule

logger = logging.getLogger(__name__)

class VoiceInputModule(InputModule):
    def __init__(self, name="voice_input", config=None):
        super().__init__(name, config)

        self.target_sample_rate = 16000 # Server expected rate
        self.native_sample_rate = 48000 # Navel expected rate

        self.audio = None
        self.stream = None
        self.navel_device_index = None

        # Simple Volume-Based VAD
        self.is_recording = False
        self.audio_frames = []
        self.silence_chunks = 0
        self.chunk_size = int(self.native_sample_rate * 0.1) # 100ms chunks

        self.stop_event = threading.Event()

    def initialize(self) -> bool:
        try:
            self.audio = pyaudio.PyAudio()
            self.navel_device_index = self._get_navel_device_index("odas_1")
            logger.info(f"🎤 Connected to Navel 'odas_1' at ALSA index {self.navel_device_index}")
            return True
        except Exception as e:
            logger.error(f"❌ Navel mic init failed: {e}")
            return False

    def _get_navel_device_index(self, device_name: str):
        """Exact implementation from Navel PDF to find ODAS microphones"""
        info = self.audio.get_host_api_info_by_type(pyaudio.paALSA)
        for i in range(info.get('deviceCount')):
            dev = self.audio.get_device_info_by_host_api_device_index(info.get('index'), i)
            if dev.get('name') == device_name:
                return dev.get('index')
        raise ValueError(f"Device {device_name} not found")

    def start(self) -> bool:
        if self.enabled: return True
        self.enabled = True
        self.stop_event.clear()

        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.native_sample_rate,
                input=True,
                input_device_index=self.navel_device_index,
            )
            threading.Thread(target=self._listen_loop, daemon=True).start()
            threading.Thread(target=self._keyboard_loop, daemon=True).start()
            return True
        except Exception as e:
            logger.error(f"❌ Audio stream failed: {e}")
            return False

    def stop(self):
        self.enabled = False
        self.stop_event.set()
        if self.stream: self.stream.close()
        if self.audio: self.audio.terminate()

    def get_data(self): return None

    def _listen_loop(self):
        """Minimal volume-based Voice Activity Detection (VAD)"""
        resample_state = None

        logger.info("⏳ Waiting 1 second for Navel ALSA buffer to prime...")
        import time
        time.sleep(1.0)

        while not self.stop_event.is_set():
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # Check if robot is currently speaking (prevent hearing itself)
                if hasattr(self.client, 'is_speaking') and self.client.is_speaking.is_set():
                    self.is_recording = False
                    self.audio_frames.clear()
                    continue

                # Downsample 48kHz -> 16kHz for the server
                resampled_data, resample_state = audioop.ratecv(
                    data, 2, 1, self.native_sample_rate, self.target_sample_rate, resample_state
                )

                # Measure Volume
                rms = np.sqrt(np.mean(np.square(np.frombuffer(resampled_data, dtype=np.int16).astype(np.float32))))

                if rms > 2500 and not self.is_recording:
                    logger.info("🗣️ Voice detected, recording...")
                    self.is_recording = True
                    self.audio_frames = [resampled_data]
                    self.silence_chunks = 0

                elif self.is_recording:
                    self.audio_frames.append(resampled_data)
                    if rms < 1500:
                        self.silence_chunks += 1
                    else:
                        self.silence_chunks = 0

                    # Stop recording after ~1.5 seconds of silence
                    if self.silence_chunks > 15:
                        logger.info("🔇 Silence detected, sending audio...")
                        self._send_audio()
                        self.is_recording = False

            except Exception as e:
                pass

    def _keyboard_loop(self):
        # Headless / boot (systemd) has no terminal — skip the interactive prompt
        # instead of hammering input() with EOFError. Voice input still works.
        if not sys.stdin or not sys.stdin.isatty():
            logger.info("⌨️  No interactive terminal — keyboard input disabled (voice only).")
            return
        while not self.stop_event.is_set() and self.enabled:
            try:
                text = input("\n💬 Type text or talk to Navel: ").strip()
            except (EOFError, OSError):
                logger.info("⌨️  stdin closed — keyboard input disabled.")
                break
            if text.lower() == 'exit':
                if self.client: self.client.running = False
                break
            elif text and self.client:
                self.client.send_to_server('chat', text)

    def _send_audio(self):
        if not self.audio_frames: return

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            with wave.open(tmp, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2) # 16-bit
                wav.setframerate(self.target_sample_rate)
                wav.writeframes(b''.join(self.audio_frames))
            tmp_path = tmp.name

        with open(tmp_path, 'rb') as f:
            wav_data = f.read()
        os.unlink(tmp_path)

        if self.client:
            self.client.send_to_server('speech', wav_data)
