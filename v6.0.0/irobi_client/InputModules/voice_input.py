# InputModules/voice_input.py - VAD + keyboard hybrid voice input
import threading
import time
import tempfile
import wave
import os
import logging
import audioop
import numpy as np
import collections
from typing import Optional, Dict
from client import InputModule

logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available - voice input disabled")

WAKEWORD_AVAILABLE = True


class VoiceInputModule(InputModule):
    def __init__(self, name: str = "voice_input", config: Dict = None):
        super().__init__(name, config)

        self.target_sample_rate = 16000
        self.native_sample_rate = 48000
        self.channels = 1
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

        self.chunk_duration = 0.08
        self.native_chunk_size = int(self.native_sample_rate * self.chunk_duration)

        self.usb_device_index = None
        self.audio = None
        self.stream = None
        self.resample_state = None

        self.mode = "listening"
        self.audio_frames = []
        self.silence_chunks = 0
        self.SILENCE_LIMIT = int((16000 / 1280) * 1.5)

        self.VOLUME_THRESHOLD = 500
        self.VAD_TRIGGER_THRESHOLD = 1000

        self.pre_roll_chunks = int(1.0 / self.chunk_duration)
        self.audio_buffer = collections.deque(maxlen=self.pre_roll_chunks)

        self.oww_model = None

        self.audio_thread = None
        self.input_thread = None
        self.stop_event = threading.Event()

    def initialize(self) -> bool:
        if not PYAUDIO_AVAILABLE:
            return False
        try:
            self.audio = pyaudio.PyAudio()
            self._find_usb_microphone()
            self.native_chunk_size = int(self.native_sample_rate * self.chunk_duration)
            logger.info(f"Mic rate: {self.native_sample_rate}Hz (auto-converts to 16000Hz for STT)")
            return True
        except Exception as e:
            logger.error(f"Voice init failed: {e}")
            return False

    def _find_usb_microphone(self):
        if not self.audio:
            return
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    name_lower = info['name'].lower()
                    usb_patterns = ['uacdemov1.0', 'usb audio', 'usb', 'microphone', 'mic', 'hw:2,0']
                    is_not_tegra = 'tegra' not in name_lower
                    if any(p in name_lower for p in usb_patterns) or (i == 11 and is_not_tegra):
                        self.usb_device_index = i
                        self.native_sample_rate = int(info['defaultSampleRate'])
                        return True
            except Exception:
                continue
        try:
            info = self.audio.get_device_info_by_index(11)
            if info['maxInputChannels'] > 0:
                self.usb_device_index = 0
                self.native_sample_rate = int(info['defaultSampleRate'])
                return True
        except Exception:
            pass

    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.resample_state = None

            if self._start_audio_stream():
                self.audio_thread = threading.Thread(
                    target=self._continuous_audio_loop, daemon=True, name="irobi-audio"
                )
                self.audio_thread.start()

            self.input_thread = threading.Thread(
                target=self._keyboard_input_loop, daemon=True, name="irobi-keyboard"
            )
            self.input_thread.start()
            return True
        return False

    def stop(self):
        self.enabled = False
        self.stop_event.set()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        logger.info("Voice input stopped")

    def get_data(self):
        return None

    def _start_audio_stream(self) -> bool:
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.native_sample_rate,
                input=True,
                input_device_index=self.usb_device_index,
                frames_per_buffer=self.native_chunk_size,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
            return False

    def _continuous_audio_loop(self):
        while not self.stop_event.is_set():
            try:
                data = self.stream.read(self.native_chunk_size, exception_on_overflow=False)

                if self.native_sample_rate != self.target_sample_rate:
                    resampled_data, self.resample_state = audioop.ratecv(
                        data, 2, 1, self.native_sample_rate, self.target_sample_rate, self.resample_state
                    )
                else:
                    resampled_data = data

                audio_np = np.frombuffer(resampled_data, dtype=np.int16)

                if hasattr(self.client, 'is_speaking') and self.client.is_speaking.is_set():
                    self.audio_buffer.clear()
                    continue

                if self.mode == "listening":
                    self.audio_buffer.append(resampled_data)
                    rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                    triggered = False

                    if self.oww_model:
                        prediction = self.oww_model.predict(audio_np)
                        for mdl, score in prediction.items():
                            if score > 0.5:
                                logger.info("Wake word detected — listening...")
                                triggered = True

                    if not triggered and rms > self.VAD_TRIGGER_THRESHOLD:
                        logger.info("Voice detected (VAD) — listening...")
                        triggered = True

                    if triggered:
                        self.mode = "auto_record"
                        self.audio_frames = list(self.audio_buffer)
                        self.audio_buffer.clear()
                        self.silence_chunks = 0

                elif self.mode == "auto_record":
                    self.audio_frames.append(resampled_data)
                    rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                    if rms < self.VOLUME_THRESHOLD:
                        self.silence_chunks += 1
                    else:
                        self.silence_chunks = 0

                    if self.silence_chunks > self.SILENCE_LIMIT:
                        logger.info("Silence detected — processing speech...")
                        self._save_and_send_audio()
                        self.mode = "listening"

                elif self.mode == "manual_record":
                    self.audio_frames.append(resampled_data)

            except Exception as e:
                logger.error(f"Audio loop error: {e}")
                time.sleep(0.1)

    def _keyboard_input_loop(self):
        logger.info("iRobi input ready.")
        logger.info("  Type text and press Enter to chat")
        logger.info("  Press Enter with no text to start/stop manual voice recording")
        logger.info("  Type 'exit' to quit")
        logger.info("-" * 50)

        while not self.stop_event.is_set() and self.enabled:
            try:
                user_input = input("\nYou (text) or Enter for voice: ").strip()

                if user_input.lower() == 'exit':
                    if self.client:
                        self.client.running = False
                    break

                elif user_input:
                    threading.Thread(
                        target=self._process_request_in_background,
                        args=('chat', user_input), daemon=True,
                    ).start()

                else:
                    if self.mode != "manual_record":
                        self.mode = "manual_record"
                        self.audio_frames = []
                        logger.info("Recording... Press Enter to stop")
                        input()
                        logger.info("Recording stopped.")
                        self._save_and_send_audio()
                        self.mode = "listening"

            except KeyboardInterrupt:
                break

    def _save_and_send_audio(self):
        if not self.audio_frames:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_file = wave.open(tmp_file, 'wb')
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wav_file.setframerate(self.target_sample_rate)
                wav_file.writeframes(b''.join(self.audio_frames))
                wav_file.close()
                tmp_path = tmp_file.name

            with open(tmp_path, 'rb') as f:
                wav_data = f.read()
            os.unlink(tmp_path)

            if self.client:
                threading.Thread(
                    target=self._process_request_in_background,
                    args=('speech', wav_data), daemon=True,
                ).start()

        except Exception as e:
            logger.error(f"Audio save error: {e}")

    def _process_request_in_background(self, data_type: str, data):
        if self.client:
            response = self.client.send_to_server(data_type, data)
            self.client.process_server_response(response, 'speech')
