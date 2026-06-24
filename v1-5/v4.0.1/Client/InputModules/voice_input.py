# modules/input/voice_input.py - Voice input module using PyAudio
import threading
import time
import tempfile
import wave
import os
import logging
from typing import Optional, Dict
from client import InputModule

logger = logging.getLogger(__name__)

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("⚠️ PyAudio not available - voice input disabled")

class VoiceInputModule(InputModule):
    """Voice input module using PyAudio"""
    
    def __init__(self, name: str = "voice_input", config: Dict = None):
        super().__init__(name, config)
        
        # Audio configuration
        self.sample_rate = self.config.get('sample_rate', 48000)
        self.channels = self.config.get('channels', 1)
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.max_record_time = self.config.get('max_record_time', 30)
        self.device_index = self.config.get('input_device_index', None)
        
        # PyAudio setup
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.audio_frames = []
        
        # Control
        self.input_thread = None
        self.stop_event = threading.Event()
    
    def initialize(self) -> bool:
        """Initialize PyAudio and find microphone"""
        if not PYAUDIO_AVAILABLE:
            logger.error("❌ PyAudio not available")
            return False
        
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("🎤 PyAudio initialized")
            self._find_microphone()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize PyAudio: {e}")
            return False
    
    def _find_microphone(self):
        """Find and configure microphone"""
        logger.info("🔍 Available input devices:")
        
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    marker = " 🎯 SELECTED" if i == self.device_index else ""
                    logger.info(f"   [{i}] {info['name']}{marker}")
                    
                    if i == self.device_index:
                        self.sample_rate = int(info['defaultSampleRate'])
            except Exception:
                continue
    
    def start(self) -> bool:
        """Start voice input thread"""
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.input_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self.input_thread.start()
            logger.info("🎤 Voice input started - press Enter to record")
            return True
        return False
    
    def stop(self):
        """Stop voice input"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self._stop_recording()
            if self.input_thread:
                self.input_thread.join(timeout=1)
            if self.audio:
                self.audio.terminate()
            logger.info("🎤 Voice input stopped")
    
    def get_data(self) -> Optional[bytes]:
        """Get recorded audio data as WAV bytes"""
        if not self.audio_frames:
            return None
        
        try:
            # Create WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_file = wave.open(tmp_file, 'wb')
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_frames))
                wav_file.close()
                
                tmp_file_path = tmp_file.name
            
            # Read WAV data
            with open(tmp_file_path, 'rb') as f:
                wav_data = f.read()
            
            # Cleanup
            os.unlink(tmp_file_path)
            
            duration = len(self.audio_frames) * self.chunk_size / self.sample_rate
            logger.info(f"🟢 Recording completed: {duration:.2f}s, {len(wav_data)} bytes")
            
            return wav_data
            
        except Exception as e:
            logger.error(f"❌ Error creating WAV file: {e}")
            return None
    
    def _voice_loop(self):
        """Main voice input loop"""
        logger.info("🎤 Voice input ready. Press Enter to start/stop recording.")
        
        while not self.stop_event.is_set() and self.enabled:
            try:
                input()  # Wait for Enter press
                
                if not self.enabled:
                    break
                
                if not self.is_recording:
                    self._start_recording()
                    input("🔴 Recording... Press Enter to stop")
                    audio_data = self._stop_recording()
                    
                    if audio_data and self.client:
                        response = self.client.send_to_server('speech', audio_data)
                        self.client.process_server_response(response, 'speech')
                
            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"❌ Voice input error: {e}")
    
    def _start_recording(self) -> bool:
        """Start audio recording"""
        if self.is_recording:
            return False
        
        try:
            self.audio_frames = []
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            
            # Start recording thread
            record_thread = threading.Thread(target=self._record_audio, daemon=True)
            record_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start recording: {e}")
            return False
    
    def _record_audio(self):
        """Record audio in background"""
        start_time = time.time()
        
        while self.is_recording:
            if time.time() - start_time > self.max_record_time:
                logger.info("⏰ Maximum recording time reached")
                break
            
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_frames.append(data)
            except Exception as e:
                logger.error(f"❌ Recording error: {e}")
                break
    
    def _stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        return self.get_data()