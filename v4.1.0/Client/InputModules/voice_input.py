# modules/input/voice_input.py - UNIFIED INPUT HANDLER (no competition)
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
    """Voice input module that ALSO handles text input - no competition"""
    
    def __init__(self, name: str = "voice_input", config: Dict = None):
        super().__init__(name, config)
        
        # Audio configuration - EXACTLY from working system
        self.sample_rate = self.config.get('sample_rate', 48000)
        self.channels = self.config.get('channels', 1)
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.audio_format = pyaudio.paInt16
        self.max_record_time = self.config.get('max_record_time', 30)
        
        # USB microphone detection
        self.usb_device_index = None
        self.prefer_usb = self.config.get('prefer_usb', True)
        
        # Recording state
        self.is_recording = False
        self.audio_frames = []
        self.audio = None
        self.stream = None
        self.record_thread = None
        
        # Control
        self.input_thread = None
        self.stop_event = threading.Event()
    
    def initialize(self) -> bool:
        """Initialize PyAudio and automatically find USB microphone"""
        if not PYAUDIO_AVAILABLE:
            logger.error("❌ PyAudio not available")
            return False
        
        try:
            self.audio = pyaudio.PyAudio()
            # logger.info("🎤 PyAudio initialized successfully")
            self._find_usb_microphone()
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize PyAudio: {e}")
            self.audio = None
            return False
    
    def _find_usb_microphone(self):
        """USB microphone auto-detection - EXACT working logic"""
        if not self.audio:
            return
    
        # logger.info("🔍 Auto-detecting USB microphone...")
        # logger.info("   📋 Available input devices:")
    
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_name = info['name']
                    # logger.info(f"      [{i}] {device_name}")
                
                    name_lower = device_name.lower()
                    usb_patterns = [
                        'uacdemov1.0', 'usb audio', 'usb', 'microphone', 'mic', 'hw:2,0'
                    ]
                    
                    is_not_tegra = 'tegra' not in name_lower
                    is_usb_device = any(pattern in name_lower for pattern in usb_patterns)
                    is_likely_usb = (i == 11 and is_not_tegra)
                
                    if is_usb_device or is_likely_usb:
                        self.usb_device_index = i
                        self.sample_rate = int(info['defaultSampleRate'])
                        # logger.info(f"   🎯 AUTO-SELECTED USB microphone:")
                        # logger.info(f"      Device index: {i}")
                        # logger.info(f"      Name: {device_name}")
                        # logger.info(f"      Sample rate: {self.sample_rate}")
                        # logger.info(f"      ✅ Will be used for all recordings")
                        return True
                    
            except Exception as e:
                logger.info(f"      [?] Device {i}: Error getting info - {e}")
                continue
    
        # Fallback to device 11
        try:
            info = self.audio.get_device_info_by_index(11)
            if info['maxInputChannels'] > 0:
                self.usb_device_index = 11
                self.sample_rate = int(info['defaultSampleRate'])
                # logger.info(f"   🎯 FALLBACK: Using device 11")
                # logger.info(f"      Name: {info['name']}")
                return True
        except Exception as e:
            logger.info(f"   ❌ Cannot access device 11: {e}")
    
    def start(self) -> bool:
        """Start unified input handler"""
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.input_thread = threading.Thread(target=self._unified_input_loop, daemon=True)
            self.input_thread.start()
            # logger.info("🎤 Unified input handler started")
            return True
        return False
    
    def stop(self):
        """Stop voice input"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            if self.is_recording:
                self.stop_recording()
            if self.input_thread:
                self.input_thread.join(timeout=1)
            if self.audio:
                self.audio.terminate()
            logger.info("🎤 Voice input stopped")
    
    def get_data(self) -> Optional[bytes]:
        """Required by base class"""
        return None
    
    def _unified_input_loop(self):
        """UNIFIED input handler - EXACT working jetson_client_test.py logic"""
        logger.info("💬 Chat Interface Ready!")
        logger.info("   📝 Type a message and press Enter for text chat")
        logger.info("   🎤 Press Enter on empty line to start voice recording")
        logger.info("   🛑 Type 'exit' to quit")
        logger.info("-" * 50)
        
        while not self.stop_event.is_set() and self.enabled:
            try:
                # SINGLE input() call - no competition!
                user_input = input("\n💬 You (text) or 🎤 Enter for voice: ").strip()
                
                if user_input.lower() == 'exit':
                    logger.info("🛑 Exit command received")
                    if self.client:
                        self.client.running = False
                    break
                
                # Text input - EXACT working logic
                elif user_input:
                    logger.info("🔄 Processing text message...")
                    if self.client:
                        response = self.client.send_to_server('chat', user_input)
                        self.client.process_server_response(response, 'text')
                
                # Voice input (empty input) - EXACT working logic
                else:
                    self._handle_voice_recording()
                       
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Input error: {e}")
    
    def _handle_voice_recording(self):
        """Voice recording - EXACT jetson_client_test.py logic"""
        if not self.audio:
            logger.error("❌ Voice recording not available")
            return
        
        # Start recording
        if not self.start_recording():
            return
        
        # DEDICATED input() for stopping - no competition!
        try:
            input()  # This is the ONLY input() call active during recording
        except KeyboardInterrupt:
            pass
        
        # Stop recording and process
        audio_data = self.stop_recording()
        
        if audio_data and self.client:
            logger.info("🔄 Processing speech message...")
            response = self.client.send_to_server('speech', audio_data)
            self.client.process_server_response(response, 'speech')
        else:
            logger.error("❌ No audio recorded")
    
    def start_recording(self) -> bool:
        """Start recording - EXACT working logic"""
        if not self.audio or self.is_recording:
            return False
        
        recording_device = self.usb_device_index
        
        try:
            self.audio_frames = []
            
            if recording_device is not None:
                device_info = self.audio.get_device_info_by_index(recording_device)
                logger.info(f"🎤 Recording with: {device_info['name']}")
            
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=recording_device,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            logger.info("🔴 Recording started... Press Enter to stop")
            
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start recording: {e}")
            return False
    
    def _record_audio(self):
        """Record audio - EXACT working logic"""
        start_time = time.time()
        
        try:
            while self.is_recording:
                if time.time() - start_time > self.max_record_time:
                    logger.info("⏰ Maximum recording time reached")
                    break
                
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_frames.append(data)
                    
                    elapsed = int(time.time() - start_time)
                    
                except Exception as e:
                    if "Unanticipated host error" not in str(e):
                        logger.error(f"❌ Error reading audio: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ Recording error: {e}")
        finally:
            self.is_recording = False
    
    def stop_recording(self) -> Optional[bytes]:
        """Stop recording - EXACT working logic"""
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if not self.audio_frames:
            return None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_file = wave.open(tmp_file, 'wb')
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_frames))
                wav_file.close()
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as f:
                wav_data = f.read()
            
            os.unlink(tmp_file_path)
            
            duration = len(self.audio_frames) * self.chunk_size / self.sample_rate
            logger.info(f"🟢 Recording completed: {duration:.2f}s, {len(wav_data)} bytes")
            
            return wav_data
            
        except Exception as e:
            logger.error(f"❌ Error creating WAV file: {e}")
            return None
