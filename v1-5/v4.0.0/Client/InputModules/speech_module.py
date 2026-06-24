# modules/input/speech_input.py - Speech Input Module
import pyaudio
import wave
import tempfile
import os
import time
import threading
from typing import Optional, Callable, Dict, Any
import numpy as np

class SpeechInputModule:
    """
    Speech input module for audio recording and sending to server for processing.
    
    Features:
    - USB microphone auto-detection
    - Voice recording with real-time feedback
    - Audio preprocessing and optimization
    - Integration with server speech processing
    - Configurable audio settings
    - Voice activity detection (optional)
    """
    
    def __init__(self, client_core, config: Dict[str, Any] = None):
        """
        Initialize speech input module
        
        Args:
            client_core: Core client instance
            config: Configuration dictionary
        """
        self.client_core = client_core
        self.config = config or client_core.get_config()
        
        # Audio settings from config
        self.audio_settings = self.config.get('audio_settings', {})
        self.sample_rate = self.audio_settings.get('sample_rate', 48000)
        self.channels = self.audio_settings.get('channels', 1)
        self.chunk_size = self.audio_settings.get('chunk_size', 1024)
        self.max_record_time = self.audio_settings.get('max_record_time', 30)
        
        # Hardware settings
        self.usb_device_index = self.config.get('hardware', {}).get('usb_mic_device', 11)
        self.prefer_usb = True
        
        # Audio format
        self.audio_format = pyaudio.paInt16
        
        # Recording state
        self.is_recording = False
        self.audio_frames = []
        self.audio = None
        self.stream = None
        self.record_thread = None
        
        # Callbacks
        self.on_recording_started = None    # Callback() - called when recording starts
        self.on_recording_stopped = None    # Callback(duration, data_size) - called when recording stops
        self.on_audio_data_ready = None     # Callback(audio_data) - called when audio is ready
        self.on_transcription_result = None # Callback(transcription, confidence) - called when transcription received
        self.on_speech_response = None      # Callback(transcription, response, confidence) - called when full response received
        self.on_recording_error = None      # Callback(error_msg) - called on recording errors
        
        # Register with core client
        self.client_core.register_callback('on_speech_response', self._handle_speech_response)
        
        # Initialize PyAudio
        self._initialize_pyaudio()
        
        print(f"🎤 Speech input module initialized")
        print(f"   📊 Sample rate: {self.sample_rate} Hz")
        print(f"   🔊 Channels: {self.channels}")
        print(f"   🎚️ USB mic device: {self.usb_device_index}")
    
    def _handle_speech_response(self, transcription: str, response: str, confidence: float):
        """Handle speech response from server"""
        if self.on_transcription_result:
            self.on_transcription_result(transcription, confidence)
        
        if self.on_speech_response:
            self.on_speech_response(transcription, response, confidence)
    
    def _initialize_pyaudio(self) -> bool:
        """Initialize PyAudio and find USB microphone"""
        try:
            self.audio = pyaudio.PyAudio()
            print("🎤 PyAudio initialized successfully")
            
            # Find and configure USB microphone
            self._find_usb_microphone()
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize PyAudio: {e}")
            self.audio = None
            if self.on_recording_error:
                self.on_recording_error(f"PyAudio initialization failed: {e}")
            return False
    
    def _find_usb_microphone(self):
        """Find and configure USB microphone"""
        if not self.audio:
            return
        
        print("🔍 Auto-detecting USB microphone...")
        print("   📋 Available input devices:")
        
        best_device = None
        best_score = 0
        
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_name = info['name']
                    print(f"      [{i}] {device_name}")
                    
                    # Enhanced USB microphone detection with scoring
                    name_lower = device_name.lower()
                    score = 0
                    
                    # USB device patterns (higher scores for better matches)
                    usb_patterns = {
                        'uacdemov1.0': 100,       # Specific USB audio device
                        'usb audio': 90,          # Generic USB audio
                        'microphone': 80,         # Contains microphone
                        'mic': 70,                # Contains mic
                        'usb': 60,                # Contains USB
                        'hw:2,0': 50,             # Hardware ID pattern
                    }
                    
                    # Check patterns
                    for pattern, pattern_score in usb_patterns.items():
                        if pattern in name_lower:
                            score += pattern_score
                    
                    # Avoid built-in audio (negative score)
                    if 'tegra' in name_lower or 'built-in' in name_lower:
                        score -= 50
                    
                    # Prefer higher device indices (often USB devices)
                    if i >= 10:
                        score += 20
                    
                    # Special case for configured device index
                    if i == self.usb_device_index:
                        score += 30
                    
                    if score > best_score:
                        best_device = i
                        best_score = score
                        
            except Exception as e:
                print(f"      [?] Device {i}: Error getting info - {e}")
                continue
        
        # Set best device
        if best_device is not None:
            self.usb_device_index = best_device
            try:
                info = self.audio.get_device_info_by_index(best_device)
                self.sample_rate = int(info['defaultSampleRate'])
                
                print(f"   🎯 AUTO-SELECTED USB microphone:")
                print(f"      Device index: {best_device}")
                print(f"      Name: {info['name']}")
                print(f"      Sample rate: {self.sample_rate}")
                print(f"      Score: {best_score}")
                return True
            except Exception as e:
                print(f"   ❌ Error accessing selected device: {e}")
        
        print("   ⚠️ No suitable USB microphone found")
        print(f"   🔧 Using fallback device index {self.usb_device_index}")
        
        return False
    
    def list_audio_devices(self):
        """List all available audio input devices"""
        if not self.audio:
            print("❌ PyAudio not available")
            return
        
        print("🎙️ Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    marker = " 🎯 SELECTED" if i == self.usb_device_index else ""
                    print(f"   [{i}] {info['name']}{marker}")
                    print(f"       Channels: {info['maxInputChannels']}, Rate: {int(info['defaultSampleRate'])}")
            except Exception as e:
                print(f"   [?] Device {i}: Error - {e}")
    
    def start_recording(self, device_index: Optional[int] = None, auto_stop_duration: Optional[float] = None) -> bool:
        """
        Start voice recording
        
        Args:
            device_index: Optional device index override
            auto_stop_duration: Auto-stop after this many seconds (None for manual stop)
            
        Returns:
            bool: True if recording started successfully
        """
        if not self.audio:
            print("❌ PyAudio not available")
            if self.on_recording_error:
                self.on_recording_error("PyAudio not available")
            return False
        
        if self.is_recording:
            print("⚠️ Already recording")
            return True
        
        if not self.client_core.is_module_enabled('speech'):
            print("❌ Speech module not enabled in client configuration")
            if self.on_recording_error:
                self.on_recording_error("Speech module not enabled")
            return False
        
        # Use specified device or auto-detected USB microphone
        recording_device = device_index if device_index is not None else self.usb_device_index
        
        try:
            self.audio_frames = []
            
            # Show which device is being used
            if recording_device is not None:
                device_info = self.audio.get_device_info_by_index(recording_device)
                print(f"🎤 Recording with: [{recording_device}] {device_info['name']}")
            else:
                print("🎤 Recording with default device")
            
            # Open audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=recording_device,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            self.auto_stop_duration = auto_stop_duration
            
            if auto_stop_duration:
                print(f"🔴 Recording started for {auto_stop_duration} seconds...")
            else:
                print("🔴 Recording started... Call stop_recording() to stop")
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()
            
            # Call callback
            if self.on_recording_started:
                self.on_recording_started()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            if self.on_recording_error:
                self.on_recording_error(f"Failed to start recording: {e}")
            return False
    
    def _record_audio(self):
        """Record audio in background thread"""
        start_time = time.time()
        
        while self.is_recording:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check auto-stop duration
            if self.auto_stop_duration and elapsed >= self.auto_stop_duration:
                print(f"\n⏰ Auto-stop duration ({self.auto_stop_duration}s) reached")
                break
            
            # Check maximum recording time
            if elapsed > self.max_record_time:
                print(f"\n⏰ Maximum recording time ({self.max_record_time}s) reached")
                break
            
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_frames.append(data)
                
                # Show recording progress every 2 seconds
                elapsed_int = int(elapsed)
                if elapsed_int > 0 and elapsed_int % 2 == 0:
                    remaining = ""
                    if self.auto_stop_duration:
                        remaining = f" ({self.auto_stop_duration - elapsed:.1f}s left)"
                    print(f"   🎤 Recording... {elapsed_int}s{remaining}", end='\r')
            except Exception as e:
                print(f"\n❌ Error while recording audio: {e}")
                if self.on_recording_error:
                    self.on_recording_error(f"Error while recording audio: {e}")
                break
        print("\n🔴 Recording stopped")
        self.stop_recording()