# modules/output/tts_output.py - Text-to-Speech Output Module
import subprocess
import re
import threading
import time
from typing import Optional, Callable, Dict, Any, List
import queue

class TextToSpeechModule:
    """
    Text-to-Speech output module using various TTS engines.
    
    Features:
    - Multiple TTS engines support (espeak, festival, pico2wave, etc.)
    - Voice customization (rate, pitch, volume, voice type)
    - Emotion tag processing and voice adaptation
    - Queue-based speech synthesis for sequential playback
    - Configurable speech parameters
    - Background speech processing
    """
    
    def __init__(self, client_core, config: Dict[str, Any] = None):
        """
        Initialize text-to-speech module
        
        Args:
            client_core: Core client instance
            config: Configuration dictionary
        """
        self.client_core = client_core
        self.config = config or client_core.get_config()
        
        # TTS settings from config
        self.tts_settings = self.config.get('tts_settings', {})
        self.voice = self.tts_settings.get('voice', 'en+f2')
        self.rate = self.tts_settings.get('rate', 155)
        self.volume = self.tts_settings.get('volume', 100)
        self.pitch = self.tts_settings.get('pitch', 60)
        self.gap = self.tts_settings.get('gap', 0)
        
        # TTS engine settings
        self.engine = self.tts_settings.get('engine', 'espeak')
        self.engine_available = False
        
        # Speech queue for sequential processing
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_worker_running = False
        self.speech_worker_thread = None
        
        # Emotion-based voice adaptation
        self.emotion_voices = {
            'happy': {'voice': 'en+f3', 'rate': 165, 'pitch': 70},
            'sad': {'voice': 'en+f1', 'rate': 130, 'pitch': 45},
            'angry': {'voice': 'en+m1', 'rate': 180, 'pitch': 80},
            'fear': {'voice': 'en+f2', 'rate': 190, 'pitch': 85},
            'surprise': {'voice': 'en+f3', 'rate': 170, 'pitch': 75},
            'neutral': {'voice': 'en+f2', 'rate': 155, 'pitch': 60},
            'default': {'voice': 'en+f2', 'rate': 155, 'pitch': 60}
        }
        
        # Callbacks
        self.on_speech_started = None   # Callback(text, emotion) - called when speech starts
        self.on_speech_finished = None  # Callback(text, emotion) - called when speech finishes
        self.on_speech_error = None     # Callback(error_msg, text) - called on speech errors
        
        # Register with core client for automatic speech responses
        self.client_core.register_callback('on_chat_response', self._handle_chat_response)
        self.client_core.register_callback('on_speech_response', self._handle_speech_response)
        
        # Initialize TTS engine
        self._initialize_tts_engine()
        
        print(f"🔊 Text-to-speech module initialized")
        print(f"   🎵 Engine: {self.engine}")
        print(f"   🗣️ Voice: {self.voice}")
        print(f"   ⚡ Rate: {self.rate} WPM")
        print(f"   🎚️ Volume: {self.volume}%")
    
    def _handle_chat_response(self, response: str, detected_emotion: str, confidence: float):
        """Handle chat response from server - auto-speak if enabled"""
        if self.config.get('features', {}).get('auto_speak_responses', True):
            self.speak_with_emotion_detection(response)
    
    def _handle_speech_response(self, transcription: str, response: str, confidence: float):
        """Handle speech response from server - auto-speak response"""
        if response and self.config.get('features', {}).get('auto_speak_responses', True):
            self.speak_with_emotion_detection(response)
    
    def _initialize_tts_engine(self):
        """Initialize and test TTS engine"""
        try:
            if self.engine == 'espeak':
                # Test espeak availability
                result = subprocess.run(['espeak', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.engine_available = True
                    print("✅ eSpeak engine available")
                else:
                    print("❌ eSpeak not available")
            
            elif self.engine == 'festival':
                # Test festival availability
                result = subprocess.run(['festival', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.engine_available = True
                    print("✅ Festival engine available")
                else:
                    print("❌ Festival not available")
            
            elif self.engine == 'pico2wave':
                # Test pico2wave availability
                result = subprocess.run(['pico2wave', '--help'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode in [0, 1]:  # pico2wave returns 1 for help
                    self.engine_available = True
                    print("✅ Pico2Wave engine available")
                else:
                    print("❌ Pico2Wave not available")
            
            else:
                print(f"❌ Unknown TTS engine: {self.engine}")
                
        except Exception as e:
            print(f"❌ Error initializing TTS engine: {e}")
            self.engine_available = False
        
        if not self.engine_available:
            print("⚠️ TTS engine not available - speech output disabled")
    
    def start_speech_worker(self):
        """Start background speech worker thread"""
        if self.speech_worker_running:
            return
        
        self.speech_worker_running = True
        self.speech_worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_worker_thread.start()
        print("🎤 Speech worker started")
    
    def stop_speech_worker(self):
        """Stop background speech worker thread"""
        self.speech_worker_running = False
        if self.speech_worker_thread:
            self.speech_worker_thread.join(timeout=2.0)
        print("🛑 Speech worker stopped")
    
    def _speech_worker(self):
        """Background worker for processing speech queue"""
        while self.speech_worker_running:
            try:
                # Get speech item from queue (blocking with timeout)
                speech_item = self.speech_queue.get(timeout=1.0)
                
                if speech_item is None:  # Shutdown signal
                    break
                
                # Process speech item
                self._process_speech_item(speech_item)
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Speech worker error: {e}")
                if self.on_speech_error:
                    self.on_speech_error(f"Speech worker error: {e}", "")
    
    def _process_speech_item(self, speech_item: Dict[str, Any]):
        """Process a single speech item"""
        text = speech_item.get('text', '')
        emotion = speech_item.get('emotion', 'default')
        custom_settings = speech_item.get('settings', {})
        
        if not text.strip():
            return
        
        try:
            self.is_speaking = True
            
            # Call callback
            if self.on_speech_started:
                self.on_speech_started(text, emotion)
            
            # Perform TTS
            success = self._synthesize_speech(text, emotion, custom_settings)
            
            if success:
                if self.on_speech_finished:
                    self.on_speech_finished(text, emotion)
            else:
                if self.on_speech_error:
                    self.on_speech_error("Speech synthesis failed", text)
            
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            if self.on_speech_error:
                self.on_speech_error(f"Speech processing error: {e}", text)
        finally:
            self.is_speaking = False
    
    def _synthesize_speech(self, text: str, emotion: str = 'default', custom_settings: Dict[str, Any] = None) -> bool:
        """
        Synthesize speech using configured TTS engine
        
        Args:
            text: Text to speak
            emotion: Emotion for voice adaptation
            custom_settings: Optional custom voice settings
            
        Returns:
            bool: True if synthesis successful
        """
        if not self.engine_available:
            print("⚠️ TTS engine not available")
            return False
        
        # Clean text (remove emotion tags)
        clean_text = self.remove_emotion_tags(text)
        
        if not clean_text.strip():
            return False
        
        # Get voice settings for emotion
        voice_settings = self._get_voice_settings_for_emotion(emotion, custom_settings)
        
        try:
            if self.engine == 'espeak':
                return self._speak_with_espeak(clean_text, voice_settings)
            elif self.engine == 'festival':
                return self._speak_with_festival(clean_text, voice_settings)
            elif self.engine == 'pico2wave':
                return self._speak_with_pico2wave(clean_text, voice_settings)
            else:
                print(f"❌ Unsupported TTS engine: {self.engine}")
                return False
                
        except Exception as e:
            print(f"❌ TTS synthesis error: {e}")
            return False
    
    def _get_voice_settings_for_emotion(self, emotion: str, custom_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get voice settings adapted for specific emotion"""
        # Start with default settings
        settings = {
            'voice': self.voice,
            'rate': self.rate,
            'volume': self.volume,
            'pitch': self.pitch,
            'gap': self.gap
        }
        
        # Apply emotion-specific settings
        if emotion in self.emotion_voices:
            emotion_settings = self.emotion_voices[emotion]
            settings.update(emotion_settings)
        
        # Apply custom settings override
        if custom_settings:
            settings.update(custom_settings)
        
        return settings
    
    def _speak_with_espeak(self, text: str, settings: Dict[str, Any]) -> bool:
        """Speak using eSpeak TTS engine"""
        try:
            cmd = [
                'espeak',
                f'-v{settings["voice"]}',
                f'-s{settings["rate"]}',
                f'-a{settings["volume"]}',
                f'-p{settings["pitch"]}',
                f'-g{settings["gap"]}',
                text
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True)
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"❌ eSpeak error: {e}")
            return False
        except Exception as e:
            print(f"❌ eSpeak execution error: {e}")
            return False
    
    def _speak_with_festival(self, text: str, settings: Dict[str, Any]) -> bool:
        """Speak using Festival TTS engine"""
        try:
            # Festival uses different parameter format
            festival_script = f'(SayText "{text}")'
            
            cmd = ['festival', '--pipe']
            
            result = subprocess.run(cmd, input=festival_script, text=True, 
                                  check=True, capture_output=True)
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Festival error: {e}")
            return False
        except Exception as e:
            print(f"❌ Festival execution error: {e}")
            return False
    
    def _speak_with_pico2wave(self, text: str, settings: Dict[str, Any]) -> bool:
        """Speak using Pico2Wave TTS engine"""
        try:
            import tempfile
            import os
            
            # Generate WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_path = tmp_file.name
            
            # Generate speech
            cmd = ['pico2wave', '-w', wav_path, text]
            result = subprocess.run(cmd, check=True, capture_output=True)
            
            if result.returncode == 0:
                # Play the generated WAV file
                play_cmd = ['aplay', wav_path]  # Linux audio player
                subprocess.run(play_cmd, check=True, capture_output=True)
                
                # Clean up
                os.unlink(wav_path)
                return True
            else:
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Pico2Wave error: {e}")
            return False
        except Exception as e:
            print(f"❌ Pico2Wave execution error: {e}")
            return False
    
    def speak(self, text: str, emotion: str = 'default', priority: bool = False, 
              custom_settings: Dict[str, Any] = None) -> bool:
        """
        Add text to speech queue
        
        Args:
            text: Text to speak
            emotion: Emotion for voice adaptation
            priority: If True, add to front of queue
            custom_settings: Custom voice settings for this speech
            
        Returns:
            bool: True if added to queue successfully
        """
        if not text.strip():
            return False
        
        if not self.speech_worker_running:
            self.start_speech_worker()
        
        speech_item = {
            'text': text,
            'emotion': emotion,
            'settings': custom_settings or {}
        }
        
        try:
            if priority:
                # Add to front of queue (not directly supported by queue.Queue)
                # Create new queue with priority item first
                temp_items = []
                while not self.speech_queue.empty():
                    temp_items.append(self.speech_queue.get_nowait())
                
                self.speech_queue.put(speech_item)
                for item in temp_items:
                    self.speech_queue.put(item)
            else:
                self.speech_queue.put(speech_item)
            
            print(f"🔊 Added to speech queue: '{text[:50]}...' (emotion: {emotion})")
            return True
            
        except Exception as e:
            print(f"❌ Error adding to speech queue: {e}")
            if self.on_speech_error:
                self.on_speech_error(f"Queue error: {e}", text)
            return False
    
    def speak_immediately(self, text: str, emotion: str = 'default', 
                         custom_settings: Dict[str, Any] = None) -> bool:
        """
        Speak text immediately (bypassing queue)
        
        Args:
            text: Text to speak
            emotion: Emotion for voice adaptation
            custom_settings: Custom voice settings
            
        Returns:
            bool: True if spoken successfully
        """
        if not text.strip():
            return False
        
        speech_item = {
            'text': text,
            'emotion': emotion,
            'settings': custom_settings or {}
        }
        
        self._process_speech_item(speech_item)
        return True
    
    def speak_with_emotion_detection(self, text: str) -> bool:
        """
        Speak text with automatic emotion detection from emotion tags
        
        Args:
            text: Text to speak (may contain emotion tags like [HAPPY])
            
        Returns:
            bool: True if processing started successfully
        """
        # Extract emotion from text
        emotion = self.extract_emotion_from_text(text)
        
        return self.speak(text, emotion)
    
    def extract_emotion_from_text(self, text: str) -> str:
        """
        Extract emotion tag from text
        
        Args:
            text: Text that may contain emotion tags like [HAPPY]
            
        Returns:
            str: Extracted emotion or 'default'
        """
        match = re.match(r"^\[(.*?)\]", text)
        if match:
            emotion_tag = match.group(1).lower()
            # Map common emotion tags to our emotion system
            emotion_mapping = {
                'happy': 'happy',
                'sad': 'sad',
                'angry': 'angry',
                'fear': 'fear',
                'surprise': 'surprise',
                'neutral': 'neutral',
                'greeting': 'happy',
                'wave': 'happy',
                'confused': 'neutral',
                'default': 'default'
            }
            return emotion_mapping.get(emotion_tag, 'default')
        
        return 'default'
    
    def remove_emotion_tags(self, text: str) -> str:
        """
        Remove emotion tags from text
        
        Args:
            text: Text with potential emotion tags
            
        Returns:
            str: Clean text without emotion tags
        """
        return re.sub(r"^\[(.*?)\]\s*", "", text)
    
    def clear_speech_queue(self):
        """Clear all pending speech items"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
        print("🗑️ Speech queue cleared")
    
    def get_queue_size(self) -> int:
        """Get number of items in speech queue"""
        return self.speech_queue.qsize()
    
    def set_voice_settings(self, voice: str = None, rate: int = None, volume: int = None, 
                          pitch: int = None, gap: int = None):
        """
        Update default voice settings
        
        Args:
            voice: Voice identifier
            rate: Speech rate (words per minute)
            volume: Volume (0-100)
            pitch: Pitch (0-100)
            gap: Gap between words (milliseconds)
        """
        if voice is not None:
            self.voice = voice
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if pitch is not None:
            self.pitch = pitch