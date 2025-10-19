# modules/output/tts_output.py - Text-to-Speech output module
import subprocess
import re
import threading
import queue
import logging
from typing import Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)

class TTSOutputModule(OutputModule):
    """Text-to-Speech output module using espeak"""
    
    def __init__(self, name: str = "tts_output", config: Dict = None):
        super().__init__(name, config)
        
        # TTS configuration
        self.voice = self.config.get('voice', 'en+f2')
        self.rate = self.config.get('rate', 155)  # Words per minute
        self.volume = self.config.get('volume', 100)  # 0-100
        self.pitch = self.config.get('pitch', 60)  # 0-100
        self.gap = self.config.get('gap', 0)  # Gap between words
        
        # Processing
        self.remove_emotion_tags = self.config.get('remove_emotion_tags', True)
        self.max_length = self.config.get('max_length', 500)  # Character limit
        
        # Threading for non-blocking TTS
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()
        
        # Test espeak availability
        self.espeak_available = self._test_espeak()
    
    def _test_espeak(self) -> bool:
        """Test if espeak is available"""
        try:
            result = subprocess.run(['espeak', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                logger.info("🔊 espeak TTS engine available")
                return True
            else:
                logger.warning("⚠️ espeak not available")
                return False
        except Exception as e:
            logger.warning(f"⚠️ espeak test failed: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize TTS output module"""
        if not self.espeak_available:
            logger.error("❌ espeak not available - TTS disabled")
            return False
        
        logger.info("🔊 Initializing TTS output module")
        logger.info(f"   🗣️ Voice: {self.voice}")
        logger.info(f"   ⚡ Rate: {self.rate} WPM")
        logger.info(f"   🔊 Volume: {self.volume}%")
        return True
    
    def start(self) -> bool:
        """Start TTS output module"""
        if not self.enabled and self.espeak_available:
            self.enabled = True
            self.stop_event.clear()
            
            # Start TTS processing thread
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
            logger.info("🔊 TTS output started")
            return True
        return False
    
    def stop(self):
        """Stop TTS output module"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            
            # Add stop signal to queue
            self.tts_queue.put(None)
            
            if self.tts_thread:
                self.tts_thread.join(timeout=2)
            
            logger.info("🔊 TTS output stopped")
    
    def process_output(self, data: Any) -> bool:
        """Process output data and add to TTS queue"""
        if not self.enabled:
            return False
        
        try:
            # Extract text from data
            if isinstance(data, dict):
                text = data.get('text', str(data))
            elif isinstance(data, str):
                text = data
            else:
                text = str(data)
            
            # Clean and prepare text
            speech_text = self._prepare_text(text)
            
            if speech_text:
                # Add to TTS queue (non-blocking)
                self.tts_queue.put(speech_text)
                logger.debug(f"🔊 Added to TTS queue: {speech_text[:50]}...")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ TTS processing error: {e}")
            return False
    
    def _prepare_text(self, text: str) -> str:
        """Clean and prepare text for TTS"""
        # Remove emotion tags if enabled
        if self.remove_emotion_tags:
            text = re.sub(r"^\[(.*?)\]\s*", "", text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if too long
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0] + "..."
        
        return text
    
    def _tts_worker(self):
        """TTS worker thread that processes the queue"""
        logger.info("🔊 TTS worker thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get text from queue (blocking with timeout)
                text = self.tts_queue.get(timeout=1)
                
                # Stop signal
                if text is None:
                    break
                
                # Speak the text
                self._speak_text(text)
                
                # Mark task as done
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ TTS worker error: {e}")
        
        logger.info("🔊 TTS worker thread stopped")
    
    def _speak_text(self, text: str):
        """Use espeak to vocalize text"""
        try:
            cmd = [
                'espeak',
                f'-v{self.voice}',
                f'-s{self.rate}',
                f'-a{self.volume}',
                f'-p{self.pitch}',
                f'-g{self.gap}',
                text
            ]
            
            # Run espeak (blocking)
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ espeak warning: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error("❌ TTS timeout - text too long")
        except Exception as e:
            logger.error(f"❌ TTS speak error: {e}")

class PyttsxTTSOutputModule(OutputModule):
    """Alternative TTS module using pyttsx3 (cross-platform)"""
    
    def __init__(self, name: str = "pyttsx_tts_output", config: Dict = None):
        super().__init__(name, config)
        
        # TTS configuration
        self.rate = self.config.get('rate', 200)
        self.volume = self.config.get('volume', 0.9)
        self.voice_index = self.config.get('voice_index', 0)
        
        # Processing
        self.remove_emotion_tags = self.config.get('remove_emotion_tags', True)
        self.max_length = self.config.get('max_length', 500)
        
        # Threading
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()
        
        # TTS engine
        self.engine = None
        self.pyttsx3_available = self._test_pyttsx3()
    
    def _test_pyttsx3(self) -> bool:
        """Test if pyttsx3 is available"""
        try:
            import pyttsx3
            return True
        except ImportError:
            logger.warning("⚠️ pyttsx3 not available")
            return False
    
    def initialize(self) -> bool:
        """Initialize pyttsx3 TTS engine"""
        if not self.pyttsx3_available:
            logger.error("❌ pyttsx3 not available - install with: pip install pyttsx3")
            return False
        
        try:
            import pyttsx3
            
            self.engine = pyttsx3.init()
            
            # Set properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_index:
                self.engine.setProperty('voice', voices[self.voice_index].id)
                logger.info(f"🗣️ Using voice: {voices[self.voice_index].name}")
            
            logger.info("🔊 pyttsx3 TTS engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start pyttsx3 TTS module"""
        if not self.enabled and self.engine:
            self.enabled = True
            self.stop_event.clear()
            
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
            logger.info("🔊 pyttsx3 TTS started")
            return True
        return False
    
    def stop(self):
        """Stop pyttsx3 TTS module"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self.tts_queue.put(None)
            
            if self.tts_thread:
                self.tts_thread.join(timeout=2)
            
            logger.info("🔊 pyttsx3 TTS stopped")
    
    def process_output(self, data: Any) -> bool:
        """Process output data for pyttsx3 TTS"""
        if not self.enabled:
            return False
        
        try:
            # Extract and prepare text
            if isinstance(data, dict):
                text = data.get('text', str(data))
            elif isinstance(data, str):
                text = data
            else:
                text = str(data)
            
            # Clean text
            if self.remove_emotion_tags:
                text = re.sub(r"^\[(.*?)\]\s*", "", text)
            
            text = re.sub(r'\s+', ' ', text).strip()
            
            if self.max_length and len(text) > self.max_length:
                text = text[:self.max_length].rsplit(' ', 1)[0] + "..."
            
            if text:
                self.tts_queue.put(text)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ pyttsx3 processing error: {e}")
            return False
    
    def _tts_worker(self):
        """pyttsx3 worker thread"""
        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=1)
                
                if text is None:
                    break
                
                self.engine.say(text)
                self.engine.runAndWait()
                
                self.tts_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ pyttsx3 worker error: {e}")
