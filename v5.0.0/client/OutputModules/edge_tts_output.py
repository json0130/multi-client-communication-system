# modules/output/edge_tts_output.py - COMPLETE CLEAN EdgeTTS Module
import subprocess
import re
import threading
import queue
import logging
import os
import tempfile
import asyncio
from typing import Dict, Any
from client import OutputModule

logger = logging.getLogger(__name__)

class EdgeTTSOutputModule(OutputModule):
    """Microsoft Edge TTS with clean text processing and default speaker support"""
    
    def __init__(self, name: str = "edge_tts_output", config: Dict = None):
        super().__init__(name, config)
        
        # Edge TTS configuration
        self.voice = self.config.get('voice', 'en-US-AriaNeural')
        self.rate = self.config.get('rate', '+0%')
        self.pitch = self.config.get('pitch', '+0Hz')
        
        # Processing
        self.remove_emotion_tags = self.config.get('remove_emotion_tags', True)
        self.max_length = self.config.get('max_length', 500)
        
        # Audio configuration - macOS default speaker
        self.working_audio_cmd = ['afplay']
        self.fallback_audio_cmd = ['ffplay', '-nodisp', '-autoexit']
        
        # Threading
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()
        
        # Test availability
        self.edge_tts_available = self._test_edge_tts()
        # ffmpeg is not needed for afplay
        self.audio_available = self._test_audio_playback()
    
    def _test_edge_tts(self) -> bool:
        """Test if edge-tts is available"""
        try:
            import edge_tts
            # logger.info("🎙️ Microsoft Edge TTS available")
            return True
        except ImportError:
            logger.warning("⚠️ Edge TTS not available - install with: pip install edge-tts")
            return False
    
    def _test_audio_playback(self) -> bool:
        """Test default audio playback with MP3 pipeline"""
        if not self.edge_tts_available:
            logger.warning("⚠️ Cannot test audio playback - missing dependencies")
            return False
        
        try:
            import edge_tts
            
            async def test_pipeline():
                # Test the complete pipeline
                test_text = "Audio test"
                temp_mp3 = '/tmp/edge_tts_pipeline_test.mp3'
                
                try:
                    # Generate MP3
                    communicate = edge_tts.Communicate(test_text, self.voice)
                    await communicate.save(temp_mp3)
                    
                    # Test playback (afplay)
                    cmd = self.working_audio_cmd + [temp_mp3]
                    play_result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                    
                    if play_result.returncode == 0:
                        # logger.info("✅ EdgeTTS → MP3 → afplay pipeline working")
                        return True
                    else:
                        # Try fallback (ffplay)
                        cmd = self.fallback_audio_cmd + [temp_mp3]
                        fallback_result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                        if fallback_result.returncode == 0:
                            # logger.info("✅ Pipeline working with ffplay fallback")
                            self.working_audio_cmd = self.fallback_audio_cmd
                            return True
                        else:
                            logger.warning("⚠️ Both afplay and ffplay failed")
                            return False
                        
                except Exception as e:
                    logger.error(f"❌ Pipeline test error: {e}")
                    return False
                    
                finally:
                    # Cleanup
                    if os.path.exists(temp_mp3):
                        os.unlink(temp_mp3)
            
            return asyncio.run(test_pipeline())
            
        except Exception as e:
            logger.error(f"❌ Audio playback pipeline test failed: {e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize Edge TTS"""
        if not self.edge_tts_available:
            logger.error("❌ Edge TTS not available")
            return False
        
        if not self.audio_available:
            logger.error("❌ Default audio playback pipeline not working")
            return False
        
        # logger.info("🎙️ Initializing Microsoft Edge TTS")
        # logger.info(f"   🗣️ Voice: {self.voice}")
        # logger.info(f"   🔊 Audio: {' '.join(self.working_audio_cmd)}")
        # logger.info("   🔧 Pipeline: EdgeTTS → MP3 → Default Speaker")
        return True
    
    def start(self) -> bool:
        """Start Edge TTS module"""
        if not self.enabled and self.edge_tts_available and self.audio_available:
            self.enabled = True
            self.stop_event.clear()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            return True
        return False
    
    def stop(self):
        """Stop Edge TTS module"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self.tts_queue.put(None)
            if self.tts_thread:
                self.tts_thread.join(timeout=2)
            # logger.info("🎙️ Edge TTS stopped")
    
    def process_output(self, data: Any) -> bool:
        """Process output for Edge TTS - CLEAN TEXT ONLY"""
        if not self.enabled:
            return False
        
        try:
            # Extract the text from the known data structure
            speech_text = ""
            
            if isinstance(data, dict):
                # From debug output, we know 'text' field contains the response
                speech_text = data.get('text', '')
                
            elif isinstance(data, str):
                speech_text = data
            else:
                speech_text = str(data)
            
            # Clean the text thoroughly
            speech_text = self._prepare_text(speech_text)
            
            if speech_text and len(speech_text.strip()) > 2:
                logger.debug(f"🎙️ Speaking: '{speech_text}'")
                self.tts_queue.put(speech_text)
                return True
            else:
                logger.warning("⚠️ No valid text to speak after cleaning")
                return False
            
        except Exception as e:
            logger.error(f"❌ EdgeTTS processing error: {e}")
            return False
    
    def _prepare_text(self, text: str) -> str:
        """Clean text thoroughly for TTS"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove ALL emotion/action tags: [GREETING], [POSE], [DEFAULT], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove any JSON artifacts
        text = re.sub(r'[{}"]', '', text)
        
        # Remove common response patterns
        text = re.sub(r'response:|text:|content:', '', text, flags=re.IGNORECASE)
        
        # Final cleanup
        text = text.strip()
        
        # Truncate if too long
        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length].rsplit(' ', 1)[0] + "..."
        
        return text
    
    def _tts_worker(self):
        """Edge TTS worker thread"""
        
        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None:
                    break
                self._speak_text(text)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Edge TTS worker error: {e}")
    
    def _speak_text(self, text: str):
        """Generate and play speech - NO SSML to avoid XML artifacts"""
        temp_mp3 = None
        
        try:
            import edge_tts
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                temp_mp3 = tmp_mp3.name
            
            # Generate speech with plain text (no SSML/XML)
            asyncio.run(self._generate_speech(text, temp_mp3))
            
            # Play converted WAV file
            self._play_audio(temp_mp3)
            
        except Exception as e:
            logger.error(f"❌ Edge TTS speak error: {e}")
        finally:
            # Cleanup temporary files
            if temp_mp3 and os.path.exists(temp_mp3):
                try:
                    os.unlink(temp_mp3)
                except:
                    pass
    
    async def _generate_speech(self, text: str, mp3_file: str):
        """Generate speech with plain text - no SSML/XML to avoid artifacts"""
        import edge_tts
        
        # Use plain text WITHOUT SSML formatting to avoid XML/HTML artifacts
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(mp3_file)
        logger.debug("✅ Speech generated successfully")

    
    def _play_audio(self, audio_file: str):
        """Play converted WAV file using default speaker"""
        try:
            # Verify file exists
            if not os.path.exists(audio_file):
                logger.error(f"❌ Audio file not found: {audio_file}")
                return

            # This signals to other modules that speech is about to start.
            if self.client and hasattr(self.client, 'tts_started_event'):
                logger.debug("Firing TTS started event")
                self.client.tts_started_event.set()
            
            # Play using working audio command (afplay or ffplay)
            cmd = self.working_audio_cmd + [audio_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.debug("🔊 Audio played successfully")
            else:
                logger.warning(f"⚠️ Audio playback warning: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Audio playback timeout")
        except Exception as e:
            logger.error(f"❌ Audio playback error: {e}")