# modules/output/edge_tts_output.py - Powered by Google TTS (Bypassing Microsoft Block)
import subprocess
import re
import threading
import queue
import logging
import os
import tempfile
from typing import Dict, Any
from client import OutputModule
from gtts import gTTS

logger = logging.getLogger(__name__)

class EdgeTTSOutputModule(OutputModule):
    """Google TTS with clean text processing and USB speaker support"""
    
    def __init__(self, name: str = "edge_tts_output", config: Dict = None):
        super().__init__(name, config)
        self.max_length = self.config.get('max_length', 500)
        self.working_audio_cmd = ['aplay', '-D', 'plughw:2,0'] # Hardcoded to your USB speaker
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.stop_event = threading.Event()
        self.audio_available = True

    def initialize(self) -> bool:
        return True
    
    def start(self) -> bool:
        if not self.enabled:
            self.enabled = True
            self.stop_event.clear()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            return True
        return False
    
    def stop(self):
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            self.tts_queue.put(None)
            if self.tts_thread:
                self.tts_thread.join(timeout=2)
    
    def process_output(self, data: Any) -> bool:
        if not self.enabled: return False
        try:
            speech_text = data.get('text', '') if isinstance(data, dict) else str(data)
            speech_text = self._prepare_text(speech_text)
            
            if speech_text and len(speech_text.strip()) > 2:
                logger.debug(f"🎙️ Speaking: '{speech_text}'")
                self.tts_queue.put(speech_text)
                return True
            return False
        except Exception as e:
            logger.error(f"❌ TTS processing error: {e}")
            return False
    
    def _prepare_text(self, text: str) -> str:
        text = re.sub(r'\[.*?\]', '', text) # Strip emotion tags for speech
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[{}"]', '', text)
        return text
    
    def _tts_worker(self):
        while not self.stop_event.is_set():
            try:
                text = self.tts_queue.get(timeout=1)
                if text is None: break
                self._speak_text(text)
                self.tts_queue.task_done()
            except queue.Empty: continue
            except Exception as e: logger.error(f"❌ TTS worker error: {e}")
    
    def _speak_text(self, text: str):
        temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        talking_speed= "1.25"        
        try:
            # Generate speech with Google
            tts = gTTS(text=text, lang='en')
            tts.save(temp_mp3)
            
            # Convert to WAV for ALSA
            subprocess.run([
                'ffmpeg', '-i', temp_mp3,
                '-filter:a', f'atempo={talking_speed}',
                '-ar', '22050', '-ac', '1', '-sample_fmt', 's16', '-y', temp_wav
            ], capture_output=True, check=True)
            
            if self.client:
                # Create the event if the main client script didn't already
                if not hasattr(self.client, 'is_speaking'):
                    self.client.is_speaking = threading.Event()
                self.client.is_speaking.set()
                logger.debug("🔇 Mutting mic (Robot is speaking)...")

                if hasattr(self.client, 'tts_started_event'):
                    self.client.tts_started_event.set()
                
            # Play Audio (This blocks until the audio finishes!)
            subprocess.run(self.working_audio_cmd + [temp_wav], capture_output=True)
            
        except Exception as e:
            logger.error(f"❌ TTS playback error: {e}")
        finally:
            # 🟢 UNLOCK THE MICROPHONE
            if self.client and hasattr(self.client, 'is_speaking'):
                self.client.is_speaking.clear()
                logger.debug("🎙️ Unmuting mic (Robot finished speaking).")

            for f in [temp_mp3, temp_wav]:
                if os.path.exists(f): os.unlink(f)
