# speech_processor.py - Speech-to-Text Component using Faster-Whisper
import os
import tempfile
import base64
import wave
import threading
from typing import Optional, Tuple
import numpy as np

class SpeechProcessor:
    """Speech-to-text processor using faster-whisper"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Model configuration
        self.model_size = self.config.get('whisper_model_size', 'base')
        self.device = self.config.get('whisper_device', 'auto')
        self.compute_type = self.config.get('whisper_compute_type', 'float16')
        self.language = self.config.get('whisper_language', None)  # Auto-detect if None
        
        # Processing configuration
        self.max_audio_length = self.config.get('max_audio_length', 30)  # seconds
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        # State
        self.model = None
        self.model_loaded = False
        self.processing_lock = threading.Lock()
        
        print(f"🎤 Initializing Speech Processor...")
        print(f"   Model: {self.model_size}")
        print(f"   Device: {self.device}")
    
    def initialize(self):
        """Initialize the faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            print(f"🔄 Loading Faster-Whisper model ({self.model_size})...")
            
            # Determine device with safer fallback logic
            device = 'cpu'  # Default to CPU for stability
            compute_type = 'int8'  # Safe default
            
            if self.device == 'auto':
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Test CUDA compatibility before using it
                        try:
                            torch.cuda.get_device_name(0)
                            device = 'cuda'
                            compute_type = 'float16'
                            print("🔍 CUDA detected, will attempt GPU acceleration")
                        except Exception as cuda_e:
                            print(f"⚠️ CUDA available but not working properly: {cuda_e}")
                            print("   Falling back to CPU")
                            device = 'cpu'
                            compute_type = 'int8'
                    else:
                        print("ℹ️ CUDA not available, using CPU")
                except ImportError:
                    print("ℹ️ PyTorch not available, using CPU")
            elif self.device == 'cuda':
                device = 'cuda'
                compute_type = self.compute_type
                print("🔍 Forcing CUDA usage as requested")
            else:
                device = self.device
                compute_type = self.compute_type
            
            # Try to load model with GPU first, fallback to CPU if it fails
            try:
                self.model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=self.config.get('model_cache_dir')
                )
                
                self.model_loaded = True
                print(f"✅ Faster-Whisper model loaded successfully")
                print(f"   Device: {device}")
                print(f"   Compute type: {compute_type}")
                
                return True
                
            except Exception as gpu_error:
                if device == 'cuda':
                    print(f"⚠️ GPU loading failed: {gpu_error}")
                    print("🔄 Attempting CPU fallback...")
                    
                    # Fallback to CPU
                    try:
                        self.model = WhisperModel(
                            self.model_size,
                            device='cpu',
                            compute_type='int8',
                            download_root=self.config.get('model_cache_dir')
                        )
                        
                        self.model_loaded = True
                        print(f"✅ Faster-Whisper model loaded successfully (CPU fallback)")
                        print(f"   Device: cpu")
                        print(f"   Compute type: int8")
                        
                        return True
                        
                    except Exception as cpu_error:
                        print(f"❌ CPU fallback also failed: {cpu_error}")
                        return False
                else:
                    print(f"❌ Error loading model on {device}: {gpu_error}")
                    return False
            
        except ImportError:
            print("❌ faster-whisper not installed. Install with: pip install faster-whisper")
            print("💡 Try: pip install faster-whisper")
            return False
        except Exception as e:
            print(f"❌ Error loading Faster-Whisper model: {e}")
            print("💡 Try installing CPU-only version: pip install faster-whisper --force-reinstall")
            return False
    
    def decode_audio_base64(self, audio_b64: str) -> Optional[bytes]:
        """Decode base64 audio data"""
        try:
            audio_bytes = base64.b64decode(audio_b64)
            return audio_bytes
        except Exception as e:
            print(f"❌ Error decoding audio: {e}")
            return None
    
    def validate_wav_file(self, audio_bytes: bytes) -> bool:
        """Validate that the audio is a proper WAV file"""
        try:
            # Create temporary file to validate
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            # Try to open with wave module
            with wave.open(tmp_file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                
                print(f"🎵 Audio info: {duration:.2f}s, {sample_rate}Hz, {frames} frames")
                
                # Check duration
                if duration > self.max_audio_length:
                    print(f"⚠️ Audio too long: {duration:.2f}s (max: {self.max_audio_length}s)")
                    return False
                
                if duration < 0.1:
                    print(f"⚠️ Audio too short: {duration:.2f}s")
                    return False
            
            # Clean up
            os.unlink(tmp_file_path)
            return True
            
        except Exception as e:
            print(f"❌ Error validating WAV file: {e}")
            return False
    
    def transcribe_audio(self, audio_bytes: bytes) -> Tuple[bool, str, float]:
        """
        Transcribe audio to text using faster-whisper
        Returns: (success, transcription, confidence)
        """
        if not self.model_loaded:
            return False, "Speech-to-text model not loaded", 0.0
        
        with self.processing_lock:
            try:
                # Validate audio
                if not self.validate_wav_file(audio_bytes):
                    return False, "Invalid audio file", 0.0
                
                # Create temporary file for whisper
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file_path = tmp_file.name
                
                print(f"🎤 Transcribing audio file: {tmp_file_path}")
                
                try:
                    # Transcribe with faster-whisper
                    segments, info = self.model.transcribe(
                        tmp_file_path,
                        language=self.language,
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        vad_filter=True,  # Voice activity detection
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    print(f"🔍 Transcription info - Language: {info.language}, Probability: {info.language_probability:.2f}")
                    
                    # Extract text and confidence
                    transcription_parts = []
                    confidences = []
                    
                    segment_count = 0
                    for segment in segments:
                        segment_count += 1
                        text = segment.text.strip()
                        print(f"   Segment {segment_count}: '{text}' (start: {segment.start:.1f}s, end: {segment.end:.1f}s, prob: {segment.avg_logprob:.2f})")
                        
                        if text:
                            transcription_parts.append(text)
                            # Use average log probability as confidence proxy
                            confidences.append(segment.avg_logprob)
                    
                    print(f"📊 Found {segment_count} segments, {len(transcription_parts)} with text")
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    if not transcription_parts:
                        print("⚠️ No speech segments with text found")
                        return False, "No speech detected in audio", 0.0
                    
                    # Combine transcription
                    full_transcription = " ".join(transcription_parts).strip()
                    
                    # Calculate overall confidence (convert log prob to percentage)
                    if confidences:
                        avg_logprob = np.mean(confidences)
                        # Convert log probability to confidence percentage (rough approximation)
                        confidence = max(0, min(100, (avg_logprob + 1) * 100))
                    else:
                        confidence = 50.0  # Default confidence
                    
                    print(f"🎯 Final transcription: '{full_transcription}'")
                    print(f"📊 Confidence: {confidence:.1f}%")
                    print(f"🌍 Detected language: {info.language} (probability: {info.language_probability:.2f})")
                    
                    return True, full_transcription, confidence
                    
                except Exception as transcription_error:
                    # Clean up temporary file on error
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
                    print(f"❌ Transcription error: {transcription_error}")
                    print(f"   Error type: {type(transcription_error).__name__}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    
                    return False, f"Transcription failed: {str(transcription_error)}", 0.0
                
            except Exception as e:
                print(f"❌ Error during transcription setup: {e}")
                print(f"   Error type: {type(e).__name__}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                return False, f"Transcription error: {str(e)}", 0.0
    
    def transcribe_audio_base64(self, audio_b64: str) -> Tuple[bool, str, float]:
        """
        Transcribe base64 encoded audio to text
        Returns: (success, transcription, confidence)
        """
        # Decode audio
        audio_bytes = self.decode_audio_base64(audio_b64)
        if audio_bytes is None:
            return False, "Failed to decode audio", 0.0
        
        # Transcribe
        return self.transcribe_audio(audio_bytes)
    
    def get_status(self):
        """Get speech processor status"""
        return {
            'model_loaded': self.model_loaded,
            'model_size': self.model_size,
            'device': self.device,
            'max_audio_length': self.max_audio_length,
            'sample_rate': self.sample_rate
        }
    
    def is_available(self):
        """Check if speech processor is available"""
        return self.model_loaded

# Test function for standalone testing
def test_speech_processor():
    """Test the speech processor with a sample file"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python speech_processor.py <audio_file.wav>")
        return
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return
    
    # Initialize processor
    processor = SpeechProcessor()
    
    if not processor.initialize():
        print("Failed to initialize speech processor")
        return
    
    # Read audio file
    with open(audio_file, 'rb') as f:
        audio_bytes = f.read()
    
    # Transcribe
    success, transcription, confidence = processor.transcribe_audio(audio_bytes)
    
    if success:
        print(f"✅ Transcription successful!")
        print(f"📝 Text: {transcription}")
        print(f"📊 Confidence: {confidence:.1f}%")
    else:
        print(f"❌ Transcription failed: {transcription}")

if __name__ == "__main__":
    test_speech_processor()