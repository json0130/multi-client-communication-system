# server.py - Individual Client Server Instance
import os
import time
import threading
from typing import Set, Dict, Any, Optional
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import modular components
from Modules.emotion_processor import EmotionProcessor
from Modules.gpt_client import GPTClient
from Modules.web_interface import WebInterface
from Modules.speech_processor import SpeechProcessor

# Configuration - CORRECTED PATH FOR YOUR SETUP
MODEL_PATH = '../models/efficientnet_HQRAF_improved_withCon.pth'  # Your existing model
API_KEY = "emotion_recognition_key_123"
load_dotenv()

class RobotServer:
    """
    Individual server instance for a single client.
    Each client gets their own instance with custom module configuration.
    
    This class is NO LONGER the main entry point - it's managed by ServerController.
    """
    
    def __init__(self, client_id: str, enabled_modules: Set[str], config: Dict[str, Any]):
        # Client identification
        self.client_id = client_id
        self.enabled_modules = enabled_modules
        self.config = config
        
        # Module instances (initialized based on enabled_modules)
        self.emotion_processor = None
        self.gpt_client = None
        self.speech_processor = None
        self.web_interface = None
        
        # State tracking
        self.components_initialized = False
        self.initialization_lock = threading.Lock()
        
        print(f"🎯 Created server instance for client '{self.client_id}' with modules: {list(self.enabled_modules)}")
    
    @classmethod
    def create_for_client(cls, client_id: str, enabled_modules: Set[str], config: Dict[str, Any]):
        """Factory method to create a server instance for a specific client"""
        return cls(client_id=client_id, enabled_modules=enabled_modules, config=config)
    
    def initialize_components(self) -> bool:
        """Initialize only the enabled modules for this client"""
        with self.initialization_lock:
            if self.components_initialized:
                return True
            
            print(f"🚀 Initializing components for client '{self.client_id}'...")
            print(f"🎯 Enabled modules: {list(self.enabled_modules)}")
            
            success_count = 0
            total_components = len(self.enabled_modules)
            
            # Initialize Emotion Processing Module
            if 'emotion' in self.enabled_modules:
                print(f"  📊 Initializing emotion processing...")
                try:
                    # Check multiple possible paths for the model
                    possible_paths = [
                        MODEL_PATH,  # Default path
                        './models/efficientnet_HQRAF_improved_withCon.pth',  # If running from v4.0.0
                        '../models/efficientnet_HQRAF_improved_withCon.pth',  # If running from ServerController
                        '../../models/efficientnet_HQRAF_improved_withCon.pth',  # If running from deeper
                        'models/efficientnet_HQRAF_improved_withCon.pth'  # Relative path
                    ]
                    
                    found_model_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            found_model_path = path
                            print(f"    ✅ Found model at: {path}")
                            break
                    
                    if not found_model_path:
                        print(f"    ❌ Model file not found in any of these locations:")
                        for path in possible_paths:
                            print(f"        - {path}")
                        print(f"    💡 Current working directory: {os.getcwd()}")
                        # Still create processor but it will fail gracefully
                        found_model_path = MODEL_PATH
                    
                    self.emotion_processor = EmotionProcessor(found_model_path, self.config)
                    emotion_success, emotion_total = self.emotion_processor.initialize()
                    
                    if emotion_success == emotion_total:
                        success_count += 1
                        print(f"    ✅ Emotion processing initialized ({emotion_success}/{emotion_total})")
                    elif emotion_success > 0:
                        success_count += 0.5  # Partial success
                        print(f"    ⚠️ Emotion processing partially initialized ({emotion_success}/{emotion_total})")
                        if emotion_success == 1 and emotion_total == 2:
                            print(f"        - Face detection: ✅ Working")
                            print(f"        - Model loading: ❌ Failed (check model file path)")
                    else:
                        print(f"    ❌ Emotion processing failed completely ({emotion_success}/{emotion_total})")
                        
                except Exception as e:
                    print(f"    ❌ Emotion processing failed: {e}")
            else:
                # Create minimal emotion processor for compatibility (other modules might need it)
                self.emotion_processor = EmotionProcessor(MODEL_PATH, self.config)
                print(f"  📊 Emotion module disabled (compatibility instance created)")
            
            # Initialize GPT Module
            if 'gpt' in self.enabled_modules:
                print(f"  🤖 Initializing GPT client...")
                try:
                    self.gpt_client = GPTClient()
                    if self.gpt_client.setup_openai():
                        success_count += 1
                        print(f"    ✅ GPT client initialized")
                    else:
                        print(f"    ❌ GPT client initialization failed")
                        
                except Exception as e:
                    print(f"    ❌ GPT initialization error: {e}")
            else:
                # Create mock GPT client for compatibility
                self.gpt_client = GPTClient()
                print(f"  🤖 GPT module disabled")
            
            # Initialize Speech Processing Module
            if 'speech' in self.enabled_modules:
                print(f"  🎤 Initializing speech processor...")
                try:
                    self.speech_processor = SpeechProcessor(self.config)
                    if self.speech_processor.initialize():
                        success_count += 1
                        print(f"    ✅ Speech processor initialized")
                    else:
                        print(f"    ❌ Speech processor initialization failed")
                        
                except Exception as e:
                    print(f"    ❌ Speech initialization error: {e}")
            else:
                # Create mock speech processor for compatibility
                self.speech_processor = SpeechProcessor(self.config)
                print(f"  🎤 Speech module disabled")
            
            # Initialize Facial Recognition Module
            if 'facial' in self.enabled_modules:
                print(f"  👤 Initializing facial recognition...")
                try:
                    # Note: Facial recognition is typically part of emotion processing
                    # but can be separate if you have additional facial features
                    if self.emotion_processor and self.emotion_processor.face_cascade_loaded:
                        success_count += 1
                        print(f"    ✅ Facial recognition initialized (via emotion processor)")
                    else:
                        print(f"    ❌ Facial recognition failed - emotion processor required")
                        
                except Exception as e:
                    print(f"    ❌ Facial recognition error: {e}")
            else:
                print(f"  👤 Facial recognition module disabled")
            
            # Web interface (always available for monitoring/debugging)
            try:
                self.web_interface = WebInterface(self.config.get('stream_fps', 30))
                print(f"  🌐 Web interface ready")
            except Exception as e:
                print(f"  ⚠️ Web interface setup warning: {e}")
            
            # Determine if initialization was successful
            self.components_initialized = success_count >= (total_components * 0.5)  # At least 50% success
            
            if self.components_initialized:
                print(f"✅ Client '{self.client_id}' initialized successfully ({success_count}/{total_components} components)")
            else:
                print(f"❌ Client '{self.client_id}' initialization failed ({success_count}/{total_components} components)")
            
            return self.components_initialized
    
    def process_chat_message(self, message: str) -> Dict[str, Any]:
        """
        Process chat message using GPT and current emotion state
        Requires: 'gpt' module enabled
        """
        if 'gpt' not in self.enabled_modules:
            raise ValueError("GPT module not enabled for this client")
        
        if not self.gpt_client or not self.gpt_client.is_available():
            raise RuntimeError("GPT client not available")
        
        try:
            # Get current emotion if emotion module is enabled
            detected_emotion = "neutral"
            emotion_confidence = 0.0
            
            if 'emotion' in self.enabled_modules and self.emotion_processor:
                detected_emotion, emotion_confidence = self.emotion_processor.get_current_emotion()
            
            print(f"💬 Processing chat for '{self.client_id}': [{detected_emotion}] {message}")
            
            # Process with GPT
            response_text = self.gpt_client.ask_chatgpt_optimized(message, detected_emotion, emotion_confidence)
            bot_emotion = self.gpt_client.extract_emotion_tag(response_text)
            
            result = {
                'response': response_text,
                'bot_emotion': bot_emotion,
                'detected_emotion': detected_emotion,
                'confidence': round(emotion_confidence, 1),
                'emotion_distribution': self.emotion_processor.get_emotion_distribution() if self.emotion_processor else {}
            }
            
            print(f"🤖 GPT response for '{self.client_id}': {response_text}")
            return result
            
        except Exception as e:
            print(f"❌ Chat processing error for '{self.client_id}': {e}")
            raise RuntimeError(f"Chat processing failed: {e}")
    
    def process_speech_input(self, audio_b64: str) -> Dict[str, Any]:
        """
        Process speech input (speech-to-text and optionally chat)
        Requires: 'speech' module enabled
        """
        if 'speech' not in self.enabled_modules:
            raise ValueError("Speech module not enabled for this client")
        
        if not self.speech_processor or not self.speech_processor.is_available():
            raise RuntimeError("Speech processor not available")
        
        try:
            print(f"🎤 Processing speech for '{self.client_id}'")
            
            # Transcribe audio
            success, transcription, speech_confidence = self.speech_processor.transcribe_audio_base64(audio_b64)
            
            if not success:
                raise RuntimeError(f"Speech transcription failed: {transcription}")
            
            if not transcription or len(transcription.strip()) < 2:
                raise ValueError(f"No meaningful speech detected: '{transcription}'")
            
            print(f"📝 Transcribed for '{self.client_id}': '{transcription}' (confidence: {speech_confidence:.1f}%)")
            
            result = {
                'transcription': transcription,
                'confidence': round(speech_confidence, 1)
            }
            
            # If GPT is also enabled, process the transcription as a chat message
            if 'gpt' in self.enabled_modules and self.gpt_client and self.gpt_client.is_available():
                try:
                    chat_result = self.process_chat_message(transcription)
                    result.update(chat_result)
                    print(f"🔄 Also processed as chat for '{self.client_id}'")
                except Exception as e:
                    print(f"⚠️ Chat processing after speech failed for '{self.client_id}': {e}")
                    # Don't fail the whole request, just return speech results
            
            return result
            
        except Exception as e:
            print(f"❌ Speech processing error for '{self.client_id}': {e}")
            raise RuntimeError(f"Speech processing failed: {e}")
    
    def process_image_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image frame for emotion/facial recognition
        Requires: 'emotion' or 'facial' module enabled
        """
        if 'emotion' not in self.enabled_modules and 'facial' not in self.enabled_modules:
            raise ValueError("Emotion or facial recognition module required for image processing")
        
        if not self.emotion_processor:
            raise RuntimeError("Emotion processor not available")
        
        try:
            frame_b64 = frame_data.get('frame', '')
            if not frame_b64:
                raise ValueError("No frame data provided")
            
            # Decode and process frame
            frame = self.emotion_processor.decode_frame_optimized(frame_b64)
            if frame is None:
                raise ValueError("Failed to decode frame")
            
            # Process emotion detection
            emotion, confidence, status = self.emotion_processor.process_emotion_detection_realtime(frame)
            
            result = {
                'emotion': emotion,
                'confidence': round(confidence, 1),
                'status': status,
                'distribution': self.emotion_processor.get_emotion_distribution(),
                'timestamp': time.time()
            }
            
            # Add facial recognition results if enabled
            if 'facial' in self.enabled_modules:
                # You can add additional facial recognition features here
                # For now, we use the face detection from emotion processor
                result['faces_detected'] = status != "no_faces"
            
            return result
            
        except Exception as e:
            print(f"❌ Image processing error for '{self.client_id}': {e}")
            raise RuntimeError(f"Image processing failed: {e}")
    
    def get_current_emotion_state(self) -> Dict[str, Any]:
        """
        Get current emotion state
        Requires: 'emotion' module enabled
        """
        if 'emotion' not in self.enabled_modules:
            raise ValueError("Emotion module not enabled for this client")
        
        if not self.emotion_processor:
            raise RuntimeError("Emotion processor not available")
        
        try:
            emotion, confidence = self.emotion_processor.get_current_emotion()
            distribution = self.emotion_processor.get_emotion_distribution()
            
            return {
                'emotion': emotion,
                'confidence': round(confidence, 1),
                'distribution': distribution,
                'last_update': self.emotion_processor.last_emotion_update
            }
            
        except Exception as e:
            print(f"❌ Emotion state error for '{self.client_id}': {e}")
            raise RuntimeError(f"Failed to get emotion state: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status of all components"""
        try:
            status = {
                'client_id': self.client_id,
                'enabled_modules': list(self.enabled_modules),
                'components_initialized': self.components_initialized,
                'components': {}
            }
            
            # Check each module
            if 'emotion' in self.enabled_modules:
                status['components']['emotion'] = {
                    'available': self.emotion_processor is not None,
                    'model_loaded': self.emotion_processor.model_loaded if self.emotion_processor else False,
                    'face_cascade_loaded': self.emotion_processor.face_cascade_loaded if self.emotion_processor else False
                }
            
            if 'gpt' in self.enabled_modules:
                status['components']['gpt'] = {
                    'available': self.gpt_client is not None,
                    'openai_ready': self.gpt_client.is_available() if self.gpt_client else False
                }
            
            if 'speech' in self.enabled_modules:
                status['components']['speech'] = {
                    'available': self.speech_processor is not None,
                    'whisper_ready': self.speech_processor.is_available() if self.speech_processor else False
                }
            
            if 'facial' in self.enabled_modules:
                status['components']['facial'] = {
                    'available': self.emotion_processor is not None,
                    'face_detection_ready': self.emotion_processor.face_cascade_loaded if self.emotion_processor else False
                }
            
            return status
            
        except Exception as e:
            print(f"❌ Health check error for '{self.client_id}': {e}")
            return {
                'client_id': self.client_id,
                'error': f"Health check failed: {e}",
                'components_initialized': False
            }
    
    def cleanup_resources(self):
        """Clean up resources when client server is being destroyed"""
        try:
            print(f"🧹 Cleaning up resources for client '{self.client_id}'")
            
            # Clear emotion tracking history
            if self.emotion_processor and hasattr(self.emotion_processor, 'emotion_tracker'):
                self.emotion_processor.emotion_tracker.emotion_history.clear()
                self.emotion_processor.emotion_tracker.confidence_history.clear()
                self.emotion_processor.emotion_tracker.emotion_counts.clear()
            
            # Clear any other resources
            self.components_initialized = False
            
            print(f"✅ Cleanup completed for client '{self.client_id}'")
            
        except Exception as e:
            print(f"❌ Cleanup error for client '{self.client_id}': {e}")

# For backward compatibility - this allows the original server.py to still work standalone
def main():
    """
    Standalone mode - creates a single default client server
    This is for backward compatibility only. 
    Use ServerController for multi-client support.
    """
    print("⚠️ WARNING: Running in standalone mode (single client)")
    print("   For multi-client support, use server_controller.py instead")
    print()
    
    # Create a default client configuration
    default_modules = {'gpt', 'emotion', 'speech', 'facial'}
    default_config = {
        'emotion_processing_interval': 0.1,
        'stream_fps': 30,
        'frame_skip_ratio': 1,
        'emotion_update_threshold': 0.05,
        'emotion_window_size': 5,
        'confidence_threshold': 30.0,
        'emotion_change_threshold': 15.0,
        'whisper_model_size': 'base',
        'whisper_device': 'auto',
        'whisper_compute_type': 'float16',
        'max_audio_length': 30,
        'sample_rate': 16000
    }
    
    # Create and initialize server
    server = RobotServer.create_for_client(
        client_id="standalone_client",
        enabled_modules=default_modules,
        config=default_config
    )
    
    if server.initialize_components():
        print("✅ Standalone server ready")
        print("   Note: This mode doesn't support HTTP/WebSocket APIs")
        print("   Use server_controller.py for full functionality")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Standalone server shutdown")
            server.cleanup_resources()
    else:
        print("❌ Standalone server initialization failed")

if __name__ == "__main__":
    main()