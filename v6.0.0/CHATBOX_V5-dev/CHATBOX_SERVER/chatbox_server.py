# chatbox_server.py - Main ChatBox Server Controller
import os
import sys
import time
import threading
from typing import Dict, Any, Optional, Set
from pathlib import Path
from dotenv import load_dotenv

# Import modules
from modules.llm_processor import OllamaClient
from modules.rag_module import RagModule
from modules.database import Database

try:
    from modules.emotion_processor import EmotionProcessor
except ImportError:
    EmotionProcessor = None

try:
    from modules.speech_processor import SpeechProcessor
except ImportError:
    SpeechProcessor = None

try:
    from modules.web_interface import WebInterface
except ImportError:
    WebInterface = None

# Load environment variables
load_dotenv()

class ChatBoxServer:
    """
    Main ChatBox server controller for single client.
    Manages all modules and handles processing requests.
    """
    
    def __init__(self):
        # Client information
        self.client_info = {}
        self.enabled_modules = set()
        self.initialized = False
        self.initialization_lock = threading.Lock()
        
        # Module instances
        self.gpt_client = None
        self.emotion_processor = None
        self.speech_processor = None
        self.web_interface = None
        self.rag_module = None
        self.database = None
        
        # State tracking
        self.latest_emotion = "neutral"
        self.latest_confidence = 0.0
        self.last_update_time = time.time()
        
        # Configuration paths
        self.model_path = self._find_model_path()
        
        print("🎯 ChatBox Server created")

    def _find_model_path(self) -> str:
        """Find the emotion recognition model file"""
        possible_paths = [
            '../models/efficientnet_HQRAF_improved_withCon.pth',
            './models/efficientnet_HQRAF_improved_withCon.pth',
            '../../models/efficientnet_HQRAF_improved_withCon.pth',
            'models/efficientnet_HQRAF_improved_withCon.pth',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found emotion model at: {path}")
                return path
        
        print("⚠️ Emotion model not found, will use fallback path")
        return '../models/efficientnet_HQRAF_improved_withCon.pth'

    def initialize_with_config(self, client_config: Dict[str, Any]) -> tuple[bool, str]:
        """Initialize server with client configuration"""
        with self.initialization_lock:
            if self.initialized:
                return True, "Server already initialized"
            
            try:
                print(f"🚀 Initializing ChatBox Server...")
                print(f"🤖 Robot: {client_config.get('robot_name')}")
                print(f"🆔 Client ID: {client_config.get('client_id')}")
                print(f"📦 Modules: {client_config.get('modules', [])}")
                
                # Store client information
                self.client_info = client_config.copy()
                self.enabled_modules = set(client_config.get('modules', []))
                
                # Initialize modules
                success_count = 0
                total_modules = len(self.enabled_modules)
                
                # GPT Module'modules', []))
                
                # Initialize modules
                success_count = 0
                total_modules = len(self.enabled_modules)
                
                # GPT Module
                if 'gpt' in self.enabled_modules:
                    print("  🤖 Initializing Ollama client...")
                    try:
                        self.gpt_client = OllamaClient(model_name="qwen2.5:7b")
                        if self.gpt_client.setup_client():
                            success_count += 1
                            print("    ✅ Ollama client ready")
                        else:
                            print("    ❌ Ollama client failed - is Ollama running?")
                    except Exception as e:
                        print(f"    ❌ Ollama initialization error: {e}")
                
                # Database Module (always initialize for RAG support)
                print("  🗄️ Initializing database...")
                try:
                    self.database = Database()
                    print("    ✅ Database ready")
                except Exception as e:
                    print(f"    ⚠️ Database initialization error: {e}")
                
                # RAG Module (if enabled)
                if 'rag' in self.enabled_modules:
                    print("  🧠 Initializing RAG module...")
                    try:
                        if self.database:
                            # Get or create user in database based on client_id
                            client_id = client_config.get('client_id', 'default')
                            robot_name = client_config.get('robot_name', 'ChatBox')
                            
                            # Try to get existing user or create new one
                            user_id = None
                            try:
                                # You might want to implement a get_user_by_client_id method
                                # For now, create a simple mapping
                                user_id = hash(client_id) % (10**6)  # Simpler hash for user_id
                                
                                # Try to get user, if not found, create one
                                user = self.database.get_user_by_user_id(user_id)
                                if not user:
                                    user_id = self.database.create_user(
                                        name=f"{robot_name} ({client_id})",
                                        interests=[],
                                        health_conditions=[]
                                    )
                                    print(f"    📝 Created new user with ID: {user_id}")
                                else:
                                    print(f"    🔍 Found existing user with ID: {user_id}")
                                
                            except Exception as e:
                                print(f"    ⚠️ User management error: {e}")
                                user_id = hash(client_id) % (10**6)  # Fallback
                            
                            # Initialize RAG with database's Supabase client
                            from supabase_client import supabase
                            self.rag_module = RagModule(user_id, supabase)
                            success_count += 1
                            print(f"    ✅ RAG module ready for user {user_id}")
                        else:
                            print("    ❌ RAG module skipped - Database not available")
                    except Exception as e:
                        print(f"    ❌ RAG initialization error: {e}")
                
                # Emotion Module
                if 'emotion' in self.enabled_modules:
                    print("  😊 Initializing emotion processor...")
                    try:
                        if EmotionProcessor:
                            config = {
                                'emotion_processing_interval': 0.1,
                                'confidence_threshold': 30.0,
                                'emotion_change_threshold': 15.0
                            }
                            self.emotion_processor = EmotionProcessor(self.model_path, config)
                            emotion_success, emotion_total = self.emotion_processor.initialize()
                            
                            if emotion_success >= emotion_total * 0.5:  # At least 50% success
                                success_count += 1
                                print(f"    ✅ Emotion processor ready ({emotion_success}/{emotion_total})")
                            else:
                                print(f"    ⚠️ Emotion processor partial ({emotion_success}/{emotion_total})")
                        else:
                            print("    ❌ EmotionProcessor not available")
                    except Exception as e:
                        print(f"    ❌ Emotion initialization error: {e}")
                
                # Speech Module
                if 'speech' in self.enabled_modules:
                    print("  🎤 Initializing speech processor...")
                    try:
                        if SpeechProcessor:
                            config = {
                                'whisper_model_size': 'base',
                                'whisper_device': 'auto',
                                'whisper_compute_type': 'float16',
                                'max_audio_length': 30,
                                'sample_rate': 16000
                            }
                            self.speech_processor = SpeechProcessor(config)
                            if self.speech_processor.initialize():
                                success_count += 1
                                print("    ✅ Speech processor ready")
                            else:
                                print("    ❌ Speech processor failed")
                        else:
                            print("    ❌ SpeechProcessor not available")
                    except Exception as e:
                        print(f"    ❌ Speech initialization error: {e}")
                
                # Web Interface (always initialize for monitoring)
                print("  🌐 Initializing web interface...")
                try:
                    if WebInterface:
                        self.web_interface = WebInterface(stream_fps=10)
                        print("    ✅ Web interface ready")
                    else:
                        print("    ⚠️ WebInterface not available")
                except Exception as e:
                    print(f"    ⚠️ Web interface error: {e}")
                
                # Check initialization success
                self.initialized = success_count >= max(1, total_modules * 0.5)  # At least 50% or 1 module
                
                if self.initialized:
                    message = f"ChatBox server initialized successfully ({success_count}/{total_modules} modules)"
                    print(f"✅ {message}")
                    return True, message
                else:
                    message = f"Insufficient modules initialized ({success_count}/{total_modules})"
                    print(f"❌ {message}")
                    return False, message
                    
            except Exception as e:
                error_message = f"Server initialization failed: {e}"
                print(f"❌ {error_message}")
                return False, error_message

    def process_chat_message(self, message: str) -> Dict[str, Any]:
        """Process chat message with GPT and optionally RAG"""
        if 'gpt' not in self.enabled_modules:
            raise ValueError("GPT module not enabled")
        
        if not self.gpt_client or not self.gpt_client.is_available():
            raise RuntimeError("GPT client not available")
        
        try:
            print(f"💬 Processing chat: [{self.latest_emotion}] {message}")
            
            # Get RAG context if available
            context_texts = []
            if self.rag_module and 'rag' in self.enabled_modules:
                try:
                    context_texts = self.rag_module.search(message, top_k=5)
                    if context_texts:
                        print(f"🧠 RAG found {len(context_texts)} relevant contexts")
                except Exception as e:
                    print(f"⚠️ RAG search error: {e}")
            
            # Enhance message with context if available
            enhanced_message = message
            if context_texts:
                context_str = "\n".join([f"- {text}" for text in context_texts[:3]])  # Top 3 contexts
                enhanced_message = f"Context from previous conversations:\n{context_str}\n\nCurrent message: {message}"
            
            my_allowed_tags = self.client_info.get('allowed_tags', [])
            print(f"Allowed tags for this client: {my_allowed_tags}")

            if self.gpt_client and self.gpt_client.is_available():
                response_text = self.gpt_client.ask_model_optimized(
                    enhanced_message, 
                    self.latest_emotion, 
                    self.latest_confidence,
                    allowed_tags=my_allowed_tags
                )
                bot_emotion = self.gpt_client.extract_emotion_tag(response_text)
            else:
                response_text = "[SAD] The language model is not available."
                bot_emotion = "SAD"

            # Store conversation in database and RAG if available
            if self.database and self.rag_module and 'rag' in self.enabled_modules:
                try:
                    # Store in database first (proper way)
                    chat_log_id = self.database.insert_chat_log(
                        user_id=self.rag_module.user_id,
                        message=message,
                        response=response_text
                    )
                    
                    # Then add to RAG index
                    self.rag_module.add(message)
                    print(f"    💾 Stored chat log {chat_log_id} and updated RAG index")
                except Exception as e:
                    print(f"    ⚠️ Storage error: {e}")
            
            result = {
                'response': response_text,
                'bot_emotion': bot_emotion,
                'detected_emotion': self.latest_emotion,
                'confidence': round(self.latest_confidence, 1),
                'context_used': len(context_texts) > 0,
                'context_count': len(context_texts),
                'timestamp': time.time()
            }
            
            print(f"🤖 Response: {response_text[:100]}...")
            return result
            
        except Exception as e:
            print(f"❌ Chat processing error: {e}")
            raise RuntimeError(f"Chat processing failed: {e}")

    def process_speech_input(self, audio_b64: str) -> Dict[str, Any]:
        """Process speech input (speech-to-text and optionally chat)"""
        if 'speech' not in self.enabled_modules:
            raise ValueError("Speech module not enabled")
        
        if not self.speech_processor or not self.speech_processor.is_available():
            raise RuntimeError("Speech processor not available")
        
        try:
            print("🎤 Processing speech input...")
            
            # Transcribe audio
            success, transcription, confidence = self.speech_processor.transcribe_audio_base64(audio_b64)
            
            if not success:
                raise RuntimeError(f"Speech transcription failed: {transcription}")
            
            if not transcription or len(transcription.strip()) < 2:
                raise ValueError(f"No meaningful speech detected: '{transcription}'")
            
            print(f"📝 Transcribed: '{transcription}' (confidence: {confidence:.1f}%)")
            
            result = {
                'transcription': transcription,
                'confidence': round(confidence, 1),
                'timestamp': time.time()
            }
            
            # If GPT is also enabled, process as chat message
            if 'gpt' in self.enabled_modules and self.gpt_client and self.gpt_client.is_available():
                try:
                    chat_result = self.process_chat_message(transcription)
                    result.update(chat_result)
                    print("🔄 Also processed as chat")
                except Exception as e:
                    print(f"⚠️ Chat processing after speech failed: {e}")
            
            return result
            
        except Exception as e:
            print(f"❌ Speech processing error: {e}")
            raise RuntimeError(f"Speech processing failed: {e}")

    def process_image_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process image frame for emotion recognition"""
        if 'emotion' not in self.enabled_modules:
            raise ValueError("Emotion module not enabled")
        
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
            
            # Update state
            self.latest_emotion = emotion
            self.latest_confidence = confidence
            self.last_update_time = time.time()
            
            result = {
                'emotion': emotion,
                'confidence': round(confidence, 1),
                'status': status,
                'distribution': self.emotion_processor.get_emotion_distribution(),
                'timestamp': self.last_update_time
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            raise RuntimeError(f"Image processing failed: {e}")

    def get_current_emotion_state(self) -> Dict[str, Any]:
        """Get current emotion state"""
        if 'emotion' not in self.enabled_modules:
            raise ValueError("Emotion module not enabled")
        
        try:
            return {
                'emotion': self.latest_emotion,
                'confidence': round(self.latest_confidence, 1),
                'last_update': self.last_update_time,
                'status': 'active' if time.time() - self.last_update_time < 10 else 'inactive',
                'distribution': self.emotion_processor.get_emotion_distribution() if self.emotion_processor else {}
            }
        except Exception as e:
            print(f"❌ Emotion state error: {e}")
            raise RuntimeError(f"Failed to get emotion state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        try:
            status = {
                'initialized': self.initialized,
                'enabled_modules': list(self.enabled_modules),
                'client_info': {
                    'robot_name': self.client_info.get('robot_name'),
                    'client_id': self.client_info.get('client_id')
                },
                'current_emotion': self.latest_emotion,
                'current_confidence': round(self.latest_confidence, 1),
                'last_activity': self.last_update_time,
                'components': {}
            }
            
            # Check each module status
            if 'gpt' in self.enabled_modules:
                status['components']['gpt'] = {
                    'available': self.gpt_client is not None,
                    'ready': self.gpt_client.is_available() if self.gpt_client else False
                }
            
            if 'emotion' in self.enabled_modules:
                status['components']['emotion'] = {
                    'available': self.emotion_processor is not None,
                    'model_loaded': self.emotion_processor.model_loaded if self.emotion_processor else False,
                    'face_detection_ready': self.emotion_processor.face_cascade_loaded if self.emotion_processor else False
                }
            
            if 'speech' in self.enabled_modules:
                status['components']['speech'] = {
                    'available': self.speech_processor is not None,
                    'ready': self.speech_processor.is_available() if self.speech_processor else False
                }
            
            if 'rag' in self.enabled_modules:
                status['components']['rag'] = {
                    'available': self.rag_module is not None,
                    'user_id': self.rag_module.user_id if self.rag_module else None,
                    'database_connected': self.database is not None
                }
            
            status['components']['web_interface'] = {
                'available': self.web_interface is not None
            }
            
            return status
            
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return {
                'initialized': False,
                'error': f"Status check failed: {e}"
            }

    def cleanup_resources(self):
        """Clean up server resources"""
        try:
            print("🧹 Cleaning up ChatBox server resources...")
            
            # Reset state
            self.latest_emotion = "neutral"
            self.latest_confidence = 0.0
            
            # Clear emotion tracking if available
            if self.emotion_processor and hasattr(self.emotion_processor, 'emotion_tracker'):
                self.emotion_processor.emotion_tracker.emotion_history.clear()
                self.emotion_processor.emotion_tracker.confidence_history.clear()
            
            # Clear other resources
            self.client_info.clear()
            self.enabled_modules.clear()
            self.initialized = False
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")


def main():
    """Main function to start ChatBox server"""
    print("branch")
    print("🤖 ChatBox Server Controller")
    print("   Simplified single-client version")
    print()
    
    try:
        # Import communication manager
        from communication_manager import CommunicationManager
        
        # Create ChatBox server
        chatbox_server = ChatBoxServer()
        
        # Create communication manager
        comm_manager = CommunicationManager(chatbox_server, port=5000)
        
        # Start the server
        comm_manager.start_server()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure communication_manager.py is in the same directory")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown by user")
        if 'chatbox_server' in locals():
            chatbox_server.cleanup_resources()
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())