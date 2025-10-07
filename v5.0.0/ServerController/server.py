# server.py - Individual Client Server Instance (Enhanced for Individual Monitoring)
import os
import time
import threading
import json
from typing import Set, Dict, Any, Optional
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# default config attributes
from database import Database
from supabase_client import SupabaseClient
db = Database()
uid = db.create_user(name="Standalone")  # Create a default user for standalone mode

# Import modular components
from Modules.emotion_processor import EmotionProcessor
from Modules.gpt_client import GPTClient
from Modules.web_interface import WebInterface
from Modules.speech_processor import SpeechProcessor
from Modules.rag_module import RagModule

# Configuration - CORRECTED PATH FOR YOUR SETUP
MODEL_PATH = '../models/efficientnet_HQRAF_improved_withCon.pth'  # Your existing model
API_KEY = "emotion_recognition_key_123"
load_dotenv()

class RobotServer:
    """
    Individual server instance for a single client with individual web monitoring.
    Each client gets their own instance with custom module configuration and individual monitor.
    """
    
    def __init__(self, client_id: str, enabled_modules: Set[str], config: Dict[str, Any], robot_registry: Dict):
        # Client identification
        self.client_id = client_id
        self.enabled_modules = enabled_modules
        self.config = config
        self.robot_name = None
        self.robot_registry = robot_registry
        
        # user identification
        self.user_id = config.get('user_id')
        
        # Module instances (initialized based on enabled_modules)
        self.emotion_processor = None
        self.gpt_client = None
        self.speech_processor = None
        self.web_interface = None  # Individual web interface for this client
        self.rag = None
        
        # Individual client monitoring data
        self.latest_frame = None
        self.latest_emotion = "neutral"
        self.latest_confidence = 0.0
        self.last_update_time = time.time()
        
        
        # State tracking
        self.components_initialized = False
        self.initialization_lock = threading.Lock()
        self.frame_lock = threading.Lock()  # Lock for frame access

        # 🚀 PERFORMANCE: Add frame optimization variables
        self.frame_skip_counter = 0
        self.frame_skip_ratio = config.get('frame_skip_ratio', 3)
        self.max_frame_age = 1.0  # Max age of frame in seconds
        self.last_monitor_frame_time = 0
        self.last_broadcast_time = 0  # For throttling broadcasts
        
        # Monitor-specific settings
        self.monitor_quality = config.get('monitor_quality', 50)
        self.monitor_resolution = config.get('monitor_resolution', (320, 240))
        self.broadcast_throttle = config.get('broadcast_throttle', 0.2)  # 5 updates per second max

        # ✅ ADD THIS DEBUG LINE:
        print(f"🔍 DEBUG [{client_id}]: config keys = {list(config.keys())}")
        print(f"🔍 DEBUG [{client_id}]: robot_role = {config.get('robot_role', 'NOT FOUND')}")
        
        print(f"🎯 Created server instance for client '{self.client_id}' with modules: {list(self.enabled_modules)}")
        print(f"🚀 Performance settings: {self.monitor_resolution} @ {self.monitor_quality}% quality, skip ratio: {self.frame_skip_ratio}")
    
    @classmethod
    def create_for_client(cls, client_id: str, enabled_modules: Set[str], config: Dict[str, Any], robot_registry: Dict):
        """Factory method to create a server instance for a specific client"""
        return cls(client_id, enabled_modules, config, robot_registry)
    
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
                
            # Initialize RAG Module
            if 'rag' in self.enabled_modules:
                try:
                    if self.user_id is not None and self.config.get("database"):
                        self.rag = RagModule(self.user_id, self.config["database"].client)
                        print("    ✅ RAG module initialized")
                        success_count += 1
                    else:
                        print("    ❌ RAG init skipped (missing user_id or database)")
                except Exception as e:
                    print(f"    ❌ RAG initialization failed: {e}")
            else:
                print("  📄 RAG module disabled")
            
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
            
            # Initialize Individual Web Interface (always for monitoring)
            print(f"  🌐 Initializing individual web interface for client '{self.client_id}'...")
            try:
                # Create web interface specifically for this client
                stream_fps = self.config.get('stream_fps', 10)  # Lower FPS for individual monitoring
                self.web_interface = WebInterface(stream_fps=stream_fps)
                
                # Store client info in web interface
                self.web_interface.client_id = self.client_id
                self.web_interface.enabled_modules = list(self.enabled_modules)
                
                print(f"    ✅ Individual web interface initialized for client '{self.client_id}'")
                print(f"    🖥️ Monitor will be available at: /client/{self.client_id}/monitor")
                
            except Exception as e:
                print(f"    ⚠️ Web interface setup warning for '{self.client_id}': {e}")
            
            # Determine if initialization was successful
            self.components_initialized = success_count >= (total_components * 0.5)  # At least 50% success
            
            if self.components_initialized:
                print(f"✅ Client '{self.client_id}' initialized successfully ({success_count}/{total_components} components)")
            else:
                print(f"❌ Client '{self.client_id}' initialization failed ({success_count}/{total_components} components)")
            
            return self.components_initialized
    
    def process_image_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process image frame for emotion/facial recognition and update individual monitoring
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
            
            # 🚀 PERFORMANCE: Smart frame updating for monitors
            current_time = time.time()
            self.frame_skip_counter += 1
            
            # Only update monitor frame based on skip ratio OR if too much time has passed
            should_update_monitor = (
                self.frame_skip_counter % self.frame_skip_ratio == 0 or 
                current_time - self.last_monitor_frame_time > 0.5
            )
            
            if should_update_monitor:
                # 🚀 PERFORMANCE: Create optimized frame for monitor display
                monitor_frame = self._prepare_monitor_frame(frame)
                
                with self.frame_lock:
                    self.latest_frame = monitor_frame
                    self.last_monitor_frame_time = current_time
            
            # Always update emotion data (this is lightweight)
            self.latest_emotion = emotion
            self.latest_confidence = confidence
            self.last_update_time = current_time
            
            result = {
                'emotion': emotion,
                'confidence': round(confidence, 1),
                'status': status,
                'distribution': self.emotion_processor.get_emotion_distribution(),
                'timestamp': self.last_update_time,
                'client_id': self.client_id
            }
            
            # Add facial recognition results if enabled
            if 'facial' in self.enabled_modules:
                result['faces_detected'] = status != "no_faces"
            
            # 🚀 PERFORMANCE: Log only significant emotion changes
            if self.frame_skip_counter % 10 == 0:  # Log every 10th frame only
                print(f"🎭 {self.client_id}: {emotion} ({confidence:.1f}%)")
            
            return result
            
        except Exception as e:
            print(f"❌ Image processing error for '{self.client_id}': {e}")
            raise RuntimeError(f"Image processing failed: {e}")

    def process_chat_message(self, message: str, is_delegated_command: bool = False) -> Dict[str, Any]:
        """
        Processes a chat message. Decides whether to use delegation or execution logic.
        """
        print(f"🧠 Robot '{self.client_id}' processing message. Delegated: {is_delegated_command}")

        # 1. Get the appropriate prompt from a helper method
        if is_delegated_command:
            final_prompt = self._get_execution_prompt(message)
        else:
            final_prompt = self._get_delegation_prompt(message)
            
        # 2. Both modes use the same flexible GPT call
        response_text = self.gpt_client.ask_with_dynamic_prompt(final_prompt)
            
        # 3. Log the interaction to the database and update RAG
        if self.config.get("database") and self.user_id is not None:
            try:
                self.config["database"].insert_chat_log(self.user_id, message, response_text)
                if self.rag:
                    self.rag.add(message)
            except Exception as e:
                print(f"[DB/RAG] Failed to log or add embedding: {e}")

        # 4. Process and return the final result
        bot_emotion = self.gpt_client.extract_emotion_tag(response_text)
            
        result = {
            'response': response_text,
            'bot_emotion': bot_emotion,
            'detected_emotion': self.latest_emotion,
            'confidence': round(self.latest_confidence, 1),
            'emotion_distribution': self.emotion_processor.get_emotion_distribution() if self.emotion_processor else {},
            'client_id': self.client_id
        }
            
        print(f"🤖 GPT response for '{self.client_id}': {response_text}")
        return result

    def _get_delegation_prompt(self, user_message: str) -> str:
        """Constructs the system prompt for DELEGATION MODE."""
        my_role = self.config.get('robot_role', 'You are a helpful robot.')

        print(f"🔍 DEBUG: Using role prompt: {my_role[:100]}...")

        rag_context = ""
        if self.rag:
            try:
                context_texts = self.rag.search(user_message, top_k=5)
                if context_texts:
                    print(f"🔍 RAG: Found {len(context_texts)} relevant messages")
                    print(f"🔍 RAG: Retrieved context:")
                    for i, text in enumerate(context_texts, 1):
                        print(f"   {i}. '{text}'")
                    
                    rag_context += "The user has previously told you:\n"
                    rag_context += "\n".join(f'- "{text}"' for text in context_texts)
                    rag_context += "\n\n Try refer to this information when answering their questions."
                    
                    print(f"🔍 RAG: Injected context into prompt")
                else:
                    print(f"🔍 RAG: No context found")
            except Exception as e:
                print(f"⚠️ RAG search failed: {e}")
        
        network_robots_overview = []
        try:
            # Fetch all robots from Supabase
            robots_data = self.config['database'].client.supabase.table('robots').select('client_id, robot_name, robot_role').execute()
            
            if robots_data.data:
                for robot in robots_data.data:
                    # Don't include yourself in the team list
                    if robot['client_id'] != self.client_id:
                        network_robots_overview.append({
                            "robot_id": robot['client_id'],
                            "robot_name": robot['robot_name'],
                            "role_description": robot['robot_role']
                        })
                
                print(f"🔍 DEBUG: Loaded {len(network_robots_overview)} teammates from database")
            else:
                print(f"⚠️ No other robots found in database")
                
        except Exception as e:
            print(f"⚠️ Failed to fetch robot registry from DB: {e}")
            # Fallback to empty list if DB fetch fails
            network_robots_overview = []

        formatted_robot_list = json.dumps(network_robots_overview, indent=2)

        return (
            f"System: Your identity is Robot ID '{self.client_id}' and your role is: '{my_role}'.\n\n"
            f"{rag_context}\n\n"
            "You have a base capability to answer simple conversational questions.\n\n"
            f"Here is your team: {formatted_robot_list}\n\n"
            "Follow these decision steps:\n"
            "1. Is the user's request a simple conversational question? If YES, answer it directly.\n"
            "2. If it's a specialized task, does it match YOUR role? If YES, perform it and respond directly.\n"
            "3. If the task is better suited for ANOTHER robot, you MUST delegate it.\n\n"
            "**CRITICAL Delegation Rule:**\n"
            "When you create the JSON command block, the value for 'target_robot_id' MUST be the exact 'robot_id' from the team list above (e.g., 'silbot_01'). Do NOT use the robot's name."
            
            "\n\n**BROADCASTING:**\n"
            "If the user asks you to send a message to 'everyone' or 'all robots', you can set the target_robot_id to the special value \"ALL\".\n"
            "Example User Request: \"Tell everyone to say hello.\"\n"
            'Example JSON: { "target_robot_id": "ALL", "task": "Team, the user wants us all to say hello." }'
            
            "\n\n**Example JSON:**\n"
            "```json\n"
            '{ "target_robot_id": "silbot_01", "task": "Silbot, please fetch water." }\n'
            "```"
            f"\nUser: {user_message}"
        )

    def _get_execution_prompt(self, task_message: str) -> str:
        """Constructs the system prompt for EXECUTION MODE."""
        # --- FIX: Fetch the role from the robot_registry for consistency ---
        robot_role = self.config.get('robot_role', 'You are a helpful robot.')

        return (
            f"System: Your persona is: \"{robot_role}\". You are acting as this robot.\n"
            f"You have received a direct order from a teammate. The order is: '{task_message}'.\n"
            "Your only job is to respond in character with a confident confirmation that you are executing this exact order now. Do not question the order or refuse the task."
            f"\nUser: {task_message}"
        )
    
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
                'confidence': round(speech_confidence, 1),
                'client_id': self.client_id
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
    
    def get_current_emotion_state(self) -> Dict[str, Any]:
        """
        Get current emotion state
        Requires: 'emotion' module enabled
        """
        if 'emotion' not in self.enabled_modules:
            raise ValueError("Emotion module not enabled for this client")
        
        try:
            return {
                'emotion': self.latest_emotion,
                'confidence': round(self.latest_confidence, 1),
                'last_update': self.last_update_time,
                'client_id': self.client_id,
                'status': 'active' if time.time() - self.last_update_time < 10 else 'inactive',
                'distribution': self.emotion_processor.get_emotion_distribution() if self.emotion_processor else {}
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
                'last_activity': self.last_update_time,
                'current_emotion': self.latest_emotion,
                'current_confidence': round(self.latest_confidence, 1),
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
            
            # Individual web interface status
            status['components']['web_interface'] = {
                'available': self.web_interface is not None,
                'monitor_url': f'/client/{self.client_id}/monitor',
                'stream_url': f'/client/{self.client_id}/live_stream'
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
            
            # Clear individual monitoring data
            self.latest_frame = None
            self.latest_emotion = "neutral"
            self.latest_confidence = 0.0
            
            # Clear emotion tracking history
            if self.emotion_processor and hasattr(self.emotion_processor, 'emotion_tracker'):
                self.emotion_processor.emotion_tracker.emotion_history.clear()
                self.emotion_processor.emotion_tracker.confidence_history.clear()
                self.emotion_processor.emotion_tracker.emotion_counts.clear()
            
            # Clean up web interface
            if self.web_interface:
                self.web_interface = None
            
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
    default_modules = {'gpt', 'emotion', 'speech', 'facial', 'rag'}
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
        'sample_rate': 16000,
        'database': db,
        'user_id': uid,  # Default user ID for standalone mode
    }
    
    # Create and initialize server
    server = RobotServer.create_for_client(
        client_id="standalone_client",
        enabled_modules=default_modules,
        config=default_config
    )
    
    if server.initialize_components():
        print("✅ Standalone server ready")
        
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