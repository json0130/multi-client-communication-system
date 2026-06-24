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
from Modules.llm_processor import OllamaClient
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
        print(f"🔍 DEBUG [{client_id}]: initial allowed_tags in config = {config.get('allowed_tags', 'NOT FOUND')}")
        
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
                    self.gpt_client = OllamaClient(model_name="qwen2.5:7b")
                    if self.gpt_client.setup_client():
                        success_count += 1
                        print(f"    ✅ GPT client initialized")
                    else:
                        print(f"    ❌ GPT client initialization failed")
                        
                except Exception as e:
                    print(f"    ❌ GPT initialization error: {e}")
            else:
                # Create mock GPT client for compatibility
                self.gpt_client = OllamaClient(model_name="qwen2.5:3b")
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
    
    def _prepare_monitor_frame(self, frame):
        """Helper for monitor frame resizing (fallback if not defined elsewhere)"""
        import cv2
        if frame is None:
            return None
        return cv2.resize(frame, self.monitor_resolution)

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

    def _get_allowed_tags_info(self, config_tags: list) -> tuple[str, str]:
        """Helper to cleanly format the allowed tags from the config."""
        # print(f"🔍 DEBUG [_get_allowed_tags_info]: Received config_tags = {config_tags}")
        if config_tags and isinstance(config_tags, list) and len(config_tags) > 0:
            allowed_list = ", ".join(config_tags)
            safe_example = config_tags[0]
            # print(f"🔍 DEBUG [_get_allowed_tags_info]: Returning list = '{allowed_list}', example = '{safe_example}'")
            return allowed_list, safe_example
        
        print("⚠️ DEBUG [_get_allowed_tags_info]: config_tags empty/invalid, falling back to [DEFAULT]")
        return "[DEFAULT]", "[DEFAULT]"

    def process_chat_message(self, message: str, is_delegated_command: bool = False) -> Dict[str, Any]:
        """
        Processes a chat message. Decides whether to use delegation or execution logic.
        """

        self._refresh_config_from_db()

        print(f"🧠 Robot '{self.client_id}' processing message. Delegated: {is_delegated_command}")

        # 1. Get the appropriate prompt from a helper method
        if is_delegated_command:
            final_prompt = self._get_execution_prompt(message)
        else:
            final_prompt = self._get_delegation_prompt(message)
            
        # 2. Both modes use the same flexible GPT call
        response_text = self.gpt_client.ask_with_dynamic_prompt(final_prompt)
        
        # 3. Log the interaction to the database and update RAG
        if self.config.get("database") and getattr(self, 'user_id', None) is not None:
            try:
                self.config["database"].insert_chat_log(self.user_id, message, response_text)
                if self.rag:
                    self.rag.add(message)
            except Exception as e:
                print(f"[DB/RAG] Failed to log or add embedding: {e}")

        # 4. Process and return the final result
        bot_emotion = self.gpt_client.extract_emotion_tag(response_text)
        print(f"🔍 DEBUG [process_chat_message] EXTRACTED TAG: '{bot_emotion}'")
            
        result = {
            'response': response_text,
            'bot_emotion': bot_emotion,
            'detected_emotion': self.latest_emotion,
            'confidence': round(self.latest_confidence, 1),
            'emotion_distribution': self.emotion_processor.get_emotion_distribution() if self.emotion_processor else {},
            'client_id': self.client_id
        }
            
        print(f"🤖 GPT response for '{self.client_id}' mapped to final output.")
        return result


    def _get_delegation_prompt(self, user_message: str) -> str:
        """Constructs the system prompt for DELEGATION MODE."""
        my_role = self.config.get('robot_role', 'You are a helpful robot.')
        my_tags_list = self.config.get('allowed_tags', ['[DEFAULT]'])
        
        allowed_tags, example_tag = self._get_allowed_tags_info(my_tags_list)
        print (f'{allowed_tags}')

        active_robots = self._get_active_robots_info()
        if not active_robots or active_robots.strip() == "":
            active_robots = "None. You are currently the only active robot on the network."
            
        print(f"🤖 Active Robots injected into prompt: \n{active_robots}")


        rag_context = ""
        if self.rag:
            try:
                context_texts = self.rag.search(user_message, top_k=5)
                if context_texts:
                    rag_context += "The user has previously told you:\n"
                    rag_context += "\n".join(f'- "{text}"' for text in context_texts)
            except Exception as e:
                pass
        
        return f"""System: You are {self.client_id}. Your role is: '{my_role}'.

*** MANDATORY FORMATTING ***
1.THE VERY FIRST CHARACTER of your response MUST be an open bracket '['. Never start with a word, greeting, or space.\n"
2. CRITICAL: You may ONLY use EXACTLY ONE tag from this list: {allowed_tags}. Do NOT add a second tag in the middle of your sentence!
3. DO NOT use angle brackets.

{rag_context}

*** TEAMMATES & DELEGATION RULES ***
CURRENTLY AVAILABLE ACTIVE ROBOTS:
{active_robots}

**CRITICAL RULES** HOW TO HANDLE REQUESTS (FOLLOW THESE STEPS IN ORDER):

STEP 1: SELF-CHECK
Can you fulfill the request using your own role? 
- If YES: Just answer the user normally. Ignore steps 2 and 3.

Step 2: Check TEAMMATES
If you CANNOT fulfill the request, carefully read the "CURRENTLY AVAILABLE ACTIVE ROBOTS" list. Is there a teammate whose role matches what the user wants?
- If NO (list is empty or no match): Politely explain that neither you nor any available teammate can do that.
  (Example: [{example_tag}] I'm sorry, but I can't build a spaceship, and none of the active robots on my team can either.)
- If YES (a teammate matches): Tell the user you cannot do it, state the teammate's name who can, and ASK the user if they want you to ask that teammate. DO NOT output JSON yet.
  (Example: [{example_tag}] I cannot cook pizzas, but ChefBot can. Would you like me to ask them to make one for you?)

STEP 3: EXECUTE DELEGATION (JSON FORMAT)
ONLY if the user is explicitly saying "yes" to your previous offer to delegate:
You MUST output your spoken confirmation AND a JSON block with the exact ID from the active list.
(Example if user says "Yes please"):
{example_tag} I will ask them right away!
```json
{{"target_robot_id": "<EXACT_ID_FROM_ACTIVE_LIST>", "task": "<Target_Name>, can you please help the user?"}}
```
      
CRITICAL DELEGATION RULES:
1. Never invent or hallucinate robot names. Must use exact names and IDs from the "CURRENTLY AVAILABLE ACTIVE ROBOTS" list.
2. Only delegate to ONE robot at a time. Do NOT ask multiple robots for help in the same response.
3. If the user agrees to delegation, you MUST respond with the exact JSON format shown above, filling in the target_robot_id and task appropriately. This is the ONLY way to delegate tasks.
4. If the user does not agree to delegation, do NOT delegate and do NOT send the JSON. Instead, offer alternative help as described in Step 2.
5. Always be polite and helpful, even if delegation is not possible.

User: {user_message}
Assistant: """


    def _get_execution_prompt(self, task_message: str) -> str:
        """Constructs the system prompt for EXECUTION MODE."""
        robot_role = self.config.get('robot_role', 'You are a helpful robot.')
        my_tags_list = self.config.get('allowed_tags', ['[DEFAULT]'])
        # print(f"🔍 DEBUG [_get_execution_prompt]: Fetched allowed_tags from config = {my_tags_list}")
        
        # DYNAMICALLY grab this specific robot's tags!
        allowed_tags, example_tag = self._get_allowed_tags_info(my_tags_list)

        return (
            f"System: Your persona is: \"{robot_role}\". You are acting as this robot.\n"
            "*** MANDATORY FORMATTING ***\n"
            f"1. The VERY FIRST CHARACTER of your response MUST be an emotion tag from this exact list: {allowed_tags}.\n"
            "2. Keep spoken responses to 1 or 2 sentences maximum.\n"
            "3. CRITICAL: DO NOT use angle brackets like <WAVE>. You must ONLY use square brackets [ ] at the very beginning of the sentence.\n\n"
            f"Correct format: {example_tag} Here is my response.\n"
            f"Incorrect format: {example_tag.strip('[]')} Here is my response.\n"
            f"You have received a direct order from a teammate. The order is: '{task_message}'.\n"
            "Your only job is to respond in character with a confident confirmation that you are executing this exact order now. Do not question the order.\n\n"
            "*** EXAMPLES OF CORRECT RESPONSES ***\n"
            f"User: '{task_message}'\n"
            f"Assistant: {example_tag} I am executing that command right away!\n\n"
            f"User: '{task_message}'\n"
            f"Assistant: "
        )
    
    def _refresh_config_from_db(self):
        """Fetches the absolute latest tags and role directly from Supabase"""
        try:
            db = self.config.get('database')
            if db and hasattr(db, 'client'):
                response = db.client.supabase.table('robots').select('robot_role, allowed_tags').eq('client_id', self.client_id).execute()
                
                if response.data and len(response.data) > 0:
                    latest_data = response.data[0]
                    
                    # Live update the RAM config!
                    if latest_data.get('robot_role'):
                        self.config['robot_role'] = latest_data['robot_role']
                    if latest_data.get('allowed_tags'):
                        self.config['allowed_tags'] = latest_data['allowed_tags']
        except Exception as e:
            print(f"⚠️ Failed to sync tags with DB: {e}")

    def _get_active_robots_info(self) -> str:
        """Fetches a formatted list of currently active robots and their roles from Supabase."""
        try:
            db = self.config.get('database')
            if db and hasattr(db, 'client'):
                # 1. Query DB for active robots, excluding itself
                response = db.client.supabase.table('robots').select(
                    'client_id, robot_name, robot_role'
                ).eq('is_active', True).neq('client_id', self.client_id).execute()
                
                active_robots = response.data
                
                # 2. If no other robots are online
                if not active_robots:
                    return "No other robots are currently online."
                
                # 3. Format the list for the LLM
                info_lines = []
                for robot in active_robots:
                    r_id = robot.get('client_id')
                    r_name = robot.get('robot_name')
                    r_role = robot.get('robot_role', 'No specific role.')
                    info_lines.append(f"- ID: '{r_id}' (Name: {r_name}) | Role: {r_role}")
                
                return "\n".join(info_lines)
                
        except Exception as e:
            print(f"⚠️ Failed to fetch active robots: {e}")
            
        return "No other robots are currently online."
    
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