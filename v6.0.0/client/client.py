
# client.py - Base Client and Abstract Classes
import json
import time
import threading
import socketio
import requests
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseModule(ABC):
    """Base class for all input/output modules"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = False
        self.client = None  # Will be set by client
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the module. Return True if successful."""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """Start the module. Return True if successful."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the module and cleanup resources."""
        pass
    
    def set_client(self, client):
        """Set reference to main client for communication"""
        self.client = client

class InputModule(BaseModule):
    """Base class for input modules (sensors, cameras, microphones, etc.)"""
    
    @abstractmethod
    def get_data(self) -> Optional[Any]:
        """Get current data from the input source"""
        pass

class OutputModule(BaseModule):
    """Base class for output modules (TTS, display, actuators, etc.)"""
    
    @abstractmethod
    def process_output(self, data: Any) -> bool:
        """Process output data. Return True if successful."""
        pass

class ServerConnection:
    """Handles WebSocket and HTTP communication with the server"""
    
    def __init__(self, server_url: str, client_config: Dict[str, Any]):
        self.server_url = server_url.rstrip('/')
        self.client_config = client_config
        self.client_id = client_config.get('client_id', 'basic_client_001')
        
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )
        
        self.session = requests.Session()
        # self.session.timeout = 30 # This might be too long, requests will set their own
        
        self.connected = False
        self.initialized = False
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.on('connect')
        def on_connect():
            logger.info("🔗 Connected to server")
            self.connected = True
            self._send_client_init()
        
        @self.sio.on('disconnect')
        def on_disconnect():
            logger.info("🔌 Disconnected from server")
            self.connected = False
            self.initialized = False
        
        @self.sio.on('client_init_response')
        def on_client_init_response(data):
            if data.get('success'):
                logger.info(f"✅ Client initialized: {data.get('client_id')}")
                logger.info(f"   🎯 Enabled modules: {data.get('enabled_modules', [])}")
                self.initialized = True
            else:
                logger.error(f"❌ Client initialization failed: {data.get('message')}")
        
        @self.sio.on('error')
        def on_error(data):
            logger.error(f"❌ Server error: {data.get('message')}")
    
    def _send_client_init(self):
        client_init_data = {
            "robot_name": self.client_config.get('robot_name', 'BasicClient'),
            "modules": self.client_config.get('modules', ['gpt']),
            "client_id": self.client_id,
            "config": self.client_config.get('server_config', {})
        }
        logger.info("📋 Sending client initialization...")
        self.sio.emit('client_init', client_init_data)

    def ensure_connected(self):
        """Checks for a connection and attempts to reconnect if dropped."""
        if not self.sio.connected:
            logger.warning("⚠️ WebSocket connection dropped. Attempting to reconnect...")
            self.connect()
    
    def connect(self) -> bool:
        try:
            logger.info(f"🔗 Connecting to server: {self.server_url}")
            self.sio.connect(
                self.server_url,
                transports=['websocket'],
                wait_timeout=10
            )
            
            timeout = 15
            start_time = time.time()
            while not self.initialized and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.initialized:
                logger.info("✅ Client ready")
                return True
            else:
                logger.error("❌ Client initialization timeout")
                self.disconnect()
                return False
        
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        if self.connected:
            self.sio.disconnect()
    
    def send_chat_message(self, message: str) -> Optional[Dict]:
        try:
            self.ensure_connected()
            url = f"{self.server_url}/client/{self.client_id}/chat"
            payload = {"message": message}
            logger.info(f"DEBUG: Sending chat payload: {payload}")
            response = self.session.post(url, json=payload, timeout=20)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Chat request failed: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error sending chat message: {e}")
            return None
    
    def send_speech_data(self, audio_data: bytes) -> Optional[Dict]:
        try:
            self.ensure_connected()
            logger.info("--- DEBUG: Preparing speech request ---")
            logger.info(f"Type of audio_data: {type(audio_data)}")
            if audio_data:
                logger.info(f"Length of audio_data: {len(audio_data)} bytes")
            else:
                logger.info("CRITICAL: audio_data is None or empty!")
                logger.info("------------------------------------")

            import base64
            url = f"{self.server_url}/client/{self.client_id}/speech"
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            payload = {"audio": audio_b64}
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Speech request failed: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error sending speech data: {e}")
            return None
    
    def send_frame_data(self, frame_data: Any) -> bool:
        if not self.connected or not self.initialized:
            return False
        
        try:
            frame_payload = {
                'frame': frame_data,
                'timestamp': time.time(),
                'source': 'client_camera'
            }
            self.sio.emit('image_frame', frame_payload)
            return True
        
        except Exception as e:
            logger.error(f"❌ Error sending frame: {e}")
            return False

class BasicClient:
    """Main client class that manages modules and server communication"""
    
    def __init__(self, config_file: str = "client_config.json"):
        self.config = self._load_config(config_file)
        if not self.config:
            raise Exception("Failed to load configuration")
        
        self.server_connection = ServerConnection(
            self.config['server_url'], 
            self.config
        )
        
        self.input_modules: Dict[str, InputModule] = {}
        self.output_modules: Dict[str, OutputModule] = {}
        
        self.running = False

        self.heartbeat_thread = None
        self.HEARTBEAT_INTERVAL = 10

        self.tts_started_event = threading.Event()
        
        logger.info(f"🤖 Initializing {self.config['robot_name']}")
        logger.info(f"   🆔 ID: {self.config['client_id']}")
        logger.info(f"   🌐 Server: {self.config['server_url']}")
        logger.info(f"   📦 Modules: {', '.join(self.config['modules'])}")
    
    def _heartbeat_thread(self):
        """
        Sends a small 'heartbeat' message to the server periodically
        to keep the WebSocket connection alive.
        """
        logger.info("❤️ Heartbeat thread started")
        while self.running:
            try:
                # Only send heartbeats if the connection is fully established
                if self.server_connection and self.server_connection.initialized:
                    self.server_connection.sio.emit('heartbeat', {'timestamp': time.time()})
                    logger.debug("❤️ Heartbeat sent")
                
                # Wait for the interval, but check for the stop signal every second
                for _ in range(self.HEARTBEAT_INTERVAL):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.warning(f"⚠️ Heartbeat error: {e}")
                time.sleep(self.HEARTBEAT_INTERVAL) # Wait before retrying on error
        logger.info("❤️ Heartbeat thread stopped")
    # -----------------------------------

    def _load_config(self, config_file: str) -> Optional[Dict]:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Configuration file {config_file} not found")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            return None
    
    def register_input_module(self, module: InputModule) -> bool:
        try:
            module.set_client(self)
            if module.initialize():
                self.input_modules[module.name] = module
                logger.info(f"✅ Input module '{module.name}' registered")
                return True
            else:
                logger.error(f"❌ Failed to initialize input module '{module.name}'")
                return False
        except Exception as e:
            logger.error(f"❌ Error registering input module '{module.name}': {e}")
            return False
    
    def register_output_module(self, module: OutputModule) -> bool:
        try:
            module.set_client(self)
            if module.initialize():
                self.output_modules[module.name] = module
                logger.info(f"✅ Output module '{module.name}' registered")
                return True
            else:
                logger.error(f"❌ Failed to initialize output module '{module.name}'")
                return False
        except Exception as e:
            logger.error(f"❌ Error registering output module '{module.name}': {e}")
            return False
    
    def send_to_server(self, data_type: str, data: Any) -> Optional[Dict]:
        if data_type == 'chat':
            return self.server_connection.send_chat_message(data)
        elif data_type == 'speech':
            return self.server_connection.send_speech_data(data)
        elif data_type == 'frame':
            success = self.server_connection.send_frame_data(data)
            return {'success': success}
        else:
            logger.warning(f"⚠️ Unknown data type: {data_type}")
            return None
    
    def on_emotion_detected(self, emotion_tag: str):
        """Hook for subclasses to handle detected emotions (e.g., sending to Arduino)"""
        pass

    def process_server_response(self, response_data: Dict, response_type: str = "chat"):
        """Process response from server and send to appropriate output modules"""
        if not response_data:
            logger.warning("⚠️ No response data received")
            return
        
        response_text = response_data.get("response", "")
        if not response_text:
            return

        logger.info(f"🤖 Server response: {response_text}")

        # --- CHECKPOINT 2 & 3 FIX: Universal Emotion Parsing ---
        # This will now catch tags from BOTH text chat and voice HTTP requests!
        match = re.search(r'\[(.*?)\]', response_text) 
        if match:
            command = match.group(1)
            logger.info(f"🎯 [CHECKPOINT] Parsed emotion tag from {response_type.upper()}: [{command}]")
            self.on_emotion_detected(command) # Trigger the Arduino
        else:
            logger.debug(f"ℹ️ [CHECKPOINT] No emotion tag found in this {response_type} response.")

        self.tts_started_event.clear()

        for module_name, module in self.output_modules.items():
            try:
                module.process_output({
                    'text': response_text,
                    'type': response_type,
                    'full_response': response_data
                })
            except Exception as e:
                logger.error(f"❌ Error in output module '{module_name}': {e}")
        
        if response_type == "speech":
            transcription = response_data.get("transcription", "")
            if transcription:
                logger.info(f"📝 Transcribed: '{transcription}'")
    
    def start(self) -> bool:
        try:
            logger.info("🏥 Checking server health...")
            if not self._check_server_health():
                logger.error("❌ Server is not healthy")
                return False

            # Register client via HTTP before WebSocket connection
            self.register_via_http()
            
            if not self.server_connection.connect():
                logger.error("❌ Failed to connect to server")
                return False
            
            self.running = True

            if not self.heartbeat_thread:
                self.heartbeat_thread = threading.Thread(target=self._heartbeat_thread, daemon=True)
                self.heartbeat_thread.start()
            
            for module_name, module in self.input_modules.items():
                try:
                    if module.start():
                        logger.info(f"🎯 Started input module: {module_name}")
                except Exception as e:
                    logger.error(f"❌ Error starting input module '{module_name}': {e}")
            
            for module_name, module in self.output_modules.items():
                try:
                    if module.start():
                        logger.info(f"🎯 Started output module: {module_name}")
                except Exception as e:
                    logger.error(f"❌ Error starting output module '{module_name}': {e}")
            
            logger.info("✅ Client system started successfully")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error starting client: {e}")
            return False
    
    def stop(self):
        logger.info("🛑 Stopping client system...")
        self.running = False
        
        all_modules = list(self.input_modules.values()) + list(self.output_modules.values())
        for module in all_modules:
            try:
                module.stop()
                logger.info(f"🛑 Stopped module: {module.name}")
            except Exception as e:
                logger.error(f"❌ Error stopping module '{module.name}': {e}")
        
        self.server_connection.disconnect()
        logger.info("✅ Client system stopped")
    
    def run(self):
        try:
            if not self.start():
                return
            
            logger.info("🚀 Client running... Press Ctrl+C to stop")
            
            while self.running:
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("🛑 Received stop signal")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.stop()
    
    def _check_server_health(self) -> bool:
        try:
            response = requests.get(f"{self.server_connection.server_url}/", timeout=5)
            if response.status_code == 200:
                server_info = response.json()
                logger.info("✅ Server health check passed")
                logger.info(f"   📊 Status: {server_info.get('status', 'unknown')}")
                logger.info(f"   🤖 Active clients: {server_info.get('active_clients', 0)}")
                return True
            else:
                logger.error(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to server: {e}")
            return False

    def register_via_http(self):
        print("📡 Registering client via HTTP...")
        url = f"{self.server_connection.server_url}/register_client"
        payload = {
            "robot_name": self.config.get('robot_name', 'BasicClient'),
            "modules": self.config.get('modules', ['gpt']),
            "client_id": self.config.get('client_id', 'basic_client_001'),
            "robot_role": self.config.get('robot_role', 'default'),
            "allowed_tags": self.config.get('allowed_tags', ['[DEFAULT]'])
        }
        try:
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                print("HTTP registration response:", response.json())
            else:
                print(f"⚠️ HTTP Registration skipped (Server returned {response.status_code})")
        except Exception as e:
            print(f"⚠️ Could not register via HTTP: {e}")
