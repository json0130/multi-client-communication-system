# client.py - Base Client and Abstract Classes
import json
import time
import threading
import socketio
import requests
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
        
        # Initialize Socket.IO client
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )
        
        # Initialize HTTP session
        self.session = requests.Session()
        self.session.timeout = 30
        
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
        """Send client initialization to server"""
        client_init_data = {
            "robot_name": self.client_config.get('robot_name', 'BasicClient'),
            "modules": self.client_config.get('modules', ['gpt']),
            "client_id": self.client_id,
            "config": self.client_config.get('server_config', {})
        }
        
        logger.info("📋 Sending client initialization...")
        self.sio.emit('client_init', client_init_data)
    
    def connect(self) -> bool:
        """Connect to server and wait for initialization"""
        try:
            logger.info(f"🔗 Connecting to server: {self.server_url}")
            
            self.sio.connect(
                self.server_url,
                transports=['websocket'],
                wait_timeout=10
            )
            
            # Wait for initialization
            timeout = 15
            start_time = time.time()
            while not self.initialized and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.initialized:
                logger.info("✅ Client ready")
                return True
            else:
                logger.error("❌ Client initialization timeout")
                return False
        
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.connected:
            self.sio.disconnect()
    
    def send_chat_message(self, message: str) -> Optional[Dict]:
        """Send chat message via HTTP"""
        try:
            url = f"{self.server_url}/client/{self.client_id}/chat"
            payload = {"message": message}
            
            response = self.session.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Chat request failed: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error sending chat message: {e}")
            return None
    
    def send_speech_data(self, audio_data: bytes) -> Optional[Dict]:
        """Send speech audio via HTTP"""
        try:
            import base64
            url = f"{self.server_url}/client/{self.client_id}/speech"
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            payload = {"audio": audio_b64}
            
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Speech request failed: {response.status_code}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Error sending speech data: {e}")
            return None
    
    def send_frame_data(self, frame_data: Any) -> bool:
        """Send frame data via WebSocket"""
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
        
        # Initialize server connection
        self.server_connection = ServerConnection(
            self.config['server_url'], 
            self.config
        )
        
        # Module storage
        self.input_modules: Dict[str, InputModule] = {}
        self.output_modules: Dict[str, OutputModule] = {}
        
        # State
        self.running = False
        
        logger.info(f"🤖 Initializing {self.config['robot_name']}")
        logger.info(f"   🆔 ID: {self.config['client_id']}")
        logger.info(f"   🌐 Server: {self.config['server_url']}")
        logger.info(f"   📦 Modules: {', '.join(self.config['modules'])}")
    
    def _load_config(self, config_file: str) -> Optional[Dict]:
        """Load client configuration from JSON file"""
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
        """Register an input module"""
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
        """Register an output module"""
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
        """Send data to server based on type"""
        if data_type == 'chat':
            return self.server_connection.send_chat_message(data)
        elif data_type == 'speech':
            return self.server_connection.send_speech_data(data)
        elif data_type == 'frame':
            # logger.info(f"frame send")
            success = self.server_connection.send_frame_data(data)
            return {'success': success}
        else:
            logger.warning(f"⚠️ Unknown data type: {data_type}")
            return None
    
    def process_server_response(self, response_data: Dict, response_type: str = "chat"):
        """Process response from server and send to appropriate output modules"""
        if not response_data:
            logger.warning("⚠️ No response data received")
            return
        
        # Extract response text
        response_text = response_data.get("response", "")
        if response_text:
            logger.info(f"🤖 Server response: {response_text}")
            
            # Send to all output modules
            for module_name, module in self.output_modules.items():
                try:
                    module.process_output({
                        'text': response_text,
                        'type': response_type,
                        'full_response': response_data
                    })
                except Exception as e:
                    logger.error(f"❌ Error in output module '{module_name}': {e}")
        
        # Log transcription if speech input
        if response_type == "speech":
            transcription = response_data.get("transcription", "")
            if transcription:
                logger.info(f"📝 Transcribed: '{transcription}'")
    
    def start(self) -> bool:
        """Start the client system"""
        try:
            # Check server health
            logger.info("🏥 Checking server health...")
            if not self._check_server_health():
                logger.error("❌ Server is not healthy")
                return False
            
            # Connect to server
            if not self.server_connection.connect():
                logger.error("❌ Failed to connect to server")
                return False
            
            # Start all modules
            self.running = True
            
            # Start input modules
            for module_name, module in self.input_modules.items():
                try:
                    if module.start():
                        logger.info(f"🎯 Started input module: {module_name}")
                    else:
                        logger.warning(f"⚠️ Failed to start input module: {module_name}")
                except Exception as e:
                    logger.error(f"❌ Error starting input module '{module_name}': {e}")
            
            # Start output modules
            for module_name, module in self.output_modules.items():
                try:
                    if module.start():
                        logger.info(f"🎯 Started output module: {module_name}")
                    else:
                        logger.warning(f"⚠️ Failed to start output module: {module_name}")
                except Exception as e:
                    logger.error(f"❌ Error starting output module '{module_name}': {e}")
            
            logger.info("✅ Client system started successfully")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error starting client: {e}")
            return False
    
    def stop(self):
        """Stop the client system"""
        logger.info("🛑 Stopping client system...")
        self.running = False
        
        # Stop all modules
        for module_name, module in list(self.input_modules.items()) + list(self.output_modules.items()):
            try:
                module.stop()
                logger.info(f"🛑 Stopped module: {module_name}")
            except Exception as e:
                logger.error(f"❌ Error stopping module '{module_name}': {e}")
        
        # Disconnect from server
        self.server_connection.disconnect()
        
        logger.info("✅ Client system stopped")
    
    def run(self):
        """Main run loop"""
        try:
            if not self.start():
                return
            
            logger.info("🚀 Client running... Press Ctrl+C to stop")
            
            # Main loop - keep running while modules are active
            while self.running:
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("🛑 Received stop signal")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.stop()
    
    def _check_server_health(self) -> bool:
        """Check if server is healthy"""
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
