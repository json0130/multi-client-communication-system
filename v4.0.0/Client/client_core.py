# client_core.py - Core Client for Multi-Client Emotion Server
import requests
import socketio
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
import os

class ClientCore:
    """
    Core client for connecting to multi-client emotion server.
    
    This is the foundation that handles:
    - Server connection and authentication
    - Client initialization with server
    - HTTP API calls (chat, speech, health)
    - WebSocket real-time communication
    - Basic error handling and reconnection
    
    Modules can extend this to add specific functionality.
    """
    
    def __init__(self, config_file: str = "client_config.json"):
        """
        Initialize core emotion client
        
        Args:
            config_file: Path to JSON configuration file
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Basic settings from config
        self.server_url = self.config.get('server_url', '').rstrip('/')
        self.robot_name = self.config.get('robot_name', 'UnknownRobot')
        self.modules = self.config.get('modules', [])
        self.client_id = self.config.get('client_id')
        self.server_config = self.config.get('server_config', {})
        
        # Connection state
        self.connected = False
        self.client_initialized = False
        self.running = False
        
        # HTTP session
        self.session = requests.Session()
        self.session.timeout = self.config.get('network', {}).get('request_timeout', 30)
        
        # WebSocket client
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=self.config.get('network', {}).get('reconnect_attempts', 5),
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )
        
        # Callbacks for modules to register
        self.callbacks = {
            'on_emotion_detected': [],      # List of callbacks for emotion detection
            'on_chat_response': [],         # List of callbacks for chat responses
            'on_speech_response': [],       # List of callbacks for speech responses
            'on_frame_result': [],          # List of callbacks for frame processing results
            'on_error': [],                 # List of callbacks for errors
            'on_connected': [],             # List of callbacks for connection events
            'on_disconnected': []           # List of callbacks for disconnection events
        }
        
        # Setup WebSocket handlers
        self._setup_websocket_handlers()
        
        print(f"🤖 Initialized {self.robot_name} client core")
        print(f"   📦 Modules: {', '.join(self.modules)}")
        print(f"   🌐 Server: {self.server_url}")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if not os.path.exists(config_file):
                print(f"❌ Configuration file not found: {config_file}")
                print("   Creating default configuration...")
                self._create_default_config(config_file)
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"✅ Configuration loaded from {config_file}")
            return config
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            print("   Using default configuration...")
            return self._get_default_config()
    
    def _create_default_config(self, config_file: str):
        """Create default configuration file"""
        default_config = self._get_default_config()
        
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            print(f"✅ Default configuration created: {config_file}")
            print("   Please edit the configuration file and restart")
        except Exception as e:
            print(f"❌ Error creating default configuration: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "robot_name": "DefaultRobot",
            "client_id": None,
            "server_url": "http://localhost:5000",
            "modules": ["gpt", "emotion"],
            "server_config": {
                "confidence_threshold": 30.0,
                "emotion_window_size": 5
            },
            "network": {
                "connection_timeout": 10,
                "request_timeout": 30,
                "reconnect_attempts": 5
            }
        }
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.sio.on('connect')
        def on_connect():
            print(f"🔗 {self.robot_name}: Connected to server, sending initialization...")
            self.connected = True
            self._send_client_init()
            self._trigger_callbacks('on_connected')
        
        @self.sio.on('disconnect')
        def on_disconnect():
            print(f"🔌 {self.robot_name}: Disconnected from server")
            self.connected = False
            self.client_initialized = False
            self._trigger_callbacks('on_disconnected')
        
        @self.sio.on('client_init_response')
        def on_client_init_response(data):
            """Handle client initialization response"""
            if data.get('success'):
                self.client_id = data.get('client_id')
                self.client_initialized = True
                print(f"✅ {self.robot_name}: Initialized successfully!")
                print(f"   🆔 Client ID: {self.client_id}")
                print(f"   🎯 Enabled modules: {data.get('enabled_modules', [])}")
            else:
                print(f"❌ {self.robot_name}: Initialization failed: {data.get('message')}")
                self.client_initialized = False
                self._trigger_callbacks('on_error', "Initialization failed", data.get('message'))
        
        @self.sio.on('frame_result')
        def on_frame_result(data):
            """Handle emotion detection results from image frames"""
            try:
                result = data.get('result', {})
                emotion = result.get('emotion', 'neutral')
                confidence = result.get('confidence', 0.0)
                distribution = result.get('distribution', {})
                
                # Trigger callbacks
                self._trigger_callbacks('on_emotion_detected', emotion, confidence, distribution)
                self._trigger_callbacks('on_frame_result', data)
                
            except Exception as e:
                print(f"❌ {self.robot_name}: Error processing frame result: {e}")
                self._trigger_callbacks('on_error', "Frame processing error", str(e))
        
        @self.sio.on('chat_response')
        def on_chat_response(data):
            """Handle chat response via WebSocket"""
            try:
                if 'error' not in data:
                    response = data.get('response', '')
                    emotion = data.get('detected_emotion', 'neutral')
                    confidence = data.get('confidence', 0.0)
                    
                    # Trigger callbacks
                    self._trigger_callbacks('on_chat_response', response, emotion, confidence)
                else:
                    error_msg = data.get('error', 'Unknown error')
                    print(f"❌ {self.robot_name}: WebSocket Chat Error: {error_msg}")
                    self._trigger_callbacks('on_error', "WebSocket chat error", error_msg)
                        
            except Exception as e:
                print(f"❌ {self.robot_name}: Error processing chat response: {e}")
                self._trigger_callbacks('on_error', "Chat response error", str(e))
        
        @self.sio.on('error')
        def on_server_error(data):
            error_msg = data.get('message', 'Unknown server error')
            print(f"❌ {self.robot_name}: Server error: {error_msg}")
            self._trigger_callbacks('on_error', "Server error", error_msg)
    
    def _trigger_callbacks(self, callback_type: str, *args):
        """Trigger all registered callbacks of a specific type"""
        for callback in self.callbacks.get(callback_type, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"❌ Callback error in {callback_type}: {e}")
    
    def register_callback(self, callback_type: str, callback: Callable):
        """
        Register a callback function for specific events
        
        Args:
            callback_type: Type of callback ('on_emotion_detected', 'on_chat_response', etc.)
            callback: Function to call when event occurs
        """
        if callback_type in self.callbacks:
            self.callbacks[callback_type].append(callback)
            print(f"📝 Registered callback for {callback_type}")
        else:
            print(f"❌ Unknown callback type: {callback_type}")
    
    def unregister_callback(self, callback_type: str, callback: Callable):
        """Unregister a callback function"""
        if callback_type in self.callbacks and callback in self.callbacks[callback_type]:
            self.callbacks[callback_type].remove(callback)
            print(f"🗑️ Unregistered callback for {callback_type}")
    
    def _send_client_init(self):
        """Send client initialization data to server"""
        client_init_data = {
            "robot_name": self.robot_name,
            "modules": self.modules,
            "config": self.server_config
        }
        
        if self.client_id:
            client_init_data["client_id"] = self.client_id
        
        print(f"📋 {self.robot_name}: Sending initialization data...")
        self.sio.emit('client_init', client_init_data)
    
    def connect(self) -> bool:
        """
        Connect to the emotion server
        
        Returns:
            bool: True if connection and initialization successful
        """
        try:
            print(f"🔗 {self.robot_name}: Connecting to server...")
            
            connection_timeout = self.config.get('network', {}).get('connection_timeout', 10)
            
            self.sio.connect(
                self.server_url,
                transports=['websocket'],
                wait_timeout=connection_timeout
            )
            
            # Wait for initialization to complete
            timeout = 15
            start_time = time.time()
            while not self.client_initialized and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.client_initialized:
                print(f"✅ {self.robot_name}: Connection and initialization complete")
                return True
            else:
                print(f"❌ {self.robot_name}: Initialization timeout")
                return False
            
        except Exception as e:
            print(f"❌ {self.robot_name}: Connection failed: {e}")
            self._trigger_callbacks('on_error', "Connection failed", str(e))
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        self.running = False
        if self.connected:
            self.sio.disconnect()
            print(f"🔌 {self.robot_name}: Disconnected from server")
    
    def send_chat_message(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Send a chat message to the server
        
        Args:
            message: Text message to send
            
        Returns:
            dict: Response from server or None if failed
        """
        if not self.client_initialized:
            print(f"❌ {self.robot_name}: Client not initialized")
            return None
        
        if 'gpt' not in self.modules:
            print(f"❌ {self.robot_name}: GPT module not enabled")
            return None
        
        try:
            payload = {"message": message}
            response = self.session.post(
                f"{self.server_url}/client/{self.client_id}/chat",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                # Trigger callbacks
                self._trigger_callbacks('on_chat_response',
                    data.get('response', ''),
                    data.get('detected_emotion', 'neutral'),
                    data.get('confidence', 0.0)
                )
                return data
            else:
                print(f"❌ {self.robot_name}: Chat request failed: {response.status_code}")
                self._trigger_callbacks('on_error', "Chat request failed", f"Status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ {self.robot_name}: Error sending chat message: {e}")
            self._trigger_callbacks('on_error', "Chat message error", str(e))
            return None
    
    def send_speech_audio(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Send audio data for speech-to-text processing
        
        Args:
            audio_data: WAV audio data as bytes
            
        Returns:
            dict: Response from server or None if failed
        """
        if not self.client_initialized:
            print(f"❌ {self.robot_name}: Client not initialized")
            return None
        
        if 'speech' not in self.modules:
            print(f"❌ {self.robot_name}: Speech module not enabled")
            return None
        
        try:
            import base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            payload = {"audio": audio_b64}
            response = self.session.post(
                f"{self.server_url}/client/{self.client_id}/speech",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                # Trigger callbacks
                self._trigger_callbacks('on_speech_response',
                    data.get('transcription', ''),
                    data.get('response', ''),
                    data.get('confidence', 0.0)
                )
                return data
            else:
                print(f"❌ {self.robot_name}: Speech request failed: {response.status_code}")
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_details = error_data.get('details', 'Unknown error')
                        print(f"   Error details: {error_details}")
                        self._trigger_callbacks('on_error', "Speech request failed", error_details)
                    except:
                        pass
                return None
                
        except Exception as e:
            print(f"❌ {self.robot_name}: Error sending speech audio: {e}")
            self._trigger_callbacks('on_error', "Speech audio error", str(e))
            return None
    
    def send_websocket_frame(self, frame_data: Dict[str, Any]) -> bool:
        """
        Send image frame via WebSocket for real-time processing
        
        Args:
            frame_data: Frame data dictionary
            
        Returns:
            bool: True if sent successfully
        """
        if not self.client_initialized:
            return False
        
        try:
            self.sio.emit('image_frame', frame_data)
            return True
        except Exception as e:
            print(f"❌ {self.robot_name}: Error sending WebSocket frame: {e}")
            self._trigger_callbacks('on_error', "WebSocket frame error", str(e))
            return False
    
    def get_health_status(self) -> Optional[Dict[str, Any]]:
        """
        Get health status from server
        
        Returns:
            dict: Health status or None if failed
        """
        if not self.client_id:
            print(f"❌ {self.robot_name}: Client ID not available")
            return None
        
        try:
            response = self.session.get(
                f"{self.server_url}/client/{self.client_id}/health",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ {self.robot_name}: Health check failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ {self.robot_name}: Error checking health: {e}")
            self._trigger_callbacks('on_error', "Health check error", str(e))
            return None
    
    def get_current_emotion(self) -> Optional[Dict[str, Any]]:
        """
        Get current emotion state from server
        
        Returns:
            dict: Emotion data or None if failed
        """
        if not self.client_initialized:
            return None
        
        if 'emotion' not in self.modules:
            return None
        
        try:
            response = self.session.get(
                f"{self.server_url}/client/{self.client_id}/emotion",
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ {self.robot_name}: Emotion request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ {self.robot_name}: Error getting emotion: {e}")
            self._trigger_callbacks('on_error', "Emotion request error", str(e))
            return None
    
    def check_server_connection(self) -> bool:
        """
        Check if server is reachable
        
        Returns:
            bool: True if server is reachable
        """
        try:
            response = self.session.get(self.server_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return "Multi-Client" in data.get('message', '')
            return False
        except:
            return False
    
    def start_keep_alive(self):
        """Start keep-alive ping to maintain WebSocket connection"""
        def ping_worker():
            ping_interval = self.config.get('network', {}).get('ping_interval', 30)
            
            while self.running and self.connected:
                try:
                    self.sio.emit('ping', {'timestamp': time.time()})
                    time.sleep(ping_interval)
                except:
                    break
        
        self.running = True
        ping_thread = threading.Thread(target=ping_worker, daemon=True)
        ping_thread.start()
        print(f"💓 Keep-alive started (interval: {self.config.get('network', {}).get('ping_interval', 30)}s)")
    
    def stop_keep_alive(self):
        """Stop keep-alive ping"""
        self.running = False
        print("💔 Keep-alive stopped")
    
    def get_config(self, key: str = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (dot notation supported, e.g., 'audio_settings.sample_rate')
            
        Returns:
            Configuration value or entire config if key is None
        """
        if key is None:
            return self.config
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def is_module_enabled(self, module: str) -> bool:
        """Check if a specific module is enabled"""
        return module in self.modules
    
    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot information"""
        return {
            'robot_name': self.robot_name,
            'client_id': self.client_id,
            'modules': self.modules,
            'connected': self.connected,
            'initialized': self.client_initialized
        }