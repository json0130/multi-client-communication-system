# client_manager.py - Client Registration and Management
import time
import threading
import uuid
import json
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass

from server import RobotServer
from database import Database

@dataclass
class ClientInfo:
    """Information about a connected client"""
    client_id: str
    robot_name: str
    modules: Set[str]
    config_overrides: Dict[str, Any]
    registration_time: float
    last_activity: float
    user_id: int
    
    def get_display_name(self) -> str:
        """Get display name for logging: [client_id] robot_name"""
        return f"[{self.client_id}] {self.robot_name}"

class ClientManager:
    """
    Manages client registration, server instance creation, and lifecycle.
    Handles client_init.json processing and server instance management.
    """

    def __init__(self, database: Optional[Database] = None):
        self.client_infos: Dict[str, ClientInfo] = {}       # client_id -> ClientInfo
        self.client_servers: Dict[str, RobotServer] = {}  # client_id -> server instance
        self.manager_lock = threading.RLock()
        self.id_map: Dict[str, int] = {}
        self.db = database

        # Valid modules for validation
        self.valid_modules = {'gpt', 'emotion', 'speech', 'facial', 'rag'}
        
        # Cleanup configuration
        self.cleanup_task_running = False
        self.inactive_threshold = 30 * 60  # 30 minutes
    
    def process_client_init(self, client_init_data: Dict[str, Any]) -> tuple[bool, str, Optional[ClientInfo]]:
        """
        Process client_init.json data and register client
        
        Expected format:
        {
            "client_id": "optional_custom_id",
            "robot_name": "HomeAssistant_Robot", 
            "modules": ["gpt", "emotion", "speech"],
            "config": {"custom_param": "value"}
        }
        
        Returns: (success, message, client_info)
        """
        try:
            # Extract and validate required fields
            robot_name = client_init_data.get('robot_name')
            modules = client_init_data.get('modules', [])
            
            if not robot_name:
                return False, "robot_name is required in client_init.json", None
            
            if not modules:
                return False, "modules list is required in client_init.json", None
            
            # Generate or use provided client_id
            client_id = client_init_data.get('client_id')
            if not client_id:
                client_id = f"{robot_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
            
            # Validate modules
            modules_set = set(modules)
            if not modules_set.issubset(self.valid_modules):
                invalid_modules = modules_set - self.valid_modules
                return False, f"Invalid modules: {invalid_modules}. Valid options: {self.valid_modules}", None
            
            # Get config overrides
            config_overrides = client_init_data.get('config', {})
            
            # Ensure user exists first
            if client_id not in self.id_map:
                user_id = self.db.create_user(name=robot_name)
                self.id_map[client_id] = user_id
            else:
                user_id = self.id_map[client_id]
            
            # Create client info
            client_info = ClientInfo(
                client_id=client_id,
                robot_name=robot_name,
                modules=modules_set,
                config_overrides=config_overrides,
                registration_time=time.time(),
                last_activity=time.time(),
                user_id=user_id
            )
            
            # Register the client
            with self.manager_lock:
                self.client_infos[client_id] = client_info

            print(f"📝 Registered client {client_info.get_display_name()} with modules: {list(modules_set)}")
            
            return True, f"Client {client_info.get_display_name()} registered successfully", client_info
            
        except Exception as e:
            return False, f"Failed to process client_init.json: {e}", None
    
    def get_or_create_server_instance(self, client_id: str) -> Optional[RobotServer]:
        """Get existing server instance or create new one for client"""
        with self.manager_lock:
            # Return existing server
            if client_id in self.client_servers:
                self.update_client_activity(client_id)
                return self.client_servers[client_id]
            
            # Check if client is registered
            if client_id not in self.client_infos:
                print(f"❌ Client '{client_id}' not registered")
                return None
            
            client_info = self.client_infos[client_id]
            
            # Create new server instance
            print(f"🚀 Creating server instance for {client_info.get_display_name()}")
            
            try:
                server = self._create_server_instance(client_info)
                
                if server and server.initialize_components():
                    self.client_servers[client_id] = server
                    self.update_client_activity(client_id)
                    print(f"✅ Server instance created for {client_info.get_display_name()}")
                    return server
                else:
                    print(f"❌ Failed to create/initialize server for {client_info.get_display_name()}")
                    return None
                    
            except Exception as e:
                print(f"❌ Error creating server for {client_info.get_display_name()}: {e}")
                return None
    
    def _create_server_instance(self, client_info: ClientInfo) -> Optional[RobotServer]:
        """Create a new RobotServer instance for a client"""
        try:
            # Create custom configuration for this client
            server_config = {
                # 🚀 SPEED IMPROVEMENTS - Emotion Processing
            'emotion_processing_interval': 0.2,    # Changed from 0.1 - process every 200ms instead of 100ms
            'confidence_threshold': 25.0,          # Changed from 30.0 - lower threshold for faster processing
            'emotion_change_threshold': 20.0,      # Changed from 15.0 - less frequent updates
            'emotion_window_size': 3,              # Changed from 5 - smaller window for faster processing
            
            # 🚀 SPEED IMPROVEMENTS - Video Streaming for Monitors
            'stream_fps': 15,                       # Changed from 30 - much lower FPS for monitors
            'monitor_quality': 85,                 # NEW - lower JPEG quality for faster encoding
            'monitor_resolution': (1280, 720),      # NEW - smaller resolution for much faster streaming
            'frame_skip_ratio': 3,                 # NEW - skip 2 out of every 3 frames for monitors
            'frame_cache_duration': 0.15,         # NEW - cache frames for 150ms to avoid re-encoding
            
            # 🚀 SPEED IMPROVEMENTS - WebSocket Settings
            'broadcast_throttle': 0.2,            # NEW - limit broadcasts to 5 per second
            'emotion_update_threshold': 0.1,      # Changed from 0.05 - less frequent emotion updates
            
            # Original settings (kept the same)
            'whisper_model_size': 'base',
            'whisper_device': 'auto', 
            'whisper_compute_type': 'float16',
            'max_audio_length': 30,
            'sample_rate': 16000,
            
            "database": self.db,     # pass Database instance
            "user_id": client_info.user_id,  # stored this earlier

                **client_info.config_overrides  # Apply client-specific overrides
            }
            
            # Create server instance with client-specific modules
            server = RobotServer.create_for_client(
                client_id=client_info.client_id,
                enabled_modules=client_info.modules,
                config=server_config
            )
            
            # Store robot name in server for logging
            server.robot_name = client_info.robot_name
            
            return server
                
        except Exception as e:
            print(f"❌ Error creating server instance for {client_info.get_display_name()}: {e}")
            return None
    
    def get_client_server(self, client_id: str) -> Optional[RobotServer]:
        """Get existing client server (don't create if doesn't exist)"""
        with self.manager_lock:
            server = self.client_servers.get(client_id)
            if server:
                self.update_client_activity(client_id)
            return server
    
    def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """Get client information"""
        with self.manager_lock:
            return self.client_infos.get(client_id)
    
    def get_client_modules(self, client_id: str) -> Set[str]:
        """Get enabled modules for a client"""
        with self.manager_lock:
            client_info = self.client_infos.get(client_id)
            return client_info.modules if client_info else set()
        
    def get_user_id(self, client_id: str) -> Optional[int]:
        """Return the numeric user_id for a given client_id, or None."""
        return self.id_map.get(client_id)
    
    def update_client_activity(self, client_id: str):
        """Update last activity timestamp for client"""
        with self.manager_lock:
            if client_id in self.client_infos:
                self.client_infos[client_id].last_activity = time.time()
    
    def get_all_clients_status(self) -> Dict[str, Any]:
        """Get status of all registered clients"""
        with self.manager_lock:
            clients_status = {}
            
            for client_id, client_info in self.client_infos.items():
                server_status = "active" if client_id in self.client_servers else "inactive"
                inactive_minutes = (time.time() - client_info.last_activity) / 60
                
                clients_status[client_id] = {
                    "robot_name": client_info.robot_name,
                    "display_name": client_info.get_display_name(),
                    "modules": list(client_info.modules),
                    "status": server_status,
                    "registration_time": client_info.registration_time,
                    "last_activity": client_info.last_activity,
                    "inactive_minutes": round(inactive_minutes, 1)
                }
            
            return clients_status
    
    def start_cleanup_task(self):
        """Start background task to cleanup inactive client server instances"""
        if self.cleanup_task_running:
            return
        
        self.cleanup_task_running = True
        
        def cleanup_worker():
            while self.cleanup_task_running:
                try:
                    self._cleanup_inactive_clients()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    print(f"❌ Cleanup task error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        print("🧹 Started cleanup task for inactive clients")
    
    def _cleanup_inactive_clients(self):
        """Remove server instances for clients inactive for >30 minutes"""
        current_time = time.time()
        
        with self.manager_lock:
            inactive_clients = []
            
            for client_id, client_info in self.client_infos.items():
                if current_time - client_info.last_activity > self.inactive_threshold:
                    inactive_clients.append(client_id)
            
            for client_id in inactive_clients:
                if client_id in self.client_servers:
                    client_info = self.client_infos[client_id]
                    print(f"🧹 Cleaning up inactive client server: {client_info.get_display_name()}")
                    try:
                        server = self.client_servers[client_id]
                        server.cleanup_resources()
                        del self.client_servers[client_id]
                    except Exception as e:
                        print(f"❌ Error cleaning up {client_info.get_display_name()}: {e}")
    
    def stop_cleanup_task(self):
        """Stop the cleanup task"""
        self.cleanup_task_running = False
    
    def cleanup_all_clients(self):
        """Cleanup all client servers (called on shutdown)"""
        with self.manager_lock:
            for client_id, server in self.client_servers.items():
                try:
                    client_info = self.client_infos.get(client_id)
                    display_name = client_info.get_display_name() if client_info else client_id
                    print(f"🧹 Cleaning up server for {display_name}")
                    server.cleanup_resources()
                except Exception as e:
                    print(f"❌ Error cleaning up '{client_id}': {e}")
            
            self.client_servers.clear()
    
    def remove_client(self, client_id: str) -> bool:
        """Remove a client and its server instance"""
        with self.manager_lock:
            removed = False
            
            # Remove server instance
            if client_id in self.client_servers:
                try:
                    server = self.client_servers[client_id]
                    server.cleanup_resources()
                    del self.client_servers[client_id]
                    removed = True
                except Exception as e:
                    print(f"❌ Error removing server for '{client_id}': {e}")
            
            # Remove client info
            if client_id in self.client_infos:
                client_info = self.client_infos[client_id]
                del self.client_infos[client_id]
                print(f"🗑️ Removed client {client_info.get_display_name()}")
                removed = True
            
            return removed