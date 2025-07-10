# server_controller.py - Main Multi-Client Server Controller (Modular)
import time
import json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

from client_manager import ClientManager
from request_router import RequestRouter
from websocket_manager import WebSocketManager

class ServerController:
    """
    Main multi-client server controller using modular architecture.
    
    This is the main entry point that orchestrates:
    - Client registration and management
    - HTTP request routing
    - WebSocket real-time communication
    - Server instance lifecycle management
    
    Clients connect by sending client_init.json with their requirements.
    """
    
    def __init__(self, port=5000):
        self.port = port
        
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=2000000,  # For image frames
            transports=['websocket', 'polling'],
            allow_upgrades=True,
            cookie=False
        )
        
        # Initialize modular components
        self.client_manager = ClientManager()
        self.request_router = RequestRouter(self.client_manager)
        self.websocket_manager = WebSocketManager(
            self.socketio, 
            self.client_manager, 
            self.request_router
        )
        
        # Setup HTTP routes
        self.setup_routes()
        
        print("🏗️ Server Controller initialized with modular architecture")
    
    def setup_routes(self):
        """Setup HTTP REST API routes"""
        
        @self.app.route('/', methods=['GET'])
        def controller_info():
            """Controller status and active clients"""
            clients_status = self.client_manager.get_all_clients_status()
            
            return jsonify({
                "message": "Multi-Client Emotion Server Controller",
                "status": "running",
                "port": self.port,
                "active_clients": len(self.client_manager.client_servers),
                "total_registered_clients": len(self.client_manager.client_infos),
                "available_modules": list(self.client_manager.valid_modules),
                "clients": clients_status,
                "endpoints": {
                    "client_chat": "/client/{client_id}/chat (POST)",
                    "client_speech": "/client/{client_id}/speech (POST)", 
                    "client_emotion": "/client/{client_id}/emotion (GET)",
                    "client_health": "/client/{client_id}/health (GET)",
                    "websocket": "/socket.io/ (send client_init event first)"
                },
                "websocket_events": {
                    "client_init": "Initialize client with robot_name and modules",
                    "image_frame": "Send image frames for real-time processing",
                    "chat_message": "Send chat messages via WebSocket",
                    "ping": "Keep connection alive"
                }
            })
        
        @self.app.route('/register_client', methods=['POST'])
        def register_client():
            """
            Register a new client (alternative to WebSocket client_init)
            
            Request body should match client_init.json format:
            {
                "robot_name": "HomeAssistant_Robot",
                "modules": ["gpt", "emotion", "speech"],
                "client_id": "optional_custom_id",
                "config": {"custom_param": "value"}
            }
            """
            try:
                data = request.json
                if not data:
                    return jsonify({"error": "JSON data required"}), 400
                
                # Process client initialization using client manager
                success, message, client_info = self.client_manager.process_client_init(data)
                
                if success:
                    return jsonify({
                        "success": True,
                        "message": message,
                        "client_id": client_info.client_id,
                        "robot_name": client_info.robot_name,
                        "display_name": client_info.get_display_name(),
                        "enabled_modules": list(client_info.modules),
                        "endpoints": {
                            "chat": f"/client/{client_info.client_id}/chat",
                            "speech": f"/client/{client_info.client_id}/speech",
                            "emotion": f"/client/{client_info.client_id}/emotion",
                            "health": f"/client/{client_info.client_id}/health"
                        }
                    }), 201
                else:
                    return jsonify({
                        "success": False,
                        "message": message
                    }), 400
                
            except Exception as e:
                print(f"❌ Client registration error: {e}")
                return jsonify({
                    "success": False,
                    "message": f"Registration failed: {e}"
                }), 500
        
        @self.app.route('/client/<client_id>/chat', methods=['POST'])
        def client_chat(client_id):
            """Handle chat request for specific client"""
            return self.request_router.route_client_request(client_id, 'chat', request)
        
        @self.app.route('/client/<client_id>/speech', methods=['POST'])
        def client_speech(client_id):
            """Handle speech-to-text request for specific client"""
            return self.request_router.route_client_request(client_id, 'speech', request)
        
        @self.app.route('/client/<client_id>/emotion', methods=['GET'])
        def client_emotion(client_id):
            """Get current emotion state for specific client"""
            return self.request_router.route_client_request(client_id, 'emotion', request)
        
        @self.app.route('/client/<client_id>/health', methods=['GET'])
        def client_health(client_id):
            """Get health status for specific client"""
            return self.request_router.route_client_request(client_id, 'health', request)
        
        @self.app.route('/clients', methods=['GET'])
        def list_clients():
            """List all registered clients and their status"""
            clients_status = self.client_manager.get_all_clients_status()
            
            return jsonify({
                "total_clients": len(clients_status),
                "active_servers": len(self.client_manager.client_servers),
                "clients": clients_status,
                "timestamp": time.time()
            })
        
        @self.app.route('/client/<client_id>/remove', methods=['DELETE'])
        def remove_client(client_id):
            """Remove a client and cleanup its resources"""
            try:
                if self.client_manager.remove_client(client_id):
                    return jsonify({
                        "message": f"Client '{client_id}' removed successfully"
                    }), 200
                else:
                    return jsonify({
                        "error": f"Client '{client_id}' not found"
                    }), 404
                    
            except Exception as e:
                print(f"❌ Error removing client '{client_id}': {e}")
                return jsonify({
                    "error": f"Failed to remove client: {e}"
                }), 500
        
        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-Client-ID')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
    
    def start(self):
        """Start the multi-client server controller"""
        try:
            print("🚀 Starting Multi-Client Server Controller...")
            print("=" * 70)
            print(f"🌐 Controller URL: http://0.0.0.0:{self.port}")
            print(f"🔌 WebSocket URL: ws://0.0.0.0:{self.port}/socket.io/")
            print(f"📦 Available Modules: {', '.join(self.client_manager.valid_modules)}")
            print("=" * 70)
            
            # Start background cleanup task
            self.client_manager.start_cleanup_task()
            
            print("✅ Server Controller ready!")
            
            self._print_usage_examples()
            
            # Start the Flask-SocketIO server
            self.socketio.run(
                self.app,
                host='0.0.0.0',
                port=self.port,
                debug=False,
                allow_unsafe_werkzeug=True,
                use_reloader=False,
                log_output=False
            )
            
        except KeyboardInterrupt:
            print("\n🛑 Server Controller shutdown")
            self._shutdown()
                        
        except Exception as e:
            print(f"❌ Server Controller error: {e}")
            self._shutdown()
    
    def _shutdown(self):
        """Cleanup resources on shutdown"""
        try:
            print("🧹 Cleaning up resources...")
            
            # Stop cleanup task
            self.client_manager.stop_cleanup_task()
            
            # Cleanup all client servers
            self.client_manager.cleanup_all_clients()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
    
    def _print_usage_examples(self):
        """Print usage examples for the API"""
        print(f"\n📋 CLIENT INITIALIZATION:")
        print(f"   WebSocket client_init event or POST /register_client:")
        print(f"""   {{
       "robot_name": "HomeAssistant_Robot",
       "modules": ["gpt", "emotion"],
       "client_id": "optional_id",
       "config": {{"confidence_threshold": 40.0}}
   }}""")
        
        print(f"\n💬 CHAT EXAMPLE:")
        print(f"   POST http://localhost:{self.port}/client/homeassistant_robot_abc123/chat")
        print(f'   {{"message": "How are you feeling today?"}}')
        
        print(f"\n🎤 SPEECH EXAMPLE:")
        print(f"   POST http://localhost:{self.port}/client/homeassistant_robot_abc123/speech")
        print(f'   {{"audio": "base64_encoded_audio_data"}}')
        
        print(f"\n📸 WEBSOCKET IMAGE FRAMES:")
        print(f"   1. Connect: ws://localhost:{self.port}/socket.io/")
        print(f"   2. Send: client_init event with robot info")
        print(f"   3. Send: image_frame events with base64 image data")
        
        print(f"\n🏥 HEALTH CHECK:")
        print(f"   GET http://localhost:{self.port}/client/homeassistant_robot_abc123/health")
        
        print(f"\n📊 LIST ALL CLIENTS:")
        print(f"   GET http://localhost:{self.port}/clients")

def main():
    """Main function to start the server controller"""
    print("🎯 Multi-Client Emotion Server Controller")
    print("   Modular architecture with client_init.json support")
    print()
    
    controller = ServerController(port=5000)
    controller.start()

if __name__ == "__main__":
    main()