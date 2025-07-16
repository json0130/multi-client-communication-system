# server_controller.py - Main Multi-Client Server Controller (Modular)
import time
import json
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit, join_room

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
        self.request_router = RequestRouter(self.client_manager, self.socketio)
        self.websocket_manager = WebSocketManager(
            self.socketio, 
            self.client_manager, 
            self.request_router
        )
        
        # Setup HTTP routes
        self.setup_routes()
        
        # Setup monitor WebSocket handlers directly
        self.setup_monitor_websocket_handlers()
        
        print("🏗️ Server Controller initialized with modular architecture")

    def setup_monitor_websocket_handlers(self):
        """Direct setup of monitor WebSocket handlers"""
        print("🔧 Setting up monitor WebSocket handlers directly...")

        self.socketio.server.eio.ping_interval = 15  # Reduce ping frequency
        self.socketio.server.eio.ping_timeout = 30   # Reduce timeout
        
        try:
            # Monitor namespace handlers
            @self.socketio.on('connect', namespace='/monitor')
            def handle_monitor_connect():
                print(f"🖥️ Monitor client connected: {request.sid} from {request.remote_addr}")
                return True

            @self.socketio.on('disconnect', namespace='/monitor')
            def handle_monitor_disconnect():
                print(f"🖥️ Monitor client disconnected: {request.sid}")

            @self.socketio.on('join_client_room', namespace='/monitor')
            def handle_join_client_room(data):
                client_id = data.get('client_id')
                print(f"🚪 Monitor {request.sid} requesting to join room for client '{client_id}'")
                
                if not client_id:
                    print(f"⚠️ Monitor {request.sid} tried to join without client_id")
                    emit('error', {'message': 'client_id is required'}, namespace='/monitor')
                    return
                
                # Verify client exists
                client_info = self.client_manager.get_client_info(client_id)
                if not client_info:
                    print(f"❌ Monitor {request.sid} tried to join room for non-existent client '{client_id}'")
                    emit('error', {'message': f'Client {client_id} not found'}, namespace='/monitor')
                    return
                
                # Join the room
                join_room(client_id, namespace='/monitor')
                print(f"✅ Monitor {request.sid} successfully joined room for client '{client_id}'")
                
                # Get current client status
                server = self.client_manager.get_client_server(client_id)
                current_emotion = "neutral"
                current_confidence = 0
                
                if server:
                    try:
                        emotion_state = server.get_current_emotion_state()
                        current_emotion = emotion_state.get('emotion', 'neutral')
                        current_confidence = emotion_state.get('confidence', 0)
                    except:
                        pass
                
                # Send confirmation
                emit('room_joined', {
                    'message': f"Successfully joined room for {client_info.get_display_name()}",
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'enabled_modules': list(client_info.modules),
                    'current_emotion': current_emotion,
                    'current_confidence': current_confidence,
                    'server_active': server is not None
                }, namespace='/monitor')

            @self.socketio.on('ping', namespace='/monitor')
            def handle_monitor_ping(data):
                client_id = data.get('client_id')
                # 🚀 PERFORMANCE: Reduce ping logging spam
                if hasattr(self, '_last_ping_log'):
                    if time.time() - self._last_ping_log < 30:  # Log ping only every 30 seconds
                        pass
                    else:
                        print(f"🏓 Monitor ping for client: {client_id}")
                        self._last_ping_log = time.time()
                else:
                    print(f"🏓 Monitor ping for client: {client_id}")
                    self._last_ping_log = time.time()
                    
                emit('pong', {
                    'timestamp': time.time(),
                    'client_id': client_id,
                    'message': 'Monitor connection alive'
                }, namespace='/monitor')
            
            print("✅ Monitor WebSocket handlers registered successfully!")
            
            # Enhance frame broadcasting
            self._enhance_frame_broadcasting()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup monitor handlers: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _enhance_frame_broadcasting(self):
        """Enhance frame processing to broadcast to monitors"""
        print("🔧 Enhancing frame broadcasting...")
        
        try:
            # Store original method
            if not hasattr(self, '_original_handle_image_frame'):
                self._original_handle_image_frame = self.request_router.handle_image_frame_processing
            
            # 🚀 PERFORMANCE: Add throttling variables
            self.last_broadcast_times = {}  # Track last broadcast time per client
            self.broadcast_throttle_time = 0.2  # 5 broadcasts per second max
            self.last_emotions = {}  # Track last emotion per client to avoid duplicate broadcasts
            
            def enhanced_handle_image_frame(client_id, frame_data):
                # Process frame normally
                result = self._original_handle_image_frame(client_id, frame_data)
                
                # 🚀 PERFORMANCE: Smart broadcasting with throttling
                if 'error' not in result and 'emotion' in result:
                    try:
                        current_time = time.time()
                        
                        # Check if enough time has passed since last broadcast
                        last_broadcast = self.last_broadcast_times.get(client_id, 0)
                        time_since_last = current_time - last_broadcast
                        
                        # Check if emotion has changed significantly
                        current_emotion = result.get('emotion', 'neutral')
                        current_confidence = result.get('confidence', 0)
                        last_emotion_data = self.last_emotions.get(client_id, {'emotion': None, 'confidence': 0})
                        
                        emotion_changed = (
                            current_emotion != last_emotion_data['emotion'] or
                            abs(current_confidence - last_emotion_data['confidence']) > 10  # 10% confidence change
                        )
                        
                        # Broadcast if enough time passed OR emotion changed significantly
                        should_broadcast = (
                            time_since_last >= self.broadcast_throttle_time or
                            (emotion_changed and time_since_last >= 0.1)  # Minimum 100ms between any broadcasts
                        )
                        
                        if should_broadcast:
                            client_info = self.client_manager.get_client_info(client_id)
                            update_data = {
                                'client_id': client_id,
                                'robot_name': client_info.robot_name if client_info else 'Unknown',
                                'emotion': current_emotion,
                                'confidence': current_confidence,
                                'status': result.get('status', 'unknown'),
                                'distribution': result.get('distribution', {}),
                                'timestamp': current_time
                            }
                            
                            # Only log significant changes to reduce spam
                            if emotion_changed or time_since_last >= 2.0:  # Log every 2 seconds or on change
                                print(f"📡 Broadcasting frame update to monitors for {client_id}: {current_emotion} ({current_confidence}%)")
                            
                            self.socketio.emit('client_frame_update', update_data, room=client_id, namespace='/monitor')
                            
                            # Update tracking
                            self.last_broadcast_times[client_id] = current_time
                            self.last_emotions[client_id] = {
                                'emotion': current_emotion,
                                'confidence': current_confidence
                            }
                            
                    except Exception as broadcast_error:
                        print(f"⚠️ Frame broadcast error: {broadcast_error}")
                
                return result
            
            # Replace the method
            self.request_router.handle_image_frame_processing = enhanced_handle_image_frame
            print("✅ Frame broadcasting enhanced with smart throttling!")
            
        except Exception as e:
            print(f"❌ Failed to enhance frame broadcasting: {e}")
    
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
                    "client_monitor": "/client/{client_id}/monitor (GET)",
                    "client_live_stream": "/client/{client_id}/live_stream (GET)",
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
            
        @self.app.route('/debug/client/<client_id>/monitor_html', methods=['GET'])
        def debug_monitor_html(client_id):
            """Debug monitor HTML generation"""
            try:
                client_info = self.client_manager.get_client_info(client_id)
                if not client_info:
                    return f"Client {client_id} not found", 404
                
                server = self.client_manager.get_or_create_server_instance(client_id)
                if not server:
                    return f"Server instance creation failed for {client_id}", 500
                
                # Get the HTML and return it as plain text to inspect
                html = server.get_individual_monitor_html()
                return f"<pre>{html.replace('<', '&lt;').replace('>', '&gt;')}</pre>", 200
                
            except Exception as e:
                return f"Debug error: {e}", 500
        
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

        @self.app.route('/client/<client_id>/monitor', methods=['GET'])
        def client_monitor(client_id):
            """Serve individual client monitor page"""
            print(f"🔍 DEBUG: Monitor route hit for client_id: {client_id}")
            
            try:
                # Get the server instance directly
                server = self.client_manager.get_client_server(client_id)
                if not server:
                    # Try to create server instance if client exists
                    client_info = self.client_manager.get_client_info(client_id)
                    if client_info:
                        server = self.client_manager.get_or_create_server_instance(client_id)
                    
                    if not server:
                        return "Server not found", 404
                    
                # Check if method exists
                if hasattr(server, 'get_individual_monitor_html'):
                    print(f"✅ Server has get_individual_monitor_html method")
                    html = server.get_individual_monitor_html()
                    print(f"📄 HTML length: {len(html)}")
                    return Response(html, mimetype='text/html')
                else:
                    print(f"❌ Server missing get_individual_monitor_html method")
                    return "Method not found", 500
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                return f"Error: {e}", 500

        @self.app.route('/client/<client_id>/live_stream', methods=['GET'])
        def client_live_stream(client_id):
            """Serve individual client live stream"""
            return self.request_router.route_client_request(client_id, 'live_stream', request)
        
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