# server_controller.py - Main Multi-Client Server Controller (Production)
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
    
    def __init__(self, port=5002):
        self.port = port
        
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=30,
            ping_interval=15,
            max_http_buffer_size=1000000,  # Optimized buffer size
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
        
        print("🏗️ Server Controller initialized")

    def setup_monitor_websocket_handlers(self):
        """Direct setup of monitor WebSocket handlers"""
        # Performance optimizations
        self.socketio.server.eio.ping_interval = 15
        self.socketio.server.eio.ping_timeout = 30
        
        try:
            # Monitor namespace handlers
            @self.socketio.on('connect', namespace='/monitor')
            def handle_monitor_connect():
                return True

            @self.socketio.on('disconnect', namespace='/monitor')
            def handle_monitor_disconnect():
                pass

            @self.socketio.on('join_client_room', namespace='/monitor')
            def handle_join_client_room(data):
                client_id = data.get('client_id')
                
                if not client_id:
                    emit('error', {'message': 'client_id is required'}, namespace='/monitor')
                    return
                
                # Verify client exists
                client_info = self.client_manager.get_client_info(client_id)
                if not client_info:
                    emit('error', {'message': f'Client {client_id} not found'}, namespace='/monitor')
                    return
                
                # Join the room
                join_room(client_id, namespace='/monitor')
                
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
                emit('pong', {
                    'timestamp': time.time(),
                    'client_id': client_id,
                    'message': 'Monitor connection alive'
                }, namespace='/monitor')
            
            # Enhance frame broadcasting
            self._enhance_frame_broadcasting()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup monitor handlers: {e}")
            return False

    def _enhance_frame_broadcasting(self):
        """Enhance frame processing to broadcast to monitors with throttling"""
        try:
            # Store original method
            if not hasattr(self, '_original_handle_image_frame'):
                self._original_handle_image_frame = self.request_router.handle_image_frame_processing
            
            # Performance throttling variables
            self.last_broadcast_times = {}
            self.broadcast_throttle_time = 0.2  # 5 broadcasts per second max
            self.last_emotions = {}
            
            def enhanced_handle_image_frame(client_id, frame_data):
                # Process frame normally
                result = self._original_handle_image_frame(client_id, frame_data)
                
                # Smart broadcasting with throttling
                if 'error' not in result and 'emotion' in result:
                    try:
                        current_time = time.time()
                        
                        # Check throttling
                        last_broadcast = self.last_broadcast_times.get(client_id, 0)
                        time_since_last = current_time - last_broadcast
                        
                        # Check emotion changes
                        current_emotion = result.get('emotion', 'neutral')
                        current_confidence = result.get('confidence', 0)
                        last_emotion_data = self.last_emotions.get(client_id, {'emotion': None, 'confidence': 0})
                        
                        emotion_changed = (
                            current_emotion != last_emotion_data['emotion'] or
                            abs(current_confidence - last_emotion_data['confidence']) > 10
                        )
                        
                        # Broadcast conditions
                        should_broadcast = (
                            time_since_last >= self.broadcast_throttle_time or
                            (emotion_changed and time_since_last >= 0.1)
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
                            
                            self.socketio.emit('client_frame_update', update_data, room=client_id, namespace='/monitor')
                            
                            # Update tracking
                            self.last_broadcast_times[client_id] = current_time
                            self.last_emotions[client_id] = {
                                'emotion': current_emotion,
                                'confidence': current_confidence
                            }
                            
                    except Exception:
                        pass  # Silent fail for broadcasts
                
                return result
            
            # Replace the method
            self.request_router.handle_image_frame_processing = enhanced_handle_image_frame
            
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
            """Register a new client (alternative to WebSocket client_init)"""
            try:
                data = request.json
                if not data:
                    return jsonify({"error": "JSON data required"}), 400
                
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

        @self.app.route('/client/<client_id>/monitor', methods=['GET'])
        def client_monitor(client_id):
            try:
                server = self.client_manager.get_client_server(client_id)
                if not server:
                    # If not found, create a temporary RobotServer instance (no modules)
                    from server import RobotServer
                    server = RobotServer.create_for_client(client_id, set(), {})
                # Always call the method (even if modules is empty)
                html = server.get_individual_monitor_html()
                return Response(html, mimetype='text/html')
            except Exception as e:
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
            print("=" * 50)
            print(f"🌐 Server: http://0.0.0.0:{self.port}")
            print(f"🔌 WebSocket: ws://0.0.0.0:{self.port}/socket.io/")
            print(f"📦 Modules: {', '.join(self.client_manager.valid_modules)}")
            print("=" * 50)
            
            # Start background cleanup task
            self.client_manager.start_cleanup_task()
            
            print("✅ Server ready!")
            print(f"🖥️  Test Monitor UI: http://localhost:{self.port}/client/test/monitor")
            
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
            print("\n🛑 Server shutdown")
            self._shutdown()
                        
        except Exception as e:
            print(f"❌ Server error: {e}")
            self._shutdown()
    
    def _shutdown(self):
        """Cleanup resources on shutdown"""
        try:
            self.client_manager.stop_cleanup_task()
            self.client_manager.cleanup_all_clients()
        except Exception as e:
            print(f"❌ Cleanup error: {e}")

def main():
    """Main function to start the server controller"""
    print("🎯 Multi-Client Emotion Server Controller")
    print("   Production optimized version")
    print()
    
    controller = ServerController(port=5002)
    controller.start()

if __name__ == "__main__":
    main()