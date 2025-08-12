# server_controller.py - Complete Multi-Client Server Controller with Individual Monitoring
import time
import json
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

from client_manager import ClientManager
from request_router import RequestRouter
from websocket_manager import WebSocketManager

def generate_placeholder_frame(client_id, server_instance):
    """Generate a placeholder frame for clients without video"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background color (dark blue)
    frame[:] = (40, 40, 100)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Client ID
    cv2.putText(frame, f"Client: {client_id}", (50, 80), font, 1, (0, 255, 0), 2)
    
    # Status
    cv2.putText(frame, "Waiting for camera feed...", (50, 130), font, 0.8, (255, 255, 255), 2)
    
    # Modules
    if hasattr(server_instance, 'enabled_modules'):
        modules_text = f"Modules: {', '.join(list(server_instance.enabled_modules))}"
        cv2.putText(frame, modules_text[:50], (50, 180), font, 0.6, (0, 255, 255), 1)
        if len(modules_text) > 50:
            cv2.putText(frame, modules_text[50:], (50, 210), font, 0.6, (0, 255, 255), 1)
    
    # Current emotion if available
    if hasattr(server_instance, 'latest_emotion') and hasattr(server_instance, 'latest_confidence'):
        emotion_text = f"Emotion: {server_instance.latest_emotion} ({server_instance.latest_confidence:.1f}%)"
        cv2.putText(frame, emotion_text, (50, 260), font, 0.7, (255, 0, 255), 2)
    
    # Timestamp
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, f"Time: {timestamp}", (50, 310), font, 0.6, (255, 255, 255), 1)
    
    # Instructions
    cv2.putText(frame, "Send image frames via WebSocket", (50, 360), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "to see live video here", (50, 380), font, 0.5, (200, 200, 200), 1)
    
    return frame

def generate_manual_stream(client_id, server_instance):
    """Generate a manual video stream for clients"""
    while True:
        try:
            # Get frame from server instance or generate placeholder
            if hasattr(server_instance, 'latest_frame') and server_instance.latest_frame is not None:
                frame = server_instance.latest_frame.copy()
                
                # Add overlay with client info
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"{client_id}", (10, 30), font, 0.7, (0, 255, 0), 2)
                
                if hasattr(server_instance, 'latest_emotion') and hasattr(server_instance, 'latest_confidence'):
                    emotion_text = f"{server_instance.latest_emotion} ({server_instance.latest_confidence:.1f}%)"
                    cv2.putText(frame, emotion_text, (10, 60), font, 0.6, (255, 0, 255), 2)
                    
                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), font, 0.5, (255, 255, 255), 1)
            else:
                frame = generate_placeholder_frame(client_id, server_instance)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
                
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control frame rate (10 FPS for efficiency)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Manual stream error for '{client_id}': {e}")
            time.sleep(1)  # Wait before retrying

def generate_error_stream(error_message):
    """Generate an error image for failed streams"""
    def error_generator():
        while True:
            try:
                # Create error frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (50, 50, 50)  # Dark gray background
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "Stream Error", (200, 200), font, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, error_message[:40], (100, 250), font, 0.7, (255, 255, 255), 2)
                if len(error_message) > 40:
                    cv2.putText(frame, error_message[40:80], (100, 280), font, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, "Check server logs for details", (150, 330), font, 0.6, (200, 200, 200), 1)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(2)  # Slow refresh for error frames
                
            except Exception as e:
                print(f"❌ Error generating error stream: {e}")
                time.sleep(2)
    
    return error_generator()

class ServerController:
    """
    Multi-client server controller with individual client monitoring.
    
    Each client gets their own RobotServer instance with individual web interface.
    Supports multiple clients with independent monitoring capabilities.
    """
    
    def __init__(self, port=5000):
        self.port = port
        self.start_time = time.time()
        
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
        
        # Connect WebSocket manager to this controller
        self.websocket_manager.set_server_controller(self)
        
        # Setup HTTP routes
        self.setup_routes()
        
        print("🏗️ Server Controller initialized for individual client monitoring")
    
    def setup_routes(self):
        """Setup HTTP REST API routes"""
        
        @self.app.route('/', methods=['GET'])
        def controller_info():
            """Controller status and active clients with individual monitor links"""
            clients_status = self.client_manager.get_all_clients_status()
            
            # Add individual monitor URLs to each client
            individual_monitors = []
            for client_id, status in clients_status.items():
                individual_monitors.append({
                    "client_id": client_id,
                    "robot_name": status.get('robot_name', 'Unknown'),
                    "enabled_modules": status.get('enabled_modules', []),
                    "status": status.get('status', 'unknown'),
                    "monitor_url": f"/client/{client_id}/monitor",
                    "stream_url": f"/client/{client_id}/live_stream",
                    "last_activity": status.get('last_activity', 0)
                })
            
            return jsonify({
                "message": "Multi-Client Server with Individual Monitoring",
                "status": "running",
                "port": self.port,
                "uptime": round(time.time() - self.start_time, 2),
                "active_clients": len(self.client_manager.client_servers),
                "total_registered_clients": len(self.client_manager.client_infos),
                "available_modules": list(self.client_manager.valid_modules),
                "clients": clients_status,
                "individual_monitors": individual_monitors,
                "endpoints": {
                    "client_registration": "/register_client (POST)",
                    "client_chat": "/client/{client_id}/chat (POST)",
                    "client_speech": "/client/{client_id}/speech (POST)", 
                    "client_emotion": "/client/{client_id}/emotion (GET)",
                    "client_health": "/client/{client_id}/health (GET)",
                    "client_monitor": "/client/{client_id}/monitor (GET - Individual monitor)",
                    "client_stream": "/client/{client_id}/live_stream (GET - Individual stream)",
                    "websocket": "/socket.io/ (send client_init event first)",
                    "all_clients": "/clients (GET - List all clients)"
                },
                "websocket_events": {
                    "client_init": "Initialize client with robot_name and modules",
                    "image_frame": "Send image frames for real-time processing",
                    "chat_message": "Send chat messages via WebSocket",
                    "ping": "Keep connection alive"
                }
            })
        
        @self.app.route('/health', methods=['GET'])
        def global_health():
            """Global server health check endpoint"""
            try:
                active_clients = len(self.client_manager.client_servers)
                total_clients = len(self.client_manager.client_infos)
                
                # Get health status of all clients
                client_health = {}
                for client_id, server_instance in self.client_manager.client_servers.items():
                    try:
                        if hasattr(server_instance, 'get_health_status'):
                            client_health[client_id] = server_instance.get_health_status()
                        else:
                            client_health[client_id] = {"status": "unknown", "client_id": client_id}
                    except Exception as e:
                        client_health[client_id] = {"status": "error", "error": str(e), "client_id": client_id}
                
                return jsonify({
                    "status": "healthy",
                    "message": "Multi-Client Server is running",
                    "timestamp": time.time(),
                    "uptime": time.time() - self.start_time,
                    "active_clients": active_clients,
                    "total_registered_clients": total_clients,
                    "available_modules": list(self.client_manager.valid_modules),
                    "server_version": "1.0.0",
                    "client_health": client_health
                }), 200
                
            except Exception as e:
                return jsonify({
                    "status": "unhealthy", 
                    "message": f"Server error: {e}",
                    "timestamp": time.time()
                }), 500
        
        @self.app.route('/register_client', methods=['POST'])
        def register_client():
            """
            Register a new client (alternative to WebSocket client_init)
            
            Request body format:
            {
                "robot_name": "ChatBox",
                "client_id": "chatbox_jetson_001",
                "modules": ["gpt", "emotion", "speech"],
                "config": {"stream_fps": 10}
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
                        "individual_monitor": f"/client/{client_info.client_id}/monitor",
                        "endpoints": {
                            "chat": f"/client/{client_info.client_id}/chat",
                            "speech": f"/client/{client_info.client_id}/speech",
                            "emotion": f"/client/{client_info.client_id}/emotion",
                            "health": f"/client/{client_info.client_id}/health",
                            "monitor": f"/client/{client_info.client_id}/monitor",
                            "stream": f"/client/{client_info.client_id}/live_stream"
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
            """
            Handle chat request for specific client
            
            Request body: {"message": "Hello, how are you?"}
            """
            try:
                return self.request_router.route_client_request(client_id, 'chat', request)
            except Exception as e:
                print(f"❌ Chat route error for client '{client_id}': {e}")
                return jsonify({"error": f"Chat processing failed: {e}"}), 500
        
        @self.app.route('/client/<client_id>/speech', methods=['POST'])
        def client_speech(client_id):
            """
            Handle speech-to-text request for specific client
            
            Request body: {"audio": "base64_encoded_audio_data"}
            """
            try:
                return self.request_router.route_client_request(client_id, 'speech', request)
            except Exception as e:
                print(f"❌ Speech route error for client '{client_id}': {e}")
                return jsonify({"error": f"Speech processing failed: {e}"}), 500
        
        @self.app.route('/client/<client_id>/emotion', methods=['GET'])
        def client_emotion(client_id):
            """Get current emotion state for specific client"""
            try:
                return self.request_router.route_client_request(client_id, 'emotion', request)
            except Exception as e:
                print(f"❌ Emotion route error for client '{client_id}': {e}")
                return jsonify({"error": f"Emotion state retrieval failed: {e}"}), 500
        
        @self.app.route('/client/<client_id>/health', methods=['GET'])
        def client_health(client_id):
            """Get health status for specific client"""
            try:
                return self.request_router.route_client_request(client_id, 'health', request)
            except Exception as e:
                print(f"❌ Health route error for client '{client_id}': {e}")
                return jsonify({"error": f"Health check failed: {e}"}), 500
        
        @self.app.route('/client/<client_id>/monitor', methods=['GET'])
        def client_individual_monitor(client_id):
            """Individual client monitor interface - serves customized HTML for each client"""
            try:
                # Get the client's server instance
                server_instance = self.client_manager.get_client_server(client_id)
                
                if not server_instance:
                    return f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Client Not Found</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; background: #f5f5f5; }}
                            .error-container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }}
                            h1 {{ color: #e74c3c; margin-bottom: 20px; }}
                            p {{ color: #666; margin: 10px 0; }}
                            a {{ color: #3498db; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="error-container">
                            <h1>❌ Client Not Found</h1>
                            <p>Client '<strong>{client_id}</strong>' not found or not initialized.</p>
                            <p>Please check that the client is connected and registered.</p>
                            <p><a href="/">← Back to Server Status</a></p>
                        </div>
                    </body>
                    </html>
                    """, 404
                
                # Get individual monitor HTML from the client's server instance
                if hasattr(server_instance, 'get_individual_monitor_html'):
                    return server_instance.get_individual_monitor_html()
                elif hasattr(server_instance, 'web_interface') and server_instance.web_interface:
                    # Fallback to basic web interface
                    base_html = server_instance.web_interface.get_monitor_html()
                    # Basic customization
                    customized_html = base_html.replace(
                        '🐱 Local Emotion Detection Monitor',
                        f'🤖 Monitor: {client_id}'
                    )
                    return customized_html
                else:
                    enabled_modules = getattr(server_instance, 'enabled_modules', set())
                    return f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Web Interface Not Available</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; background: #f5f5f5; }}
                            .warning-container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }}
                            h1 {{ color: #f39c12; margin-bottom: 20px; }}
                            p {{ color: #666; margin: 10px 0; }}
                            a {{ color: #3498db; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                            .modules {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; }}
                        </style>
                    </head>
                    <body>
                        <div class="warning-container">
                            <h1>⚠️ Web Interface Not Available</h1>
                            <p>Client '<strong>{client_id}</strong>' does not have web interface enabled.</p>
                            <div class="modules">
                                <strong>Enabled modules:</strong> {', '.join(list(enabled_modules)) if enabled_modules else 'Unknown'}
                            </div>
                            <p>The client may need to be restarted with web interface support.</p>
                            <p><a href="/">← Back to Server Status</a></p>
                        </div>
                    </body>
                    </html>
                    """, 404
                
            except Exception as e:
                print(f"❌ Error serving monitor for client '{client_id}': {e}")
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Monitor Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; background: #f5f5f5; }}
                        .error-container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }}
                        h1 {{ color: #e74c3c; margin-bottom: 20px; }}
                        p {{ color: #666; margin: 10px 0; }}
                        a {{ color: #3498db; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                        .error-details {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; font-family: monospace; font-size: 14px; text-align: left; }}
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <h1>❌ Monitor Error</h1>
                        <p>Error loading monitor for client '<strong>{client_id}</strong>'</p>
                        <div class="error-details">{e}</div>
                        <p><a href="/">← Back to Server Status</a></p>
                    </div>
                </body>
                </html>
                """, 500
        
        @self.app.route('/client/<client_id>/live_stream', methods=['GET'])
        def client_individual_live_stream(client_id):
            """Individual client live video stream - simplified working version"""
            try:
                print(f"🎥 Live stream request for client: {client_id}")
                
                # Get or create server instance
                server_instance = self.client_manager.get_or_create_server_instance(client_id)
                
                if not server_instance:
                    print(f"❌ Client '{client_id}' not found for live stream")
                    return Response(
                        generate_error_stream(f"Client '{client_id}' not found"),
                        mimetype='multipart/x-mixed-replace; boundary=frame'
                    ), 404
                
                print(f"✅ Starting live stream for client '{client_id}'")
                
                # Use manual stream generation (most reliable)
                return Response(
                    generate_manual_stream(client_id, server_instance),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )
                
            except Exception as e:
                print(f"❌ Live stream error for client '{client_id}': {e}")
                import traceback
                traceback.print_exc()
                return Response(
                    generate_error_stream(f"Stream error: {e}"),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                ), 500
        
        @self.app.route('/clients', methods=['GET'])
        def list_clients():
            """List all registered clients and their status with individual monitor links"""
            try:
                clients_status = self.client_manager.get_all_clients_status()
                
                # Add individual monitor URLs and additional info to each client
                enhanced_clients = {}
                for client_id, status in clients_status.items():
                    enhanced_status = dict(status)  # Copy existing status
                    enhanced_status.update({
                        'individual_monitor': f"/client/{client_id}/monitor",
                        'live_stream': f"/client/{client_id}/live_stream",
                        'api_endpoints': {
                            'chat': f"/client/{client_id}/chat",
                            'speech': f"/client/{client_id}/speech",
                            'emotion': f"/client/{client_id}/emotion",
                            'health': f"/client/{client_id}/health"
                        }
                    })
                    enhanced_clients[client_id] = enhanced_status
                
                # Create summary for easy access
                individual_monitors = [
                    {
                        "client_id": client_id,
                        "robot_name": status.get('robot_name', 'Unknown'),
                        "monitor_url": f"/client/{client_id}/monitor",
                        "stream_url": f"/client/{client_id}/live_stream",
                        "enabled_modules": status.get('enabled_modules', []),
                        "status": status.get('status', 'unknown'),
                        "last_activity": status.get('last_activity', 0)
                    }
                    for client_id, status in enhanced_clients.items()
                ]
                
                return jsonify({
                    "total_clients": len(enhanced_clients),
                    "active_servers": len(self.client_manager.client_servers),
                    "clients": enhanced_clients,
                    "individual_monitors": individual_monitors,
                    "server_info": {
                        "uptime": time.time() - self.start_time,
                        "available_modules": list(self.client_manager.valid_modules),
                        "server_url": request.host_url.rstrip('/')
                    },
                    "timestamp": time.time()
                })
                
            except Exception as e:
                print(f"❌ Error listing clients: {e}")
                return jsonify({"error": f"Failed to list clients: {e}"}), 500
        
        @self.app.route('/client/<client_id>/remove', methods=['DELETE'])
        def remove_client(client_id):
            """Remove a client and cleanup its resources"""
            try:
                if self.client_manager.remove_client(client_id):
                    return jsonify({
                        "message": f"Client '{client_id}' removed successfully",
                        "timestamp": time.time()
                    }), 200
                else:
                    return jsonify({
                        "error": f"Client '{client_id}' not found",
                        "timestamp": time.time()
                    }), 404
                    
            except Exception as e:
                print(f"❌ Error removing client '{client_id}': {e}")
                return jsonify({
                    "error": f"Failed to remove client: {e}",
                    "timestamp": time.time()
                }), 500
        
        @self.app.after_request
        def after_request(response):
            """Add CORS headers to all responses"""
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
            print(f"🎯 Individual Client Monitor: http://0.0.0.0:{self.port}/client/{{client_id}}/monitor")
            print(f"📊 Server Status: http://0.0.0.0:{self.port}/")
            print(f"📋 All Clients: http://0.0.0.0:{self.port}/clients")
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
       "robot_name": "ChatBox",
       "client_id": "chatbox_jetson_001",
       "modules": ["gpt", "emotion", "speech"],
       "config": {{"stream_fps": 10}}
   }}""")
        
        print(f"\n💬 CHAT EXAMPLE:")
        print(f"   POST http://localhost:{self.port}/client/chatbox_jetson_001/chat")
        print(f'   {{"message": "How are you feeling today?"}}')
        
        print(f"\n🎤 SPEECH EXAMPLE:")
        print(f"   POST http://localhost:{self.port}/client/chatbox_jetson_001/speech")
        print(f'   {{"audio": "base64_encoded_audio_data"}}')
        
        print(f"\n📸 WEBSOCKET IMAGE FRAMES:")
        print(f"   1. Connect: ws://localhost:{self.port}/socket.io/")
        print(f"   2. Send: client_init event with robot info")
        print(f"   3. Send: image_frame events with base64 image data")
        
        print(f"\n🖥️ INDIVIDUAL CLIENT MONITORING:")
        print(f"   Monitor: http://localhost:{self.port}/client/chatbox_jetson_001/monitor")
        print(f"   Stream: http://localhost:{self.port}/client/chatbox_jetson_001/live_stream")
        print(f"   Health: http://localhost:{self.port}/client/chatbox_jetson_001/health")
        print(f"   Emotion: http://localhost:{self.port}/client/chatbox_jetson_001/emotion")
        
        print(f"\n📊 SERVER MANAGEMENT:")
        print(f"   Status: http://localhost:{self.port}/")
        print(f"   All Clients: http://localhost:{self.port}/clients")
        print(f"   Health: http://localhost:{self.port}/health")
        
        print(f"\n🔧 CLIENT MANAGEMENT:")
        print(f"   Register: POST http://localhost:{self.port}/register_client")
        print(f"   Remove: DELETE http://localhost:{self.port}/client/chatbox_jetson_001/remove")

def main():
    """Main function to start the server controller"""
    print("🎯 Multi-Client Server with Individual Client Monitoring")
    print("   Each client gets their own RobotServer instance with individual web interface")
    print("   Full REST API and WebSocket support for multiple concurrent clients")
    print()
    
    controller = ServerController(port=5000)
    controller.start()

if __name__ == "__main__":
    main()