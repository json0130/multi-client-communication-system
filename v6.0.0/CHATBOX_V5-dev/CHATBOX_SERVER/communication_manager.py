# communication_manager.py - ChatBox WebSocket + REST Communication Manager
import time
import json
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response, session
from flask_socketio import SocketIO, emit, disconnect

class CommunicationManager:
    """
    Handles WebSocket and REST API communication for single ChatBox client.
    Simplified version without multi-client complexity.
    """
    
    def __init__(self, chatbox_server, port=5000):
        self.chatbox_server = chatbox_server
        self.port = port
        
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'chatbox_secret_key'
        
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=30,
            ping_interval=15,
            max_http_buffer_size=1000000,
            transports=['websocket', 'polling'],
            allow_upgrades=True,
            cookie=False
        )
        
        # Client connection state
        self.client_connected = False
        self.client_info = {}
        self.connection_time = None
        
        # Setup handlers
        self.setup_websocket_handlers()
        self.setup_rest_api()
        
        print("📡 Communication Manager initialized")

    def setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔌 Client attempting connection from {request.remote_addr}")
            return True

        @self.socketio.on('client_init')
        def handle_client_init(data):
            """Handle client initialization - expects config from ChatBox"""
            try:
                print(f"🚀 Client init request: {data}")
                
                # Validate required fields
                robot_name = data.get('robot_name')
                client_id = data.get('client_id')
                modules = data.get('modules', [])
                
                if not robot_name or not client_id:
                    emit('client_init_response', {
                        'success': False, 
                        'message': 'robot_name and client_id are required'
                    })
                    return
                
                # Store client info
                self.client_info = {
                    'robot_name': robot_name,
                    'client_id': client_id,
                    'robot_role': data.get('robot_role', 'a friendly companion'),
                    'allowed_tags': data.get('allowed_tags', ['[DEFAULT]']),
                    'modules': modules,
                    'voice_config': data.get('voice_config', {}),
                    'arduino_output': data.get('arduino_output', {}),
                    'hardware': data.get('hardware', {})
                }
                
                # Initialize ChatBox server with client config
                success, message = self.chatbox_server.initialize_with_config(self.client_info)
                
                if success:
                    self.client_connected = True
                    self.connection_time = time.time()
                    session['authenticated'] = True
                    session['client_id'] = client_id
                    
                    print(f"✅ ChatBox '{robot_name}' ({client_id}) connected successfully")
                    
                    emit('client_init_response', {
                        'success': True,
                        'message': f'ChatBox {robot_name} initialized successfully',
                        'client_id': client_id,
                        'robot_name': robot_name,
                        'enabled_modules': modules,
                        'server_status': self.chatbox_server.get_status()
                    })
                else:
                    emit('client_init_response', {
                        'success': False,
                        'message': f'Server initialization failed: {message}'
                    })
                    
            except Exception as e:
                error_msg = f"Client initialization error: {e}"
                print(f"❌ {error_msg}")
                emit('client_init_response', {
                    'success': False, 
                    'message': error_msg
                })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            if self.client_connected:
                client_id = self.client_info.get('client_id', 'unknown')
                print(f"🔌 ChatBox {client_id} disconnected")
                self.client_connected = False
                self.connection_time = None

        @self.socketio.on('ping')
        def handle_ping(data):
            if not self._check_auth():
                return
            
            emit('pong', {
                'timestamp': time.time(),
                'client_id': self.client_info.get('client_id'),
                'robot_name': self.client_info.get('robot_name'),
                'server_status': 'active'
            })

        @self.socketio.on('image_frame')
        def handle_image_frame(data):
            """Handle image frame for emotion processing"""
            if not self._check_auth():
                return
            
            if 'emotion' not in self.client_info.get('modules', []):
                emit('error', {'message': 'Emotion module not enabled'})
                return
            
            try:
                result = self.chatbox_server.process_image_frame(data)
                
                if 'error' in result:
                    emit('error', {
                        'message': result['error'], 
                        'details': result.get('details', '')
                    })
                else:
                    emit('frame_result', {
                        'client_id': self.client_info.get('client_id'),
                        'result': result,
                        'timestamp': time.time()
                    })
                    
            except Exception as e:
                emit('error', {
                    'message': 'Frame processing failed', 
                    'details': str(e)
                })

        @self.socketio.on('chat_message')
        def handle_chat_message(data):
            """Handle chat message via WebSocket"""
            if not self._check_auth():
                return
            
            if 'gpt' not in self.client_info.get('modules', []):
                emit('chat_response', {
                    'error': 'GPT module not enabled',
                    'enabled_modules': self.client_info.get('modules', []),
                    'timestamp': time.time()
                })
                return
            
            try:
                message = data.get('message', '')
                if not message:
                    emit('chat_response', {
                        'error': 'No message provided',
                        'timestamp': time.time()
                    })
                    return
                
                # Process with ChatBox server
                result = self.chatbox_server.process_chat_message(message)
                
                emit('chat_response', {
                    'client_id': self.client_info.get('client_id'),
                    'robot_name': self.client_info.get('robot_name'),
                    'response': result.get('response', ''),
                    'detected_emotion': result.get('detected_emotion'),
                    'confidence': result.get('confidence'),
                    'timestamp': time.time()
                })
                
            except Exception as e:
                emit('chat_response', {
                    'error': 'Chat processing failed',
                    'details': str(e),
                    'timestamp': time.time()
                })

        @self.socketio.on('speech')
        def handle_speech(data):
            """Handle speech audio via WebSocket — transcribes and optionally generates a reply"""
            if not self._check_auth():
                return

            if 'speech' not in self.client_info.get('modules', []):
                emit('speech_response', {'error': 'Speech module not enabled'})
                return

            try:
                audio_b64 = data.get('audio', '')
                if not audio_b64:
                    emit('speech_response', {'error': 'No audio data provided'})
                    return

                result = self.chatbox_server.process_speech_input(audio_b64)

                emit('speech_response', {
                    'client_id':     self.client_info.get('client_id'),
                    'robot_name':    self.client_info.get('robot_name'),
                    'transcription': result.get('transcription', ''),
                    'confidence':    result.get('confidence'),
                    'response':      result.get('response'),
                    'timestamp':     time.time()
                })
            except Exception as e:
                emit('speech_response', {
                    'error':     'Speech processing failed',
                    'details':   str(e),
                    'timestamp': time.time()
                })

        @self.socketio.on('get_status')
        def handle_get_status():
            """Get server status"""
            if not self._check_auth():
                return
            
            try:
                status = self.chatbox_server.get_status()
                
                emit('status_response', {
                    'client_id': self.client_info.get('client_id'),
                    'robot_name': self.client_info.get('robot_name'),
                    'enabled_modules': self.client_info.get('modules', []),
                    'server_active': True,
                    'connection_time': self.connection_time,
                    'server_status': status,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                emit('status_response', {
                    'error': f'Status check failed: {e}',
                    'timestamp': time.time()
                })

    def setup_rest_api(self):
        """Setup REST API endpoints"""
        
        @self.app.route('/', methods=['GET'])
        def server_info():
            """Server status and information"""
            status_data = {
                "message": "ChatBox Server",
                "status": "running",
                "port": self.port,
                "client_connected": self.client_connected,
                "connection_time": self.connection_time
            }
            
            if self.client_connected:
                status_data.update({
                    "robot_name": self.client_info.get('robot_name'),
                    "client_id": self.client_info.get('client_id'),
                    "enabled_modules": self.client_info.get('modules', []),
                    "server_status": self.chatbox_server.get_status() if self.chatbox_server else None
                })
            
            return jsonify(status_data)
        
        @self.app.route('/register_client', methods=['POST'])
        def rest_register_client():
            """Register an HTTP client and their allowed tags"""
            try:
                data = request.json
                client_id = data.get('client_id')
                robot_name = data.get('robot_name', 'HTTP_ChatBox')
                modules = data.get('modules', ['gpt']) # Give HTTP clients GPT access by default
                
                if not client_id:
                    return jsonify({"error": "Missing client_id"}), 400
                
                # Store client info AND the dynamic tags
                self.client_info = {
                    'robot_name': robot_name,
                    'client_id': client_id,
                    'modules': modules,
                    'allowed_tags': data.get('allowed_tags', ['[DEFAULT]']) # 🚀 TAGS CAPTURED HERE
                }
                
                # Initialize ChatBox server with this config
                success, message = self.chatbox_server.initialize_with_config(self.client_info)
                
                if success:
                    self.client_connected = True
                    self.connection_time = time.time()
                    print(f"✅ HTTP Client Registered: {client_id} | Tags: {self.client_info['allowed_tags']}")
                    return jsonify({
                        "success": True, 
                        "message": message,
                        "tags_registered": self.client_info['allowed_tags']
                    }), 200
                else:
                    return jsonify({"error": message}), 500
                    
            except Exception as e:
                return jsonify({"error": f"Registration failed: {str(e)}"}), 500

        @self.app.route('/chat', methods=['POST'])
        def rest_chat():
            """REST API for chat messages"""
            if not self.client_connected:
                return jsonify({"error": "No client connected"}), 400
            
            if 'gpt' not in self.client_info.get('modules', []):
                return jsonify({
                    "error": "GPT module not enabled",
                    "enabled_modules": self.client_info.get('modules', [])
                }), 403
            
            try:
                data = request.json
                message = data.get('message', '')
                
                if not message:
                    return jsonify({"error": "No message provided"}), 400
                
                result = self.chatbox_server.process_chat_message(message)
                
                return jsonify({
                    "client_id": self.client_info.get('client_id'),
                    "robot_name": self.client_info.get('robot_name'),
                    "response": result.get('response', ''),
                    "detected_emotion": result.get('detected_emotion'),
                    "confidence": result.get('confidence'),
                    "timestamp": time.time()
                }), 200
                
            except Exception as e:
                return jsonify({
                    "error": "Chat processing failed", 
                    "details": str(e)
                }), 500

        @self.app.route('/speech', methods=['POST'])
        def rest_speech():
            """REST API for speech processing"""
            if not self.client_connected:
                return jsonify({"error": "No client connected"}), 400
            
            if 'speech' not in self.client_info.get('modules', []):
                return jsonify({
                    "error": "Speech module not enabled",
                    "enabled_modules": self.client_info.get('modules', [])
                }), 403
            
            try:
                data = request.json
                audio_b64 = data.get('audio', '')
                
                if not audio_b64:
                    return jsonify({"error": "No audio data provided"}), 400
                
                result = self.chatbox_server.process_speech_input(audio_b64)
                
                return jsonify({
                    "client_id": self.client_info.get('client_id'),
                    "robot_name": self.client_info.get('robot_name'),
                    "transcription": result.get('transcription', ''),
                    "confidence": result.get('confidence'),
                    "response": result.get('response'),  # If GPT also enabled
                    "timestamp": time.time()
                }), 200
                
            except Exception as e:
                return jsonify({
                    "error": "Speech processing failed", 
                    "details": str(e)
                }), 500

        @self.app.route('/emotion', methods=['GET'])
        def rest_emotion():
            """REST API for current emotion state"""
            if not self.client_connected:
                return jsonify({"error": "No client connected"}), 400
            
            if 'emotion' not in self.client_info.get('modules', []):
                return jsonify({
                    "error": "Emotion module not enabled",
                    "enabled_modules": self.client_info.get('modules', [])
                }), 403
            
            try:
                emotion_data = self.chatbox_server.get_current_emotion_state()
                
                return jsonify({
                    "client_id": self.client_info.get('client_id'),
                    "robot_name": self.client_info.get('robot_name'),
                    "emotion": emotion_data.get('emotion'),
                    "confidence": emotion_data.get('confidence'),
                    "distribution": emotion_data.get('distribution'),
                    "timestamp": time.time()
                }), 200
                
            except Exception as e:
                return jsonify({
                    "error": "Emotion processing failed", 
                    "details": str(e)
                }), 500

        @self.app.route('/health', methods=['GET'])
        def rest_health():
            """REST API for health check"""
            try:
                health_data = {
                    "status": "healthy",
                    "client_connected": self.client_connected,
                    "timestamp": time.time()
                }
                
                if self.client_connected:
                    health_data.update({
                        "client_id": self.client_info.get('client_id'),
                        "robot_name": self.client_info.get('robot_name'),
                        "enabled_modules": self.client_info.get('modules', []),
                        "connection_time": self.connection_time,
                        "server_status": self.chatbox_server.get_status()
                    })
                
                return jsonify(health_data), 200
                
            except Exception as e:
                return jsonify({
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }), 500
            
        # Backward compatibility routes for old client API
        @self.app.route('/client/<client_id>/chat', methods=['POST'])
        def rest_chat_compat(client_id):
            """Backward compatibility for old /client/{id}/chat endpoint"""
            return rest_chat()

        @self.app.route('/client/<client_id>/speech', methods=['POST']) 
        def rest_speech_compat(client_id):
            """Backward compatibility for old /client/{id}/speech endpoint"""
            return rest_speech()

        @self.app.route('/client/<client_id>/emotion', methods=['GET'])
        def rest_emotion_compat(client_id):
            """Backward compatibility for old /client/{id}/emotion endpoint"""
            return rest_emotion()

        @self.app.route('/client/<client_id>/health', methods=['GET'])
        def rest_health_compat(client_id):
            """Backward compatibility for old /client/{id}/health endpoint"""
            return rest_health()

        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response

    def _check_auth(self) -> bool:
        """Check if client is authenticated"""
        if not self.client_connected:
            emit('error', {'message': 'Client not initialized. Send client_init first.'})
            return False
        return True

    def broadcast_to_client(self, event: str, data: Dict[str, Any]):
        """Broadcast message to connected client"""
        if self.client_connected:
            try:
                self.socketio.emit(event, data)
            except Exception as e:
                print(f"⚠️ Broadcast error: {e}")

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'client_connected': self.client_connected,
            'client_info': self.client_info.copy() if self.client_connected else {},
            'connection_time': self.connection_time,
            'uptime': time.time() - self.connection_time if self.connection_time else 0
        }

    def start_server(self):
        """Start the communication server"""
        try:
            print("🚀 Starting ChatBox Communication Server...")
            print("=" * 50)
            print(f"🌐 Server: http://0.0.0.0:{self.port}")
            print(f"🔌 WebSocket: ws://0.0.0.0:{self.port}/socket.io/")
            print("=" * 50)
            print("✅ Server ready for ChatBox connection!")
            
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
        except Exception as e:
            print(f"❌ Server error: {e}")