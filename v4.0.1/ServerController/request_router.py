# request_router.py - Request Routing and Processing Logic
import time
from typing import Dict, Any
from flask import jsonify, Response

from client_manager import ClientManager

class RequestRouter:
    """
    Handles routing of HTTP requests to appropriate client server instances.
    Contains all the business logic for processing different types of requests.
    """
    
    def __init__(self, client_manager: ClientManager, socketio=None):  # ✅ FIXED: Add socketio parameter
        self.client_manager = client_manager
        self.socketio = socketio  # ✅ FIXED: Store socketio for broadcasting
    
    def route_client_request(self, client_id: str, endpoint: str, flask_request) -> tuple:
        """Route request to appropriate client server instance"""
        try:
            # Get client info for display name
            client_info = self.client_manager.get_client_info(client_id)
            if not client_info:
                return jsonify({"error": f"Client '{client_id}' not registered"}), 404
            
            display_name = client_info.get_display_name()
            
            # Get or create client server instance
            server = self.client_manager.get_or_create_server_instance(client_id)
            if not server:
                return jsonify({
                    "error": f"Failed to create server instance for {display_name}"
                }), 500
            
            # Route to appropriate endpoint handler
            if endpoint == 'chat':
                return self._handle_chat_request(server, flask_request, display_name)
            elif endpoint == 'speech':
                return self._handle_speech_request(server, flask_request, display_name)
            elif endpoint == 'health':
                return self._handle_health_request(server, display_name)
            elif endpoint == 'emotion':
                return self._handle_emotion_request(server, display_name)
            elif endpoint == 'monitor':  # ✅ NEW: Monitor page
                return self._handle_monitor_request(server, display_name)
            elif endpoint == 'live_stream':  # ✅ NEW: Live stream
                return self._handle_live_stream_request(server, display_name)
            else:
                return jsonify({"error": f"Unknown endpoint: {endpoint}"}), 404
                
        except Exception as e:
            client_info = self.client_manager.get_client_info(client_id)
            display_name = client_info.get_display_name() if client_info else client_id
            print(f"❌ Request routing error for {display_name}: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500


    def _handle_chat_request(self, server, flask_request, display_name: str) -> tuple:
        """Handle chat request - requires GPT module"""
        try:
            client_modules = self.client_manager.get_client_modules(server.client_id)
            
            if 'gpt' not in client_modules:
                return jsonify({
                    "error": "GPT module not enabled for this client",
                    "enabled_modules": list(client_modules)
                }), 403
            
            data = flask_request.json
            message = data.get('message', '')
            
            if not message:
                return jsonify({"error": "No message provided"}), 400
            
            print(f"💬 {display_name}: Processing chat message: '{message}'")
            
            result = server.process_chat_message(message)
            
            print(f"🤖 {display_name}: Response: '{result.get('response', 'No response')}'")
            
            # ✅ FIXED: Broadcast HTTP chat messages ONLY to the correct monitor via its room and namespace
            if self.socketio:
                try:
                    # The event name 'client_chat_message' matches the new JS in server.py
                    # The room=server.client_id ensures it only goes to the right monitor.
                    # The namespace='/monitor' ensures it's sent on the channel monitors are listening to.

                    # Broadcast user message
                    self.socketio.emit('client_chat_message', {
                        'type': 'user',
                        'content': message,
                        'client_id': server.client_id,
                    }, room=server.client_id, namespace='/monitor')
                    
                    # Broadcast bot response
                    self.socketio.emit('client_chat_message', {
                        'type': 'bot',
                        'content': result.get('response', ''),
                        'client_id': server.client_id,
                    }, room=server.client_id, namespace='/monitor')
                    
                    print(f"📡 {display_name}: Chat broadcasted to its specific monitor room.")
                    
                except Exception as broadcast_error:
                    print(f"⚠️ {display_name}: Chat broadcast error: {broadcast_error}")
            
            return jsonify({
                "client_id": server.client_id,
                "response": result.get('response', ''),
                "detected_emotion": result.get('detected_emotion'),
                "timestamp": time.time()
            }), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Chat request error: {e}")
            return jsonify({"error": "Chat processing failed", "details": str(e)}), 500
    
    def _handle_speech_request(self, server, flask_request, display_name: str) -> tuple:
        """Handle speech-to-text request - requires Speech module"""
        try:
            client_modules = self.client_manager.get_client_modules(server.client_id)
            
            if 'speech' not in client_modules:
                return jsonify({
                    "error": "Speech module not enabled for this client",
                    "enabled_modules": list(client_modules)
                }), 403
            
            data = flask_request.json
            audio_b64 = data.get('audio', '')
            
            if not audio_b64:
                return jsonify({"error": "No audio data provided"}), 400
            
            print(f"🎤 {display_name}: Processing speech input")
            
            # Process with the client's server instance
            result = server.process_speech_input(audio_b64)
            
            transcription = result.get('transcription', '')
            print(f"📝 {display_name}: Transcribed: '{transcription}'")
            
            if result.get('response'):
                print(f"🤖 {display_name}: GPT Response: '{result.get('response')}'")
            
            return jsonify({
                "client_id": server.client_id,
                "robot_name": getattr(server, 'robot_name', 'Unknown'),
                "transcription": transcription,
                "confidence": result.get('confidence'),
                "response": result.get('response'),  # If GPT is also enabled
                "timestamp": time.time()
            }), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Speech request error: {e}")
            return jsonify({"error": "Speech processing failed", "details": str(e)}), 500
    
    def _handle_health_request(self, server, display_name: str) -> tuple:
        """Handle health check request"""
        try:
            client_modules = self.client_manager.get_client_modules(server.client_id)
            client_info = self.client_manager.get_client_info(server.client_id)
            
            return jsonify({
                "status": "healthy",
                "client_id": server.client_id,
                "robot_name": getattr(server, 'robot_name', 'Unknown'),
                "enabled_modules": list(client_modules),
                "server_status": server.get_health_status(),
                "last_activity": client_info.last_activity if client_info else 0,
                "registration_time": client_info.registration_time if client_info else 0,
                "timestamp": time.time()
            }), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Health request error: {e}")
            return jsonify({"error": "Health check failed", "details": str(e)}), 500
    
    def _handle_emotion_request(self, server, display_name: str) -> tuple:
        """Handle emotion state request - requires Emotion module"""
        try:
            client_modules = self.client_manager.get_client_modules(server.client_id)
            
            if 'emotion' not in client_modules:
                return jsonify({
                    "error": "Emotion module not enabled for this client",
                    "enabled_modules": list(client_modules)
                }), 403
            
            emotion_data = server.get_current_emotion_state()
            
            print(f"🎭 {display_name}: Current emotion: {emotion_data.get('emotion')} ({emotion_data.get('confidence', 0):.1f}%)")
            
            return jsonify({
                "client_id": server.client_id,
                "robot_name": getattr(server, 'robot_name', 'Unknown'),
                "emotion": emotion_data.get('emotion'),
                "confidence": emotion_data.get('confidence'),
                "distribution": emotion_data.get('distribution'),
                "timestamp": time.time()
            }), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Emotion request error: {e}")
            return jsonify({"error": "Emotion processing failed", "details": str(e)}), 500
    
    # ✅ NEW: Monitor page handler
    def _handle_monitor_request(self, server, display_name: str) -> tuple:
        """Handle individual client monitor page request"""
        try:
            print(f"🖥️ {display_name}: Monitor page request")

            # DEBUG: Check what methods the server has
            print(f"🔍 DEBUG: Server type: {type(server)}")
            print(f"🔍 DEBUG: Has get_individual_monitor_html: {hasattr(server, 'get_individual_monitor_html')}")
            
            # Get individual monitor HTML
            monitor_html = server.get_individual_monitor_html()
            
            return Response(monitor_html, mimetype='text/html'), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Monitor page error: {e}")
            
            # Return error HTML page
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Monitor Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 40px; text-align: center; background: #f5f5f5; }}
                    .error-container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 500px; margin: 0 auto; }}
                    h1 {{ color: #e74c3c; margin-bottom: 20px; }}
                    p {{ color: #666; margin: 10px 0; }}
                    .error-details {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 15px 0; font-family: monospace; font-size: 14px; }}
                </style>
            </head>
            <body>
                <div class="error-container">
                    <h1>❌ Monitor Error</h1>
                    <p>Error loading monitor for {display_name}</p>
                    <div class="error-details">{str(e).replace('<', '&lt;').replace('>', '&gt;')}</div>
                    <p><a href="/">← Back to Controller</a></p>
                </div>
            </body>
            </html>
            """
            return Response(error_html, mimetype='text/html', status=500), 500
    
    # ✅ NEW: Live stream handler
    def _handle_live_stream_request(self, server, display_name: str) -> tuple:
        """Handle individual client live stream request"""
        try:
            print(f"📺 {display_name}: Live stream request")
            
            # Check if emotion module is enabled (required for live stream)
            client_modules = self.client_manager.get_client_modules(server.client_id)
            if 'emotion' not in client_modules and 'facial' not in client_modules:
                return jsonify({
                    'error': 'Emotion or facial recognition module required for live stream',
                    'enabled_modules': list(client_modules),
                    'timestamp': time.time()
                }), 400
            
            # Generate individual live stream
            stream_response = server.generate_individual_live_stream()
            
            if stream_response is None:
                return jsonify({
                    'error': 'Live stream not available for this client',
                    'details': 'Web interface not initialized',
                    'timestamp': time.time()
                }), 503
            
            return stream_response, 200
            
        except Exception as e:
            print(f"❌ {display_name}: Live stream error: {e}")
            return jsonify({
                'error': 'Live stream failed',
                'details': str(e),
                'timestamp': time.time()
            }), 500
    
    def handle_image_frame_processing(self, client_id: str, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle image frame processing via WebSocket
        Returns result dict or error dict
        """
        try:
            # Get client info for display name
            client_info = self.client_manager.get_client_info(client_id)
            if not client_info:
                return {"error": f"Client '{client_id}' not registered"}
            
            display_name = client_info.get_display_name()
            
            # Get server instance
            server = self.client_manager.get_or_create_server_instance(client_id)
            if not server:
                return {"error": f"No server instance for {display_name}"}
            
            # Check if client has emotion or facial modules
            client_modules = self.client_manager.get_client_modules(client_id)
            if 'emotion' not in client_modules and 'facial' not in client_modules:
                return {"error": f"Emotion or facial module required for {display_name}"}
            
            # Process image frame
            result = server.process_image_frame(frame_data)
            
            # Log the result
            emotion = result.get('emotion', 'unknown')
            confidence = result.get('confidence', 0)
            print(f"📸 {display_name}: Frame processed - {emotion} ({confidence:.1f}%)")
            
            # Add client info to result
            result.update({
                'client_id': client_id,
                'robot_name': getattr(server, 'robot_name', 'Unknown')
            })
            
            return result
            
        except Exception as e:
            client_info = self.client_manager.get_client_info(client_id)
            display_name = client_info.get_display_name() if client_info else client_id
            print(f"❌ {display_name}: Image frame processing error: {e}")
            return {
                "error": "Frame processing failed",
                "details": str(e)
            }