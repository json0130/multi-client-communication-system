# request_router.py - Request Routing and Processing Logic
import time
import json
import re
import threading
import requests
from typing import Dict, Any, Optional
from flask import jsonify, Response

from client_manager import ClientManager
from database import Database
from Modules.gpt_client import GPTClient

class RequestRouter:
    """
    Handles routing of HTTP requests to appropriate client server instances.
    Contains all the business logic for processing different types of requests.
    """
    
    def __init__(self, client_manager: ClientManager, socketio=None, database: Optional[Database] = None):
        self.client_manager = client_manager
        self.socketio = socketio  # ✅ FIXED: Store socketio for broadcasting
        self.db = database or Database()
    
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
        
    def route_user_request(self, user_id: int, endpoint: str, flask_request):
        """
        Handle routes that operate on the user record itself (no client server).
        """
        try:
            if endpoint == "infer_topics":
                return self._handle_infer_topics(user_id, flask_request)
            return jsonify({"error": f"Unknown user endpoint: {endpoint}"}), 404
        except Exception as e:
            print(f"❌ User-route error for {user_id}: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500

    def _handle_chat_request(self, server, flask_request, display_name: str) -> tuple:
        """
        Handles a chat request by routing it to the server's processing method
        and managing any resulting delegation commands.
        """
        try:
            message = flask_request.json.get('message', '')
            if not message:
                return jsonify({"error": "Message cannot be empty"}), 400

            # 1. Call the new, centralized processing method in server.py
            # The 'is_delegated_command' flag is False by default, putting it in DELEGATION MODE.
            result = server.process_chat_message(message)
            
            llm_response_text = result.get('response', '')
            user_facing_response = llm_response_text # Start with the full response

            # 2. Check if the LLM decided to delegate a task.
            # This logic correctly stays in the router, which manages inter-robot communication.
            # Check if the LLM decided to delegate a task.
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response_text, re.DOTALL)
            if match:
                try:
                    command_data = json.loads(match.group(1))
                    target_id = command_data.get("target_robot_id")
                    task_message = command_data.get("task")

                    if target_id and task_message:
                        
                        # Instead of making an HTTP request, we directly emit the WebSocket event.
                        # This is guaranteed to have the correct context.
                        print(f" M-DEBUG -> Relaying spoken command directly to {target_id}: '{task_message}'")
                        self.socketio.emit(
                            'execute_command',                  # The event name the client is listening for
                            {'command': task_message},          # The data payload
                            room=target_id                      # Send only to this specific client's room
                        )

                        # The background task now needs the socketio instance to report back
                        def execute_and_report_back(server_instance, task, socketio_instance):
                            try:
                                print(f"🚀 Executing delegated task for '{server_instance.client_id}'...")
                                
                                # 1. Execute the task and get the result
                                result = server_instance.process_chat_message(task, is_delegated_command=True)
                                response_content = result.get('response', 'Task completed.')
                                
                                print(f"✅ Delegated task complete. Response from '{server_instance.client_id}': {response_content}")

                                # 2. Report the result back to the correct client's UI
                                if socketio_instance:
                                    # Send to the specific client's monitor
                                    socketio_instance.emit('client_chat_message', 
                                        {'type': 'bot', 'content': response_content, 'client_id': server_instance.client_id},
                                        room=server_instance.client_id, 
                                        namespace='/monitor'
                                    )
                                    # Also send to the global Mission Control monitor
                                    socketio_instance.emit('global_log',
                                        {'type': 'bot', 'content': f"(Delegated) {response_content}", 'client_id': server_instance.client_id},
                                        namespace='/monitor'
                                    )

                            except Exception as e:
                                print(f"❌ Error in background delegation thread for '{server_instance.client_id}': {e}")
                        
                        target_server = self.client_manager.get_client_server(target_id)
                        if target_server:
                            # Start the task in a thread, passing socketio to it
                            delegation_thread = threading.Thread(
                                target=execute_and_report_back,
                                args=(target_server, task_message, self.socketio) # Pass self.socketio
                            )
                            delegation_thread.start()

                        # Clean the JSON block from the response sent to the original user
                        user_facing_response = llm_response_text.replace(match.group(0), "").strip()
                
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON found in response from '{server.client_id}'")


            # 3. Prepare and send the final response to the user
            final_result = {
                "client_id": server.client_id,
                "response": user_facing_response, # Send the cleaned response
                "detected_emotion": result.get('detected_emotion'),
                "timestamp": time.time()
            }

            # (Optional) Broadcast to your monitoring UI if you have one
            if self.socketio:
                self.socketio.emit('client_chat_message', {'type': 'user', 'content': message, 'client_id': server.client_id}, room=server.client_id, namespace='/monitor')
                self.socketio.emit('client_chat_message', {'type': 'bot', 'content': user_facing_response, 'client_id': server.client_id}, room=server.client_id, namespace='/monitor')

            return jsonify(final_result), 200

        except Exception as e:
            print(f"❌ {display_name}: Chat request error: {e}")
            import traceback
            traceback.print_exc()
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
            
    def _handle_infer_topics(self, user_id: int, flask_request):
        sample_size = int(flask_request.json.get("sample_size", 100)) if flask_request.is_json else 100

        # 1) fetch latest messages
        messages = self.db.get_chat_messages(user_id, limit=sample_size)
        if not messages:
            return jsonify({"error": "No chat logs for user"}), 404

        # 2) GPT inference
        gpt = GPTClient()
        if not gpt.setup_openai():
            return jsonify({"error": "OPENAI_API_KEY not configured"}), 500

        interests, conditions = gpt.infer_topics_and_conditions(messages)

        # 3) update Supabase
        self.db.update_user(user_id, interests=interests, health_conditions=conditions)

        return jsonify({
            "user_id": user_id,
            "interests": interests,
            "health_conditions": conditions,
            "updated": True
        }), 200