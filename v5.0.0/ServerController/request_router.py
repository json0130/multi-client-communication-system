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
        try:
            message = flask_request.json.get('message', '')
            if not message:
                return jsonify({"error": "Message cannot be empty"}), 400

            result = server.process_chat_message(message)
            llm_response_text = result.get('response', '')
            
            # --- START: CORRECTED LOGIC WITH CLEAR IF/ELSE ---
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response_text, re.DOTALL)
            
            # PATH 1: The response contains a delegation command.
            if match and self.socketio:
                try:
                    command_data = json.loads(match.group(1))
                    target_id = command_data.get("target_robot_id")
                    task_message = command_data.get("task")

                    if target_id and task_message:
                        # 1. Construct the ONE speech for the ORIGINAL robot (e.g., chatbox).
                        user_facing_response = llm_response_text.replace(match.group(0), "").strip()
                        delegation_speech = f"{user_facing_response} {task_message}"
                        
                        # 2. Emit the single, combined speech to the ORIGINAL robot.
                        print(f"🗣️ Sending delegation speech to {server.client_id}: '{delegation_speech}'")
                        self.socketio.emit('chat_response', {'response': delegation_speech}, room=server.client_id)

                        # 3. Emit the silent command to the TARGET robot (e.g., silbot).
                        print(f"📤 Relaying silent command to {target_id}: '{task_message}'")
                        self.socketio.emit('execute_command', {'command': task_message}, room=target_id)

                        # 4. Handle the background task for the target's response.
                        def execute_and_respond(target_server_instance, task, socket_io_instance):
                            try:
                                print(f"🚀 Executing background task for '{target_server_instance.client_id}'...")
                                delegated_result = target_server_instance.process_chat_message(task, is_delegated_command=True)
                                final_response = delegated_result.get('response', 'Task completed.')
                                print(f"✅ Background task complete. Response from '{target_server_instance.client_id}': {final_response}")

                                # 5. Emit the final response for the TARGET robot to speak.
                                socket_io_instance.emit('chat_response', {'response': final_response}, room=target_server_instance.client_id)
                            except Exception as e:
                                print(f"❌ Error in background delegation thread: {e}")

                        target_server = self.client_manager.get_client_server(target_id)
                        if target_server:
                            threading.Thread(target=execute_and_respond, args=(target_server, task_message, self.socketio)).start()
                        
                        # The HTTP response only contains the user-facing part.
                        return jsonify({"response": user_facing_response, "client_id": server.client_id}), 200

                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON in response. Treating as a normal message.")
                    # If JSON fails, the code will now correctly fall through to the 'else' block.

            # PATH 2: The response is a normal, non-delegated message.
            # This 'else' block ensures this code ONLY runs if the 'if match' above was false.
            if self.socketio:
                print(f"🗣️ Sending normal response to {server.client_id}: '{llm_response_text}'")
                self.socketio.emit('chat_response', {'response': llm_response_text}, room=server.client_id)
            
            return jsonify({"response": llm_response_text, "client_id": server.client_id}), 200
            # --- END: CORRECTED LOGIC ---

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