# request_router.py - COMPLETE FIX with unified delegation handling

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
        self.socketio = socketio
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
            elif endpoint == 'monitor': 
                return self._handle_monitor_request(server, display_name)
            elif endpoint == 'live_stream': 
                return self._handle_live_stream_request(server, display_name)
            else:
                return jsonify({"error": f"Unknown endpoint: {endpoint}"}), 404
                
        except Exception as e:
            client_info = self.client_manager.get_client_info(client_id)
            display_name = client_info.get_display_name() if client_info else client_id
            print(f"❌ Request routing error for {display_name}: {e}")
            return jsonify({"error": "Internal server error", "details": str(e)}), 500
    
    def _process_delegation_logic(self, server, llm_response_text: str) -> Dict[str, Any]:
        """
        Unified delegation processing logic used by both chat and speech handlers.
        
        Returns:
            Dict with keys:
            - 'is_delegation': bool - whether this is a delegation
            - 'user_facing_text': str - text to show/speak to user (if not delegation)
            - 'delegation_speech': str - speech for delegating robot (if delegation)
            - 'target_id': str - target robot ID (if delegation)
            - 'task_message': str - task for target robot (if delegation)
        """
        result = {
            'is_delegation': False,
            'user_facing_text': llm_response_text,
            'delegation_speech': None,
            'target_id': None,
            'task_message': None
        }
        
        # Check for delegation command
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response_text, re.DOTALL)
        
        if match and self.socketio:
            try:
                command_data = json.loads(match.group(1))
                target_id = command_data.get("target_robot_id")
                task_message = command_data.get("task")

                if target_id and task_message:
                    # Extract user-facing response (without JSON block)
                    user_facing_response = llm_response_text.replace(match.group(0), "").strip()
                    
                    # Construct delegation speech (for the ORIGINAL robot to speak)
                    delegation_speech = f"{user_facing_response} {task_message}"
                    
                    result.update({
                        'is_delegation': True,
                        'user_facing_text': user_facing_response,
                        'delegation_speech': delegation_speech,
                        'target_id': target_id,
                        'task_message': task_message
                    })
                    
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON in response. Treating as normal message.")
        
        return result
    
    def _execute_delegation(self, server, delegation_info: Dict[str, Any]):
        """
        Execute the delegation: send to original robot and target robot.
        
        Args:
            server: The originating server instance
            delegation_info: Dict from _process_delegation_logic with delegation details
        """
        if not delegation_info['is_delegation']:
            return
        
        target_id = delegation_info['target_id']
        task_message = delegation_info['task_message']
        delegation_speech = delegation_info['delegation_speech']
        
        # 1. Send delegation speech to ORIGINAL robot via SocketIO
        print(f"🗣️ Sending delegation speech to {server.client_id}: '{delegation_speech}'")
        self.socketio.emit('chat_response', {'response': delegation_speech}, room=server.client_id)

        # 2. Send silent command to TARGET robot
        print(f"📤 Relaying silent command to {target_id}: '{task_message}'")
        # self.socketio.emit('execute_command', {'command': task_message}, room=target_id)

        # 3. Background task for target's response
        def execute_and_respond(target_server_instance, task, socket_io_instance):
            import time
            time.sleep(10)  # Brief pause to ensure command is registered
            try:
                print(f"🚀 Executing background task for '{target_server_instance.client_id}'...")
                delegated_result = target_server_instance.process_chat_message(task, is_delegated_command=True)
                final_response = delegated_result.get('response', 'Task completed.')
                print(f"✅ Background task complete. Response from '{target_server_instance.client_id}': {final_response}")
                
                # Emit final response for TARGET robot to speak
                socket_io_instance.emit('chat_response', {'response': final_response}, room=target_server_instance.client_id)
            except Exception as e:
                print(f"❌ Error in background delegation thread: {e}")

        target_server = self.client_manager.get_client_server(target_id)
        if target_server:
            threading.Thread(target=execute_and_respond, args=(target_server, task_message, self.socketio)).start()
        else:
            print(f"⚠️ Target robot '{target_id}' not found or not active")

    def _handle_chat_request(self, server, flask_request, display_name: str) -> tuple:
        """Handle chat request with delegation support"""
        try:
            message = flask_request.json.get('message', '')
            if not message:
                return jsonify({"error": "Message cannot be empty"}), 400

            # Process the message
            result = server.process_chat_message(message)
            llm_response_text = result.get('response', '')
            
            # Process delegation logic
            delegation_info = self._process_delegation_logic(server, llm_response_text)
            
            if delegation_info['is_delegation']:
                # Execute delegation
                self._execute_delegation(server, delegation_info)
                
                # Return metadata only
                return jsonify({
                    "status": "delegated",
                    "client_id": server.client_id,
                    "target_robot": delegation_info['target_id'],
                    "message": "Task delegated successfully"
                }), 200
            else:
                # Normal response - send via SocketIO
                if self.socketio:
                    print(f"🗣️ Sending normal response to {server.client_id}: '{llm_response_text}'")
                    self.socketio.emit('chat_response', {'response': llm_response_text}, room=server.client_id)
                
                # Return metadata
                return jsonify({
                    "status": "completed",
                    "client_id": server.client_id,
                    "message": "Response sent successfully"
                }), 200

        except Exception as e:
            print(f"❌ {display_name}: Chat request error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Chat processing failed", "details": str(e)}), 500
    
    def _handle_speech_request(self, server, flask_request, display_name: str) -> tuple:
        """Handle speech-to-text request with delegation support"""
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
            
            # Process speech input
            result = server.process_speech_input(audio_b64)
            
            transcription = result.get('transcription', '')
            print(f"📝 {display_name}: Transcribed: '{transcription}'")
            
            # Check if GPT response is present (means GPT module is enabled)
            llm_response_text = result.get('response')
            
            if llm_response_text:
                # Process delegation logic for the GPT response
                delegation_info = self._process_delegation_logic(server, llm_response_text)
                
                if delegation_info['is_delegation']:
                    # Execute delegation
                    print(f"🔄 {display_name}: Speech triggered delegation")
                    self._execute_delegation(server, delegation_info)
                    
                    # Return speech result with delegation status
                    return jsonify({
                        "status": "delegated",
                        "client_id": server.client_id,
                        "robot_name": getattr(server, 'robot_name', 'Unknown'),
                        "transcription": transcription,
                        "confidence": result.get('confidence'),
                        "target_robot": delegation_info['target_id'],
                        "message": "Speech processed and task delegated",
                        "timestamp": time.time()
                    }), 200
                else:
                    # Normal response - send via SocketIO
                    if self.socketio:
                        print(f"🗣️ Sending speech response to {server.client_id}: '{llm_response_text}'")
                        self.socketio.emit('chat_response', {'response': llm_response_text}, room=server.client_id)
            
            # Return speech processing result
            return jsonify({
                "status": "completed",
                "client_id": server.client_id,
                "robot_name": getattr(server, 'robot_name', 'Unknown'),
                "transcription": transcription,
                "confidence": result.get('confidence'),
                "message": "Speech processed successfully",
                "timestamp": time.time()
            }), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Speech request error: {e}")
            import traceback
            traceback.print_exc()
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