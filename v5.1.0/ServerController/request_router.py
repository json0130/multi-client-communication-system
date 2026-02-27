# request_router.py - COMPLETE FIX with unified delegation handling and server-side role templates

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

# Role templates for server-side processing
ROLE_TEMPLATES = {
    "cooking_robot": "{robot_name}, a cooking robot. You are friendly, patient, and knowledgeable about food. Your purpose is to assist users with cooking-related tasks—such as preparing meals like breakfast, lunch, or dinner, suggesting recipes, and offering guidance in the kitchen—while keeping interactions warm and engaging.",
    
    "guide": "{robot_name}, an enthusiastic and knowledgeable tour guide robot. Your purpose is to provide engaging information about exhibits, landmarks, or facilities, answer visitor questions, and help people navigate spaces. You communicate clearly, adapt explanations to different age groups, and make learning fun and memorable.",
    
    "assistant": "{robot_name}, a versatile and helpful assistant robot. You are professional, efficient, and friendly. Your purpose is to help people with a wide variety of tasks—from answering questions and providing information to scheduling, reminders, and general support. You adapt to different situations and prioritize being helpful while maintaining a warm, approachable demeanor.",
    
    "greeter": "{robot_name}, a friendly and welcoming greeter robot. Your purpose is to welcome visitors, provide initial information about the facility or event, help with directions, and create a positive first impression. You are warm, enthusiastic, and attentive to people's needs, making everyone feel valued and comfortable.",
    
    "security": "{robot_name}, a vigilant and professional security robot. Your purpose is to monitor areas, detect unusual activities, provide safety information, and assist with emergency procedures. You are calm, authoritative when needed, and focused on maintaining a safe environment. You balance being approachable with being observant and responsive to security concerns.",
    
    "cleaning": "{robot_name}, an efficient and thorough cleaning robot. Your purpose is to maintain clean and hygienic spaces, identify areas that need attention, and optimize cleaning schedules. You are detail-oriented, systematic, and take pride in creating comfortable environments. You communicate clearly about cleaning status and work around people's activities with minimal disruption."
}

AVAILABLE_ROLES = ["guide", "cooking_robot", "assistant", "greeter", "security", "cleaning"]

def get_role_prompt(role_name: str, robot_name: str) -> str:
    """
    Generate a role prompt based on the role name and robot name.
    
    Args:
        role_name: The role identifier (e.g., 'cooking_robot', 'guide')
        robot_name: The actual name of the robot (e.g., 'Pepper', 'Alex')
    
    Returns:
        A formatted role prompt string with "You are" prefix
    """
    if role_name in ROLE_TEMPLATES:
        template = ROLE_TEMPLATES[role_name]
        return f"You are {template.format(robot_name=robot_name)}"
    else:
        # Default fallback role
        return f"You are {robot_name}, a helpful robot assistant. You are friendly, professional, and ready to help with various tasks. You communicate clearly and adapt to different situations to provide the best assistance possible."


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
            elif endpoint == 'modules':
                return self._handle_modules_request(server, display_name)
            elif endpoint == 'monitor': 
                return self._handle_monitor_request(server, display_name)
            elif endpoint == 'live_stream': 
                return self._handle_live_stream_request(server, display_name)
            elif endpoint == 'update':
                return self._handle_update_request(server, flask_request, display_name)
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
        
    def _handle_modules_request(self, server, display_name: str) -> tuple:
        """Handle request to get enabled modules for the client"""
        try:
            client_modules = self.client_manager.get_client_modules(server.client_id)
            
            return jsonify({
                "client_id": server.client_id,
                "robot_name": getattr(server, 'robot_name', 'Unknown'),
                "enabled_modules": list(client_modules),
                "timestamp": time.time()
            }), 200
            
        except Exception as e:
            print(f"❌ {display_name}: Modules request error: {e}")
            return jsonify({"error": "Failed to retrieve modules", "details": str(e)}), 500
    
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
    
    def _handle_update_request(self, server, flask_request, display_name: str) -> tuple:
        """Handle client config update request with role template + character/voice processing"""
        try:
            update_data = flask_request.json
            if not update_data:
                return jsonify({"error": "No update data provided"}), 400

            # Get robot name for role template processing
            robot_name = update_data.get('robot_name') or getattr(server, 'robot_name', 'Robot')

            # === ROLE HANDLING (unchanged) ===
            if 'robot_role' in update_data:
                role_name = update_data['robot_role']
                if role_name in AVAILABLE_ROLES:
                    robot_role = get_role_prompt(role_name, robot_name)
                    update_data['robot_role'] = robot_role
                    print(f"🎭 Generated role for {display_name}: {role_name}")
                else:
                    return jsonify({"error": f"Invalid role_name. Available: {', '.join(AVAILABLE_ROLES)}"}), 400

            # === NEW: CHARACTER → EDGE_TTS_CONFIG ===
            if 'character' in update_data:
                character_id = update_data['character']
                voice_map = {
                    "male_friendly": "en-US-GuyNeural",
                    "female_friendly": "en-US-JennyNeural",
                    "male_professional": "en-NZ-MitchellNeural",
                    "female_professional": "en-US-AriaNeural",
                    "child_friendly": "en-US-AnaNeural",
                    "elderly_friendly": "en-NZ-MollyNeural",
                }

                if character_id in voice_map:
                    voice = voice_map[character_id]
                    edge_tts_config = {
                        "voice": voice,
                        "playback_device": "hw:2,0"          # change if you want per-robot device later
                    }
                    update_data['edge_tts_config'] = edge_tts_config

                    print(f"🎙️ Character '{character_id}' → voice '{voice}' for {display_name}")
                else:
                    return jsonify({"error": f"Invalid character ID: {character_id}"}), 400

            # === MODULES VALIDATION (unchanged) ===
            if 'modules' in update_data:
                modules = update_data['modules']
                valid_modules = self.client_manager.valid_modules
                if not isinstance(modules, list) or not all(m in valid_modules for m in modules):
                    return jsonify({"error": f"Invalid modules. Valid: {list(valid_modules)}"}), 400
                if 'rag' not in modules:
                    modules.append('rag')

            # === DB UPDATE (now also stores character) ===
            db_update = {}
            if 'robot_name' in update_data:
                db_update['robot_name'] = update_data['robot_name']
            if 'robot_role' in update_data:
                db_update['robot_role'] = update_data['robot_role']
            if 'role_name' in update_data:
                db_update['role_name'] = update_data['role_name']
            if 'modules' in update_data:
                db_update['modules'] = update_data['modules']
            if 'character' in update_data:                     # ← NEW
                db_update['character'] = update_data['character']
            if 'ocean_traits' in update_data:
                db_update['ocean_traits'] = update_data['ocean_traits']

            if db_update:
                try:
                    self.db.client.supabase.table('robots').update(db_update).eq('client_id', server.client_id).execute()
                    print(f"📝 Updated DB for {display_name}: {db_update}")
                except Exception as e:
                    # Check for missing column error (PGRST204)
                    if "ocean_traits" in str(e) and "column" in str(e):
                        print(f"⚠️ Warning: 'ocean_traits' column missing in Supabase. Retrying update without it.")
                        if "ocean_traits" in db_update:
                            del db_update["ocean_traits"]
                        
                        try:
                            self.db.client.supabase.table('robots').update(db_update).eq('client_id', server.client_id).execute()
                            print(f"📝 Updated DB for {display_name} (without ocean_traits): {db_update}")
                        except Exception as retry_e:
                            print(f"❌ Error updating DB (retry failed): {retry_e}")
                    else:
                        print(f"❌ Error updating DB: {e}")

            # === Update client_manager (now carries edge_tts_config) ===
            self.client_manager.update_client_config(server.client_id, update_data)

            # === Broadcast full config to client ===
            if self.socketio:
                client_update = update_data.copy()                     # now contains edge_tts_config
                self.socketio.emit('config_updated', client_update, room=server.client_id)
                print(f"📢 Broadcasted config update (incl. voice) to {display_name}")

            return jsonify({
                "message": f"Client {display_name} updated successfully",
                "updated_fields": list(update_data.keys()),
                "character": update_data.get('character'),
                "voice": update_data.get('edge_tts_config', {}).get('voice')
            }), 200

        except Exception as e:
            print(f"❌ Update error for {display_name}: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": "Update failed", "details": str(e)}), 500