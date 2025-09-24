# request_router.py - Request Routing and Processing Logic
import time
import json
import re
import threading
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
            
            # 1. Process the user message. The RobotServer will decide whether to delegate or not.
            # We are NOT passing the is_delegated_command flag, so it defaults to False (Delegation Mode).
            result = server.process_chat_message(message)
            
            llm_response_text = result.get('response', '')
            user_facing_response = llm_response_text
            
            # 2. Check if the response contains a delegation command.
            match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response_text, re.DOTALL)
            if match:
                command_data = json.loads(match.group(1))
                target_id = command_data.get("target_robot_id")
                task_message = command_data.get("task")
                
                if target_id and task_message:

                    # --- This is the new, simpler background task ---
                    def execute_delegated_command(target_server, task_msg):
                        try:
                            # Call process_chat_message, but this time, set the flag to True!
                            result = target_server.process_chat_message(task_msg, is_delegated_command=True)
                            
                            # Send the final execution response back to the client
                            response_content = result.get('response', 'Task processed.')
                            self.socketio.emit('chat_response', 
                                               {'response': response_content}, 
                                               room=target_server.client_id)
                        except Exception as e:
                            print(f"❌ Error in background task for '{target_server.client_id}': {e}")
                    
                    # 1. Define a function to handle the broadcast in the background
                    def broadcast_command_async(command, source_name, target_name):
                        # Add a small delay to give the first robot time to speak
                        time.sleep(0.5) 
                        print(f"📣 Broadcasting command to all clients: '{command}'")
                        self.socketio.emit('chat_response', {
                            'response': f"[{source_name} -> {target_name}] {command}"
                        })

                    # 2. Start the broadcast in a new thread so it doesn't block the main response
                    broadcast_thread = threading.Thread(
                        target=broadcast_command_async, 
                        args=(task_message, server.robot_name, target_id)
                    )
                    broadcast_thread.start()
                    
                    # 3. The existing logic to execute the command is still correct
                    target_server_instance = self.client_manager.get_client_server(target_id)
                    if target_server_instance:
                        execute_thread = threading.Thread(
                            target=execute_delegated_command, 
                            args=(target_server_instance, task_message)
                        )
                        execute_thread.start()
                    
                    # Clean the JSON out of the response for the original user
                    user_facing_response = llm_response_text.replace(match.group(0), "").strip()
                
            result['response'] = user_facing_response
            
            if self.socketio:
                try:
                    self.socketio.emit('client_chat_message', {
                        'type': 'user',
                        'content': message,
                        'client_id': server.client_id,
                    }, room=server.client_id, namespace='/monitor')
                    
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
    
    def generate_role_based_prompt(self, current_robot_id, user_message, rag_context: str = ""):
        """
        Generates a context-rich prompt for the LLM to decide delegation based on robot roles.
        Now includes RAG context if available.
        """
        try:
            my_robot_info = MOCK_ROBOT_REGISTRY[current_robot_id]
        except KeyError:
            my_robot_info = {"role": "You are a helpful assistant."}

        network_robots_overview = []
        for robot_id, info in MOCK_ROBOT_REGISTRY.items():
            network_robots_overview.append({
                "robot_id": robot_id,
                "role_description": info["role"]
            })

        formatted_robot_list = json.dumps(network_robots_overview, indent=2)

        # --- NEW: Dynamically create a RAG context block if context is provided ---
        rag_block = ""
        if rag_context:
            rag_block = f"""
    --- START RELEVANT CONTEXT ---
    Here is some relevant information from past conversations to help you understand the user's request:
    {rag_context}
    --- END RELEVANT CONTEXT ---
    """

        system_prompt = f"""You are an AI controller for a specific robot.
    Your assigned identity is:
    - Your Robot ID: "{current_robot_id}"
    - Your Role: "{my_robot_info['role']}"

    You are part of a team of robots. Here is the current list of all robots on the network:
    --- START ROBOT TEAM ---
    {formatted_robot_list}
    --- END ROBOT TEAM ---

    {rag_block} 
    Your primary task is to analyze the user's request and decide if YOU should perform the task based on YOUR role, or if you should delegate it to a teammate.

    1.  **Analyze Request & YOUR Role:** First, look at the user's request. Then, look at YOUR role description above. Can YOU fulfill this request?
    2.  **Decide:**
        * If the task fits YOUR role (e.g., you are a mobile robot and the task is to move), then execute it yourself and respond directly to the user.
        * If the task is better suited for ANOTHER robot, you MUST delegate it. Do NOT delegate tasks to yourself.

    **Delegation Rules:**
    When you delegate, your response MUST contain two parts:
    1. A friendly message for the user explaining who will handle the task.
    2. A JSON command block for the target robot, enclosed in ```json ... ```.
    - The JSON MUST have a "target_robot_id" key.
    - It MUST have a "task" key containing a **natural language command as if you were speaking directly to the other robot.** # <-- CHANGE THIS LINE

    **Example Delegation:**
    User Request: "Can you bring me a coffee from the kitchen?"
    Your Correct Response (because your role is a stationary assistant):
    I can't get that for you myself, but I will ask Silbot, our mobile robot, to bring you a coffee.
    ```json
    {{
    "target_robot_id": "silbot_01",
    "task": "Silbot, please bring a coffee from the kitchen for the user." # <-- CHANGE THIS LINE
    }}
    """
        final_prompt = f"{system_prompt}\n\nNow, please process this user request:\nUser: {user_message}"
        return final_prompt