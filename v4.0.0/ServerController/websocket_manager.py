# websocket_manager.py - Updated for Individual Client Monitoring
import time
import json
import base64
import cv2
import numpy as np
from typing import Dict, Any
from flask import request, session
from flask_socketio import emit, disconnect

from client_manager import ClientManager
from request_router import RequestRouter

class WebSocketManager:
    """
    Manages WebSocket connections and events for real-time communication.
    Handles client initialization via client_init.json and image frame processing.
    Updated to support individual client monitoring.
    """
    
    def __init__(self, socketio, client_manager: ClientManager, request_router: RequestRouter):
        self.socketio = socketio
        self.client_manager = client_manager
        self.request_router = request_router
        
        # Reference to server controller for individual client monitoring
        self.server_controller = None
        
        # Setup WebSocket event handlers
        self.setup_handlers()
    
    def set_server_controller(self, controller):
        """Set reference to server controller for individual client monitoring integration"""
        self.server_controller = controller
        print("🔗 WebSocket manager connected to server controller for individual monitoring")
    
    def setup_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            """Handle WebSocket connection"""
            print(f"🔌 WebSocket connection attempt from {request.remote_addr}")
            
            # Don't require client_id on connect - wait for client_init
            return True
        
        @self.socketio.on('client_init')
        def handle_client_init(data):
            """
            Handle client initialization with client_init.json data
            
            Expected data format:
            {
                "robot_name": "HomeAssistant_Robot",
                "modules": ["gpt", "emotion", "speech"],
                "client_id": "optional_custom_id",
                "config": {"custom_param": "value"}
            }
            """
            try:
                print(f"📋 Received client_init from {request.remote_addr}")
                
                # Process client initialization
                success, message, client_info = self.client_manager.process_client_init(data)
                
                if success:
                    # ✅ FIX: Store client_id in Flask-SocketIO session (persistent across events)
                    session['client_id'] = client_info.client_id
                    print(f"🔍 DEBUG: Stored client_id '{client_info.client_id}' in session")
                    
                    print(f"✅ {client_info.get_display_name()}: WebSocket initialized successfully")
                    
                    # Send success response
                    emit('client_init_response', {
                        'success': True,
                        'message': message,
                        'client_id': client_info.client_id,
                        'robot_name': client_info.robot_name,
                        'enabled_modules': list(client_info.modules),
                        'timestamp': time.time()
                    })
                    
                    # Optionally create server instance immediately for faster first requests
                    try:
                        server = self.client_manager.get_or_create_server_instance(client_info.client_id)
                        if server:
                            print(f"🚀 {client_info.get_display_name()}: Server instance pre-created")
                            print(f"🖥️ Individual monitor available at: /client/{client_info.client_id}/monitor")
                    except Exception as e:
                        print(f"⚠️ {client_info.get_display_name()}: Server pre-creation warning: {e}")
                
                else:
                    print(f"❌ Client initialization failed: {message}")
                    emit('client_init_response', {
                        'success': False,
                        'message': message,
                        'timestamp': time.time()
                    })
                    # Don't disconnect - let client retry
                
            except Exception as e:
                error_msg = f"Client initialization error: {e}"
                print(f"❌ {error_msg}")
                emit('client_init_response', {
                    'success': False,
                    'message': error_msg,
                    'timestamp': time.time()
                })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle WebSocket disconnection"""
            # ✅ FIX: Get client_id from session instead of request
            client_id = session.get('client_id')
            if client_id:
                client_info = self.client_manager.get_client_info(client_id)
                if client_info:
                    print(f"🔌 {client_info.get_display_name()}: WebSocket disconnected")
                    self.client_manager.update_client_activity(client_id)
                else:
                    print(f"🔌 Client '{client_id}': WebSocket disconnected")
            else:
                print(f"🔌 Unknown client WebSocket disconnected")
        
        @self.socketio.on('image_frame')
        def handle_image_frame(data):
            """
            Handle image frame for real-time processing with individual client monitoring
            
            Expected data format:
            {
                "frame": "base64_encoded_image",
                "timestamp": unix_timestamp,
                "metadata": {...}
            }
            """
            # ✅ FIX: Get client_id from session instead of request
            client_id = session.get('client_id')
            print(f"🔍 DEBUG: Retrieved client_id '{client_id}' from session for image_frame")
            
            if not client_id:
                print("❌ DEBUG: No client_id found in session")
                emit('error', {
                    'message': 'Client not initialized. Send client_init first.',
                    'timestamp': time.time()
                })
                return
            
            # Get client info for display name
            client_info = self.client_manager.get_client_info(client_id)
            if not client_info:
                print(f"❌ DEBUG: No client_info found for client_id '{client_id}'")
                emit('error', {
                    'message': f'Client {client_id} not found',
                    'timestamp': time.time()
                })
                return
            
            try:
                print(f"📸 DEBUG: Processing image frame for {client_info.get_display_name()}")
                
                # Decode frame for individual client monitoring
                frame = None
                try:
                    if 'frame' in data:
                        frame_data = base64.b64decode(data['frame'])
                        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                        print(f"📸 DEBUG: Decoded frame shape: {frame.shape if frame is not None else 'None'}")
                except Exception as decode_error:
                    print(f"⚠️ Frame decode warning: {decode_error}")
                
                # Process image frame using request router
                result = self.request_router.handle_image_frame_processing(client_id, data)
                
                if 'error' in result:
                    emit('error', {
                        'message': result['error'],
                        'details': result.get('details', ''),
                        'timestamp': time.time()
                    })
                else:
                    # Extract emotion and confidence for individual monitoring
                    emotion = result.get('emotion', 'neutral')
                    confidence = result.get('confidence', 0.0)
                    
                    print(f"🎭 DEBUG: {client_info.get_display_name()}: {emotion} ({confidence:.1f}%)")
                    
                    # Update individual client's monitoring data via their RobotServer instance
                    server_instance = self.client_manager.get_client_server(client_id)
                    if server_instance and frame is not None:
                        # Update the individual client's monitoring data
                        if hasattr(server_instance, 'latest_frame'):
                            server_instance.latest_frame = frame.copy()
                        if hasattr(server_instance, 'latest_emotion'):
                            server_instance.latest_emotion = emotion
                        if hasattr(server_instance, 'latest_confidence'):
                            server_instance.latest_confidence = confidence
                        if hasattr(server_instance, 'last_update_time'):
                            server_instance.last_update_time = time.time()
                        
                        print(f"📊 DEBUG: Updated individual monitoring data for {client_info.get_display_name()}")
                    
                    # Send successful result back to client
                    emit('frame_result', {
                        'client_id': client_id,
                        'robot_name': result.get('robot_name', 'Unknown'),
                        'result': result,
                        'timestamp': time.time()
                    })
                
            except Exception as e:
                print(f"❌ {client_info.get_display_name()}: Image frame error: {e}")
                emit('error', {
                    'message': 'Frame processing failed',
                    'details': str(e),
                    'timestamp': time.time()
                })
        
        @self.socketio.on('ping')
        def handle_ping(data):
            """Handle ping for keeping connection alive"""
            # ✅ FIX: Get client_id from session instead of request
            client_id = session.get('client_id')
            
            if client_id:
                self.client_manager.update_client_activity(client_id)
                client_info = self.client_manager.get_client_info(client_id)
                robot_name = client_info.robot_name if client_info else 'Unknown'
            else:
                robot_name = 'Unknown'
            
            emit('pong', {
                'timestamp': time.time(),
                'client_id': client_id,
                'robot_name': robot_name
            })
        
        @self.socketio.on('get_status')
        def handle_get_status():
            """Handle status request via WebSocket"""
            # ✅ FIX: Get client_id from session instead of request
            client_id = session.get('client_id')
            
            if not client_id:
                emit('status_response', {
                    'error': 'Client not initialized',
                    'timestamp': time.time()
                })
                return
            
            try:
                client_info = self.client_manager.get_client_info(client_id)
                if not client_info:
                    emit('status_response', {
                        'error': f'Client {client_id} not found',
                        'timestamp': time.time()
                    })
                    return
                
                server = self.client_manager.get_client_server(client_id)
                server_active = server is not None
                
                # Include individual monitoring URLs in status
                status_response = {
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'enabled_modules': list(client_info.modules),
                    'server_active': server_active,
                    'last_activity': client_info.last_activity,
                    'registration_time': client_info.registration_time,
                    'server_status': server.get_health_status() if server else None,
                    'individual_monitor': f'/client/{client_id}/monitor',
                    'live_stream': f'/client/{client_id}/live_stream',
                    'timestamp': time.time()
                }
                
                # Add current emotion state if available
                if server and hasattr(server, 'latest_emotion'):
                    status_response['current_emotion'] = {
                        'emotion': getattr(server, 'latest_emotion', 'neutral'),
                        'confidence': getattr(server, 'latest_confidence', 0.0),
                        'last_update': getattr(server, 'last_update_time', 0)
                    }
                
                emit('status_response', status_response)
                
            except Exception as e:
                emit('status_response', {
                    'error': f'Status check failed: {e}',
                    'timestamp': time.time()
                })
        
        @self.socketio.on('chat_message')
        def handle_chat_message(data):
            """Handle chat message via WebSocket (alternative to HTTP)"""
            # ✅ FIX: Get client_id from session instead of request
            client_id = session.get('client_id')
            
            if not client_id:
                emit('chat_response', {
                    'error': 'Client not initialized',
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
                
                # Get client info
                client_info = self.client_manager.get_client_info(client_id)
                if not client_info:
                    emit('chat_response', {
                        'error': f'Client {client_id} not found',
                        'timestamp': time.time()
                    })
                    return
                
                # Check if GPT module is enabled
                if 'gpt' not in client_info.modules:
                    emit('chat_response', {
                        'error': 'GPT module not enabled for this client',
                        'enabled_modules': list(client_info.modules),
                        'timestamp': time.time()
                    })
                    return
                
                # Get server instance
                server = self.client_manager.get_or_create_server_instance(client_id)
                if not server:
                    emit('chat_response', {
                        'error': f'Failed to create server instance for {client_info.get_display_name()}',
                        'timestamp': time.time()
                    })
                    return
                
                print(f"💬 {client_info.get_display_name()}: WebSocket chat: '{message}'")
                
                # Process chat message
                result = server.process_chat_message(message)
                
                print(f"🤖 {client_info.get_display_name()}: WebSocket response: '{result.get('response', 'No response')}'")
                
                emit('chat_response', {
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'response': result.get('response', ''),
                    'detected_emotion': result.get('detected_emotion'),
                    'confidence': result.get('confidence'),
                    'timestamp': time.time()
                })
                
            except Exception as e:
                client_info = self.client_manager.get_client_info(client_id)
                display_name = client_info.get_display_name() if client_info else client_id
                print(f"❌ {display_name}: WebSocket chat error: {e}")
                emit('chat_response', {
                    'error': 'Chat processing failed',
                    'details': str(e),
                    'timestamp': time.time()
                })
        
        @self.socketio.on('get_individual_monitor_info')
        def handle_get_individual_monitor_info():
            """Get information needed for individual client monitoring"""
            client_id = session.get('client_id')
            
            if not client_id:
                emit('monitor_info_response', {
                    'error': 'Client not initialized',
                    'timestamp': time.time()
                })
                return
            
            try:
                client_info = self.client_manager.get_client_info(client_id)
                server_instance = self.client_manager.get_client_server(client_id)
                
                if not client_info:
                    emit('monitor_info_response', {
                        'error': f'Client {client_id} not found',
                        'timestamp': time.time()
                    })
                    return
                
                monitor_info = {
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'enabled_modules': list(client_info.modules),
                    'monitor_url': f'/client/{client_id}/monitor',
                    'stream_url': f'/client/{client_id}/live_stream',
                    'has_web_interface': server_instance and hasattr(server_instance, 'web_interface') and server_instance.web_interface is not None,
                    'timestamp': time.time()
                }
                
                # Add current state if available
                if server_instance:
                    if hasattr(server_instance, 'latest_emotion'):
                        monitor_info['current_emotion'] = server_instance.latest_emotion
                    if hasattr(server_instance, 'latest_confidence'):
                        monitor_info['current_confidence'] = server_instance.latest_confidence
                    if hasattr(server_instance, 'last_update_time'):
                        monitor_info['last_update'] = server_instance.last_update_time
                
                emit('monitor_info_response', monitor_info)
                
            except Exception as e:
                emit('monitor_info_response', {
                    'error': f'Failed to get monitor info: {e}',
                    'timestamp': time.time()
                })
    
    def broadcast_to_client(self, client_id: str, event: str, data: Dict[str, Any]):
        """Broadcast message to specific client if connected"""
        try:
            # Note: This would require tracking WebSocket sessions by client_id
            # For now, we'll use room-based approach
            self.socketio.emit(event, data, room=client_id)
        except Exception as e:
            print(f"❌ Failed to broadcast to client {client_id}: {e}")
    
    def broadcast_to_all_clients(self, event: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        try:
            self.socketio.emit(event, data, broadcast=True)
        except Exception as e:
            print(f"❌ Failed to broadcast to all clients: {e}")
    
    def get_connected_clients_count(self) -> int:
        """Get number of connected WebSocket clients"""
        try:
            # This would require tracking active sessions
            # For now, return estimated count
            return len(self.client_manager.client_servers)
        except Exception:
            return 0
    
    def notify_individual_monitor_update(self, client_id: str, emotion: str, confidence: float):
        """Notify individual client monitor of emotion updates"""
        try:
            if self.server_controller:
                # This could be used for real-time monitor updates if needed
                pass
        except Exception as e:
            print(f"❌ Failed to notify monitor update for client {client_id}: {e}")