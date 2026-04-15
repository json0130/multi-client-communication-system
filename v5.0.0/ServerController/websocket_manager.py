# websocket_manager.py - Production Version
import time
import json
from typing import Dict, Any, Optional
from flask import request, session
from flask_socketio import emit, disconnect, join_room, leave_room

from client_manager import ClientManager
from request_router import RequestRouter
from database import Database

class WebSocketManager:
    """
    Manages WebSocket connections and events for real-time communication.
    Handles client initialization via client_init.json and image frame processing.
    """
    
    def __init__(self, socketio, client_manager: ClientManager, request_router: RequestRouter, database: Optional[Database] = None,  on_connect_callback=None, on_disconnect_callback=None):
        self.socketio = socketio
        self.client_manager = client_manager
        self.request_router = request_router
        self.database = database

        # Store the callbacks
        self.on_connect_callback = on_connect_callback
        self.on_disconnect_callback = on_disconnect_callback
        
        # Track monitor connections
        self.monitor_connections = {}
        
        # Setup WebSocket event handlers
        self.setup_handlers()

    
    def setup_handlers(self):
        """Setup WebSocket event handlers"""

        # Monitor namespace handlers
        @self.socketio.on('connect', namespace='/monitor')
        def handle_monitor_connect(auth=None): # Fixed signature
            return True

        @self.socketio.on('disconnect', namespace='/monitor')
        def handle_monitor_disconnect():
            if request.sid in self.monitor_connections:
                del self.monitor_connections[request.sid]

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
            self.monitor_connections[request.sid] = client_id
            
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

        # Robot client handlers (global namespace)
        @self.socketio.on('connect')
        def handle_connect(auth=None): # Fixed signature - Prevents AssertionError write() before start_response
            return True

        @self.socketio.on('client_init')
        def handle_client_init(data):
            try:
                success, message, client_info = self.client_manager.process_client_init(data)

                if success:
                    session['client_id'] = client_info.client_id
                    client_id = client_info.client_id

                    join_room(client_id)
                    print(f"✅ Client '{client_id}' has joined its personal room.")

                    if self.on_connect_callback:
                        self.on_connect_callback(client_id)
                    
                    # Send success response
                    emit('client_init_response', {
                        'success': True,
                        'message': message,
                        'client_id': client_info.client_id,
                        'robot_name': client_info.robot_name,
                        'enabled_modules': list(client_info.modules)
                    })
                    
                    # Pre-create server instance
                    server = self.client_manager.get_or_create_server_instance(client_info.client_id)
                    
                    # Notify monitors
                    self.socketio.emit('client_connected', {
                        'client_id': client_info.client_id,
                        'robot_name': client_info.robot_name,
                        'enabled_modules': list(client_info.modules),
                        'timestamp': time.time()
                    }, room=client_info.client_id, namespace='/monitor')
                    
                else:
                    emit('client_init_response', {'success': False, 'message': message})

            except Exception as e:
                error_msg = f"Client initialization error: {e}"
                emit('client_init_response', {'success': False, 'message': error_msg})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = session.get('client_id')
            if client_id:
                client_info = self.client_manager.get_client_info(client_id)
                self.client_manager.update_client_activity(client_id)

                if self.on_disconnect_callback:
                    self.on_disconnect_callback(client_id)
                
                # Notify monitors
                self.socketio.emit('client_disconnected', {
                    'client_id': client_id,
                    'timestamp': time.time()
                }, room=client_id, namespace='/monitor')

        @self.socketio.on('image_frame')
        def handle_image_frame(data):
            client_id = session.get('client_id')
            if not client_id:
                emit('error', {'message': 'Client not initialized. Send client_init first.'})
                return

            client_info = self.client_manager.get_client_info(client_id)
            if not client_info:
                emit('error', {'message': f'Client {client_id} not found.'})
                return
            
            try:
                if hasattr(self, 'request_router') and hasattr(self.request_router, 'handle_image_frame_processing'):
                    result = self.request_router.handle_image_frame_processing(client_id, data)
                    emit('emotion_update', result, room=client_id)
            except Exception as e:
                 emit('error', {'message': f'Error processing frame: {e}'})