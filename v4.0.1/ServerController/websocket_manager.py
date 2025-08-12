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
    
    def __init__(self, socketio, client_manager: ClientManager, request_router: RequestRouter, database: Optional[Database] = None):
        self.socketio = socketio
        self.client_manager = client_manager
        self.request_router = request_router
        self.database = database
        
        # Track monitor connections
        self.monitor_connections = {}
        
        # Setup WebSocket event handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup WebSocket event handlers"""

        # Monitor namespace handlers
        @self.socketio.on('connect', namespace='/monitor')
        def handle_monitor_connect():
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
        def handle_connect(auth):
            return True

        @self.socketio.on('client_init')
        def handle_client_init(data):
            try:
                success, message, client_info = self.client_manager.process_client_init(data)

                if success:
                    session['client_id'] = client_info.client_id
                    
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
                emit('error', {'message': f'Client {client_id} not found'})
                return

            try:
                # Process the frame
                result = self.request_router.handle_image_frame_processing(client_id, data)

                if 'error' in result:
                    emit('error', {'message': result['error'], 'details': result.get('details', '')})
                else:
                    # Send result back to the robot client
                    emit('frame_result', {
                        'client_id': client_id,
                        'result': result,
                        'timestamp': time.time()
                    })

            except Exception as e:
                emit('error', {'message': 'Frame processing failed', 'details': str(e)})

        @self.socketio.on('join_stream')
        def handle_join_stream():
            client_id = session.get('client_id')
            emit('stream_joined', {
                'message': 'Joined live stream',
                'timestamp': time.time()
            })
        
        @self.socketio.on('ping')
        def handle_ping(data):
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
                
                emit('status_response', {
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'enabled_modules': list(client_info.modules),
                    'server_active': server_active,
                    'last_activity': client_info.last_activity,
                    'registration_time': client_info.registration_time,
                    'server_status': server.get_health_status() if server else None,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                emit('status_response', {
                    'error': f'Status check failed: {e}',
                    'timestamp': time.time()
                })
        
        @self.socketio.on('chat_message')
        def handle_chat_message(data):
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
                
                # Process chat message
                result = server.process_chat_message(message)

                # persist chat
                uid = self.client_manager.get_user_id(client_id)
                if uid:
                    try:
                        self.db.insert_chat_log(uid, message, result.get('response', ''))
                    except Exception as db_err:
                        print(f"⚠️  Chat log insert failed: {db_err}")

                # Send to robot client
                emit('chat_response', {
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'response': result.get('response', ''),
                    'detected_emotion': result.get('detected_emotion'),
                    'confidence': result.get('confidence'),
                    'timestamp': time.time()
                })
                
                # Broadcast to monitors
                # User message
                self.socketio.emit('client_chat_message', {
                    'type': 'user',
                    'content': message,
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'emotion': result.get('detected_emotion'),
                    'timestamp': time.time()
                }, room=client_id, namespace='/monitor')
                
                # Bot response
                self.socketio.emit('client_chat_message', {
                    'type': 'bot',
                    'content': result.get('response', ''),
                    'client_id': client_id,
                    'robot_name': client_info.robot_name,
                    'timestamp': time.time()
                }, room=client_id, namespace='/monitor')
                
            except Exception as e:
                client_info = self.client_manager.get_client_info(client_id)
                emit('chat_response', {
                    'error': 'Chat processing failed',
                    'details': str(e),
                    'timestamp': time.time()
                })
    
    def broadcast_frame_update_to_monitors(self, client_id: str, update_data: Dict[str, Any]):
        """Broadcast frame update specifically to monitors"""
        try:
            self.socketio.emit('client_frame_update', update_data, room=client_id, namespace='/monitor')
        except Exception:
            pass
    
    def broadcast_chat_to_monitors(self, client_id: str, message_type: str, content: str, extra_data: Dict[str, Any] = None):
        """Broadcast chat message specifically to monitors"""
        try:
            client_info = self.client_manager.get_client_info(client_id)
            robot_name = client_info.robot_name if client_info else 'Unknown'
            
            chat_data = {
                'type': message_type,
                'content': content,
                'client_id': client_id,
                'robot_name': robot_name,
                'timestamp': time.time()
            }
            
            if extra_data:
                chat_data.update(extra_data)
            
            self.socketio.emit('client_chat_message', chat_data, room=client_id, namespace='/monitor')
            
        except Exception:
            pass
    
    def broadcast_to_client(self, client_id: str, event: str, data: Dict[str, Any]):
        """Broadcast message to specific client if connected"""
        try:
            self.socketio.emit(event, data, room=client_id)
        except Exception:
            pass
    
    def broadcast_to_all_clients(self, event: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        try:
            self.socketio.emit(event, data, broadcast=True)
        except Exception:
            pass
    
    def get_connected_clients_count(self) -> int:
        """Get number of connected WebSocket clients"""
        try:
            return len(self.client_manager.client_servers)
        except Exception:
            return 0
    
    def get_monitor_connections_count(self) -> int:
        """Get number of connected monitor clients"""
        return len(self.monitor_connections)
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get detailed monitor connection status"""
        return {
            'total_monitors': len(self.monitor_connections),
            'monitored_clients': list(set(self.monitor_connections.values())),
            'connections': {sid: client_id for sid, client_id in self.monitor_connections.items()}
        }