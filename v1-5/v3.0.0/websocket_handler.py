# websocket_handler.py - WebSocket Event Handlers
import time
from flask_socketio import emit, join_room
from flask import request

class WebSocketHandler:
    """WebSocket event handler for real-time communication"""
    
    def __init__(self, socketio, emotion_processor, gpt_client, config=None):
        self.socketio = socketio
        self.emotion_processor = emotion_processor
        self.gpt_client = gpt_client
        self.config = config or {}
        
        # Rate limiting
        self.connection_timestamps = {}
        self.message_counts = {}
        
        # Streaming variables
        self.latest_frame = None
        self.frame_buffer = []
        self.frame_lock = None
        
        # Configuration
        self.stream_fps = self.config.get('stream_fps', 30)
        
        # Setup event handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection with enhanced config"""
            print(f"Client connected: {request.sid}")
            try:
                emit('connected', {
                    'status': 'Connected to real-time emotion server',
                    'sid': request.sid,
                    'server_ip': self.config.get('server_ip', 'unknown'),
                    'config': {
                        'max_fps': self.stream_fps,
                        'emotion_interval': self.config.get('emotion_processing_interval', 0.1),
                        'emotion_window_size': self.config.get('emotion_window_size', 5),
                        'confidence_threshold': self.config.get('confidence_threshold', 30.0)
                    },
                    'components_status': self.emotion_processor.get_status()
                })
            except Exception as e:
                print(f"Error sending connect confirmation: {e}")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"Client disconnected: {request.sid}")
            if request.sid in self.connection_timestamps:
                del self.connection_timestamps[request.sid]
            if request.sid in self.message_counts:
                del self.message_counts[request.sid]

        @self.socketio.on('emotion_frame')
        def handle_emotion_frame(data):
            """Enhanced real-time emotion detection with frequent updates"""
            try:
                if not self.rate_limit_check(request.sid):
                    emit('error', {'message': 'Rate limit exceeded'})
                    return

                frame_b64 = data.get('frame')
                if not frame_b64:
                    emit('error', {'message': 'No frame data provided'})
                    return

                frame = self.emotion_processor.decode_frame_optimized(frame_b64)
                if frame is None:
                    emit('error', {'message': 'Invalid frame data'})
                    return

                emotion, confidence, status = self.emotion_processor.process_emotion_detection_realtime(frame)

                if status in ["success", "throttled"] or time.time() - self.emotion_processor.last_emotion_update > 0.5:
                    distribution = self.emotion_processor.get_emotion_distribution()

                    emit('emotion_result', {
                        'emotion': emotion,
                        'confidence': round(confidence, 1),
                        'status': status,
                        'timestamp': time.time(),
                        'distribution': distribution,
                        'components_status': self.emotion_processor.get_status()
                    })

            except Exception as e:
                print(f"WebSocket emotion error: {e}")
                try:
                    emit('error', {'message': str(e)})
                except:
                    pass

        @self.socketio.on('stream_frame')
        def handle_stream_frame(data):
            """Enhanced frame streaming with face detection visualization"""
            try:
                if not self.rate_limit_check(request.sid):
                    return

                frame_b64 = data.get('frame')
                if not frame_b64:
                    return

                frame = self.emotion_processor.decode_frame_optimized(frame_b64)
                if frame is None:
                    return

                # Add face detection overlay
                frame = self.emotion_processor.add_face_overlay(frame)

                # Store latest frame
                self.latest_frame = frame
                
                # Broadcast emotion data
                if self.emotion_processor.frame_counter % 2 == 0:
                    emotion, confidence = self.emotion_processor.get_current_emotion()
                    self.socketio.emit('live_frame_update', {
                        'emotion': emotion,
                        'confidence': confidence,
                        'timestamp': time.time(),
                        'distribution': self.emotion_processor.get_emotion_distribution()
                    }, room='stream_viewers', skip_sid=request.sid)

            except Exception as e:
                print(f"WebSocket stream error: {e}")

        @self.socketio.on('join_stream')
        def handle_join_stream():
            """Allow clients to join the stream viewers room"""
            join_room('stream_viewers')
            emit('joined_stream', {'status': 'Joined stream viewers'})

        @self.socketio.on('chat_message')
        def handle_chat_message(data):
            """Broadcast chat messages to all monitoring clients"""
            self.socketio.emit('chat_message', data, room='stream_viewers')
    
    def rate_limit_check(self, sid):
        """Enhanced rate limiting per connection"""
        current_time = time.time()

        if sid not in self.connection_timestamps:
            self.connection_timestamps[sid] = current_time
            self.message_counts[sid] = 1
            return True

        if current_time - self.connection_timestamps[sid] < 1:
            self.message_counts[sid] += 1
            if self.message_counts[sid] > 20:
                return False
        else:
            self.connection_timestamps[sid] = current_time
            self.message_counts[sid] = 1

        return True
    
    def get_latest_frame(self):
        """Get the latest frame for streaming"""
        return self.latest_frame
    
    def broadcast_chat_message(self, message_data):
        """Broadcast chat message to all viewers"""
        self.socketio.emit('chat_message', message_data, room='stream_viewers')