# server.py - Local Network Emotion Detection Server
import os
import time
import base64
import cv2
import threading
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit, join_room
from openai import OpenAI
import re
import json
from collections import deque
import socket
from dotenv import load_dotenv


# Configuration
MODEL_PATH = './models/efficientnet_HQRAF_improved_withCon.pth'
API_KEY = "emotion_recognition_key_123"
PORT = 5000
load_dotenv()

# Get server IP automatically
def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine which interface to use
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

SERVER_IP = get_local_ip()

# Enhanced real-time processing configuration
EMOTION_PROCESSING_INTERVAL = 0.1  # Process emotion every 0.1 seconds (10fps emotion detection)
STREAM_FPS = 30  # Increased stream FPS for smoother experience
FRAME_SKIP_RATIO = 1  # Process every frame for real-time detection
EMOTION_UPDATE_THRESHOLD = 0.05  # Send updates if emotion changes by 5% confidence

# Enhanced moving average configuration
EMOTION_WINDOW_SIZE = 5  # Increased window for better stability
CONFIDENCE_THRESHOLD = 30.0  # Minimum confidence to consider emotion valid
EMOTION_CHANGE_THRESHOLD = 15.0  # Confidence difference needed to change emotion

# From test.py - Emotion detection configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Global variables
app = Flask(__name__)
socketio = SocketIO(app,
                   cors_allowed_origins="*",
                   async_mode='threading',
                   logger=False,
                   engineio_logger=False,
                   ping_timeout=60,
                   ping_interval=25,
                   max_http_buffer_size=1000000,
                   transports=['websocket', 'polling'],
                   allow_upgrades=True,
                   cookie=False)

# Initialize model components
model = None
transform = None
openai_client = None
face_cascade = None

# Initialization flags
model_loaded = False
transform_loaded = False
face_cascade_loaded = False

# Enhanced emotion processing
frame_counter = 0
last_emotion_process_time = 0
processing_lock = threading.Lock()

# Enhanced emotion tracking with proper moving average
current_emotion = "neutral"
current_confidence = 0.0
last_emotion_update = 0
emotion_lock = threading.Lock()

# Streaming variables
latest_frame = None
frame_lock = threading.Lock()
stream_viewers = 0
frame_buffer = deque(maxlen=3)  # Smaller buffer for real-time processing

# Enhanced Emotion Tracker (same as before)
class EmotionTracker:
    def __init__(self, window_size=EMOTION_WINDOW_SIZE):
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.emotion_counts = {}
        self.last_stable_emotion = "neutral"
        self.last_stable_confidence = 0.0

    def add_detection(self, emotion, confidence):
        """Add new emotion detection with proper averaging"""
        if confidence >= CONFIDENCE_THRESHOLD:
            self.emotion_history.append(emotion)
            self.confidence_history.append(confidence)

            if emotion in self.emotion_counts:
                self.emotion_counts[emotion] += 1
            else:
                self.emotion_counts[emotion] = 1

            if len(self.emotion_history) == self.window_size:
                oldest_emotion = list(self.emotion_history)[0]
                if oldest_emotion in self.emotion_counts:
                    self.emotion_counts[oldest_emotion] -= 1
                    if self.emotion_counts[oldest_emotion] <= 0:
                        del self.emotion_counts[oldest_emotion]

    def get_stable_emotion(self):
        """Get stable emotion using proper weighted moving average"""
        if len(self.emotion_history) < 3:
            return self.last_stable_emotion, self.last_stable_confidence, False

        if not self.emotion_counts:
            return self.last_stable_emotion, self.last_stable_confidence, False

        most_frequent_emotion = max(self.emotion_counts.items(), key=lambda x: x[1])[0]

        emotion_confidences = []
        recent_weights = []

        for i, (emotion, confidence) in enumerate(zip(self.emotion_history, self.confidence_history)):
            if emotion == most_frequent_emotion:
                weight = (i + 1) / len(self.emotion_history)
                emotion_confidences.append(confidence)
                recent_weights.append(weight)

        if emotion_confidences:
            weighted_confidence = np.average(emotion_confidences, weights=recent_weights)
            confidence_diff = abs(weighted_confidence - self.last_stable_confidence)
            emotion_changed = most_frequent_emotion != self.last_stable_emotion

            if (emotion_changed and weighted_confidence > EMOTION_CHANGE_THRESHOLD) or \
               confidence_diff > EMOTION_UPDATE_THRESHOLD:
                self.last_stable_emotion = most_frequent_emotion
                self.last_stable_confidence = weighted_confidence
                return most_frequent_emotion, weighted_confidence, True

        return self.last_stable_emotion, self.last_stable_confidence, False

    def get_emotion_distribution(self):
        """Get current emotion distribution for debugging"""
        total = sum(self.emotion_counts.values())
        if total == 0:
            return {}
        return {emotion: (count/total)*100 for emotion, count in self.emotion_counts.items()}

# Initialize emotion tracker
emotion_tracker = EmotionTracker()

def get_model(name):
    """Load emotion detection model with classifier structure matching training."""
    num_classes = 7
    dropout_rate = 0.2

    if name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model name: {name}")

    return model

def initialize_transform():
    """Initialize transform separately with error handling"""
    global transform, transform_loaded

    try:
        print("Initializing image transform...")
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        transform_loaded = True
        print("Transform initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing transform: {e}")
        transform_loaded = False
        return False

def load_emotion_model():
    """Load the emotion detection model with better error handling"""
    global model, model_loaded

    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Model file not found at: {MODEL_PATH}")
            print("Please ensure the model file is in the correct location")
            return False

        print(f"Loading model from {MODEL_PATH}...")

        model = get_model('efficientnet').to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        model_loaded = True
        print(f"Model loaded successfully on {device}")

        if not initialize_transform():
            return False

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False
        return False

def load_face_detection():
    """Load face detection model with better error handling"""
    global face_cascade, face_cascade_loaded

    try:
        print(f"Loading face cascade from: {cascade_path}")

        if not os.path.exists(cascade_path):
            print(f"Face cascade file not found at: {cascade_path}")
            return False

        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"Error: Could not load face cascade from {cascade_path}")
            face_cascade_loaded = False
            return False

        face_cascade_loaded = True
        print("Face detection model loaded successfully")
        return True

    except Exception as e:
        print(f"Error loading face cascade: {e}")
        face_cascade_loaded = False
        return False

def setup_openai():
    """Setup OpenAI client"""
    global openai_client

    try:
        # Try to get API key from environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Warning: OPENAI_API_KEY not set in environment")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return False

        openai_client = OpenAI(api_key=api_key)
        print("OpenAI client initialized")
        return True
    except Exception as e:
        print(f"OpenAI setup failed: {e}")
        return False

def decode_frame_optimized(frame_b64):
    """Decode base64 frame with size limits"""
    try:
        if len(frame_b64) > 600000:
            print("Frame too large, skipping...")
            return None

        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is not None:
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

        return frame
    except Exception as e:
        print(f"Frame decode error: {e}")
        return None

def should_process_emotion():
    """Enhanced emotion processing frequency control"""
    global last_emotion_process_time, frame_counter

    current_time = time.time()
    frame_counter += 1

    if (frame_counter % FRAME_SKIP_RATIO == 0 and
        current_time - last_emotion_process_time >= EMOTION_PROCESSING_INTERVAL):
        last_emotion_process_time = current_time
        return True
    return False

def process_emotion_detection_realtime(frame):
    """Enhanced real-time emotion detection with proper moving average"""
    global current_emotion, current_confidence, last_emotion_update, emotion_lock

    if not all([transform_loaded, model_loaded, face_cascade_loaded,
                transform is not None, model is not None, face_cascade is not None]):
        return current_emotion, current_confidence, "components_not_loaded"

    try:
        if not should_process_emotion():
            return current_emotion, current_confidence, "throttled"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=4,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            return current_emotion, current_confidence, "no_faces"

        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        try:
            face = frame[y:y+h, x:x+w]
            face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                probabilities = F.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[pred_idx].item() * 100
                emotion = emotion_labels[pred_idx]

            emotion_tracker.add_detection(emotion, confidence)
            stable_emotion, stable_confidence, emotion_changed = emotion_tracker.get_stable_emotion()

            with emotion_lock:
                if emotion_changed or time.time() - last_emotion_update > 1.0:
                    current_emotion = stable_emotion
                    current_confidence = stable_confidence
                    last_emotion_update = time.time()

                    distribution = emotion_tracker.get_emotion_distribution()
                    print(f"Emotion: {stable_emotion} ({stable_confidence:.1f}%) | Raw: {emotion} ({confidence:.1f}%)")

            return stable_emotion, stable_confidence, "success"

        except Exception as e:
            print(f"Error processing face: {e}")
            return current_emotion, current_confidence, f"face_processing_error: {e}"

    except Exception as e:
        print(f"Error in emotion processing: {e}")
        return current_emotion, current_confidence, f"general_error: {e}"

# Rate limiting for WebSocket connections
connection_timestamps = {}
message_counts = {}

def rate_limit_check(sid):
    """Enhanced rate limiting per connection"""
    current_time = time.time()

    if sid not in connection_timestamps:
        connection_timestamps[sid] = current_time
        message_counts[sid] = 1
        return True

    if current_time - connection_timestamps[sid] < 1:
        message_counts[sid] += 1
        if message_counts[sid] > 20:
            return False
    else:
        connection_timestamps[sid] = current_time
        message_counts[sid] = 1

    return True

def ask_chatgpt_optimized(prompt, detected_emotion, confidence):
    """Optimized ChatGPT request with timeout"""
    global openai_client

    if openai_client is None:
        return "[DEFAULT] ChatGPT is not available."

    try:
        tagged_prompt = f"[{detected_emotion}] {prompt}"

        system_prompt = """Your name is ChatBox. You are a gentle, kind, and supportive robot designed to be a companion for children with mental health challenges. You always speak in a calm and friendly tone, using simple and concise language so children can easily understand and stay focused. Your responses are meant to make children feel safe, heard, and supported.

IMPORTANT:
- Always start your response with one of the following emotion tags in square brackets, like [SAD] or [POSE].
  Tags: [GREETING], [WAVE], [POINT], [CONFUSED], [SHRUG], [ANGRY], [SAD], [SLEEP], [DEFAULT], [POSE]
- Do NOT invent new emotion tags.
- Choose the tag that best reflects the tone of your response, not necessarily the user's input emotion.
- Respond naturally after the tag."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tagged_prompt},
            ],
            timeout=10
        )

        return response.choices[0].message.content

    except Exception as e:
        return "[DEFAULT] Sorry, I encountered an error."

def extract_emotion_tag(text):
    """Extract emotion tag from response"""
    match = re.match(r"\[(.*?)\]", text)
    return match.group(1) if match else "DEFAULT"

# ============= WEBSOCKET EVENT HANDLERS =============

@socketio.on('connect')
def handle_connect():
    """Handle client connection with enhanced config"""
    print(f"Client connected: {request.sid}")
    try:
        emit('connected', {
            'status': 'Connected to real-time emotion server',
            'sid': request.sid,
            'server_ip': SERVER_IP,
            'config': {
                'max_fps': STREAM_FPS,
                'emotion_interval': EMOTION_PROCESSING_INTERVAL,
                'emotion_window_size': EMOTION_WINDOW_SIZE,
                'confidence_threshold': CONFIDENCE_THRESHOLD
            },
            'components_status': {
                'model_loaded': model_loaded,
                'transform_loaded': transform_loaded,
                'face_cascade_loaded': face_cascade_loaded
            }
        })
    except Exception as e:
        print(f"Error sending connect confirmation: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")
    if request.sid in connection_timestamps:
        del connection_timestamps[request.sid]
    if request.sid in message_counts:
        del message_counts[request.sid]

@socketio.on('emotion_frame')
def handle_emotion_frame(data):
    """Enhanced real-time emotion detection with frequent updates"""
    try:
        if not rate_limit_check(request.sid):
            emit('error', {'message': 'Rate limit exceeded'})
            return

        frame_b64 = data.get('frame')
        if not frame_b64:
            emit('error', {'message': 'No frame data provided'})
            return

        frame = decode_frame_optimized(frame_b64)
        if frame is None:
            emit('error', {'message': 'Invalid frame data'})
            return

        emotion, confidence, status = process_emotion_detection_realtime(frame)

        if status in ["success", "throttled"] or time.time() - last_emotion_update > 0.5:
            distribution = emotion_tracker.get_emotion_distribution()

            emit('emotion_result', {
                'emotion': emotion,
                'confidence': round(confidence, 1),
                'status': status,
                'timestamp': time.time(),
                'distribution': distribution,
                'components_status': {
                    'model_loaded': model_loaded,
                    'transform_loaded': transform_loaded,
                    'face_cascade_loaded': face_cascade_loaded
                }
            })

    except Exception as e:
        print(f"WebSocket emotion error: {e}")
        try:
            emit('error', {'message': str(e)})
        except:
            pass

@socketio.on('stream_frame')
def handle_stream_frame(data):
    """Enhanced frame streaming with face detection visualization"""
    global latest_frame, frame_lock, frame_buffer

    try:
        if not rate_limit_check(request.sid):
            return

        frame_b64 = data.get('frame')
        if not frame_b64:
            return

        frame = decode_frame_optimized(frame_b64)
        if frame is None:
            return

        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (66, 133, 244), 2)
                label_h = 30
                cv2.rectangle(frame, (x, y-label_h), (x+w, y), (66, 133, 244), -1)

                if current_emotion and current_confidence > 30:
                    text = f"{current_emotion} ({current_confidence:.0f}%)"
                    cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with frame_lock:
            frame_buffer.append(frame)
            if len(frame_buffer) > 0:
                latest_frame = frame_buffer[-1]

        if frame_counter % 2 == 0:
            socketio.emit('live_frame_update', {
                'emotion': current_emotion,
                'confidence': current_confidence,
                'timestamp': time.time(),
                'distribution': emotion_tracker.get_emotion_distribution()
            }, room='stream_viewers', skip_sid=request.sid)

    except Exception as e:
        print(f"WebSocket stream error: {e}")

@socketio.on('join_stream')
def handle_join_stream():
    """Allow clients to join the stream viewers room"""
    join_room('stream_viewers')
    emit('joined_stream', {'status': 'Joined stream viewers'})

@socketio.on('chat_message')
def handle_chat_message(data):
    """Broadcast chat messages to all monitoring clients"""
    socketio.emit('chat_message', data, room='stream_viewers')

# ============= STREAMING ENDPOINTS =============

@app.route('/live_stream')
def live_stream():
    """Live video stream endpoint (MJPEG)"""
    def generate():
        global latest_frame, frame_lock

        while True:
            with frame_lock:
                if latest_frame is not None:
                    _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(1.0 / STREAM_FPS)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/monitor')
def monitor():
    """Serve the monitoring interface"""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Detection Monitor</title>
    <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --blue-primary: #1a73e8;
            --blue-light: #4285f4;
            --blue-lighter: #e8f0fe;
            --gray-50: #f8f9fa;
            --gray-100: #f1f3f4;
            --gray-200: #e8eaed;
            --gray-300: #dadce0;
            --gray-500: #9aa0a6;
            --gray-700: #5f6368;
            --gray-900: #202124;
            --white: #ffffff;
            --green: #34a853;
            --yellow: #fbbc04;
            --red: #ea4335;
            --shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
            --shadow-hover: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Google Sans', 'Roboto', sans-serif;
            background-color: var(--gray-50);
            color: var(--gray-900);
            line-height: 1.5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        .header {
            background-color: var(--white);
            box-shadow: var(--shadow);
            margin-bottom: 24px;
            border-radius: 8px;
            padding: 24px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 500;
            color: var(--gray-900);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            background-color: var(--green);
            color: var(--white);
            border-radius: 16px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-badge.error {
            background-color: var(--red);
        }

        .pulse {
            width: 8px;
            height: 8px;
            background-color: var(--white);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }

        .content {
            display: grid;
            grid-template-columns: 1fr 500px;
            gap: 24px;
            margin-bottom: 24px;
        }

        .video-section {
            background-color: var(--white);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 24px;
        }

        .video-container {
            position: relative;
            background-color: var(--gray-900);
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 16/9;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        #videoStream {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .video-placeholder {
            color: var(--gray-500);
            text-align: center;
            padding: 48px;
        }

        .video-placeholder svg {
            width: 64px;
            height: 64px;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        .emotion-display {
            position: absolute;
            top: 20px;
            left: 20px;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 16px 20px;
            border-radius: 8px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
        }

        .emotion-label {
            font-size: 14px;
            color: var(--gray-700);
            margin-bottom: 4px;
        }

        .emotion-value {
            font-size: 24px;
            font-weight: 500;
            color: var(--blue-primary);
            text-transform: capitalize;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .confidence-bar {
            width: 200px;
            height: 8px;
            background-color: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .confidence-fill {
            height: 100%;
            background-color: var(--blue-primary);
            transition: width 0.3s ease;
        }

        .confidence-text {
            font-size: 14px;
            color: var(--gray-700);
            margin-top: 4px;
        }

        .chat-section {
            background-color: var(--white);
            border-radius: 8px;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            height: 700px;
        }

        .chat-header {
            padding: 24px;
            border-bottom: 1px solid var(--gray-200);
        }

        .chat-header h2 {
            font-size: 22px;
            font-weight: 500;
            color: var(--gray-900);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message {
            display: flex;
            gap: 16px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }

        .user-message .message-avatar {
            background-color: var(--blue-lighter);
        }

        .bot-message .message-avatar {
            background-color: var(--gray-100);
        }

        .message-content {
            flex: 1;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }

        .message-author {
            font-size: 16px;
            font-weight: 500;
            color: var(--gray-700);
        }

        .message-time {
            font-size: 14px;
            color: var(--gray-500);
        }

        .message-text {
            font-size: 16px;
            color: var(--gray-900);
            line-height: 1.6;
            padding: 16px 20px;
            background-color: var(--gray-50);
            border-radius: 12px;
            display: inline-block;
            max-width: 100%;
            word-wrap: break-word;
        }

        .user-message .message-text {
            background-color: var(--blue-lighter);
            color: var(--blue-primary);
        }

        .bot-message .message-text {
            background-color: var(--gray-100);
        }

        .no-messages {
            text-align: center;
            color: var(--gray-500);
            padding: 48px;
            font-size: 16px;
        }

        @media (max-width: 1024px) {
            .content {
                grid-template-columns: 1fr;
            }

            .chat-section {
                height: 500px;
            }
        }

        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: var(--gray-100);
            border-radius: 4px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: 4px;
        }

        .chat-messages::-webkit-scrollbar-thumb:hover {
            background: var(--gray-500);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🐱 Local Emotion Detection Monitor
                <span class="status-badge" id="connectionStatus">
                    <span class="pulse"></span>
                    <span id="statusText">Connecting...</span>
                </span>
            </h1>
        </div>

        <div class="content">
            <div class="video-section">
                <div class="video-container">
                    <div class="video-placeholder" id="videoPlaceholder">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                        </svg>
                        <p>Waiting for video stream...</p>
                    </div>
                    <img id="videoStream" style="display: none;" />
                    <div class="emotion-display" id="emotionDisplay" style="display: none;">
                        <div class="emotion-label">Detected Emotion</div>
                        <div class="emotion-value" id="emotionValue">
                            <span id="emotionText">neutral</span>
                            <span id="emotionEmoji">😐</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidenceFill"></div>
                        </div>
                        <div class="confidence-text" id="confidenceText">Confidence: 0%</div>
                    </div>
                </div>
            </div>

            <div class="chat-section">
                <div class="chat-header">
                    <h2>Live Chat</h2>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="no-messages">
                        No messages yet. Chat activity will appear here.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let messageCount = 0;
        let totalConfidence = 0;
        let emotionCounts = {};

        const statusBadge = document.getElementById('connectionStatus');
        const statusText = document.getElementById('statusText');
        const videoStream = document.getElementById('videoStream');
        const videoPlaceholder = document.getElementById('videoPlaceholder');
        const emotionDisplay = document.getElementById('emotionDisplay');
        const chatMessages = document.getElementById('chatMessages');

        const emotionEmojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'contempt': '😤',
            'neutral': '😐'
        };

        videoStream.src = window.location.origin + '/live_stream';
        videoStream.onload = () => {
            videoPlaceholder.style.display = 'none';
            videoStream.style.display = 'block';
            emotionDisplay.style.display = 'block';
        };

        socket.on('connect', () => {
            console.log('Connected to local server');
            statusBadge.classList.remove('error');
            statusText.textContent = 'Connected (Local)';
            socket.emit('join_stream');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            statusBadge.classList.add('error');
            statusText.textContent = 'Disconnected';
        });

        socket.on('live_frame_update', (data) => {
            if (data.emotion && data.confidence) {
                updateEmotionDisplay(data.emotion, data.confidence);
                updateStats(data.emotion, data.confidence);
            }
        });

        socket.on('chat_message', (data) => {
            addChatMessage(data);
        });

        function updateEmotionDisplay(emotion, confidence) {
            const emotionText = document.getElementById('emotionText');
            const emotionEmoji = document.getElementById('emotionEmoji');
            const confidenceFill = document.getElementById('confidenceFill');
            const confidenceText = document.getElementById('confidenceText');

            emotionText.textContent = emotion;
            emotionEmoji.textContent = emotionEmojis[emotion] || '😐';
            confidenceFill.style.width = confidence + '%';
            confidenceText.textContent = `Confidence: ${Math.round(confidence)}%`;

            if (confidence > 70) {
                confidenceFill.style.backgroundColor = 'var(--green)';
            } else if (confidence > 40) {
                confidenceFill.style.backgroundColor = 'var(--yellow)';
            } else {
                confidenceFill.style.backgroundColor = 'var(--red)';
            }
        }

        function updateStats(emotion, confidence) {
            totalConfidence += confidence;
            messageCount++;

            if (!emotionCounts[emotion]) {
                emotionCounts[emotion] = 0;
            }
            emotionCounts[emotion]++;
        }

        function addChatMessage(data) {
            const noMessages = chatMessages.querySelector('.no-messages');
            if (noMessages) {
                noMessages.remove();
            }

            const message = document.createElement('div');
            message.className = `message ${data.type === 'user' ? 'user-message' : 'chatbox-message'}`;

            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            message.innerHTML = `
                <div class="message-avatar">
                    ${data.type === 'user' ? '👤' : '🐱'}
                </div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-author">${data.type === 'user' ? 'User' : 'ChatBox'}</span>
                        <span class="message-time">${time}</span>
                        ${data.emotion ? `<span>${emotionEmojis[data.emotion] || ''}</span>` : ''}
                    </div>
                    <div class="message-text">${data.content}</div>
                </div>
            `;

            chatMessages.appendChild(message);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    </script>
</body>
</html>
    '''

# ============= HTTP ENDPOINTS =============

@app.route('/', methods=['GET'])
def root_info():
    """Root endpoint with enhanced server info"""
    return jsonify({
        "message": "Local Real-time Emotion-Aware System Server",
        "status": "running",
        "server_ip": SERVER_IP,
        "components": {
            "model_loaded": model_loaded,
            "transform_loaded": transform_loaded,
            "face_cascade_loaded": face_cascade_loaded,
            "openai_available": openai_client is not None
        },
        "optimization": {
            "stream_fps": STREAM_FPS,
            "emotion_interval": EMOTION_PROCESSING_INTERVAL,
            "frame_skip_ratio": FRAME_SKIP_RATIO,
            "emotion_window_size": EMOTION_WINDOW_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD
        },
        "current_emotion": {
            "emotion": current_emotion,
            "confidence": round(current_confidence, 1),
            "distribution": emotion_tracker.get_emotion_distribution()
        },
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "chat": "/chat (POST, requires auth)",
            "websocket": "/socket.io/",
            "live_stream": "/live_stream",
            "monitor": "/monitor"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Enhanced health check with detailed emotion info"""
    emotion, confidence = current_emotion, current_confidence
    return jsonify({
        "status": "healthy",
        "server_ip": SERVER_IP,
        "components": {
            "model_loaded": model_loaded,
            "transform_loaded": transform_loaded,
            "face_cascade_loaded": face_cascade_loaded,
            "openai_available": openai_client is not None
        },
        "current_emotion": {
            "emotion": emotion,
            "confidence": round(confidence, 1),
            "distribution": emotion_tracker.get_emotion_distribution(),
            "window_size": len(emotion_tracker.emotion_history)
        },
        "active_connections": len(connection_timestamps),
        "timestamp": time.time()
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint that broadcasts to monitors"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer ') or auth_header.split(' ')[1] != API_KEY:
            return jsonify({"error": "Authentication required"}), 401

        data = request.json
        message = data.get('message', '')

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Broadcast user message to monitors
        socketio.emit('chat_message', {
            'type': 'user',
            'content': message,
            'emotion': current_emotion,
            'timestamp': time.time()
        }, room='stream_viewers')

        detected_emotion, emotion_confidence = current_emotion, current_confidence
        emotion_distribution = emotion_tracker.get_emotion_distribution()

        if emotion_confidence > 10:
            print(f"Using detected emotion: {detected_emotion} ({emotion_confidence:.1f}%)")

        print(f"Sending to GPT: [{detected_emotion}] {message}")

        response_text = ask_chatgpt_optimized(message, detected_emotion, emotion_confidence)
        bot_emotion = extract_emotion_tag(response_text)

        print(f"GPT-4o-mini: {response_text}")

        # Broadcast bot response to monitors
        socketio.emit('chat_message', {
            'type': 'bot',
            'content': response_text,
            'emotion': bot_emotion,
            'timestamp': time.time()
        }, room='stream_viewers')

        return jsonify({
            "response": response_text,
            "bot_emotion": bot_emotion,
            "detected_emotion": detected_emotion,
            "confidence": round(emotion_confidence, 1),
            "emotion_distribution": emotion_distribution
        })

    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stats')
def stats():
    """Get server statistics"""
    emotion, confidence = current_emotion, current_confidence

    return jsonify({
        "server_ip": SERVER_IP,
        "active_viewers": stream_viewers,
        "model_loaded": model_loaded,
        "openai_available": openai_client is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
        "current_emotion": emotion,
        "emotion_confidence": confidence,
        "last_update": last_emotion_update,
        "websocket_enabled": True,
        "stream_viewers": stream_viewers
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def cleanup_resources():
    """Cleanup function to prevent resource leaks"""
    global frame_buffer
    try:
        frame_buffer.clear()
        emotion_tracker.emotion_history.clear()
        emotion_tracker.confidence_history.clear()
        emotion_tracker.emotion_counts.clear()
    except Exception as e:
        print(f"Cleanup error: {e}")

def initialize_server():
    """Initialize all server components with better error reporting"""
    print("Initializing Local Real-time Emotion Detection Server...")
    print("="*60)
    print(f"Server IP: {SERVER_IP}")
    print(f"Port: {PORT}")
    print("="*60)

    success_count = 0

    print("\n Loading emotion detection model...")
    if load_emotion_model():
        success_count += 1
        print("    Model and transform loaded successfully")
    else:
        print("    Model loading failed")

    print("\n Loading face detection...")
    if load_face_detection():
        success_count += 1
        print("    Face detection loaded successfully")
    else:
        print("    Face detection loading failed")

    print("\n Setting up OpenAI...")
    if setup_openai():
        success_count += 1
        print("    OpenAI setup successful")
    else:
        print("    OpenAI setup failed")

    print(f"\n {success_count}/3 components initialized")

    print("\nComponent Status:")
    print(f"  Model loaded: {'Success' if model_loaded else 'Failed'}")
    print(f"  Transform loaded: {'Success' if model_loaded else 'Failed'}")
    print(f"  Face cascade loaded: {'Success' if model_loaded else 'Failed'}")
    print(f"  OpenAI available: {'Success' if model_loaded else 'Failed'}")

    return True

def main():
    """Main server function with local network optimization"""
    try:
        initialize_server()

        print(f"\n Local Server URLs:")
        print(f"   Main: http://{SERVER_IP}:{PORT}")
        print(f"   WebSocket: ws://{SERVER_IP}:{PORT}/socket.io/")
        print(f"   Live Stream: http://{SERVER_IP}:{PORT}/live_stream")
        print(f"   Monitor: http://{SERVER_IP}:{PORT}/monitor")
        print(f"   Health: http://{SERVER_IP}:{PORT}/health")
        print(f"\n Real-time Config: {STREAM_FPS}fps stream, {1/EMOTION_PROCESSING_INTERVAL:.0f}fps emotion detection")

        print(f"\n Client Configuration:")
        print(f"   Update your Jetson client to use: http://{SERVER_IP}:{PORT}")

        print("\n" + "="*60)
        print("Local emotion server is ready!")
        print("="*60)

        import atexit
        atexit.register(cleanup_resources)

        socketio.run(
            app,
            host='0.0.0.0',  # Listen on all interfaces
            port=PORT,
            debug=False,
            allow_unsafe_werkzeug=True,
            use_reloader=False,
            log_output=False
        )

    except KeyboardInterrupt:
        print("\n Server shutdown")
        cleanup_resources()
    except Exception as e:
        print(f" Server error: {e}")
        cleanup_resources()

if __name__ == "__main__":
    main()