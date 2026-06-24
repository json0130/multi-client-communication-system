# Install required packages
!pip install flask pyngrok pillow torch torchvision matplotlib opencv-python ipywidgets

# colab_server.py - Server integrating test.py and chatbot.py logic
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
from openai import OpenAI
import re
import json
from collections import deque
import pyngrok.ngrok as ngrok

from google.colab import userdata
from google.colab import drive
drive.mount('/content/drive')


# Configuration - Based on your original files
MODEL_PATH = '/content/drive/Shareddrives/COMPSYS 731 - JScript/model/efficientnet_HQRAF_improved_withCon.pth'
API_KEY = "emotion_recognition_key_123"
PORT = 5000

# From test.py - Emotion detection configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Global variables
app = Flask(__name__)
model = None
transform = None
openai_client = None
face_cascade = None

# Server-side emotion tracking (similar to your original emotion_file concept)
current_emotion = "neutral"
current_confidence = 0.0
last_emotion_update = 0
emotion_lock = threading.Lock()

# Streaming variables
latest_frame = None
frame_lock = threading.Lock()
stream_viewers = 0

def setup_ngrok():
    """Setup ngrok tunnel"""
    try:
        ngrok.set_auth_token("2xTKU9VrlsBCwm95KbrhpycsjK5_828oyzrb2cfBHDsA2bQbZ")
        tunnel = ngrok.connect(PORT)
        public_url = tunnel.public_url  # Get clean URL string
        print(f"🌐 Public URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ ngrok setup failed: {e}")
        return None

# From test.py - Model setup
def get_model(name):
    """Load emotion detection model with classifier structure matching training."""
    num_classes = 7
    dropout_rate = 0.2

    if name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    elif name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    elif name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    elif name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    elif name == 'efficientnetv2':
        model = models.efficientnet_v2_s(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model name: {name}")

    return model

def load_emotion_model():
    """Load the emotion detection model - adapted from test.py"""
    global model, transform

    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = get_model('efficientnet').to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Model loaded successfully on {device}")

        # Setup transform - from test.py
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing without emotion detection")
        return False

def load_face_detection():
    """Load face detection model - from test.py"""
    global face_cascade

    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"Error: Could not load face cascade from {cascade_path}")
            return False
        print("Face detection model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading face cascade: {e}")
        return False

def setup_openai():
    """Setup OpenAI client - from chatbot.py"""
    global openai_client

    try:
        api_key = userdata.get('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set")
            return False

        openai_client = OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
        return True
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
        return False

def decode_frame(frame_b64):
    """Decode base64 frame to OpenCV format"""
    try:
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Frame decode error: {e}")
        return None

def format_confidence(confidence):
    """Format confidence for display - from test.py"""
    return f"{confidence:.1f}%"

def process_emotion_detection(frame):
    """Process emotion detection - adapted from test.py process_emotion function"""
    global current_emotion, current_confidence, last_emotion_update, emotion_lock

    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            # No faces detected
            return "neutral", 0.0, "No faces detected"

        # Process the largest face - from test.py logic
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]

        if model is not None:
            try:
                # Extract face region
                face = frame[y:y+h, x:x+w]

                # Convert to PIL Image for transformation
                face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                # Apply transformations and predict - from test.py
                face_tensor = transform(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(face_tensor)

                    # Get probabilities with softmax
                    probabilities = F.softmax(output, dim=1)[0]

                    # Get prediction and confidence
                    pred_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_idx].item() * 100
                    emotion = emotion_labels[pred_idx]

                    # Get top 3 emotions with confidence - from test.py
                    top3_values, top3_indices = torch.topk(probabilities, 3)
                    top3_emotions = [(emotion_labels[idx.item()], val.item() * 100)
                                   for idx, val in zip(top3_indices, top3_values)]

                # Update server-side emotion state (thread-safe)
                with emotion_lock:
                    if (emotion != current_emotion or abs(confidence - current_confidence) > 5.0):
                        current_emotion = emotion
                        current_confidence = confidence
                        last_emotion_update = time.time()

                        # Terminal display like in test.py
                        print(f"Detected Emotion: {emotion} ({format_confidence(confidence)})")
                        print("Top Emotions:")
                        for i, (em, conf) in enumerate(top3_emotions):
                            print(f"{i+1}. {em}: {format_confidence(conf)}")

                return emotion, confidence, "success"

            except Exception as e:
                print(f"Error processing face: {e}")
                return "neutral", 0.0, f"Processing error: {e}"
        else:
            return "neutral", 0.0, "Model not loaded"

    except Exception as e:
        print(f"Error in emotion processing: {e}")
        return "neutral", 0.0, f"Error: {e}"

def get_current_server_emotion():
    """Get current emotion detected by the server"""
    global current_emotion, current_confidence, emotion_lock

    with emotion_lock:
        return current_emotion, current_confidence

# From chatbot.py - ChatGPT integration
def extract_emotion_tag(text):
    """Extract emotion tag from response - from chatbot.py"""
    match = re.match(r"\[(.*?)\]", text)
    return match.group(1) if match else "DEFAULT"

def ask_chatgpt(prompt, detected_emotion, confidence):
    """Ask ChatGPT with emotion context - adapted from chatbot.py"""
    global openai_client

    if openai_client is None:
        return "[DEFAULT] Sorry, ChatGPT is not available right now."

    try:
        # Use detected emotion in prompt - from chatbot.py logic
        tagged_prompt = f"[{detected_emotion}] {prompt}"

        # System prompt from chatbot.py
        system_prompt = """Your name is ChatBox and You are a helpful assistant that responds to users based on their emotional state and always tags your response with one of the following emotion categories and their meaning:

- [GREETING]
- [WAVE]
- [POINT]
- [CONFUSED]
- [SHRUG]
- [ANGRY]
- [SAD]
- [SLEEP]
- [DEFAULT]
- [POSE]

IMPORTANT:
- Always start your response with one of the above emotion tags in square brackets, like [SAD] or [POSE].
- Do NOT invent new emotion tags.
- Choose the tag that best reflects the tone of your response, not necessarily the user's input emotion.
- Respond naturally after the tag. For example:
  - [GREETING] Hi there! How can I assist you today?
  - [CONFUSED] I'm not quite sure I follow — could you rephrase that?
  - [ANGRY] That doesn't seem fair at all!
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tagged_prompt},
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        return "[DEFAULT] Sorry, I encountered an error while processing your request."

# Authentication middleware
def require_auth():
    """Check API key authentication"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False

    token = auth_header.split(' ')[1]
    return token == API_KEY

@app.before_request
def check_auth():
    """Authenticate requests (except public endpoints)"""
    # Allow access to these endpoints without authentication
    public_endpoints = ['health', 'live_stream', 'root_info']

    if request.endpoint in public_endpoints:
        return

    if not require_auth():
        return jsonify({"error": "Authentication required"}), 401

# API Routes
@app.route('/', methods=['GET'])
def root_info():
    """Root endpoint with server info"""
    return jsonify({
        "message": "Distributed Emotion-Aware System Server",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "live_stream": "/live_stream",
            "stats": "/stats",
            "detect_emotion": "/detect_emotion (POST, requires auth)",
            "chat": "/chat (POST, requires auth)",
            "stream_frame": "/stream_frame (POST, requires auth)"
        },
        "timestamp": time.time()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "openai_available": openai_client is not None,
        "current_emotion": current_emotion,
        "emotion_confidence": current_confidence,
        "timestamp": time.time()
    })

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    """Emotion detection endpoint - processes frame and updates server state"""
    try:
        data = request.json
        frame_b64 = data.get('frame')

        if not frame_b64:
            return jsonify({"error": "No frame provided"}), 400

        # Decode frame
        frame = decode_frame(frame_b64)
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400

        # Process emotion detection
        emotion, confidence, status = process_emotion_detection(frame)

        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "status": status,
            "timestamp": time.time()
        })

    except Exception as e:
        print(f"Emotion detection endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint - uses server's current emotion state"""
    try:
        data = request.json
        message = data.get('message', '')

        if not message:
            return jsonify({"error": "No message provided"}), 400

        # Get current emotion state from server
        detected_emotion, emotion_confidence = get_current_server_emotion()

        # Debug output like in chatbot.py
        if emotion_confidence > 10:
            print(f"Using detected emotion: {detected_emotion} ({emotion_confidence:.1f}%)")

        print(f">>> Sending to GPT: [{detected_emotion}] {message}")

        # Process with ChatGPT using server's detected emotion
        response_text = ask_chatgpt(message, detected_emotion, emotion_confidence)
        bot_emotion = extract_emotion_tag(response_text)

        print(f"GPT-4o-mini: {response_text}")

        return jsonify({
            "response": response_text,
            "bot_emotion": bot_emotion,
            "detected_emotion": detected_emotion,
            "emotion_confidence": emotion_confidence,
            "timestamp": time.time()
        })

    except Exception as e:
        print(f"Chat endpoint error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stream_frame', methods=['POST'])
def stream_frame():
    """Receive frame for live streaming"""
    global latest_frame, frame_lock

    try:
        data = request.json
        frame_b64 = data.get('frame')
        emotion = data.get('emotion')
        confidence = data.get('confidence')

        if frame_b64:
            frame = decode_frame(frame_b64)
            if frame is not None:
                # Add emotion overlay if provided
                if emotion and emotion != "neutral" and confidence and confidence > 10:
                    emotion_text = f"{emotion} ({confidence:.1f}%)"
                    cv2.putText(frame, emotion_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Store latest frame for streaming
                with frame_lock:
                    latest_frame = frame

        return jsonify({"status": "success"})

    except Exception as e:
        print(f"Stream frame error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/live_stream')
def live_stream():
    """Live video stream endpoint"""
    global stream_viewers
    stream_viewers += 1

    def generate():
        global latest_frame, frame_lock, stream_viewers

        try:
            while True:
                with frame_lock:
                    if latest_frame is not None:
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_bytes = buffer.tobytes()

                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                time.sleep(0.1)  # 10 FPS

        except GeneratorExit:
            stream_viewers -= 1
            print(f"Stream viewer disconnected. Active viewers: {stream_viewers}")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """Get server statistics"""
    emotion, confidence = get_current_server_emotion()

    return jsonify({
        "active_viewers": stream_viewers,
        "model_loaded": model is not None,
        "openai_available": openai_client is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
        "current_emotion": emotion,
        "emotion_confidence": confidence,
        "last_update": last_emotion_update
    })

# Keep-alive function for Colab
def keep_alive():
    """Thread to keep Colab from disconnecting"""
    while True:
        print(".", end="", flush=True)
        time.sleep(60)  # Print every minute

def initialize_server():
    """Initialize all server components"""
    print("🚀 Initializing Colab Server...")
    print("="*50)

    # Setup components
    success_count = 0

    if load_emotion_model():
        success_count += 1

    if load_face_detection():
        success_count += 1

    if setup_openai():
        success_count += 1

    print(f"\n✅ {success_count}/3 components initialized successfully")

    # Setup ngrok tunnel
    public_url = setup_ngrok()

    return public_url

def main():
    """Main server function"""
    # Initialize server
    public_url = initialize_server()

    if public_url:
        print(f"\n🌐 Server accessible at: {public_url}")
        print(f"📹 Live stream: {public_url}/live_stream")
        print(f"📊 Stats: {public_url}/stats")
        print(f"🏥 Health: {public_url}/health")

    # Start keep-alive thread
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()

    # Print header like in original files
    print("\n" + "\033[1;36m" + "=" * 50 + "\033[0m")
    print("\033[1;36m     Distributed Emotion-Aware System\033[0m")
    print("\033[1;36m" + "=" * 50 + "\033[0m")

    # Start Flask server
    print(f"\n🔥 Starting Flask server on port {PORT}...")
    print("📡 Ready to receive requests from Jetson Nano!")

    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()