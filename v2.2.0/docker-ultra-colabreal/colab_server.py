# Install required packages
!pip install flask pyngrok pillow torch torchvision

# Import necessary libraries
import os
import io
import base64
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response
import time
import cv2
import threading
import queue
from pyngrok import ngrok
import logging

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== CONFIG ==========
MODEL_PATH = '/content/drive/Shareddrives/COMPSYS 731 - JScript/efficientnet_opencv.pth'
print(f"Using model from: {MODEL_PATH}")

PORT = 5000
API_KEY = "emotion_recognition_key_123"  # This must match your client's SERVER_API_KEY
INPUT_SIZE = 224
EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MAX_FRAMES = 10  # Maximum number of frames to store

# Create Flask app
app = Flask(__name__)

# Frame storage - we'll keep the latest frame
latest_frame = None
latest_frame_time = 0
latest_metadata = {}
frame_lock = threading.Lock()

# ========== MODEL LOADING ==========
def get_model(name='efficientnet'):
    if name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(EMOTION_LABELS))
    elif name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(EMOTION_LABELS))
    else:
        raise ValueError("Unsupported model name")
    return model

# Initialize device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Verify the model file exists
if os.path.exists(MODEL_PATH):
    print(f"Model file found at {MODEL_PATH}")
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = get_model('efficientnet').to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating dummy model for testing")
        model = get_model('efficientnet').to(device)
        model.eval()
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}")
    print("Using a dummy model for testing")
    model = get_model('efficientnet').to(device)
    model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== AUTHENTICATION ==========
def is_valid_api_key(request_api_key):
    """Validate API key"""
    if not API_KEY:  # If no API key is set, skip authentication
        return True
    return request_api_key == API_KEY

# ========== PREDICTION FUNCTION ==========
def predict_emotion(image):
    """Predict emotion from image"""
    if model is None:
        return {"error": "Model not loaded"}

    try:
        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor)
            inference_time = time.time() - start_time

            # Get probabilities with softmax
            probabilities = F.softmax(output, dim=1)[0]

            # Get prediction and confidence
            pred_idx = torch.argmax(probabilities).item()
            confidence = probabilities[pred_idx].item() * 100
            emotion = EMOTION_LABELS[pred_idx]

            # Get top 3 emotions with confidence
            top3_values, top3_indices = torch.topk(probabilities, 3)
            top3_emotions = [
                {"emotion": EMOTION_LABELS[idx.item()], "confidence": val.item() * 100}
                for idx, val in zip(top3_indices, top3_values)
            ]

        # Return results
        return {
            "emotion": emotion,
            "confidence": confidence,
            "top_emotions": top3_emotions,
            "inference_time_ms": inference_time * 1000
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

# ========== STREAMING FUNCTIONS ==========
def process_streamed_frame(frame_data, metadata):
    """Process an incoming streamed frame"""
    global latest_frame, latest_frame_time, latest_metadata

    try:
        # Convert base64 to numpy array
        img_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Draw faces if provided
        if 'faces' in metadata and metadata['faces']:
            for face in metadata['faces']:
                # Face is (x, y, w, h)
                if len(face) == 4:
                    x, y, w, h = face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Add emotion text if available
                    if 'emotion' in metadata and metadata['emotion'] and 'confidence' in metadata:
                        emotion_text = f"{metadata['emotion']} ({metadata['confidence']:.1f}%)"
                        cv2.putText(frame, emotion_text, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update latest frame with lock to avoid race conditions
        with frame_lock:
            latest_frame = frame
            latest_frame_time = time.time()
            latest_metadata = metadata

        return True
    except Exception as e:
        logger.error(f"Error processing streamed frame: {e}")
        return False

# Generate video frames for streaming
def generate_frames():
    """Generator function to yield video frames for streaming"""
    last_frame_emitted = None

    while True:
        with frame_lock:
            current_frame = latest_frame

        if current_frame is not None:
            # Only send a new frame if it's different from the last one
            if last_frame_emitted is None or not np.array_equal(current_frame, last_frame_emitted):
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()

                # Save the current frame as the last emitted
                last_frame_emitted = current_frame.copy()

                # Yield the frame in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, send a placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for video...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Small delay to control frame rate
        time.sleep(0.1)  # Maximum 10 FPS

# ========== API ENDPOINTS ==========
@app.route('/predict', methods=['POST'])
def predict():
    """Emotion prediction endpoint"""
    logger.info(f"Received predict request")

    # Check API key
    api_key = request.headers.get('X-API-Key')
    if not is_valid_api_key(api_key):
        logger.warning("Invalid API key")
        return jsonify({"error": "Invalid API key"}), 401

    # Parse request
    if not request.json or 'image' not in request.json:
        logger.warning("Missing image data")
        return jsonify({"error": "Missing image data"}), 400

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.json['image'])
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"Decoded image: {image.width}x{image.height}")

        # Get prediction
        result = predict_emotion(image)
        logger.info(f"Prediction result: {result.get('emotion')} ({result.get('confidence'):.2f}%)")

        # Return JSON response
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/stream', methods=['POST'])
def stream():
    """Receive streamed frames from client"""
    # Check API key
    api_key = request.headers.get('X-API-Key')
    if not is_valid_api_key(api_key):
        logger.warning("Invalid API key")
        return jsonify({"error": "Invalid API key"}), 401

    # Parse request
    if not request.json or 'frame' not in request.json:
        logger.warning("Missing frame data")
        return jsonify({"error": "Missing frame data"}), 400

    try:
        # Extract frame and metadata
        frame_data = request.json['frame']

        # Extract metadata
        metadata = {
            'faces': request.json.get('faces', []),
            'emotion': request.json.get('emotion'),
            'confidence': request.json.get('confidence', 0),
            'timestamp': request.json.get('timestamp', time.time())
        }

        # Process the frame
        success = process_streamed_frame(frame_data, metadata)

        if success:
            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"error": "Failed to process frame"}), 500
    except Exception as e:
        logger.error(f"Error processing streamed frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check request received")
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "server_time": time.time(),
        "has_latest_frame": latest_frame is not None,
        "latest_frame_age": time.time() - latest_frame_time if latest_frame is not None else None
    })

@app.route('/video_feed')
def video_feed():
    """Route for streaming video feed - this returns a multipart response"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET'])
def index():
    """Main visualization page - simplified with direct video streaming"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotion Recognition Stream</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; }
            .video-container { width: 100%; text-align: center; margin-bottom: 20px; }
            #videoFeed { max-width: 100%; max-height: 600px; border: 1px solid #ddd; }
            .info-panel { background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .emotion-info { font-size: 18px; margin-bottom: 15px; }
            .status { margin-top: 10px; font-size: 14px; color: #666; }
            .controls { margin-top: 20px; }
            .emotion-label { font-weight: bold; }
            .confidence { color: #4CAF50; }
            h1 { color: #333; }
            button { padding: 8px 16px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RealSense Emotion Recognition Stream</h1>
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Video Stream">
            </div>
            <div class="info-panel">
                <div class="emotion-info">
                    <span class="emotion-label">Server Status:</span> <span id="status">Connected</span>
                </div>
                <div class="status" id="lastUpdate">Streaming...</div>
                <div class="controls">
                    <button onclick="refreshStream()">Refresh Stream</button>
                </div>
            </div>
        </div>

        <script>
            // Time display updater
            function updateTime() {
                document.getElementById('lastUpdate').textContent = 'Last update: ' + new Date().toLocaleTimeString();
            }
            setInterval(updateTime, 1000);

            // Function to refresh the stream
            function refreshStream() {
                const videoFeed = document.getElementById('videoFeed');
                videoFeed.src = '/video_feed?' + new Date().getTime();
                document.getElementById('status').textContent = 'Stream refreshed';
            }

            // Auto-refresh the stream every 30 seconds
            setInterval(refreshStream, 30000);
        </script>
    </body>
    </html>
    """)

# ========== START SERVER ==========
# Set up ngrok authentication
ngrok.set_auth_token("2xIRdc89wBeyfK9pNWvhh4x2ENp_24LvRBjQN6CAHWqVG4WX6")  # Your ngrok auth token

# Setup ngrok to expose the Flask app
public_url = ngrok.connect(PORT)
print(f" * Ngrok tunnel running at: {public_url}")
print(f" * Emotion recognition endpoint: {public_url}/predict")
print(f" * Streaming endpoint: {public_url}/stream")
print(f" * VIDEO FEED URL: {public_url}/video_feed")  # Direct video stream URL
print(f" * Visualization page: {public_url}")
print(" * IMPORTANT: Keep this tab open to maintain the connection")

# Log all routes before starting
print("Available routes:")
for rule in app.url_map.iter_rules():
    print(f" - {rule}")

# Add a keep alive routine
import IPython
import threading

def keep_alive():
    while True:
        print(".", end="", flush=True)
        time.sleep(60)  # Print every minute to keep Colab from disconnecting

# Start keep-alive thread
keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

# Start the Flask app
app.run(host='0.0.0.0', port=PORT)