# client_test.py - Modified to stream video to Colab
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import os
import sys
import requests
import json
import base64
from io import BytesIO
import threading
import queue

# Import our streaming module
from realsense_stream import RealSenseStreamer

# ========== CONFIG ==========
# Server configuration
EMOTION_SERVER_URL = "http://fac7-35-247-51-166.ngrok-free.app/predict"  # UPDATE THIS with your Colab URL
STREAM_SERVER_URL = "http://fac7-35-247-51-166.ngrok-free.app/stream"    # URL for streaming frames to Colab
SERVER_API_KEY = "emotion_recognition_key_123"  # Must match API_KEY on server
REQUEST_TIMEOUT = 5  # seconds
DEBUG_MODE = False  # Set to True for more verbose output
STREAM_QUALITY = 50  # JPEG compression quality (0-100)
STREAM_FPS = 5  # Target frames per second to stream
STREAM_ENABLED = True  # Enable/disable streaming to Colab

# Local configuration
input_size = 224
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Define the emotion file path
emotion_file = "/app/emotion_file.txt"  # Use absolute path for Docker

# For tracking emotion changes
last_emotion = None
last_confidence = 0
emotion_print_time = 0
emotion_print_interval = 0.5  # Print every 0.5 seconds when emotion change

# Queue for streaming frames
frame_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues

# Print configuration if in debug mode
if DEBUG_MODE:
    print(f"Emotion Server URL: {EMOTION_SERVER_URL}")
    print(f"Stream Server URL: {STREAM_SERVER_URL}")
    print(f"Using API key: {'[SET]' if SERVER_API_KEY else '[NOT SET]'}")
    print(f"Emotion file path: {emotion_file}")
    print(f"Streaming enabled: {STREAM_ENABLED}")

# ========== FILE OPERATIONS ==========
# Initialize emotion file with neutral
def initialize_emotion_file():
    global emotion_file
    
    try:
        with open(emotion_file, "w") as f:
            f.write("neutral:0.0")
        print(f"Initialized emotion file: {emotion_file}")
    except Exception as e:
        print(f"Error initializing emotion file: {e}")
        # Try alternate path
        try:
            alt_path = "./emotion_file.txt"
            with open(alt_path, "w") as f:
                f.write("neutral:0.0")
            print(f"Initialized alternate emotion file: {alt_path}")
            emotion_file = alt_path  # Update the global variable
        except Exception as e2:
            print(f"Error with alternate path: {e2}")

# Update the emotion file with current emotion and confidence
def update_emotion_file(emotion, confidence):
    global emotion_file
    
    try:
        with open(emotion_file, "w") as f:
            f.write(f"{emotion}:{confidence}")
        if DEBUG_MODE:
            print(f"Updated file with {emotion}:{confidence}")
    except Exception as e:
        print(f"Error writing to emotion file: {e}")

# Function to read the emotion file
def read_emotion_file():
    global emotion_file
    
    try:
        with open(emotion_file, "r") as f:
            content = f.read().strip()
            if DEBUG_MODE:
                print(f"Current emotion file content: {content}")
            return content
    except Exception as e:
        print(f"Error reading emotion file: {e}")
        return None

# ========== FACE DETECTION ==========
# Load face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {cascade_path}")
        sys.exit(1)
except Exception as e:
    print(f"Error loading face cascade: {e}")
    sys.exit(1)

# ========== SERVER COMMUNICATION ==========
def send_image_to_server(face_image):
    """
    Send face image to remote server for emotion recognition
    
    Args:
        face_image: PIL Image object containing the face
    
    Returns:
        tuple: (emotion, confidence) or (None, 0) if error
    """
    try:
        # Convert PIL image to bytes
        buffered = BytesIO()
        face_image.save(buffered, format="JPEG", quality=90)
        img_bytes = buffered.getvalue()
        
        # Encode image to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prepare payload
        payload = {
            "image": img_base64,
        }
        
        # Add API key if needed
        headers = {
            "Content-Type": "application/json"
        }
        if SERVER_API_KEY:
            headers["X-API-Key"] = SERVER_API_KEY
        
        # Send request to server
        if DEBUG_MODE:
            print(f"Sending face to emotion server")
            start_time = time.time()
            
        response = requests.post(
            EMOTION_SERVER_URL,
            data=json.dumps(payload),
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        
        if DEBUG_MODE:
            elapsed = time.time() - start_time
            print(f"Server response time: {elapsed:.2f}s")
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            emotion = result.get("emotion")
            confidence = result.get("confidence", 0)
            if DEBUG_MODE:
                print(f"Server predicted: {emotion} ({confidence:.1f}%)")
            return emotion, confidence
        else:
            print(f"Server error: {response.status_code} - {response.text}")
            return None, 0
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None, 0
    except Exception as e:
        print(f"Error sending image to server: {e}")
        return None, 0

# Function to stream frames to Colab server
def stream_frame_to_server(frame, faces=None, emotion=None, confidence=None):
    """
    Stream frame to Colab server
    
    Args:
        frame: CV2 frame to stream
        faces: List of (x,y,w,h) face coordinates
        emotion: Current detected emotion
        confidence: Confidence score
    """
    if not STREAM_ENABLED:
        return
    
    try:
        # Add the frame to queue for the streaming thread
        if not frame_queue.full():
            frame_info = {
                "frame": frame.copy(),
                "faces": faces if faces is not None else [],
                "emotion": emotion,
                "confidence": confidence
            }
            frame_queue.put(frame_info, block=False)
    except Exception as e:
        print(f"Error queueing frame: {e}")

# Thread function to send frames to server
def frame_streaming_thread():
    """Background thread to stream frames to Colab server"""
    last_stream_time = 0
    min_interval = 1.0 / STREAM_FPS  # Minimum time between frames
    
    while True:
        try:
            # Get frame from queue
            frame_info = frame_queue.get(timeout=1.0)
            
            # Rate limiting
            current_time = time.time()
            if current_time - last_stream_time < min_interval:
                time.sleep(min_interval - (current_time - last_stream_time))
            
            # Extract frame and metadata
            frame = frame_info["frame"]
            faces = frame_info["faces"]
            emotion = frame_info["emotion"]
            confidence = frame_info["confidence"]
            
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare payload
            payload = {
                "frame": img_base64,
                "faces": faces,
                "emotion": emotion,
                "confidence": confidence if confidence else 0.0,
                "timestamp": time.time()
            }
            
            # Add API key if needed
            headers = {
                "Content-Type": "application/json"
            }
            if SERVER_API_KEY:
                headers["X-API-Key"] = SERVER_API_KEY
            
            # Send to server
            response = requests.post(
                STREAM_SERVER_URL,
                data=json.dumps(payload),
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200 and DEBUG_MODE:
                print(f"Stream error: {response.status_code} - {response.text[:100]}")
            
            # Update last stream time
            last_stream_time = time.time()
            
        except queue.Empty:
            # Queue is empty, just continue
            pass
        except Exception as e:
            print(f"Error in streaming thread: {e}")
            time.sleep(1)  # Avoid tight loop on error

# Function to format confidence for display
def format_confidence(confidence):
    return f"{confidence:.1f}%"

# ========== FRAME PROCESSOR FUNCTION ==========
def process_emotion(original_frame, display_frame):
    """Process a frame for emotion detection with confidence percentage"""
    global last_emotion, last_confidence, emotion_print_time
    
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        current_time = time.time()
        
        if len(faces) == 0:
            # No faces detected
            streamer.update_metadata("emotion", "No faces detected")
            
            # Still stream the frame (without faces)
            stream_frame_to_server(original_frame)
            
            # Print to console if state changed
            if last_emotion is not None and (current_time - emotion_print_time) > emotion_print_interval:
                emotion_print_time = current_time
        else:
            # Process each face (we'll focus on the first/largest face)
            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            face_coords = []
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_coords.append((int(x), int(y), int(w), int(h)))  # Convert to int for JSON serialization
                
                try:
                    # Extract face region
                    face = original_frame[y:y+h, x:x+w]
                    
                    # Convert to PIL Image for sending
                    face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    
                    # Send to server for processing
                    emotion, confidence = send_image_to_server(face_img)
                    
                    if emotion:
                        # Update emotion metadata with confidence
                        emotion_display = f"{emotion} ({format_confidence(confidence)})"
                        streamer.update_metadata("emotion", emotion_display)
                        
                        # Add emotion text with confidence above the face
                        cv2.putText(display_frame, emotion_display, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Stream the frame with emotion data
                        stream_frame_to_server(original_frame, face_coords, emotion, confidence)
                        
                        # Update emotion file if changed significantly
                        if (emotion != last_emotion or abs(confidence - last_confidence) > 5.0):
                            update_emotion_file(emotion, confidence)
                            
                        # Update last known values
                        last_emotion = emotion
                        last_confidence = confidence
                        emotion_print_time = current_time
                    else:
                        # Use last known emotion if server failed
                        if last_emotion:
                            cv2.putText(display_frame, f"{last_emotion} (cached)", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)
                        
                        # Stream the frame with cached emotion data
                        stream_frame_to_server(original_frame, face_coords, last_emotion, last_confidence)
                    
                    # Only process the first (largest) face
                    break
                        
                except Exception as e:
                    print(f"Error processing face: {e}")
                    # Still stream the frame even if face processing fails
                    stream_frame_to_server(original_frame, face_coords)
        
        return display_frame
    except Exception as e:
        print(f"Error in emotion processing: {e}")
        return None

# ========== MAIN FUNCTION ==========
def main():
    global streamer

    # Initialize the emotion file
    initialize_emotion_file()
    
    # Print header
    print("\033[1;36m" + "=" * 50 + "\033[0m")
    print("\033[1;36m     Cloud-Based Emotion Detection System\033[0m")
    print("\033[1;36m" + "=" * 50 + "\033[0m")
    print(f"\033[1;33mConnected to emotion server: {EMOTION_SERVER_URL}\033[0m")
    
    # Start streaming thread if enabled
    if STREAM_ENABLED:
        print(f"\033[1;33mStreaming video to: {STREAM_SERVER_URL}\033[0m")
        print(f"\033[1;33mStream quality: {STREAM_QUALITY}, FPS: {STREAM_FPS}\033[0m")
        stream_thread = threading.Thread(target=frame_streaming_thread, daemon=True)
        stream_thread.start()
        print(f"\033[1;32mStream thread started\033[0m")
    
    # Initialize the RealSense streamer with our processor
    streamer = RealSenseStreamer(web_port=8080, frame_processor=process_emotion)
    
    if not streamer.start():
        print("Failed to start streamer")
        return
    
    try:
        # Keep the main thread running to monitor
        check_counter = 0
        while True:
            time.sleep(1)
            # Read the emotion file every few seconds to verify
            check_counter += 1
            if check_counter >= 1:  # Check every 3 seconds
                read_emotion_file()
                check_counter = 0
    except KeyboardInterrupt:
        print("\n\033[1;33mInterrupted by user\033[0m")
    except Exception as e:
        print(f"\033[1;31mError in main loop: {e}\033[0m")
    finally:
        # Clean up
        print("Stopping streamer...")
        streamer.stop()
        print("\033[1;32mDone\033[0m")

if __name__ == "__main__":
    main()
