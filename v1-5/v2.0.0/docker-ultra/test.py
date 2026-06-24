# test.py (optimized version with confidence percentage)
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import time
import os
import sys

# Import our streaming module
from realsense_stream import RealSenseStreamer

# ========== CONFIG ==========
model_path = 'efficientnet_opencv.pth'  # Path to trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
    #print(f"CUDA device: {torch.cuda.get_device_name(0)}")
input_size = 224
emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# For tracking emotion changes
emotion_file = "/app/emotion_file.txt"
last_emotion = None
last_confidence = 0
emotion_print_time = 0
emotion_print_interval = 0.5  # Print every 0.5 seconds when emotion change

# Initialize emotion file with neutral
def initialize_emotion_file():
    with open(emotion_file, "w") as f:
        f.write("neutral:0.0")
        print("I am trying to write neutral and initialize")
    # print(f"Initialized emotion file: {emotion_file}")

# Update the emotion file with current emotion and confidence
def update_emotion_file(emotion, confidence):
    try:
        # Make sure emotion_file is defined as a global variable
        global emotion_file
        
        # Check if emotion_file is properly defined
        if not emotion_file:
            print("Why tf are you going in here")
            emotion_file = "/app/emotion_file.txt"
            
        # Write to the file
        with open(emotion_file, "w") as f:
            # print("Are you writing tge emotion???")
            f.write(f"{emotion}:{confidence}")
        
        # Print successful write
        # print(f"Updated file with {emotion}:{confidence}")
    except Exception as e:
        print(f"Error writing to emotion file: {e}")

# Function to read the emotion file and print its contents
def read_emotion_file():
    try:
        global emotion_file
        with open(emotion_file, "r") as f:
            content = f.read().strip()
            # print(f"Current emotion file content: {content}")
            return content
    except Exception as e:
        print(f"Error reading emotion file: {e}")
        return None

# ========== MODEL ==========
def get_model(name='efficientnet'):
    if name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(emotion_labels))
    elif name == 'mobilenetv3':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(emotion_labels))
    else:
        raise ValueError("Unsupported model name")
    return model

# Load model with error handling
try:
    # print(f"Loading model from {model_path}...")
    model = get_model('efficientnet').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Continuing without emotion detection")
    model = None

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== FACE DETECTION ==========
# Load face cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {cascade_path}")
        sys.exit(1)
    # print("Face detection model loaded successfully")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    sys.exit(1)

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
            
            # Print to console if state changed
            if last_emotion is not None and (current_time - emotion_print_time) > emotion_print_interval:
                # print("No faces detected")
                emotion_print_time = current_time
        else:
            # Process each face (we'll focus on the first/largest face)
            # Sort faces by area (largest first)
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                if model is not None:
                    try:
                        # Extract face region
                        face = original_frame[y:y+h, x:x+w]
                        
                        # Convert to PIL Image for transformation
                        face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                        
                        # Apply transformations and predict
                        face_tensor = transform(face_img).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output = model(face_tensor)
                            
                            # Get probabilities with softmax
                            probabilities = F.softmax(output, dim=1)[0]
                            
                            # Get prediction and confidence
                            pred_idx = torch.argmax(probabilities).item()
                            confidence = probabilities[pred_idx].item() * 100
                            emotion = emotion_labels[pred_idx]
                            
                            # Get top 3 emotions with confidence
                            top3_values, top3_indices = torch.topk(probabilities, 3)
                            top3_emotions = [(emotion_labels[idx.item()], val.item() * 100) 
                                            for idx, val in zip(top3_indices, top3_values)]
                        
                        # Update emotion metadata with confidence
                        emotion_display = f"{emotion} ({format_confidence(confidence)})"
                        streamer.update_metadata("emotion", emotion_display)
                        
                        # Add emotion text with confidence above the face
                        cv2.putText(display_frame, emotion_display, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
			
			# Update emotion file if changed significantly
                        if (emotion != last_emotion or abs(confidence - last_confidence) > 5.0):
                            update_emotion_file(emotion, confidence)

                        
                        # Print to console if emotion changed or interval elapsed
                        if (emotion != last_emotion or abs(confidence - last_confidence) > 5.0 
                                or (current_time - emotion_print_time) > 2.0):
                            # Terminal display with ANSI colors for better visibility
                            # print("\033[1m" + "=" * 40 + "\033[0m")  # Bold separator
                            # print(f"\033[1;32mDetected Emotion: {emotion} ({format_confidence(confidence)})\033[0m")
                            
                            # Print top 3 emotions
                            # print("\033[1mTop Emotions:\033[0m")
                            for i, (em, conf) in enumerate(top3_emotions):
                                color = '\033[1;32m' if i == 0 else '\033[0;36m'  # Green for top, cyan for others
                                # print(f"{color}{i+1}. {em}: {format_confidence(conf)}\033[0m")
                            
                            last_emotion = emotion
                            last_confidence = confidence
                            emotion_print_time = current_time
                        
                        # Only process the first (largest) face
                        break
                        
                    except Exception as e:
                        print(f"Error processing face: {e}")
        
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
    print("\033[1;36m     RealSense Emotion Detection System\033[0m")
    print("\033[1;36m" + "=" * 50 + "\033[0m")
    
    # Initialize the RealSense streamer with our processor
    # print("Initializing RealSense streamer...")
    streamer = RealSenseStreamer(web_port=8080, frame_processor=process_emotion)
    
    if not streamer.start():
        print("Failed to start streamer")
        return
    
    # print("\033[1;32mStreamer started successfully.\033[0m")
    # print("\033[1;32mEmotion detection is running...\033[0m")
    # print(f"\033[1;33mView stream at: http://localhost:8080\033[0m")
    # print("\033[1;33mPress Ctrl+C to exit\033[0m")
    
    try:
        # Keep the main thread running to monitor
        check_counter = 0
        while True:
            time.sleep(1)
            # Read the emotion file every 5 seconds to verify
            check_counter += 1
            if check_counter >= 3:
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
