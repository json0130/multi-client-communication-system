# emotion_processor.py - Emotion Detection Component
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
from collections import deque

class EmotionTracker:
    """Enhanced emotion tracking with proper moving average"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.emotion_counts = {}
        self.last_stable_emotion = "neutral"
        self.last_stable_confidence = 0.0

    def add_detection(self, emotion, confidence, confidence_threshold=30.0):
        """Add new emotion detection with proper averaging"""
        if confidence >= confidence_threshold:
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

    def get_stable_emotion(self, emotion_change_threshold=15.0, emotion_update_threshold=0.05):
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

            if (emotion_changed and weighted_confidence > emotion_change_threshold) or \
               confidence_diff > emotion_update_threshold:
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

class EmotionProcessor:
    """Main emotion processing class"""
    
    def __init__(self, model_path=None, config=None):
        # Configuration
        if model_path is None:
            model_path = os.path.join("models", "efficientnet_HQRAF_improved_withCon.pth")
        self.model_path = model_path
        self.config = config or {}
        
        # Processing configuration
        self.emotion_processing_interval = self.config.get('emotion_processing_interval', 0.1)
        self.frame_skip_ratio = self.config.get('frame_skip_ratio', 1)
        self.confidence_threshold = self.config.get('confidence_threshold', 30.0)
        self.emotion_change_threshold = self.config.get('emotion_change_threshold', 15.0)
        self.emotion_update_threshold = self.config.get('emotion_update_threshold', 0.05)
        self.emotion_window_size = self.config.get('emotion_window_size', 5)
        
        # Model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 224
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Initialize components
        self.model = None
        self.transform = None
        self.face_cascade = None
        
        # State tracking
        self.model_loaded = False
        self.transform_loaded = False
        self.face_cascade_loaded = False
        
        # Processing state
        self.frame_counter = 0
        self.last_emotion_process_time = 0
        self.processing_lock = threading.Lock()
        
        # Current emotion state
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.last_emotion_update = 0
        self.emotion_lock = threading.Lock()
        
        # ✅ NEW: Frame storage for live streaming
        self.latest_processed_frame = None
        self.frame_storage_lock = threading.Lock()
        
        # Initialize emotion tracker
        self.emotion_tracker = EmotionTracker(self.emotion_window_size)
        
    def get_model(self):
        """Load emotion detection model with classifier structure matching training."""
        num_classes = 7
        dropout_rate = 0.2

        # Use EfficientNet B0 model
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

        return model
    
    def initialize_transform(self):
        """Initialize transform separately with error handling"""
        try:
            print("Initializing image transform...")
            self.transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            self.transform_loaded = True
            print("Transform initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing transform: {e}")
            self.transform_loaded = False
            return False
    
    def load_emotion_model(self):
        """Load the emotion detection model with better error handling"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found at: {self.model_path}")
                print("Please ensure the model file is in the correct location")
                return False

            print(f"Loading EfficientNet B0 model from {self.model_path}...")

            # Load EfficientNet B0 model
            self.model = self.get_model().to(self.device)
            
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model from 'model_state_dict' key")
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    print("Loaded model from 'state_dict' key")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("Loaded model state dict directly")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
            
            self.model.eval()
            self.model_loaded = True
            print(f"EfficientNet B0 model loaded successfully on {self.device}")

            if not self.initialize_transform():
                return False

            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def load_face_detection(self):
        """Load face detection model with better error handling"""
        try:
            print(f"Loading face cascade from: {self.cascade_path}")

            if not os.path.exists(self.cascade_path):
                print(f"Face cascade file not found at: {self.cascade_path}")
                return False

            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.face_cascade.empty():
                print(f"Error: Could not load face cascade from {self.cascade_path}")
                self.face_cascade_loaded = False
                return False

            self.face_cascade_loaded = True
            print("Face detection model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading face cascade: {e}")
            self.face_cascade_loaded = False
            return False
    
    def initialize(self):
        """Initialize all emotion processing components"""
        success_count = 0
        
        print("Loading emotion detection model...")
        if self.load_emotion_model():
            success_count += 1
            print("    Model and transform loaded successfully")
        else:
            print("    Model loading failed")

        print("Loading face detection...")
        if self.load_face_detection():
            success_count += 1
            print("    Face detection loaded successfully")
        else:
            print("    Face detection loading failed")
            
        return success_count, 2  # Return (success_count, total_count)
    
    def decode_frame_optimized(self, frame_b64):
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
    
    def should_process_emotion(self):
        """Enhanced emotion processing frequency control"""
        current_time = time.time()
        self.frame_counter += 1

        if (self.frame_counter % self.frame_skip_ratio == 0 and
            current_time - self.last_emotion_process_time >= self.emotion_processing_interval):
            self.last_emotion_process_time = current_time
            return True
        return False
    
    def process_emotion_detection_realtime(self, frame):
        """Enhanced real-time emotion detection with proper moving average"""
        if not all([self.transform_loaded, self.model_loaded, self.face_cascade_loaded,
                    self.transform is not None, self.model is not None, self.face_cascade is not None]):
            return self.current_emotion, self.current_confidence, "components_not_loaded"

        try:
            # ✅ NEW: Store the frame for live streaming (with face overlay)
            processed_frame = frame.copy()
            
            if not self.should_process_emotion():
                # ✅ NEW: Still store frame even if not processing emotion
                processed_frame = self.add_face_overlay(processed_frame)
                with self.frame_storage_lock:
                    self.latest_processed_frame = processed_frame
                return self.current_emotion, self.current_confidence, "throttled"

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.15,
                minNeighbors=4,
                minSize=(40, 40)
            )

            if len(faces) == 0:
                # ✅ NEW: Store frame even when no faces detected
                with self.frame_storage_lock:
                    self.latest_processed_frame = processed_frame
                return self.current_emotion, self.current_confidence, "no_faces"

            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            try:
                face = frame[y:y+h, x:x+w]
                face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(face_tensor)
                    probabilities = F.softmax(output, dim=1)[0]
                    pred_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[pred_idx].item() * 100
                    emotion = self.emotion_labels[pred_idx]

                self.emotion_tracker.add_detection(emotion, confidence, self.confidence_threshold)
                stable_emotion, stable_confidence, emotion_changed = self.emotion_tracker.get_stable_emotion(
                    self.emotion_change_threshold, self.emotion_update_threshold)

                with self.emotion_lock:
                    if emotion_changed or time.time() - self.last_emotion_update > 1.0:
                        self.current_emotion = stable_emotion
                        self.current_confidence = stable_confidence
                        self.last_emotion_update = time.time()

                        distribution = self.emotion_tracker.get_emotion_distribution()
                        print(f"Emotion: {stable_emotion} ({stable_confidence:.1f}%) | Raw: {emotion} ({confidence:.1f}%)")

                # ✅ NEW: Add face overlay and store processed frame
                processed_frame = self.add_face_overlay(processed_frame)
                with self.frame_storage_lock:
                    self.latest_processed_frame = processed_frame

                return stable_emotion, stable_confidence, "success"

            except Exception as e:
                print(f"Error processing face: {e}")
                # ✅ NEW: Store frame even when face processing fails
                with self.frame_storage_lock:
                    self.latest_processed_frame = processed_frame
                return self.current_emotion, self.current_confidence, f"face_processing_error: {e}"

        except Exception as e:
            print(f"Error in emotion processing: {e}")
            # ✅ NEW: Store frame even when general processing fails
            with self.frame_storage_lock:
                self.latest_processed_frame = frame.copy()
            return self.current_emotion, self.current_confidence, f"general_error: {e}"
    
    def get_current_emotion(self):
        """Get current emotion state thread-safely"""
        with self.emotion_lock:
            return self.current_emotion, self.current_confidence
    
    def get_emotion_distribution(self):
        """Get current emotion distribution"""
        return self.emotion_tracker.get_emotion_distribution()
    
    def get_status(self):
        """Get processor status"""
        return {
            'model_loaded': self.model_loaded,
            'transform_loaded': self.transform_loaded,
            'face_cascade_loaded': self.face_cascade_loaded,
            'current_emotion': self.current_emotion,
            'current_confidence': self.current_confidence,
            'window_size': len(self.emotion_tracker.emotion_history)
        }
    
    def get_latest_frame(self):
        """✅ NEW: Get the latest processed frame for live streaming"""
        with self.frame_storage_lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
            return None
    
    def add_face_overlay(self, frame):
        """Add face detection bounding box and emotion overlay to frame"""
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(40, 40))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (66, 133, 244), 2)
                label_h = 30
                cv2.rectangle(frame, (x, y-label_h), (x+w, y), (66, 133, 244), -1)

                if self.current_emotion and self.current_confidence > 30:
                    text = f"{self.current_emotion} ({self.current_confidence:.0f}%)"
                    cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame