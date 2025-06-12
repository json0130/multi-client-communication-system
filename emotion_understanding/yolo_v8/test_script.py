import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from torchvision import transforms
from ultralytics import YOLO
from train_script import get_model
from collections import deque, Counter

# Config
MODEL_NAME = 'efficientnet'  # Change to 'resnet18' or 'efficientnet'
EMOTION_MODEL_PATH = 'efficientnet_model.pth'
NUM_CLASSES = 8
VOTING_WINDOW = 20  # Number of frames for majority vote

recent_predictions = deque(maxlen=VOTING_WINDOW)
recent_probs = deque(maxlen=VOTING_WINDOW)

# ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load emotion recognition model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = get_model(MODEL_NAME)
emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location=device))
emotion_model.eval().to(device)

# Load YOLO face detection model
yolo = YOLO("yolov8m-face.pt")
yolo.to(device)

# Emotion labels
emotion_labels = ["happy", "sad", "anger", "surprise", "fear", "disgust", "contempt", "neutral"]

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Moving average buffer
recent_predictions = deque(maxlen=VOTING_WINDOW)
display_label = "..."

# New method to check frame quality (blurriness)
def is_blurry(frame, threshold=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        h, w, _ = frame.shape

        # Check if the frame is blurry
        # if is_blurry(frame):
        #     print("Frame is blurry, skipping...")
        #     continue

        # Face detection
        results = yolo.predict(source=frame, device=device, verbose=False)

        for r in results:
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)


                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                # Add padding around the detected face
                padding_ratio = 0.5
                box_w, box_h = x2 - x1, y2 - y1
                pad_w, pad_h = int(box_w * padding_ratio), int(box_h * padding_ratio)

                px1 = max(0, x1 - pad_w)
                py1 = max(0, y1 - pad_h)
                px2 = min(w, x2 + pad_w)
                py2 = min(h, y2 + pad_h)

                face = frame[py1:py2, px1:px2]

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_face = transforms.functional.to_pil_image(face_rgb)
                input_tensor = preprocess(pil_face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = emotion_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
                    pred = np.argmax(probs)
                    recent_predictions.append(pred)
                    recent_probs.append(probs[0])

                    display_label = ""
                    # Only update display label every 10 frames
                    if len(recent_predictions) == VOTING_WINDOW:
                        counts = Counter(recent_predictions)
                        if counts:
                            top_pred, _ = counts.most_common(1)[0]
                            avg_conf = np.mean([p[top_pred] for p in recent_probs])
                            display_label = f"{emotion_labels[top_pred]} {avg_conf:.2f}"
                        
                    # Print the current prediction and confidence
                    print(f"Detected: {emotion_labels[pred]} ({probs[0][pred]*100:.2f}%)")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
