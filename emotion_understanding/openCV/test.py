# real_time_emotion_realsense.py
import pyrealsense2 as rs
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn

# ========== CONFIG ==========
model_path = 'efficientnet_opencv.pth'  # Path to trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 224
emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # adjust as needed
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

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

model = get_model('efficientnet').to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== REALSENSE SETUP ==========
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

face_cascade = cv2.CascadeClassifier(cascade_path)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                pred = torch.argmax(output, dim=1).item()
                emotion = emotion_labels[pred]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
