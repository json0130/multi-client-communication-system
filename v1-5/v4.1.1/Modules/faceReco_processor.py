# faceReco_processor.py - Face Recognition Processor Module
import cv2
import numpy as np
from dlib import face_recognition

class FaceRecoProcessor:
    def __init__(self, model_path):
        self.model = face_recognition.FaceRecognitionModel(model_path)

    def process_image(self, image):
        # Convert the image to RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        face_locations = self.model.detect_faces(rgb_image)
        
        # Extract face encodings
        face_encodings = []
        for face_location in face_locations:
            encoding = self.model.compute_face_encoding(rgb_image, face_location)
            if encoding is not None:
                face_encodings.append(encoding)
        
        return face_locations, face_encodings

    def recognize_faces(self, image, known_face_encodings):
        locations, encodings = self.process_image(image)
        recognized_faces = []
        
        for encoding in encodings:
            matches = self.model.compare_faces(known_face_encodings, encoding)
            recognized_faces.append(matches)
        
        return locations, recognized_faces
        