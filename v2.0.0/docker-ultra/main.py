from emotion_model import detect_emotion
from chatbot import query_chatgpt
from arduino_handler import send_to_arduino
import cv2
import pyrealsense2 as rs
import numpy as np

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        # If frame is empty, continue to next iteration
        if not color_frame:
            continue
            
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        
        # Process frame for emotion detection
        emotion = detect_emotion(color_image)
        
        # Display the color frame
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Get user input and process through ChatGPT
        user_input = input("User: ")
        response = query_chatgpt(user_input, emotion)
        print("ChatGPT:", response)
        
        # Send emotion data to Arduino
        send_to_arduino(emotion)
        
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
