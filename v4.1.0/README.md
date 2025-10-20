# Release Notes v4.1.0

## Input Modules
- Depth Stream Support: The RealSense module can be configured to capture and stream both RGB and Depth data, enabling future capabilities in 3D perception and spatial awareness.
- Standard Webcam Support: A generic CameraInputModule using OpenCV has been added for standard USB webcams, providing a simple way to stream video to the server.

## Output Modules
- Arduino/ESP32 Hardware Control: An ArduinoOutputModule has been added to send commands to microcontrollers (like Arduino or ESP32) connected via a serial port.
- Emotion & Servo Control: The module is designed to control servos, translating the bot's internal emotional state (e.g., "HAPPY", "CONFUSED", "WAVE") into physical actions and expressions.
- Robust Connection Management: The module includes logic for reliable serial communication, connection monitoring, and graceful handling of device disconnections.
