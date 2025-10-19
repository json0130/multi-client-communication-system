## CHATBOX
An Emotion-Aware Social Robot for Children with Mental Health Issues. With real-time emotion detection and a chatbot system that combines computer vision, natural language processing, and hardware integration, it is designed to be used for children.

This project develops a comprehensive human-robot interaction (HRI) system that combines real-time emotion recognition with intelligent conversational capabilities. The system is specifically designed to support children experiencing mental health challenges, providing natural engagement through facial emotion detection and meaningful interactive experiences.

### Key Features

- **Real-time Emotion Detection**: Facial emotion recognition using EfficientNet
- **Contextual AI Responses**: ChatGPT integration with emotion-aware prompting
- **Hardware Integration**: Chatbot integrated with Arduino-controlled physical responses and feedback
- **Live Video Streaming**: WebSocket-based real-time video transmission
- **Server Hosted**: Streaming hosted on the Server through Google Colab
- **Portability**: Chatbot is portable through implementation on the Jetson Nano board

### System Architecture

**Hardware**
- **Jetson Nano Board**: Primary processing unit
- **Intel RealSense Camera**: Depth-aware facial capture and real-time video streaming
- **ChatBox Robot Platform**: Physical interaction and gesture execution
- **Bluetooth Speaker**: Audio output for text-to-speech functionality

**Software Stack**
- **Emotion Recognition**: Custom ChatBox_V1 model based on EfficientNet V2-S architecture
- **Computer Vision**: WebSocket connections for real-time data transmission
- **Text-to-Speech**: EDGE TTS for natural speech generation

## Versioning History

### V1.0.0 - Local Processing System (28/04/2025)

- **Features**: Arduino robot platform with local emotion recognition and ChatGPT integration
- **Limitations**: Laptop attachment reduced portability and clean aesthetics

### V2.0.0 - Jetson Nano Migration (8/05/2025)

- **Change**: Migrated processing from laptop to Jetson Nano board for improved portability
- **Features**: Jetson Nano board processing, Docker containerisation, portable robot design
- **Limitations**: Limited computational power, memory constraints

### V2.0.1 - Streaming Service Addition (10/05/2025)

- **Change**: Added video streaming capability for remote monitoring
- **Features**: Video streaming capability for caregiver/doctor monitoring
- **Limitations**: Local streaming only, no remote access capability

### V2.1.0 - Server-Based Processing Colab (16/05/2025)

- **Change**: Moved emotion recognition processing to Google Colab server
- **Features**: Emotion recognition on Google Colab server, remote streaming via ngrok
- **Limitations**: High latency, rapid token exhaustion, ChatGPT is still processed on Jetson

### V2.2.0 - Full Server Processing (18/05/2025)

- **Change**: Moved ChatGPT processing to the server alongside emotion detection
- **Features**: Both emotion detection and ChatGPT processing on the server, modular design
- **Limitations**: Streaming performance issues, continued token usage problems

### V2.3.0 - WebSocket Implementation (20/05/2025)

- **Change**: Implemented WebSocket communication to replace HTTP requests
- **Features**: WebSocket communication, significant reduction in token usage, real-time streaming
- **Limitations**: Initial implementation without emotion stability features

### V2.3.1 - Moving Average Emotion Detection (21/05/2025)

- **Change**: Added moving average algorithm for emotion detection stability
- **Features**: Moving average algorithm using 5 latest results for emotion stability
- **Limitations**: Still utilising the basic monitoring interface for the live streaming implementation without detailed visual feedback

### V2.3.2 - Enhanced Monitoring System (22/05/2025)

- **Change**: Improved monitoring interface with comprehensive visual feedback
- **Features**: Chat history, face detection bounding boxes, emotion confidence display, improved monitoring interface
  <br>**_This is the current stable version_**</br>

## System Architecture

(Insert System Architecture Diagram here)

## Setting Up

### Prerequisites

- **Jetson Nano Devkit/Orin** with Intel RealSense camera
- **Arduino** (optional, for physical feedback)
- **Python 3.8+** on Jetson
- **OpenAI API Key**

### 2. Jetson Setup

On the Jetson Nano (with Ubuntu)

```bash
# Clone the repository
git clone https://github.com/CS731-2025/cs731-2025-project-jscript.git
cd cs731-2025-project-jscript/vx.x.x.

### 3. Arduino Setup

1. Connect the ESP32 board to the Jetson Board through a USB port connection.

### 4. Run the System

**On Jetson:**

```bash
# Building the Docker image
docker build -t docker-chatbox .

# Run the Docker image
./run_docker.sh

python3 robot.py

```

**Once the robot.py is running:**

1. Chat/input into the Jetson board, to receive a GPT output


## Emotion Detection

The system recognises 7 primary emotions:

- **Happy** 😊
- **Sad** 😢
- **Angry** 😠
- **Fear** 😨
- **Surprise** 😲
- **Disgust** 🤢
- **Neutral** 😐

## ChatGPT Integration

The system uses emotion-aware prompting to generate contextually appropriate responses:

```python
# Example emotion tag usage
User Input: "I'm feeling overwhelmed"
Detected Emotion: sad (confidence: 78%)
GPT Prompt: "[sad] I'm feeling overwhelmed"
Bot Response: "[COMFORT] I understand that feeling overwhelmed can be really difficult..."
```

### Action Tags for Arduino

The bot responds with emotion tags that trigger Arduino actions:

- `[GREETING]` 👋
- `[CONFUSED]` 😕 
- `[DEFAULT]` 😐
- `[WAVE]` 🖐️
- `[POINT]` 🫵
- `[SHRUG]` 🤷
- `[ANGRY]` 😡
- `[SAD]` 😢
