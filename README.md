An Emotion-Aware Social Robot for Children with Mental Health Issues. With real-time emotion detection and a chatbot system that combines computer vision, natural language processing, and hardware integration, designed to be used for children.

## Project Overview

This project develops a comprehensive human-robot interaction (HRI) system that combines real-time emotion recognition with intelligent conversational capabilities. The system is specifically designed to support children experiencing mental health challenges, providing natural engagement through facial emotion detection and meaningful interactive experiences.

### Key Features

- **Real-time Emotion Detection**: Facial emotion recognition using EfficientNet
- **Contextual AI Responses**: ChatGPT integration with emotion-aware prompting
- **Hardware Integration**: Chatbot integrated with Arduino-controlled physical responses and feedback
- **Live Video Streaming**: WebSocket-based real-time video transmission
- **Server Hosted**: Streaming hosted on the Server through AWS Cloud
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
- **Server Infrastructure**: Google Colab backend with ngrok tunnelling
- **Text-to-Speech**: pyttsx3 with espeak engine for natural speech generation

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
- **Limitations**: High latency, rapid token exhaustion, ChatGPT still processed on Jetson

### V2.2.0 - Full Server Processing (18/05/2025)

- **Change**: Moved ChatGPT processing to server alongside emotion detection
- **Features**: Both emotion detection and ChatGPT processing on server, modular design
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

### V2.3.2.HC - Health Care Specific System (Ongoing Development)

- **Change**: Added Health Care Specific functions such as, Drug-Drug Interaction.
- **Features**: Chat history, face detection bounding boxes, emotion confidence display, improved monitoring interface, Drug-Drug Interaction Checker, Patient database through SQLite.
  <br>**_This is the current stable version_**</br>

## System Architecture

(Insert System Architecture Diagram here)

## Setting Up

### Prerequisites

- **Jetson Nano Devkit/Orin** with Intel RealSense camera
- **Arduino** (optional, for physical feedback)
- **Python 3.8+** on Jetson
- **OpenAI API Key**

### 1. Jetson Setup

On the Jetson Nano (with Ubuntu)

```bash
# Clone the repository
git clone https://github.com/CS731-2025/cs731-2025-project-jscript.git
cd cs731-2025-project-jscript/vx.x.x.

# Create an .env file
touch .env

# Input the below .env file information
```

#### .env file setup:

**_Important:_** The ngrok URL from the Colab Server Setup stage should be added to this .env file

```env

# Hardware
ARDUINO_PORT=/dev/ttyUSB0
REALSENSE_WIDTH=640
REALSENSE_HEIGHT=480
REALSENSE_FPS=30

# Performance
EMOTION_PROCESSING_INTERVAL=0.1
STREAM_FPS=30
FRAME_SKIP_RATIO=1
```

### 2. Arduino Setup

Connect the Arduino Uno board to the Jetson Board through a USB port connection.

### 3. Run the System

**On Jetson:**

```bash
# Building the Docker image
docker build -t docker-chatbot-socket .

# Run the Docker image
docker run -it --runtime nvidia --gpus all --rm docker-chatbot-socket

python3 jetson_client.py

```

**Once the jetson_client.py is running:**

1. Chat/input into the Jetson board, to receive a GPT output
2. You can access the live stream through: `https://aaaa-bb-cc-dd-e.ngrok-free.app/monitor`

## Project Structure (FIX w Versions)

```
├── emotion_understanding/          # Main application code
│   ├── model/                      # Pre-trained models
│   ├── dataset/                    # Dataset
│   ├── yolo_v8/                    # YOLO
│   └── openCV/                     # OpenCV 
│        ├── test.py                # Testing script
│        ├── train.py
│        ├── train1.py
│        ├── train_effi.py
│        ├── train_efficientnetV2.py
│        ├── train_im.py  
│        ├── train_V2.py
│        ├── train_V3.py    
│        └── train_final.py       
├── v1.0.0/                       # Version 1.0.0 release
├── v2.0.0/                       # Version 2.0.0 release
├── v2.1.0/                       # Version 2.1.0 release
├── v2.2.0/                       # Version 2.2.0 release
├── v2.3.0/                       # Version 2.3.0 release
├── v2.3.2/                       # Version 2.3.2 release (latest)
├── jetson_client.py           # Main Jetson client with WebSocket integration
│   ├── realsense_stream.py        # Optimised RealSense camera streaming
│   ├── arduino_handler.py         # Arduino communication and control
│   ├── dockerfile                 # Docker configuration
│   ├── efficientnet_opencv.pth    # Pre-trained model weights
│   ├── emotion_file.txt           # Emotion data file
│   ├── haarcascade_frontalface_default.xml  # Face detection cascade
│   ├── requirements.txt           # Python dependencies
│   └── run_docker.sh              # Docker execution script
├── .gitignore                     # Git ignore patterns
└── README.md                      # Main project documentation
```

## Emotion Detection

The system recognises 7 primary emotions:

- **Neutral** 😐
- **Happy** 😊
- **Sad** 😢
- **Angry** 😠
- **Fear** 😨
- **Surprise** 😲
- **Disgust** 🤢

## ChatGPT Integration

The system uses emotion-aware prompting to generate contextually appropriate responses:

```python
# Example emotion tag usage
User Input: "I'm feeling overwhelmed"
Detected Emotion: sad (confidence: 78%)
Drugs from this message: [Drug A, Drug B]
Extracted drugs: [Drug A, Drug B]
Checking Database: Drug A vs Drug B
Match Found: Drug A + Drug B
GPT Prompt: "[sad] I'm feeling overwhelmed"
Bot Response: "[COMFORT] Hi! Drug A and Drug B has a Major Interaction, Do not take these medications together ..."
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

## API Endpoints

### WebSocket Events

- `emotion_frame` - Send frame for emotion detection
- `stream_frame` - Send frame for live streaming
- `chat_message` - Real-time chat message broadcasting

### HTTP Endpoints

- `POST /chat` - Send chat message and get response
- `GET /health` - System health check
- `GET /stats` - Performance statistics
- `GET /live_stream` - MJPEG video stream

## Contact

For any queries or for general contact

- Jay Song  | Email: json941@aucklanduni.ac.nz
- Seth Yoo  | Email: syoo881@aucklanduni.ac.nz
- Isaac Lee | Email: mlee633@aucklanduni.ac.nz
