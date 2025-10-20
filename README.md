# Role-Adaptive Multi-Robot Communication Framework

This repository presents the **Role-Adaptive Communication Framework** for multi-robot systems powered by **Large Language Models (LLMs)**. The framework enables heterogeneous robots to communicate adaptively according to their assigned roles, personalities, and interaction contexts. It integrates **Retrieval-Augmented Generation (RAG)** for contextual memory, **adaptive persona configuration** via system prompts, and a **centralised orchestration architecture** for efficient coordination and scalability.

Developed as part of the University of Auckland Part IV Research Project (2025), this system combines emotion-aware communication, dynamic LLM configuration, and robot-to-robot (RRI) task delegation across multiple platforms.

---

## 1. Project Overview

The **Role-Adaptive Multi-Robot Framework** represents an evolution from a one-to-one adaptive learning system to a fully integrated **multi-agent communication system**.  

It allows each robot to operate under a **unique persona and functional configuration**, while maintaining shared context through a centralised server.  
The system supports both **Human-Robot Interaction (HRI)** and **Robot-Robot Interaction (RRI)** for cooperative operations across heterogeneous platforms.

### Key Capabilities

- **Centralised Orchestration Server** for multi-robot management  
- **Role-Adaptive Communication** powered by modular LLM integration  
- **Persona Assignment** through detailed JSON-based configuration files  
- **Dynamic Task Delegation** between heterogeneous robot agents  
- **WebSocket Communication** for low-latency, bidirectional streaming  
- **RAG-based Memory System** for personalised and context-aware interactions  
- **Scalable Client Architecture** supporting simultaneous robot connections  

---

## 2. System Architecture

The system consists of a **Centralised Server** and multiple **Robot Clients**, communicating via asynchronous WebSockets.  
Each robot registers to the server using a JSON configuration file that defines its role, LLM model, and modules (e.g., emotion recognition, speech, face recognition).

### Core Components

| Layer | Description |
|-------|--------------|
| **Central Server** | Hosts orchestration logic, LLM APIs, RAG memory, and module routing. |
| **Robot Client** | Connects to the server with a JSON-based configuration defining its functional modules and personality. |
| **RAG Database** | Stores conversation history and key user information for context retrieval. |
| **LLM Integration** | Supports ChatGPT, Gemini, DeepSeek, and Llama for flexible persona configurations. |
| **RRI Module** | Handles robot-to-robot task delegation and message routing. |

### Communication Flow

1. Each robot loads its configuration (`robot_config.json`) and connects to the server.  
2. The server assigns a session and activates modules (LLM, emotion, STT, etc.).  
3. Robots communicate through real-time WebSocket channels.  
4. Robots may exchange context and tasks via RRI through the central server.  

---

## 3. Hardware Platforms

| Platform | Description |
|-----------|-------------|
| **CHATBOX** | Compact companion robot with 12 DOF and emotion-aware interaction capabilities. |
| **Silbot** | Mobile humanoid robot used for navigation and guiding roles, with expressive arm gestures. |
| **IrobiQ** | Healthcare-focused service robot for elderly care and chronic disease management. |
| **Pepper** | Semi-humanoid social robot with 20 DOF for customer service and education tasks. |
| **EveR-4 H22** | High-fidelity humanoid robotic head capable of advanced emotional expression. |

---

## 4. Software Architecture

### 4.1. Server Modules

- **LLM Orchestrator** – routes queries to appropriate LLMs (GPT, Gemini, DeepSeek, Llama)
- **RAG Engine** – retrieves relevant information from stored conversations
- **Emotion Recognition** – processes facial input for emotional context
- **Speech Modules** – includes Whisper STT engines
- **RRI Handler** – manages robot-to-robot task exchanges and message routing

### 4.2. Client Configuration

Each robot client is defined by a JSON configuration file specifying:
- Role and personality prompt
- Functional modules
- LLM model selection
- Hardware settings (e.g., Arduino, camera, ports)

Example:

```json
{
  "robot_name": "CHATBOX",
  "client_id": "robot_01",
  "server_url": "ws://<server-ip>:5000",
  "modules": ["gpt", "emotion", "speech"],
  "robot_role": "You are ChatBox, a compassionate companion robot for children. You respond with empathy and encouragement.",
  " llm_config ": {
    "provider": " openrouter " ,
    "model": " openrouter - model - name - here !" ,
    "temperature": 0.7 ,
    "max_tokens": 1000
  },
  "hardware_configuration": {
    "arduino_port": "/dev/ttyUSB0",
    "arduino_baud": 115200,
    "arduino_timeout": 2.0
  }
}
```

### 5. Role-Adaptive Communication

Each robot client exhibits distinct communication characteristics derived from its **role definition** and **personality prompt** within the configuration file.  
This ensures that the robot’s dialogue, tone, and behavioural output align naturally with its assigned function in the environment.

### Communication Personalities

| Role | Communication Style | Description |
|------|----------------------|-------------|
| **Receptionist Robot** | Polite, informative, and concise | Welcomes users, provides guidance, and delegates navigation tasks. |
| **Guide Robot** | Friendly, directive, and confident | Physically escorts users, provides directional support. |
| **Companion Robot** | Warm, empathetic, and nurturing | Engages in supportive dialogue with emotional awareness. |

The system integrates both **linguistic adaptation** and **contextual consistency**, using LLM configuration parameters such as temperature, top-p, and role-specific system prompts.

---

## 6. Robot-to-Robot Communication

The **Robot-to-Robot Interaction (RRI)** protocol enables contextual and cooperative task management between heterogeneous robots.  
Messages are exchanged via the **central WebSocket server**, ensuring synchronised task flow and shared dialogue context.

### Key Features

- **Task Delegation:** Transfers a user’s request to the most suitable robot.
- **Context Retention:** Maintains dialogue state and emotional context across robots.
- **Verbal Coordination:** Expresses transitions through human-understandable speech.

### Example Dialogue

Receptionist Robot: “The nearest restroom is at the end of the hallway. Silbot can guide you there.”
Receptionist Robot: “Silbot, please escort the user to the restroom.”
Silbot: “Certainly! Please follow me.”

This interaction flow demonstrates seamless task delegation between role-specific robots while preserving conversational context.

---

## 7. System Flow Diagram




## 8. Project Structure
```bash
multi-client-communication-system/
├── Modules/
│ ├── gpt_client.py
│ ├── rag_module.py 
│ ├── emotion_processor.py
│ └── speech_processor.py 
│
├── ServerController/
│ ├── client_manager.py
│ ├── database.py 
│ ├── supabase_client.py 
│ ├── request_router.py 
│ ├── server_controller.py 
│ ├── server.py 
│ └── websocket_manager.py
│
├── models/
│ └── emotion_recognition.pth
│
├── client/
│ ├── InputModules/
| │ ├── camera_input.py 
| │ ├── realsense_input.py 
| │ ├── text_input.py
| │ └── voice_input.py 
| |
│ ├── OutputModules/
| │ ├── arduino_output.py 
| │ ├── console_output.py
| │ ├── edge_tts_output.py
| │ └── tts_output.py
| |
│ ├── dockerfile 
│ ├── .env 
│ ├── client_config.json 
│ ├── client.py 
│ ├── robot.py 
│ ├── run_docker.sh
│ └── requirements.txt
│
└── README.md # Project documentation

```
---

## 9. Academic Significance

The **Role-Adaptive Communication Framework** contributes to the field of **Human–Robot Interaction (HRI)** and **Multi-Agent Coordination** by demonstrating:

- Scalable and modular design for multi-robot collaboration  
- Adaptive LLM-driven dialogue aligned with robot roles and personalities  
- Persistent contextual awareness through RAG-based long-term memory  
- Seamless robot-to-robot communication for cooperative task execution  

This research advances multi-agent conversational systems by introducing adaptive, personality-based communication across heterogeneous service robots.

### Achievements
- 🏆 **Best Context Design Award** – IEEE RO-MAN 2025 (Amsterdam, Netherlands)  
- 🏆 **Innovative Solution Award** – ICSR 2025 (Naples, Italy)

---

## 10. Setup Instructions

### 1. Server Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:JaySong/multi-client-communication-system.git
   cd multi-client-communication-system

3. Run the server_controller:
   ```bash
   conda create --name multi-robot-server python=3.10
   conda activate multi-robot-server
   cd v5.0.0
   pip install -r requirement.txt
   cd ServerController
   python server_controller.py

2. Build and run the Docker container:
   ```bash
   docker build -t multi-robot-server .
   ./run_docker.sh
   python3 robot.py


4. Once the container is running, the central orchestration server will automatically initialise:

- WebSocket listener on port 5000

- RAG memory module (FAISS + MongoDB backend)

- LLM manager (OpenAI, Gemini, DeepSeek, Llama)

- RRI communication handler for inter-robot coordination

4. The server console will display connection logs for each connected client, confirming successful session registration.


### 2. Robot-to-Robot Demonstration

The Robot-to-Robot Interaction (RRI) protocol allows different robots to coordinate naturally through the central server.
This enables smooth task delegation and consistent conversation flow between robots of different roles.

Interact with CHATBOX (the receptionist robot):

Ask: “Where is the nearest restroom?”

CHATBOX will respond with:

“The restroom is located at the end of the hallway. Silbot can guide you there.”


CHATBOX delegates the task via RRI:

“Silbot, please escort the user to the restroom.”


SILBOT acknowledges and executes the action:

“Certainly! Please follow me.”


This example demonstrates adaptive, role-consistent collaboration between robots within a shared communication environment.
