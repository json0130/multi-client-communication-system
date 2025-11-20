# Key Changes from v4.x

- Complete rewrite of server architecture to support multiple concurrent robot instances
- New modular client framework with pluggable I/O system
- Improved conversation memory using RAG (Retrieval Augmented Generation)
- Flexible LLM backend supporting OpenAI and OpenRouter models
- Supabase integration for persistent data storage
- Enhanced emotion processing with stability improvements

# Architecture
## Server Components
```
ServerController/
├── server_controller.py   # Main multi-client orchestrator
├── client_manager.py      # Client lifecycle management
├── request_router.py      # HTTP request routing
├── websocket_manager.py   # Real-time communication
├── server.py             # Individual robot server instances
└── database.py          # Supabase data persistence
```

## Core Modules
```
Modules/
├── emotion_processor.py   # Emotion detection & analysis
├── gpt_client.py         # LLM integration 
├── rag_module.py         # Conversation memory
└── speech_processor.py   # Speech recognition/synthesis
```

## Client Framework
```
client/
├── InputModules/         # Sensor inputs
├── OutputModules/        # Actuator outputs
├── client.py            # Base client classes
└── robot.py            # Main robot implementation
```

# Main Features
## Multi-Client Support
- Each robot gets its own server instance
- Isolated conversation and emotion context
- Individual module configuration

## RAG-Enhanced Memory
- FAISS vector store for conversation indexing
- Per-user conversation history
- Semantic search for relevant context

## Flexible LLM Integration
- Support for multiple LLM providers
- Configurable models via client config
- Improved prompt engineering

## Modular I/O System
- Pluggable input/output modules
- Hot-reload capability
- Hardware abstraction layer

# Usage
Start Server
```python
cd v5.0.0/ServerController
python server_controller.py
```

Start Client
```python
cd v5.0.0/client
./run_docker.sh
python robot.py
```

Endpoints
- /register_client - Robot registration
- /client/<id>/chat - Per-client chat
- /client/<id>/stream - Client video stream
WebSocket Events
- client_connected - New client connection
- emotion_update - Emotion state change
- chat_message - Chat interaction

## Requirements
- Python 3.10+
- Docker
- NVIDIA GPU (optional)
- Intel RealSense camera (optional)
