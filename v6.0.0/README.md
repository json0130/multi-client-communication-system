# Multi-Robot Communication System — v6.0.0

A modular server that manages multiple robots simultaneously. The server initiates WebSocket connections **to** robots (not the other way around), processes speech, emotion, and chat, and exposes a REST API for a web management UI.

---

## What changed from v5.x

| v5.x | v6.0.0 |
|---|---|
| Robot dials into server | Server dials out to robot |
| God-object `RobotServer` class | Lean `RobotInstance` + separate modules |
| DB queries scattered across 4 files | Single `data/` layer, one repo per table |
| Two near-identical LLM clients | One `LLMProvider` interface, two backends |
| Monitor WebSocket registered twice | Single gateway, no duplication |
| Side effects on import in `server.py` | Clean imports everywhere |
| Streaming video monitor | Removed (not needed) |

---

## Folder structure

```
v6.0.0/
│
├── app.py                   # Entry point — wires everything, starts Flask
│
├── core/
│   └── config.py            # Single source of truth for all env vars + defaults
│                            # Everything imports from here; nothing else calls os.getenv
│
├── data/                    # Database layer — ONLY place that touches Supabase
│   ├── connection.py        # Shared Supabase client singleton
│   ├── robot_repo.py        # All queries on the robots table
│   ├── user_repo.py         # All queries on the users table
│   └── chat_log_repo.py     # All queries on the chat_logs table
│
├── modules/                 # Pluggable capability modules
│   ├── base.py              # BaseModule ABC (initialize, is_available, get_status)
│   ├── llm/
│   │   ├── llm_provider.py  # LLMProvider interface + LLMResponse dataclass
│   │   ├── llm_module.py    # Auto-selects Ollama → OpenAI fallback
│   │   ├── ollama_provider.py
│   │   └── openai_provider.py
│   ├── speech/
│   │   └── speech_module.py # Faster-Whisper STT, accepts base64 WAV
│   ├── emotion/
│   │   ├── emotion_tracker.py  # Sliding-window smoothing (pure logic, no CV)
│   │   └── emotion_module.py   # EfficientNet B0 + Haar cascade face detection
│   └── rag/
│       └── rag_module.py    # FAISS index + Ollama embeddings, per-user
│
├── robot/                   # Business logic layer
│   ├── prompt_builder.py    # All LLM prompt construction (delegation + execution)
│   ├── robot_instance.py    # One instance per robot — coordinates modules
│   └── robot_registry.py   # Creates, stores, cleans up robot instances
│
├── gateway/                 # External-facing layer (no business logic)
│   ├── delegation_handler.py  # Detects JSON delegation block, routes to target robot
│   ├── websocket_gateway.py   # Connection pool — server dials OUT to robots
│   └── http_gateway.py        # Flask REST routes for web UI + management
│
└── tools/                   # Checkpoint scripts (run these to verify each layer)
    ├── check_config.py
    ├── check_supabase.py
    ├── check_data.py
    ├── check_llm.py
    ├── check_speech.py
    ├── check_emotion.py
    ├── check_rag.py
    ├── check_robot.py
    ├── check_gateway.py
    └── check_app.py
```

---

## Supabase schema

Run this in the Supabase SQL editor before first boot:

```sql
CREATE TABLE IF NOT EXISTS users (
    user_id           SERIAL PRIMARY KEY,
    name              TEXT,
    interests         TEXT[] DEFAULT '{}',
    health_conditions TEXT[] DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS robots (
    client_id    TEXT PRIMARY KEY,
    robot_name   TEXT NOT NULL,
    robot_role   TEXT DEFAULT 'You are a helpful robot.',
    is_active    BOOLEAN DEFAULT FALSE,
    allowed_tags TEXT[] DEFAULT '{[DEFAULT]}',
    modules      TEXT[] DEFAULT '{}',
    ip_address   TEXT,
    ws_port      INTEGER
);

CREATE TABLE IF NOT EXISTS chat_logs (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    message    TEXT,
    response   TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Then apply the migrations in `server/data/migrations/`, in order:

| File | Adds |
|---|---|
| `001_projects.sql` | `projects`, `robot_project_access` |
| `002_rbac.sql` | `robots.access_level` / `scenario_id`, RBAC columns on `chat_logs` |
| `003_rbac_audit.sql` | `rbac_audit_log` + denial-count views |

Each has a matching `*_down.sql`.

---

## Scenario Profiles

A Scenario Profile declares, per deployment, which robots take part and what each
one's Social Identity is — role, persona, and its RBAC access level. Profiles live
in `server/profiles/*.yaml`, one per deployment, and are loaded and validated once
at boot by `core.profiles.ProfileRegistry`.

```yaml
scenario_id: lab_demo
description: >
  Optional free text.

robots:
  - id: pepper_01              # must match client_id in client_config.json
    role: "Guide"
    access_level: global       # Manager

  - id: silbot_01
    role: "Human-aware navigation researcher"
    access_level: local        # Worker
    default_visibility: global # visibility stamped on records this robot writes
```

### Keys

| Key | Required | Values | Meaning |
|---|---|---|---|
| `scenario_id` | yes | string | Deployment identifier. Cross-robot visibility never crosses scenarios. |
| `description` | no | string | Free text. |
| `robots[].id` | yes | string | The robot's `client_id`. Must be unique across all profiles. |
| `robots[].role` | no | string | Persona role. **Not** used for access decisions. |
| `robots[].access_level` | yes | `global` \| `local` | `global` = Manager (cross-client visibility over global records in its scenario). `local` = Worker (local isolation — its own records only). |
| `robots[].default_visibility` | no | `global` \| `local` \| `restricted` | Visibility stamped on records this robot *writes*. Defaults to `local`. Does **not** widen what the robot can read. |
| `robots[].persona` | no | string | Optional persona name. |

`default_visibility: global` on a Worker is what gives the Manager its unified view
of the user across the team. It is a deliberate, per-deployment decision: the
default is `local`, so a robot added without thought stays isolated.

### Boot validation

The server refuses to start on an invalid profile, rather than failing at the first
request. It rejects: an unknown `access_level` or `default_visibility`, a duplicate
robot `id` (within or across profiles), a duplicate `scenario_id`, a missing
`scenario_id`, an empty `robots` list, and any scenario with no `global` robot.

At boot the registry also reconciles `robots.access_level` and `robots.scenario_id`
in Supabase from the profile. Access hierarchies are therefore adjustable as data —
edit the profile and restart, or change the column directly. Levels are **not**
hot-swappable at runtime.

---

## Environment variables (`.env`)

```env
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# LLM (Ollama is default, OpenAI is fallback)
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen2.5:7b
OPENAI_API_KEY=            # optional

# Speech (Whisper)
WHISPER_MODEL_SIZE=base    # tiny | base | small | medium
WHISPER_DEVICE=auto        # auto | cpu | cuda

# Emotion
EMOTION_MODEL_PATH=./models/efficientnet_HQRAF_improved_withCon.pth

# RAG
RAG_EMBED_MODEL=nomic-embed-text

# Server
SERVER_PORT=5000
```

---

## Installation

```bash
pip install flask supabase python-dotenv openai \
            faster-whisper faiss-cpu websocket-client \
            torch torchvision opencv-python pillow numpy
```

For the RAG embedding model:
```bash
ollama pull nomic-embed-text
```

---

## Running

```bash
# Verify each layer first (run from project root)
python3 tools/check_config.py
python3 tools/check_supabase.py
python3 tools/check_data.py
python3 tools/check_llm.py
python3 tools/check_speech.py
python3 tools/check_emotion.py
python3 tools/check_rag.py
python3 tools/check_robot.py
python3 tools/check_gateway.py
python3 tools/check_app.py

# Start the server
python3 app.py
```

---

## HTTP API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Server status + connected robots |
| `GET` | `/robots` | List all registered robots |
| `POST` | `/robots/register` | Register a robot (sets role, tags, address) |
| `PUT` | `/robots/<id>/role` | Update role + allowed emotion tags |
| `POST` | `/robots/<id>/connect` | Server opens WebSocket to robot |
| `POST` | `/robots/<id>/disconnect` | Close WebSocket to robot |
| `GET` | `/robots/<id>/health` | Module health for a connected robot |
| `POST` | `/robots/<id>/chat` | Send a chat message to a robot |

---

## Robot client message format

The robot runs a WebSocket **server** on its local port. The central server connects to it.

Messages the **robot sends to the server**:
```json
{ "type": "chat",        "message": "Hello" }
{ "type": "speech",      "audio": "<base64 WAV>" }
{ "type": "image_frame", "frame": "<base64 JPEG>" }
```

Messages the **server sends to the robot**:
```json
{ "event": "chat_response",  "response": "[WAVE] Hello!", "emotion_tag": "WAVE", "clean_text": "Hello!" }
{ "event": "speech_response","transcription": "hello", "confidence": 87.3, "response": "..." }
{ "event": "emotion_update", "emotion": "happy", "confidence": 76.2, "changed": true }
```

---

## Module on/off per robot

When registering a robot, set `modules` to any combination:

```json
{ "modules": ["gpt", "speech", "emotion", "rag"] }
```

| Module | What it enables |
|---|---|
| `gpt` | LLM chat responses (Ollama or OpenAI) |
| `speech` | Speech-to-text via Whisper |
| `emotion` | Face detection + emotion classification from camera frames |
| `rag` | Conversation memory, injects past context into prompts |

Modules not in the list are simply not initialised — the rest of the system is unaffected.