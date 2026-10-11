# Multi-Robot Communication System — v6.0.0

A modular server that runs several robots at once: Pepper, ChatBox, Navel,
SilBot, iRobi, and a Gazebo simulation client. It runs the **CARES lab demo**,
a guided tour where one robot guides visitors and the project robots present
their research and take questions.

The server connects **to** each robot over WebSocket, not the other way round.
It turns what a robot hears into replies from an LLM, grounded in stored
facts, and decides when to move the tour on. It records every decision and
timing in Supabase, and serves a REST API for the robot dashboard.

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
├── server/                     # the central server (run everything from here)
│   ├── app.py                  # entry point: wires every layer, starts Flask
│   ├── core/
│   │   ├── config.py           # ALL env vars + defaults; nothing else reads os.getenv
│   │   ├── profiles/           # scenario profiles: load + boot validation
│   │   └── rbac/               # access levels, filter, grants, audit
│   ├── data/                   # ONLY place that touches Supabase: one repo per table
│   │   ├── migrations/         # NNN_*.sql + matching *_down.sql
│   │   ├── transfers/          # recorded sim-to-real transfers (tools/model_transfer.py)
│   │   └── archive/            # legacy one-off scripts
│   ├── modules/                # per-robot capabilities, each a BaseModule
│   │   ├── llm/                # LLMProvider interface; Ollama (default) / OpenAI
│   │   ├── speech/             # Faster-Whisper speech-to-text
│   │   ├── emotion/            # camera emotion (EfficientNet + face detection)
│   │   └── rag/                # per-visitor conversation memory (FAISS)
│   ├── robot/                  # robot_registry, robot_instance, prompt_builder
│   ├── conversation/turn.py    # the ONE pipeline for a visitor turn (chat, speech, dashboard)
│   ├── decision/               # pure decision logic, no I/O (see "How it works")
│   ├── demo/                   # demo_orchestrator (state machine) + demo_script (the tour)
│   ├── gateway/                # HTTP routes + the WebSocket connection pool
│   ├── profiles/lab_demo.yaml  # the lab demo's scenario profile
│   ├── robot-dashboard/        # React dashboard (Vite)
│   ├── tools/                  # check_*.py checkpoints, eval / harness / stats, model_transfer
│   └── tests/                  # hermetic pytest suite
├── client/                     # the SHARED robot client (BasicClient: WebSocket server + protocol)
├── navel_client/  pepper_client/  chatbox_client/  irobi_client/
│                               # per-robot entry script + config + robot-specific I/O only;
│                               # they import ../client, so deploy client/ beside them
├── gazebo/  gazebo_client/     # simulated lab world + its bridge client
└── README.md
```

---

## How it works

```
 Dashboard ──HTTP──► gateway/http_gateway · demo_gateway · kg · flow · persona · project
                          │
                demo/demo_orchestrator ── steps, ack waits, Q&A windows, interruptions,
                          │                resume, plan revision, run clock
                conversation/turn.py ──── one visitor turn, end to end
                          │
                decision/  policy (Q&A advance / close) · kg_policy + kg_infer (topic routing)
                           planner (time budget) · grounding (facts) · presence · style_fit
                           visitor_profile · recorder (decision + correction log)
                          │
                robot/ RobotInstance per robot ── prompt_builder, history, RBAC-filtered memory
                          │
                modules/ llm · speech · rag · emotion          data/ ──► Supabase
                          │
     gateway/websocket_gateway ── the server dials OUT to each robot's WebSocket server
```

**Connecting a robot.** At start-up the robot calls `POST /robots/register`
with its id, persona (`robot_role`), gesture tags, modules, IP and WebSocket
port. When you press Connect (`POST /robots/<id>/connect`), the server opens
the WebSocket and the registry creates the robot's `RobotInstance` with only
the modules it declared.

**A visitor turn** (`conversation/turn.py`) runs off the WebSocket reader
thread, so acks and barge-ins are never blocked behind a reply being generated:
1. **Barge-in:** if the robot is speaking, `tts_stop` stops it, and a running
   demo opens a Q&A window.
2. **Q&A decision:** an advance phrase, then a lone "yes"/"okay", then the
   question heuristic, then the LLM classifier. Every decision is logged with
   the mechanism that fired.
3. **Plan revision:** "we're running out of time", "skip this", "tell me more
   about X".
4. **Routing:** the topic graph picks the robot that should answer, which may
   not be the one whose mic heard the question.
5. **Grounding:** the verified facts for that topic and its neighbours go into
   the prompt.
6. **Reply:** the LLM reply is streamed to the robot as `chat_sentence`
   events.
7. **Auto-close:** if the robot signed off, the Q&A window closes after a
   pause.

**The demo.** `/demo/start` builds the tour from the robots selected on the
dashboard. The first robot is the guide, the rest present in the order picked.
The orchestrator sends each step as `demo_step` and waits for the robot's
`ack` before moving on.

---

## Supabase schema

The base tables predate the migrations. On a fresh database, create them first:

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

Then apply `server/data/migrations/` in order. Each one has a matching
`*_down.sql`. `apply_all.sql` combines 002–012 only; apply 013 onwards one
file at a time.

| Migration | What it does |
|---|---|
| `001_projects` | `projects`, `robot_project_access` |
| `002_rbac` | `robots.access_level` / `scenario_id`; RBAC columns on `chat_logs` (`visibility`, `source_robot_id`, `subject_user_id`, `session_id`, `scenario_id`) |
| `003_rbac_audit` | `rbac_audit_log` + denial-count views |
| `004_demo_decisions` | `demo_decision_log`, `demo_correction_log` + views by step / mechanism / plan revision |
| `005_log_rls` | write policies for the log tables |
| `006_demo_kg` | `demo_topics`, `demo_topic_links`, `demo_robot_topic` + view `demo_kg_edges` |
| `007`, `009`, `010` | `demo_robot_topic.n_displaced`, `.eligible`, `.specialised` (declared scope) |
| `008_demo_durations` | `demo_step_durations`, `demo_qa_durations` + stats views |
| `011_robot_style_fit` | `demo_robot_style` + view `demo_style_fit` |
| `012_topic_facts` | `demo_topic_facts` + view `demo_topic_fact_coverage` |
| `013_archive_legacy`, `013b` | moves legacy rows into an `archive` schema (copied first; nothing dropped) |
| `014_v2_add` | **Schema v2:** `scenarios`, `sessions`, `supervisors`, `demo_runs`; `run_id` on the logs and `chat_logs`; view `log_integrity`; duration stats by mode. Log tables lose their foreign keys. |
| `015_run_format_modes` | `demo_runs.format` (`full_tour` / `solo`), more run modes; `robots.embodiment`, `sim_twin`, `retired_at` |
| `016_navel_project56` | Navel's Project 56 topics, declared scope and facts (for the twin `navel_01`) |
| `017_sim_to_real_transfer` | `model_transfers`, `duration_priors`, view `current_duration_priors`; `prior_weight` / `prior_n` / `transfer_id` columns |
| `018_v2_backfill` | fills the v2 tables and columns from data already logged (re-runnable) |
| `019_exclude_from_priors` | `demo_runs.exclude_from_priors`; the by-mode views skip flagged runs |

What the tables hold after migration 019:

| Group | Tables | Notes |
|---|---|---|
| Robots and people | `robots`, `users`, `supervisors` | `robots`: persona, tags, modules, address, `access_level`, `scenario_id`, `embodiment` (`sim` / `physical`), `sim_twin`, `retired_at` (a delete only deactivates the robot) |
| Runs | `scenarios`, `demo_runs`, `sessions` | `demo_runs.mode`: `live` (every robot physical), `gazebo`, `mock`, `scripted_visitor`, `headless_sim`, `sim_unknown`. Also `format`, `supervisor_id`, `notes`, `exclude_from_priors`. |
| Conversation | `chat_logs` | RBAC-stamped, tied to its run (`run_id`) |
| Decisions | `demo_decision_log`, `demo_correction_log` | every Q&A and plan decision, and operator corrections to them |
| Timing | `demo_step_durations`, `demo_qa_durations` | the planner prices a tour from these, using the views by mode |
| Knowledge graph | `demo_topics`, `demo_topic_links`, `demo_robot_topic` | learned `weight` / counts, `eligible`, `specialised`, sim-to-real `prior_*`. Routing reads the view `demo_kg_edges`. |
| Facts | `demo_topic_facts` | per topic and robot: `kind`, `verified`, `source`, `transfer_id` |
| Audience fit | `demo_robot_style` | how well each robot suits each visitor style |
| Sim to real | `model_transfers`, `duration_priors` | numbered copies of a twin's knowledge to its physical robot (`tools/model_transfer.py`) |
| Access and projects | `rbac_audit_log`, `projects`, `robot_project_access` | |
| Archive | `archive.robots`, `archive.users`, `archive.chat_logs` | pre-v2 rows |

The v2 tables have row-level security with no policies, so the server must use
the **`service_role` key** as `SUPABASE_KEY`. Log tables no longer have
foreign keys; the view `log_integrity` reports ids that point at missing rows.

---

## Scenario Profiles

A Scenario Profile declares, per deployment, which robots take part and what
each one's Social Identity is: role, persona, and its RBAC access level.
Profiles live in `server/profiles/*.yaml`, one per deployment, and
`core.profiles.ProfileRegistry` loads and validates them once at boot.

```yaml
scenario_id: lab_demo
description: >
  Optional free text.
separate_stations: true        # each project robot has its own station

robots:
  - id: pepper_01              # must match client_id in client_config.json
    role: "Guide"
    access_level: global       # Manager

  - id: silbot_01
    role: "Human-aware navigation researcher"
    access_level: local        # Worker
    default_visibility: global # visibility stamped on records this robot writes
    importance: 0.6            # how much the lab wants this project kept in a short tour

  - id: navel_001              # a physical robot ...
    sim_twin: navel_01         # ... and the simulation robot it was trained on
```

### Keys

| Key | Required | Values | Meaning |
|---|---|---|---|
| `scenario_id` | yes | string | Deployment identifier. Cross-robot visibility never crosses scenarios. |
| `description` | no | string | Free text. |
| `separate_stations` | no | bool | Project robots stand at separate stations, so a question for a robot that isn't nearby is deferred to its station, or the guide answers it. |
| `robots[].id` | yes | string | The robot's `client_id`. Must be unique across all profiles. |
| `robots[].role` | no | string | Persona role, also used when the guide introduces the robot. **Not** used for access decisions. |
| `robots[].access_level` | yes* | `global` \| `local` | `global` = Manager (cross-client visibility over global records in its scenario). `local` = Worker (local isolation: its own records only). |
| `robots[].default_visibility` | no | `global` \| `local` \| `restricted` | Visibility stamped on records this robot *writes*. Defaults to `local`. Does **not** widen what the robot can read. |
| `robots[].importance` | no | 0–1 | The lab's priority for this project, and the planner's first input when a tour runs short. |
| `robots[].sim_twin` | no | robot id | Marks the robot as **physical**. It inherits role, access level, visibility and importance from its twin unless it sets them. Knowledge reaches it only through a recorded transfer. |
| `robots[].persona` | no | string | Optional persona name. |

\* A robot with a `sim_twin` can inherit `access_level` instead.

`default_visibility: global` on a Worker is what gives the Manager its unified
view of the user across the team. It's a deliberate per-deployment choice. The
default is `local`, so a robot added without thought stays isolated.

### Boot validation

The server refuses to start on an invalid profile, rather than failing at the
first request. It rejects:
- an unknown `access_level` or `default_visibility`;
- a duplicate robot `id` (within or across profiles) or a duplicate `scenario_id`;
- a missing `scenario_id`, or an empty `robots` list;
- any scenario with no `global` robot.

At boot the registry also reconciles `robots.access_level` and
`robots.scenario_id` in Supabase from the profile, and mirrors embodiment and
`sim_twin`. Access hierarchies are therefore adjustable as data: edit the
profile and restart, or change the column directly. Levels are **not**
hot-swappable at runtime.

---

## Environment variables (`server/.env`)

```env
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key      # service_role: the v2 tables deny anon

# LLM (Ollama is default, OpenAI is fallback)
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_MODEL=qwen2.5:7b
OPENAI_API_KEY=                         # optional fallback
OPENAI_MODEL=gpt-4o-mini

# Speech (Whisper)
WHISPER_MODEL_SIZE=base                 # tiny | base | small | medium
WHISPER_DEVICE=auto                     # auto | cpu | cuda
WHISPER_COMPUTE_TYPE=int8
WHISPER_MAX_AUDIO_SEC=30
WHISPER_SAMPLE_RATE=16000

# Emotion
EMOTION_MODEL_PATH=./models/efficientnet_HQRAF_improved_withCon.pth
# also EMOTION_FRAME_SKIP, EMOTION_INTERVAL, EMOTION_WINDOW_SIZE,
#      EMOTION_CONFIDENCE_THRESHOLD, EMOTION_CHANGE_THRESHOLD, EMOTION_UPDATE_THRESHOLD

# RAG
RAG_EMBED_MODEL=nomic-embed-text
RAG_INDEX_DIR=./rag_indexes

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
API_TOKEN=                 # set it: required as X-API-Token on every non-GET request
DASHBOARD_HOST=            # read by the dashboard (vite.config.js): unset = localhost only, 0.0.0.0 = whole LAN
```

Robots and tools send `API_TOKEN` from their own environment, never from
`client_config.json`, which is committed. Run `export API_TOKEN=...` on each
robot machine, or set `Environment=API_TOKEN=...` in its systemd unit. The
dashboard's Vite proxy attaches it from this `.env`, so the browser never sees
it.

---

## Installation

```bash
cd server
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull nomic-embed-text           # RAG embeddings
cd robot-dashboard && npm install
```

---

## Running

```bash
# 1. Server
cd server && python3 app.py

# 2. Dashboard (http://localhost:5173 by default)
cd server/robot-dashboard && npm run dev

# 3. Each robot: its client registers itself, then press Connect on the dashboard
cd navel_client && python3 run.py          # needs ../client beside it
```

To check one layer at a time:

```bash
python3 tools/check_config.py
python3 tools/check_supabase.py
python3 tools/check_data.py        # writes real rows to the database
python3 tools/check_llm.py
python3 tools/check_speech.py
python3 tools/check_emotion.py
python3 tools/check_rag.py
python3 tools/check_robot.py
python3 tools/check_gateway.py
python3 tools/check_app.py
python3 tools/check_rbac.py        # no DB / Ollama / hardware needed
```

---

## Tests

There are two separate kinds of checks:

**`tools/check_*.py`** are checkpoint scripts, run by hand, one per layer. Most
need live infrastructure (Supabase, Ollama, robot hardware), and
`check_data.py` writes real rows to your database. `check_rbac.py` is the
exception and needs none of it.

**`tests/`** is the automated suite: 42 test files, about 1,300 tests. It's
hermetic: no network, no database, no Ollama and no `.env`. A guard fixture
fails any test that reaches for a real Supabase client.

```bash
cd server
python3 -m pytest tests/          # ~12 s
python3 -m pytest tests/ -v -m integration
```

Run from `server/`: `pytest.ini` limits collection to `tests/` on purpose. The
shared robot client has its own few tests: `cd client && python3 -m pytest test_*.py`.

Main areas covered:
- RBAC policy, grants and the FAISS top-k regression;
- delegation end to end;
- profile validation;
- the Q&A decision policy and the turn pipeline;
- the orchestrator, plan revision and the flow planner;
- topic-graph routing, feedback and inference;
- grounding, presence and style fit;
- duration trust and sim-to-real transfer.

---

## HTTP API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/`, `/health` | Server status, connected robots |
| **Robots** | | |
| `GET` | `/robots` | Registered robots; retired ones only with `?include_retired=1` |
| `POST` | `/robots/register` | A robot registers itself (role, tags, modules, address) |
| `PUT` / `DELETE` | `/robots/<id>` | Update a robot / retire it (deactivate, not delete) |
| `PUT` | `/robots/<id>/role` | Update role + allowed gesture tags |
| `POST` | `/robots/<id>/persona` | Apply a persona |
| `POST` | `/robots/<id>/connect` · `/disconnect` | Open / close the server's WebSocket to the robot |
| `GET` | `/robots/<id>/health` | Module health for a connected robot |
| `POST` | `/robots/<id>/chat` | Say something to a robot as if into its mic (`"source": "visitor"`) |
| `GET` / `POST` | `/robots/<id>/presence` | Where a robot is relative to the group (station layout) |
| **Demo** | | |
| `POST` | `/demo/start` | `robot_ids` (guide first), optional `time_budget_sec`, `visitor_style`, `visitor_interest`, `mode`, `supervisor_id` |
| `POST` | `/demo/stop` · `/pause` · `/resume` · `/next` | Run control (`/next` = skip the current step) |
| `POST` | `/demo/qa` · `/qa_end` | Open an extra Q&A window / end the current one |
| `POST` | `/demo/revise` | Change the remaining plan (skip, compress, reorder, extend Q&A) |
| `GET` | `/demo/status` | State, current step, the full step list, run clock |
| `GET` | `/demo/transcript?since=<seq>` | Everything the robots were told to say outside the script |
| `GET` | `/supervisors` | Who can be recorded as running a demo |
| **Knowledge graph** | | |
| `GET` | `/kg/graph`, `/kg/graph/<robot_id>`, `/kg/topics`, `/kg/summary` | Robot→topic competence and the vocabulary |
| `POST` | `/kg/seed` · `/kg/observe` · `/kg/reroute` | Build the vocabulary · add one observation · correct a routing |
| `GET` / `POST` | `/kg/style` | Audience style fit |
| **Planner** | | |
| `GET` | `/flow/status` · `POST /flow/plan` | The tour as a flow graph · preview a plan for a time budget |
| **Personas, projects** | | |
| `GET` / `POST` / `PUT` / `DELETE` | `/personas[/<id>]` | Persona CRUD |
| `GET` / `POST` / `PUT` / `DELETE` | `/projects[/<id>]`, `/projects/for/<robot_id>`, `/projects/<id>/access[/<robot_id>]` | Projects and which robots may use them |

Every non-GET request needs `X-API-Token` when `API_TOKEN` is set.

---

## Robot client protocol

Each robot runs a WebSocket **server** on its own port (default 8765), and the
central server connects to it. The shared implementation is
`client/client.py` (`BasicClient`). Per-robot clients extend it.

Messages the **server sends to the robot**:

```json
{ "event": "demo_step",     "step_id": "navel_01_greeting", "text": "[GREET] Hello everyone!", "require_ack": true }
{ "event": "chat_sentence", "text": "Method B plans one gesture per segment.", "emotion_tag": "DEFAULT", "turn_id": 12 }
{ "event": "tts_stop",      "turn_id": 13 }
{ "event": "demo_nav",      "step_id": "navigate_to_chatbox_01", "target": [{"x": 9.7, "y": 1.0, "frame_id": "map"}], "final_yaw": 1.57 }
{ "event": "persona_update", "persona_name": "...", "voice_config": {"...": "..."} }
{ "event": "speech_response", "transcription": "hello", "confidence": 87.3 }
{ "event": "chat_response",  "response": "[WAVE] Hello!", "emotion_tag": "WAVE", "clean_text": "Hello!" }
{ "event": "emotion_update", "emotion": "happy", "confidence": 76.2, "changed": true }
```

- `demo_step`: speak a scripted line, then reply `ack` (if `require_ack`).
- `chat_sentence`: one sentence of a streamed answer. Sentences arrive and
  must be spoken in order.
- `tts_stop`: stop speaking. Also drop any sentence whose `turn_id` is older.
- `demo_nav`: move along the waypoints, then `ack`. Only clients that declare
  `navigation` (the Gazebo bridge) move; the others ack at once.

Messages the **robot sends to the server**:

```json
{ "type": "chat",            "message": "Navel, over to you" }
{ "type": "speech",          "audio": "<base64 WAV>" }
{ "type": "ack",             "step_id": "navel_01_greeting" }
{ "type": "tts_state",       "speaking": false }
{ "type": "tts_interrupted", "step_id": "navel_01_project_approach", "remaining_text": "..." }
{ "type": "image_frame",     "frame": "<base64 JPEG>" }
```

- `chat` / `speech`: what the mic heard, either as text the robot transcribed
  itself or as audio for Whisper.
- `ack`: a step is finished.
- `tts_state`: the robot reports exactly when it is speaking, so the server
  doesn't have to estimate from word counts.
- `tts_interrupted`: what a cut-off step never said, so the server can resume
  it.

---

## Module on/off per robot

When registering a robot, set `modules` to any combination:

```json
{ "modules": ["gpt", "speech", "emotion", "rag"] }
```

| Module | What it enables |
|---|---|
| `gpt` | LLM replies (Ollama or OpenAI) |
| `speech` | Speech-to-text via Whisper |
| `emotion` | Face detection and emotion classification from camera frames |
| `rag` | Conversation memory: past context is added to prompts |
| `navigation` | The robot can actually move (Gazebo bridge). Navigation steps are only timed for these robots. |

Modules not in the list aren't initialised, and the rest of the system works
without them.
