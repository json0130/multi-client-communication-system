# Knowledge-Graph-Driven Companion Robot: R&D System Report

**Working title (paper draft):** *A Per-Person Knowledge Graph for Face-Aware, Emotion-Aware
Companion-Robot Dialogue*

**Status:** research prototype (ChatBox / ElleBot demo, `feature/integration`).
**Scope of this document:** the vision → knowledge-graph → dialogue pipeline as currently
implemented — face recognition, emotion sensing, the dual-timescale knowledge graph
(FAST / SLOW / RELATIONSHIP), retrieval-augmented prompt assembly, end-of-session knowledge
extraction, and the live visualization. This is the technical baseline for the research paper.

> Note on configuration: the PAD (Pleasure-Arousal-Dominance) persona engine and per-turn emotion→PAD
> coupling exist in the codebase but are **disabled** in the configuration reported here. The current
> pass isolates *face recognition + knowledge-graph-through-conversation*, with emotion used **only** to
> drive the FAST mood signal. Where PAD is relevant it is described but flagged as *(disabled)*.

---

## 1. Overview

The system is a socially-aware companion robot that (a) recognizes *who* it is talking to from a webcam,
(b) retrieves what it already knows about that person from a knowledge graph (KG), (c) holds a live spoken
conversation whose prompt is conditioned on that person's memory and current affect, and (d) at the end of
each session distills the conversation back into the KG so the relationship and knowledge accumulate across
meetings.

The central design commitment is a **per-person knowledge graph with an explicit timescale on every
edge**. Fast-changing affective state (mood, attention, current topic) and slow, stable knowledge
(interests, traits, relationship closeness) live in the *same* graph but are tagged `FAST` or `SLOW`
(plus a `RELATIONSHIP` band for the person↔robot bond), so downstream components can treat volatile and
durable signals differently.

### 1.1 Component map

```
                         ┌──────────────────────── Webcam frame ───────────────────────┐
                         │                                                              │
                  ┌──────▼───────┐        ┌──────────────────┐                          │
                  │ FaceIdentifier│───────▶│ EmotionDetector  │  (optional, mood only)   │
                  │ MTCNN+FaceNet │ box    │ HSEmotion (ONNX) │                          │
                  └──────┬───────┘        └────────┬─────────┘                          │
                person_id│                emotion / valence,arousal                     │
                         ▼                          ▼                                    │
                  ┌───────────────────────────────────────────────┐                     │
                  │                KNOWLEDGE GRAPH                  │                     │
                  │  InMemoryGraphStore (Pydantic v2, JSON on disk) │◀──── seed(spec) ────┘
                  │  nodes: person/robot/topic/interest/…          │
                  │  edges: FAST | SLOW | RELATIONSHIP (provenance) │
                  └───────┬───────────────────────────────┬────────┘
             retrieve     │                               │  write-back
   (interests, shared     │                               │  (mood, session turns,
    topics, notes, tier)  ▼                               │   rapport/trust, interests)
                  ┌───────────────────┐          ┌────────▼─────────────────┐
                  │  Prompt assembler  │          │  Knowledge extraction    │
                  │  IDENTITY / RULES /│          │  transcript → LLM → guard │
                  │  WHO + mood/emotion│          │  → interests + Δrapport   │
                  └─────────┬─────────┘          └──────────────────────────┘
                            ▼                                  ▲ (end of session)
                     ┌────────────┐   reply + [TAG]            │
                     │  LLM (Ollama│──────────────────────────►│  session transcript
                     │  qwen2.5:7b)│   → ESP32 face expression  │  (SessionNode.turns)
                     └────────────┘
                            │
                            ▼
                  Live visualizer (HTTP + d3 force graph, polls kg_state.json)
```

### 1.2 Software stack

| Concern | Choice |
|---|---|
| Face detection + embedding | `facenet-pytorch` — MTCNN + InceptionResnetV1 (VGGFace2) |
| Emotion | HSEmotion EfficientNet-B0 (ONNX, AffectNet) |
| LLM (dialogue, extraction, topic label) | Ollama, `qwen2.5:7b`, OpenAI-compatible API |
| Text embeddings (topic↔capability) | Ollama, `nomic-embed-text` |
| Graph model | Pydantic v2 discriminated unions; in-memory store, JSON persistence |
| Visualization | stdlib HTTP server + d3.js v7 force-directed graph |
| Vision / IO | OpenCV; optional ESP32 over TCP for robot facial expressions |

---

## 2. Face Recognition Subsystem

File: `modules/face_webcam/face_id.py`

**Pipeline.** Faces are *located* by one of two detectors (`--detector`): **opencv** (default) runs the
OpenCV Haar cascades — frontal plus profile plus profile-on-a-mirrored-image, so a turned head is still
found — and dedupes overlapping detections of one head by IoU **and** centre-containment; **mtcnn** uses
facenet-pytorch's landmark-aligned detector. Either way the crop (margin **0.20** frontal / **0.30**
profile) is embedded by InceptionResnetV1 (VGGFace2) into a **512-dimensional L2-normalized** vector, so
cosine similarity reduces to a dot product. The same box is reused for emotion — one detection feeds both.

**Identification (multi-view gallery).** Each person is stored as **several prototype views**, not one
vector: matching takes the *best* view (`retain`) and the *mean of the top two* (`acquire`, once ≥4 views)
so a single freak prototype cannot admit a stranger. A face is accepted iff `acquire ≥` **0.75**
(`threshold`) — plus a runner-up margin of 0.05 when ≥2 people are enrolled. Averaging every pose into one
centroid (the previous design) produced a blurry mean that matched no pose well; that was the root cause of
"recognition only works at one angle".

**Hysteresis.** `identify_all(..., sticky=<held id>)` keeps the identity the caller already holds down to
`retain_threshold` **0.62**, and only lets a rival take over if it clears `threshold` *and* beats the held
score by `switch_margin` **0.08**. Applied to the largest face only, so a second person in frame is never
pulled toward the held identity. The loop layers a sampled duty cycle on top: identity is voted over a
**1 s** window and then held for **3 s** (`--id-sample` / `--id-interval`), with a miss-grace of 2 windows
for hard poses — but an empty frame or a rival vote drops the label at once.

**Enrollment.** `enroll(name, frame)` adds an ENROLLED view: a frame close to an existing view (≥ `tau_dup`
**0.92**) refines it in place (weight-capped at 50), a genuinely different pose becomes a new view, and a
full tier evicts its most redundant member so the gallery keeps *spanning* poses. Guided enrollment
(`--mode enroll`) prompts for front / left / right / up / down and rejects frames that merely repeat a view
already stored — pose coverage, not frame count, is what buys accuracy.

**Adaptive capture.** During conversation the loop can learn new views of an already-known person. The
embedding is computed anyway, so this costs no extra inference. Adoption is decided once per vote window
and gated on: a confirmed identity, exactly one face in frame, unanimous votes, a quality check (box fully
in frame, ≥90 px, Laplacian sharpness ≥40), a per-person interval (20 s) and session cap (20), and a
similarity in `[adapt_floor, tau_dup)` — `adapt_floor` defaults to `threshold`, so by default it only
learns views it already recognises. Learned views live in a **separate tier** that can never displace
enrolled ones and can be wiped with `--reset-adaptive`. The worker only *queues* adoptions; the main thread
applies them, keeping a single writer on the gallery.

**Persistence.** `faces.npz` schema 2: `schema`, `names`, `counts`, `protos` (M×512, all people's views
stacked), `proto_owner` (M, index into `names`), `proto_meta` (M×4: weight, origin, created, last-hit), plus
a legacy `embeddings` (P×512) per-person centroid kept so older readers still work. All arrays are
fixed-width numeric, so it loads with `allow_pickle=False`. A schema-1 file (single averaged embedding per
person) is **migrated on load** into one enrolled view per person — no data loss, no forced re-enrollment.

**Fallbacks / robustness.** If `facenet-pytorch` is unavailable, an OpenCV Haar-cascade path degrades to a
coarse pixel-level identity. All heavy model construction happens once at start-up.

**Identity contract.** A recognized identity is a stable **string** (e.g. `"jay"`). This string is used
directly as the KG `PersonNode.id`, which is what makes face recognition and the graph interoperate with
no ID-mapping layer.

---

## 3. Emotion Recognition Subsystem

File: `modules/face_webcam/emotion_detector.py`

**Model.** HSEmotion (EfficientNet-B0, ONNX runtime, trained on AffectNet). Backends: `hsemotion`
(default, ~5 ms/frame CPU), `hsemotion-b2` (larger), `efficientnet` (PyTorch). Labels:
`{Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise}`.

**Output.** `detect(frame, box)` → `(emotion_label, confidence[0–100], valence, arousal)`, where
`valence`/`arousal` are the continuous Russell (1980) circumplex coordinates. A **5-frame sliding-window**
smoother stabilizes the reading; smoothing is **identity-aware** (one smoother per recognized person), so a
second face never contaminates the first person's emotion trace.

**Role in the current configuration.** Emotion is used **only** to drive the FAST mood signal (§6.2) and to
annotate the current dialogue turn (§9). It does *not* drive PAD in this pass.

---

## 4. The Knowledge Graph

Files: `modules/graph_relationship/schema.py`, `store.py`, `interactions.py`, `topics.py`.

The KG is the heart of the system: a typed, provenance-tracked property graph that stores, per person, both
volatile affect and durable knowledge, plus the person↔robot relationship.

### 4.1 Node types

| Node | Key fields | Purpose |
|---|---|---|
| `PersonNode` | `id`, `display_name` | one child/user; `id` is the face-recognition identity |
| `RobotNode` | `id`, `name`, `embodiment` (CAT/ELEPHANT) | one robot persona anchor |
| `TopicNode` | `id`=`topic:<slug>`, `label`, `notes[]` | a shared concept; `notes` accumulate per-person summaries |
| `InteractionNode` | `id`=`interaction:{person}:{robot}`, `rapport`, `trust`, `interaction_count` | the single abstraction of one person↔robot relationship |
| `SessionNode` | `id`, `turns[]`, `turn_count`, `extracted_turns` | one meetup's conversation transcript |
| `PersonaNode` / `RoleNode` | `descriptor` | authored persona ("introverted, shy") / role ("companion") |
| `CapabilityNode` | `id`=`{robot}:capability`, `items[]` | robot skills as one node holding a list |
| `InterestNode` | `id`=`interest:{person}:{slug}`, `label` | a person's interest area (bridges person→topic) |
| `ConversationNode` | `id`=`conversation:{person}:{robot}`, `topics[]`, `mood`, `emotion` | **live status**: rolling recent topics + current affect (§8) |

Deterministic IDs (topics, interests, interactions, conversations) guarantee that re-seeding or
re-mentioning the same concept collapses to one node rather than duplicating.

### 4.2 Edge types and the FAST / SLOW / RELATIONSHIP model

Every edge carries a `Timescale` tag. This is the paper's core representational idea: **one graph,
two clocks, plus a relationship band.**

| Band | Edge types | Semantics | Update rule |
|---|---|---|---|
| **FAST** | `mood`, `attention`, `current_topic`, `has_conversation` | affect / focus that changes within a session and decays between sessions | replace-on-newer |
| **SLOW** | `trait`, `preference`, `has_persona`, `has_role`, `has_capability`, `has_interest`, `about` | stable identity and knowledge, persists across sessions | replace-on-newer |
| **RELATIONSHIP** | `rapport`, `trust`, `interaction_count`, `has_interaction`, `has_session` | the accumulating person↔robot bond | closeness = set/delta; count **accumulates** |

- `mood` (FAST): a self-edge on the person carrying **valence ∈ [−1,1]**, plus an optional `label` (the
  emotion word) so the visualizer can show "happy (+0.55)".
- `attention` (FAST): engagement ∈ [0,1].
- `current_topic` (FAST): superseded by the `ConversationNode` (kept in schema, no longer written; see §8).
- `about` (SLOW): `Interest→Topic` (person side) or `Capability→Topic` (robot side); the robot-side edge's
  `label` records *which* capability item matched (e.g. "knows jazz").
- Closeness (`rapport`, `trust`) and `interaction_count` are **fields on the `InteractionNode`**, not
  separate person→robot edges — one hub node per relationship keeps reads O(1).

**Provenance.** Every edge carries `Provenance{source, timestamp, confidence}`. On upsert, a write whose
timestamp is *older* than the stored edge is **discarded** (stale-write protection); `interaction_count`
is the sole edge whose value **accumulates** rather than replaces.

### 4.3 Store, indexing, persistence

File: `store.py`. `GraphStore` (ABC) + `InMemoryGraphStore`.

- **Reads are O(neighbors), not O(graph):** an endpoint-type index maps `(src, dst, edge_type)` and
  `(node, edge_type)` so `query_neighbors(node, edge_type)` and `get_person_context(person)` never scan the
  whole graph. This matters because retrieval runs every turn.
- `get_person_context(person)` returns the person's attribute edges (mood/attention/trait/preference) and
  relationship edges in one indexed call.
- **Persistence** is a single JSON file (`kg_state.json`) via Pydantic `model_dump`; `save`/`load`/`reload`
  round-trip through discriminated `TypeAdapter`s. `reload` validates the whole file *before* replacing
  in-memory state, so a partially-written file never corrupts the store. The visualizer polls this file.

### 4.4 Seeding

File: `seed.py`. Authored identity is loaded from YAML specs (`specs/robot_spec.yaml`,
`specs/human_spec.yaml`): the robot's persona/role/capabilities/known-topics and any authored human
interests. Seeding is **idempotent** (deterministic IDs) and runs on start-up so a recognized person has
retrievable context from the very first turn.

---

## 5. Relationship Model (tiers)

File: `kg_bridge.py`. The person↔robot bond is summarized into a **tier** used to modulate behavior:

```
score = (rapport + trust) / 2
  score > 0.70            → "close"
  score > 0.45            → "known"
  interaction_count > 5   → "known"   (familiar by exposure, even if score is low)
  interaction_count > 0   → "visitor"
  else                    → "unknown"
```

Rapport/trust rise from (a) positive live interaction and (b) the end-of-session extraction's
`rapport_delta`/`trust_delta` (clamped, §7). The tier is a cheap, interpretable handle that (in the PAD
configuration) drives the Dominance axis of the robot's affect, and in the current configuration is
available to the prompt/logic layer.

---

## 6. From Perception to Graph (write-back)

### 6.1 Session lifecycle

On first recognition of a person in a run, the loop ensures `PersonNode`, the pair's `InteractionNode`,
and a fresh `SessionNode` exist (`get_or_create_interaction`, `start_session`). Each **conversation turn**
(`{turn, ts, emotion, child, reply}`) is appended to `SessionNode.turns` via `append_turn`. The session
transcript is therefore the ground truth that end-of-session extraction consumes.

### 6.2 Mood tick (emotion → FAST MoodEdge)

Per pipeline tick, the current emotion's **valence** is written as the person's `MoodEdge` value (with the
emotion label attached). To avoid I/O thrash from frame-to-frame jitter, disk persistence is **dirty-gated**:
a save is triggered only when the emotion *label* changes or the valence shifts by ≥ 0.15. The same tick
mirrors mood/emotion onto the live `ConversationNode` (§8) if one exists.

*(Disabled path)* In the PAD configuration, `KGBridge.pre_turn` blends valence as **0.7·camera + 0.3·graph
mood** (arousal is camera-only), derives the tier, and passes SLOW-edge text as `structured_memory`;
`post_turn` writes mood + attention and the session turn. Dominance is **re-derived every turn and never
persisted**, preventing feedback loops.

---

## 7. Knowledge Extraction (session → graph)

File: `extraction.py`. At session end (or on demand), the accumulated transcript is distilled into durable
graph updates:

1. **Format.** `format_transcript(turns)` renders `child:` / `robot:` lines.
2. **Propose.** The LLM is asked for strict JSON: a list of `interests` (`label`, `topics[]`, `summary`)
   plus `rapport_delta` and `trust_delta`.
3. **Guard.** Deterministic validation (`normalize`) never trusts the model: ≤ 6 interests, ≤ 6 topics
   each, deltas clamped to **[−0.2, +0.2]**; malformed JSON → empty update (never raises).
4. **Apply.** `apply_update` writes: `Person→Interest→(about)→Topic` (creating shared topic nodes),
   attaches the `summary` as a per-person **note** on each topic, applies the closeness deltas to the
   `InteractionNode`, and — via the matcher (§7.1) — links any robot capability that covers a mentioned
   topic (`Capability→(about)→Topic`), so *shared* topics between person and robot emerge automatically.
5. **Mark.** `mark_session_extracted` advances `extracted_turns`, so a later extraction of the same session
   processes only *new* turns.

This closes the loop: interests/notes learned in one session are retrievable (§8) in the next.

### 7.1 Embedding-based capability↔topic matching

File: `embedding.py`. Whether a robot capability "covers" a child-raised topic is decided by an injected
`Matcher`. Default is keyword overlap; the **embedding matcher** (`make_embedding_matcher` over
`nomic-embed-text` via Ollama) embeds each capability item (filler-stripped, e.g. "knows about math" →
"math") and each topic, and links the **single best** capability by cosine similarity **iff it clears a
floor** (default **0.62**). Argmax-plus-floor (rather than a global threshold) is used because short-phrase
embeddings cluster tightly; the floor was raised from 0.50→0.62 after observing false links (e.g.
*tennis*↔*"good at math"*). The matcher degrades gracefully (returns none) if the embedding backend is down.

---

## 8. Live Conversation-Status Node (FAST layer)

File: `topics.py::update_conversation`. A dedicated `ConversationNode` per person↔robot holds:

- `topics[]` — a **rolling list of the most recent topic keywords** (capped at 3, de-duplicated,
  updated **in place**),
- `mood` (valence) and `emotion` (label) — the current affect,

and is linked to *both* the person and the robot by FAST `has_conversation` edges. Each turn, a small,
disposable LLM call (`_detect_topic`) labels the utterance in 1–3 words and pushes it onto the list.

**Why a dedicated node (design decision).** An earlier version reused shared `TopicNode`s for the "current
topic". That entangled volatile chat topics with the durable interest/capability graph and let the
embedding matcher attach spurious `about` edges (the *tennis*↔*"good at math"* artifact). A `ConversationNode`
is structurally incapable of receiving capability/interest edges, so the live status stays clean and
updates in place instead of spawning a node per topic.

---

## 9. Prompt Structure (retrieval-augmented, affect-aware)

File: `webcam_loop.py::_build_system_prompt`, `_person_memory`. Each turn the system prompt is **rebuilt**
from the current graph state in three labelled blocks, then the rolling history and the (emotion-annotated)
current user turn are appended.

**Block 1 — IDENTITY** (from `RobotNode` + persona/role/capability nodes):
```
You are ChatBox, a companion robot chatting with someone through a webcam.
Personality: introverted, shy.
You can: tells stories, knows jazz, knows about space, knows about coding, good at math.
```

**Block 2 — HOW TO REPLY** (static behavioral contract; the `[TAG]` drives the ESP32 face):
```
• Keep it short, warm and spoken — one or two sentences.
• Begin every reply with an emotion tag in square brackets, e.g. [HAPPY], [CURIOUS].
• Weave in what you know about them naturally — never list it back.
• Match their mood: if they seem low or upset, be gentle and reassuring.
```

**Block 3 — WHO YOU'RE TALKING TO** (retrieved per-person memory, **capped**: top 4 interests, ≤ 3 topics
each, 3 most-recent notes one-per-topic; plus the mood line):
```
Interests: music (jazz, r&b, hiphop) · math (math problems, multiplication) · … · sports (tennis)
Common ground: jazz, math problems
Recently about them:
  – tennis: …a fan of Roger Federer but preferring Rafael Nadal.
  – hiphop: …interest in listening to hip-hop today.
  – favorite songs: …'Open Arms' by SZA.
Right now they seem positive 🙂 (mood +0.55).
```

Then the conversation:
```
{system: <blocks above>}
{user: "hi"} {assistant: "[HI] Hey there!…"}        ← rolling history (≤5 turns, stored plain)
…
{user: "i like tennis  (they appear: Happiness)"}    ← current turn, emotion-tagged
```

**Two affect signals, by design.** The **mood valence** sits in the context block (smoothed, more stable),
while the **instantaneous emotion label** is tagged on the current user turn. If a single-frame emotion read
is wrong, the mood trend can still steer the reply — and vice-versa.

**Two LLM calls per turn.** (1) the reply (`qwen2.5:7b`, `temperature=0.8`, `max_tokens=140`, last-5
history); its leading `[TAG]` is stripped and mapped to an ESP32 facial expression, and the plain
`(msg, reply)` is stored to history + transcript. (2) the disposable topic-label call (§8).

**Caching/structuring note.** IDENTITY + HOW-TO-REPLY are static; keeping them first makes the prompt
prefix stable and cache-friendly (llama.cpp KV reuse). A future refactor can hoist them into a constant
system message or an Ollama `Modelfile` so only the dynamic blocks are assembled per turn.

---

## 10. End-to-End Pipeline

**Per frame (display loop):** capture → submit to background `_DetectionWorker` (MTCNN + FaceNet [+ emotion])
→ read latest results non-blocking → render overlay. The worker processes only the newest frame (stale
frames dropped), caps at 4 faces, and downscales detection ×0.5, so detection latency never stalls the UI.

**Per tick (≈1 s):** for each recognized person — ensure session, derive tier, write the FAST `MoodEdge`
(dirty-gated save), refresh the overlay.

**Per conversation turn (user presses talk):**
1. Assemble the retrieval-augmented, affect-aware prompt (§9).
2. LLM reply → parse `[TAG]` → dispatch ESP32 expression.
3. Append the turn to `SessionNode.turns`; push to per-person history.
4. `_detect_topic` → `update_conversation` (rolling topics + mood).
5. Persist `kg_state.json`.

**Per session end (quit / hotkey):** `extract_and_apply` over the session's *conversation* turns →
interests + topic notes + capability links + rapport/trust deltas; `mark_session_extracted`; final persist.

**Across sessions:** the accumulated interests, notes, and closeness are seeded into the next session's
prompt — the system remembers people and grows the relationship.

---

## 11. Visualization

Files: `modules/graph_relationship/viz/server.py`, `index.html`. A standalone, decoupled HTTP server
**only reads** `kg_state.json` (no in-process coupling) and serves a d3.js force-directed graph that the
browser polls every second. Features relevant to analysis and demos:

- **Node shapes by type** (person circle, robot square, topic diamond, interaction octagon, session
  triangle, persona/role/capability/interest glyphs, live conversation burst).
- **Edges colored per person** (each person a distinct hue), **shaded by timescale** (FAST lighter, SLOW
  darker, RELATIONSHIP base); **all robot edges a single blue**. Ownership is inferred from the edge's
  source node (person / `interest:`/`conversation:`/`interaction:` prefix, or robot / `*:capability`).
- **Live status node** rendered as `▶ topic1 · topic2 · topic3   🙂 emotion (+0.55)`.
- **Self-edges** (mood/attention) folded onto the person's label rather than drawn as zero-length loops.
- **Obsidian-style force sliders** (repel, link length, link force, center gravity) for interactive layout.
- **Click-to-inspect** panels (session transcript, topic notes) and node/edge deletion.

---

## 12. Configuration & Reproducibility

**Run (current configuration — face + KG, emotion→mood, PAD off):**
```
ollama serve                                   # models: qwen2.5:7b, nomic-embed-text
python3 -m modules.face_webcam.webcam_loop --mode enroll --name <person>
python3 -m modules.face_webcam.webcam_loop --mode run --llm --enable-emotion
python3 -m modules.graph_relationship.viz.server --kg-path kg_state.json   # http://127.0.0.1:8765
```

**Key parameters (defaults):** face threshold 0.75; detection cap 4 faces, scale 0.5; emotion smoothing
5 frames; tick 1.0 s; embedding floor 0.62; LLM temp 0.8 / max_tokens 140 / history 5; prompt caps
4 interests, 3 topics/interest, 3 notes; extraction guards ≤6 interests, ≤6 topics, deltas ±0.2; tier
thresholds 0.70 / 0.45 / count>5.

**In-window controls:** `T` chat · `E` enroll · `B` boost closeness · `K` dump KG · `X` extract now ·
`S` save faces · `Q` quit (+extract). CLI: `--enable-pad`, `--enable-emotion`, `--no-seed`, `--no-embed`,
`--embed-model`, `--embed-floor`, `--robot {chatbox,ellebot}`, `--esp32-host`.

---

## 13. Design Decisions & Rationale (paper-relevant)

1. **One graph, explicit timescales.** Volatile affect and durable knowledge coexist but are tagged
   FAST/SLOW so retrieval and update policies differ without maintaining two stores.
2. **Face-ID string = graph node ID.** No mapping layer; recognition and memory interoperate directly.
3. **Relationship as a single hub node.** rapport/trust/count as fields on one `InteractionNode` gives
   O(1) closeness reads and a clean tier function.
4. **Retrieval-augmented, capped prompt.** Memory is injected but bounded (top-k interests + recent notes)
   so the prompt stays small as the graph grows.
5. **Dual affect signals.** Stable mood in context + instantaneous emotion on the turn hedges against
   single-frame emotion errors.
6. **Dedicated live status node.** Keeps volatile "current topic" out of the durable topic graph, avoiding
   spurious capability links.
7. **Guarded LLM extraction.** The model proposes; deterministic code disposes (caps, clamps, JSON
   fallback) — the graph is never corrupted by a bad generation.
8. **Argmax-plus-floor semantic matching.** Robust to the tight clustering of short-phrase embeddings.
9. **Decoupled visualization.** The viz only reads the JSON snapshot, keeping the graph module
   copy-pasteable and the UI unable to corrupt state.

---

## 14. Limitations & Future Work

- **Emotion↔mood are one source.** Currently both derive from the same detector; the "two-signal"
  robustness is structural, not yet from independent sensors. Multi-modal affect (prosody, text sentiment)
  would make them genuinely independent.
- **PAD persona engine is disabled** in the reported configuration; re-enabling it (tier→Dominance,
  emotion→Pleasure/Arousal, persona OCEAN→PAD) is the next integration step.
- **Topic labeling costs a second LLM call** per turn (latency). A lightweight classifier or piggy-backed
  tag could remove it.
- **Single-user tuning.** Multi-face concurrency is implemented (per-person smoothers, per-person graph
  clusters) but the dialogue loop currently drives one active interlocutor at a time.
- **Extraction is offline (end-of-session).** Incremental/interleaved extraction and forgetting/decay of
  FAST edges between sessions are open directions.
- **No formal evaluation yet.** Planned: identification accuracy under pose/lighting, retrieval relevance,
  extraction precision/recall vs. human annotation, and a user study on perceived personalization and
  relationship growth.

---

## 15. File Index (implementation pointers)

| Concern | File |
|---|---|
| Face recognition | `modules/face_webcam/face_id.py` |
| Emotion | `modules/face_webcam/emotion_detector.py` |
| Main loop / prompt / pipeline | `modules/face_webcam/webcam_loop.py` |
| Graph schema (nodes, edges, timescale) | `modules/graph_relationship/schema.py` |
| Store + persistence + indexing | `modules/graph_relationship/store.py` |
| Interactions / sessions | `modules/graph_relationship/interactions.py` |
| Topics / interests / conversation node | `modules/graph_relationship/topics.py` |
| Knowledge extraction | `modules/graph_relationship/extraction.py` |
| Embedding matcher | `modules/graph_relationship/embedding.py` |
| KG↔PAD bridge, tiers | `modules/graph_relationship/kg_bridge.py` |
| Seeding from specs | `modules/graph_relationship/seed.py`, `specs/*.yaml` |
| Visualization | `modules/graph_relationship/viz/{server.py,index.html}` |
| PAD persona engine *(disabled)* | `modules/pad_persona/*` |
