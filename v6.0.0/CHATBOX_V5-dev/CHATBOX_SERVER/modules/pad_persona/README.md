# PAD Persona Engine

A self-contained module that gives a robot persona a dynamic affective state using the **PAD (Pleasure–Arousal–Dominance)** emotion model.  The module is currently **detached** — it does not touch the live pipeline.

---

## What is PAD?

PAD is a three-dimensional emotion space defined by Russell & Mehrabian (1977):

| Dimension  | Low (−1)  | High (+1) |
|------------|-----------|-----------|
| Pleasure   | unhappy   | happy     |
| Arousal    | calm      | excited   |
| Dominance  | submissive | confident |

PAD is useful for robots because it separates *how positive* a response should feel (P) from *how energetic* it should be (A) and *how assertive* (D), giving independent knobs for language tone, gesture speed, and posture.

---

## Pipeline Overview

```
OCEAN traits (config.py)
        │
        ▼
ocean_to_pad.py  ──► baseline PAD (fixed per persona)
                                │
affect_stream.py ──► (dP, dA)  │  per-turn live offsets
                                │
relationship.py  ──► (dP,dA,dD)│  per-user tier offsets
                                │
                         pad_engine.py
                           update()
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
        to_language_descriptors()   to_gesture_params()
                    │                     │
            prompt_builder.py      arduino_output (future)
                    │
               LLM system prompt
```

**Key behaviours:**
- After each turn, PAD **decays** back toward the persona's OCEAN-derived baseline (`alpha_decay`).
- Live camera affect (`AffectStream`) shifts P and A only.
- Relationship tier shifts D primarily (and A slightly for close bonds).

---

## How to Attach It

The host pipeline needs to call the following, in order, once per conversation turn:

```python
from modules.pad_persona.config import CHATBOX_PERSONA, PADWeights
from modules.pad_persona.pad_engine import PADEngine
from modules.pad_persona.affect_stream import AffectStream
from modules.pad_persona.prompt_builder import build_system_prompt

# --- initialise once (e.g. in ChatBoxServer.__init__) ---
engine = PADEngine(CHATBOX_PERSONA, PADWeights())
stream = AffectStream()

# --- per turn (e.g. inside process_chat_message) ---
dP, dA = stream.update(valence, arousal)          # from emotion_processor output
pad    = engine.update((dP, dA), relationship_tier)  # tier from user DB / face-id
descriptors   = engine.to_language_descriptors()
gesture_params = engine.to_gesture_params()
system_prompt = build_system_prompt(
    persona_name="ChatBox",
    descriptors=descriptors,
    relationship_tier=relationship_tier,
    memory_context=rag_snippets,               # from RagModule.search()
)
# pass system_prompt to llm_processor instead of the hardcoded string
```

`emotion_processor.py` already tracks `latest_emotion` and `latest_confidence`.  Map those to a (valence, arousal) pair (e.g. happy → +0.8/+0.5, sad → −0.6/−0.3) and pass them to `stream.update()`.

---

## How to Detach It (Fallback)

If this module is absent or disabled, the host pipeline should fall back to:

```python
descriptors    = {"pleasure": "neutral", "arousal": "moderate", "dominance": "neutral"}
gesture_params = {"amplitude": 0.5, "tempo": 0.5, "posture": 0.5, "idle_freq": 0.5, "expression": 0.5}
```

And continue using the existing hardcoded system prompt in `llm_processor.py` unchanged.  No other code path is affected.

---

## Interface Contract

### Inputs (per turn)
| Name | Type | Source |
|------|------|--------|
| `valence` | `float [-1, 1]` | Mapped from `emotion_processor.latest_emotion` |
| `arousal` | `float [-1, 1]` | Mapped from `emotion_processor.latest_confidence` (scaled) |
| `relationship_tier` | `str` | User DB lookup; one of `"close"`, `"family"`, `"known"`, `"visitor"`, `"unknown"` |

### Outputs (per turn)
| Name | Type | Consumer |
|------|------|----------|
| `descriptors` | `dict[str, str]` | `prompt_builder.build_system_prompt()` → LLM |
| `gesture_params` | `dict[str, float]` | `arduino_output.send_command()` (future mapping) |
| `system_prompt` | `str` | `llm_processor.ask_model_optimized()` messages payload |

---

## Standalone Verification

```bash
# From the CHATBOX_SERVER directory:
python -m modules.pad_persona.ocean_to_pad
# Expected output:
#   chatbox    P=+0.xxx  A=+0.xxx  D=+0.xxx
#   ellebot    P=+0.xxx  A=+0.xxx  D=+0.xxx
```
