# PAD Adapter — Integration Notes

This document says exactly where to cut the existing pipeline and what to paste in.
No existing file has been modified yet; this is the recipe for the wiring PR.

---

## 1. Where to call `process_turn()`

**File:** `CHATBOX_SERVER/chatbox_server.py`  
**Function:** `ChatBoxServer.process_chat_message()` — line 248  
**Insert after:** line 275 (`my_allowed_tags = self.client_info.get('allowed_tags', [])`)  
**Insert before:** line 279 (the `ask_model_optimized` call)

```python
# --- PAD persona turn (insert after line 275) ---
_valence, _arousal = self.pad_adapter.emotion_label_to_va(self.latest_emotion)
_pad_result = self.pad_adapter.process_turn(
    valence=_valence,
    arousal=_arousal,
    relationship_tier=self.client_info.get("relationship_tier", "unknown"),
    memory_context="\n".join(context_texts[:3]) if context_texts else "",
)
_pad_system_prompt = _pad_result["system_prompt"]
# -------------------------------------------------
```

Then change the LLM call at line 279 so `ask_model_optimized` can accept an
override prompt.  Two sub-steps:

**Sub-step A** — add one parameter to `OllamaClient.ask_model_optimized` in
`Modules/llm_processor.py:42`:

```python
# Before (line 42):
def ask_model_optimized(self, message, user_emotion="neutral",
                        confidence=0.0, allowed_tags=None):

# After:
def ask_model_optimized(self, message, user_emotion="neutral",
                        confidence=0.0, allowed_tags=None,
                        system_prompt_override: str | None = None):
```

**Sub-step B** — replace line 53 (the `system_prompt = (` assignment) with:

```python
system_prompt = system_prompt_override or (
    f"You are CHAT BOX, a gentle …"
    # rest of the existing f-string unchanged
)
```

**Sub-step C** — update the call site at line 279:

```python
response_text = self.gpt_client.ask_model_optimized(
    enhanced_message,
    self.latest_emotion,
    self.latest_confidence,
    allowed_tags=my_allowed_tags,
    system_prompt_override=_pad_system_prompt,   # ← add this line
)
```

---

## 2. Where to call `enrich_hardware_command()`

**File:** `CHATBOX_CLIENT/robot.py`  
**Function:** `SimpleConcurrentClient.send_robot_emotion()` — line 89  
**Current code (lines 91–93):**

```python
command = EMOTION_MAP.get(emotion.upper(), "default")
if self.arduino_module and self.arduino_module.is_connected():
    return self.arduino_module.send_command(command)
```

**Replace with:**

```python
command = EMOTION_MAP.get(emotion.upper(), "default")
# PAD enrichment: gesture_params are forwarded for future firmware use;
# current firmware only uses the plain 'command' string.
enriched = self.pad_adapter.enrich_hardware_command(command, self._last_gesture_params)
if self.arduino_module and self.arduino_module.is_connected():
    return self.arduino_module.send_command(enriched["command"])
```

`self._last_gesture_params` should be set on the client after each turn.
The server can forward `gesture_params` inside the `chat_response` Socket.IO
payload — add it to the result dict in `process_chat_message()`:

```python
# In process_chat_message() result dict (line 306):
result = {
    ...
    "gesture_params": _pad_result["gesture_params"],   # ← add
}
```

Then in `robot.py:on_chat_response()` (line 170), cache it:

```python
self._last_gesture_params = data.get("gesture_params", {})
```

---

## 3. Where the camera/emotion update feeds in

**File:** `CHATBOX_SERVER/chatbox_server.py`  
**Function:** `ChatBoxServer.process_image_frame()` — line 366  
**Current code (lines 387–390):**

```python
emotion, confidence, status = self.emotion_processor.process_emotion_detection_realtime(frame)
self.latest_emotion = emotion
self.latest_confidence = confidence
self.last_update_time = time.time()
```

No change needed here. The PAD adapter reads `self.latest_emotion` at each
`process_turn()` call (in `process_chat_message`), so the camera loop already
feeds the adapter indirectly via the server's state fields.

When a dedicated V-A regression head is available, replace the
`emotion_label_to_va()` lookup with the model's direct output and pass the
`(valence, arousal)` floats straight into `process_turn()`.

---

## 4. Import + instantiation snippet

Paste these two lines into `ChatBoxServer.__init__()` (after line 66):

```python
from modules.pad_persona.pipeline_adapter import PADPipelineAdapter
self.pad_adapter = PADPipelineAdapter(
    robot_id=client_config.get("robot_id", "chatbox")
)
```

Or, to defer until `initialize_with_config()` has the config available
(recommended — add after line 98 where `self.client_info` is set):

```python
from modules.pad_persona.pipeline_adapter import PADPipelineAdapter
robot_id = client_config.get("robot_id", "chatbox")
self.pad_adapter = PADPipelineAdapter(robot_id=robot_id)
```

---

## 5. One-line swap to disable the module

Replace the instantiation line with:

```python
from modules.pad_persona.pipeline_adapter import NullPADAdapter
self.pad_adapter = NullPADAdapter()
```

`NullPADAdapter` has an identical interface; `system_prompt` will be `None`,
which causes the existing hardcoded prompt in `llm_processor.py` to be used
unchanged.  No other code path is affected.

---

## 6. Logging to add

Add these lines after the `process_turn()` call in `process_chat_message()`
to track PAD state over a session:

```python
p, a, d = _pad_result["pad_state"]
print(
    f"[PAD] turn={len(self.gpt_client.history)//2:03d}  "
    f"emotion={self.latest_emotion:<8s}  tier={_tier:<8s}  "
    f"P={p:+.3f} A={a:+.3f} D={d:+.3f}  "
    f"→ {_pad_result['descriptors']['pleasure']}/"
    f"{_pad_result['descriptors']['arousal']}/"
    f"{_pad_result['descriptors']['dominance']}"
)
```

For production, route this through `logging.getLogger(__name__).debug(...)`.
The log line format is designed to be grep-able: `grep '\[PAD\]' server.log`
gives the full per-turn affect trajectory.

---

## Summary table

| What                   | File                     | Function                    | Line  | Action                      |
|------------------------|--------------------------|-----------------------------|-------|-----------------------------|
| Instantiate adapter    | `chatbox_server.py`      | `initialize_with_config()`  | ~98   | Paste import + init snippet |
| Call `process_turn()`  | `chatbox_server.py`      | `process_chat_message()`    | ~275  | Insert after `my_allowed_tags` |
| Pass prompt to LLM     | `Modules/llm_processor.py` | `ask_model_optimized()`   | 42/53 | Add `system_prompt_override` param |
| Forward gesture_params | `chatbox_server.py`      | `process_chat_message()`    | ~306  | Add key to result dict      |
| Cache gesture_params   | `robot.py`               | `on_chat_response()`        | ~170  | Store `data["gesture_params"]` |
| Enrich hw command      | `robot.py`               | `send_robot_emotion()`      | 91    | Wrap `send_command()` call  |
| Camera feeds adapter   | `chatbox_server.py`      | `process_image_frame()`     | 387   | No change; indirect via state |
