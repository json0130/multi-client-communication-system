# project56 — non-verbal interaction (Method B), parked

Navel's Project 56 co-speech gesture planner, copied in so the persona/role UI can
drive gestures on the ChatBox later. **Not wired to anything yet** — nothing in the
viz, hub or webcam loop imports it.

Source: `UoA-CARES-Student/robot-robot-interaction`, branch `feat/navel-demo-client`,
commit `49cdde5`, folder `navel-server/modules/`. Code is verbatim except the import
prefix: `modules.X` → `modules.project56.X` (so it can't clash with V5's `modules/`).

| Folder | What it does |
|---|---|
| `co_speech/` | segment the reply → time each segment → plan; `navel_executor.py` sends segments to a robot over WebSocket |
| `action_selector/` | duration + repetition filters, language scoring: which gesture fits a segment |
| `robot_profile/` | per-robot action catalogue: Navel + ChatBox (`chatbox_profile.py`, id `chatbox_jetson_001`) |
| `base.py`, `llm/llm_provider.py` | the two helpers the above import |

Left in the Navel repo: `tools/method_b_plan_service.py` (HTTP planner on :8001), the
evaluation scripts, and the other 20 test files.

Tests: `python3 -m pytest modules/project56/tests -q` → 123 passed, 1 skipped (that one
needs a 690 KB evaluation file). Needs `websockets` only for `NavelSegmentExecutor`.

ChatBox profile (`robot_profile/chatbox_profile.py`): 6 actions (DEFAULT, GREET, POINT, UNCERTAIN, SAD,
HAPPY), each mapped to a Jetson `EMOTION_MAP` tag in `CHATBOX_NATIVE_TAG` (GREET→WAVE, UNCERTAIN→CONFUSED,
HAPPY→HANDS_CLAP...). Two numbers are NOT measured and are marked as knobs in that file:
`EXPRESSION_SECONDS` (4.5, from the firmware's 5×900 ms steps) and the speech timing (Navel's values).
AGREE/DENY wait for a `NOD`/`SHAKE` entry in the client's `EMOTION_MAP` (firmware already has head_nod/head_shake).

Used by `modules/action_tag_generator.py` (branch `action_tag_generator`): reply -> one native tag per
sentence -> sent to the Jetson one `demo_step` at a time (`RobotHub.say_steps`). Enable in the viz with
`--robot-hub --action-tags`. Not wired: the Navel executor, lookahead.
