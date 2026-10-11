# `feat/navel-demo`: the Navel solo lab demo

This branch is **`feat/server-refactor`** (Schema v2 + the server refactor,
merged in) **plus the Navel solo demo**. It's the branch to run for Navel's
Display Day demo. It stays separate from `main` and from `feat/server-refactor`;
nothing goes to `main` before live testing.

`v6.0.0/README.md` describes the server in general. This file covers only what
this branch adds: 7 commits, 18 files.

---

## The demo

A **human** is the lab guide. Navel presents **Project 56, Human-Comprehensible
Robot-to-Robot Communication via Co-Speech Motion Generation**, on its own.

| # | Step | Who | What happens |
|---|---|---|---|
| 1 | Wait for the handoff | the guide talks | The demo state is `awaiting_cue`. Navel hears everything but doesn't reply, and nothing it hears counts as an interruption. |
| 2 | Handoff | the guide | "Navel, over to you", "can you explain your project?", "Navel, go ahead"… or **Skip** on the dashboard |
| 3 | Greeting | Navel | Answers the actual handoff words ("Of course! Hi everyone…"), with the `[GREET]` gesture |
| 4–6 | Project talk | Navel | Three separate points: **problem → approach → impact**, each grounded on verified facts. An interruption doesn't lose the points not yet covered. |
| 7 | Q&A | visitors and the guide | Normal Q&A rules. It ends with **Move On**, "okay thanks", "no more questions", or Navel signing off. |
| 8 | Wrap-up | Navel | Thanks everyone and hands back to the guide. The demo is `completed`; there's no open floor. |

**To run it:** on the dashboard, press **Start**, select **only Navel**, and
press Start Demo. Selecting exactly one robot is what starts the solo script.

---

## What this branch adds

### Solo script and orchestrator (`demo/`)
- **`build_solo_script(robot_id)`** (`demo_script.py`) builds the 8 steps
  above. Every step is spoken by Navel, with no walking or turning. The
  greeting is a `HANDOFF` step, so time-budget compression never drops it.
  The project points come from the shared `PROJECT_CHECKLIST`.
- **`DemoOrchestrator.start()`:** when exactly one robot is selected, it runs
  the solo script.
- **Cue step.** A new `step_kind="cue"` and state **`awaiting_cue`**, with no
  timeout. `cue_received(text)` ends the wait, and so does `/demo/next` (Skip),
  without also skipping the later Q&A. The guide's exact words are added to the
  greeting instruction.
- **Grounded project talk.** `facts_lookup` adds the robot's top 12 verified
  facts to every `PROJECT` step (`app.py` → `facts_for_robot` + `format_facts`,
  the same rows Q&A answers use). Other steps are worded as before, and a failed
  lookup never stops the talk.

### Handoff detection (`decision/policy.py`, `robot/robot_instance.py`, gateways)
1. **The phrase list** `HANDOFF_CUE_PHRASES` is checked first and is instant.
   Every phrase is *addressed to* the robot: "over to you", "take it away",
   "your project", "can you explain", "let's hear from", "what are you working
   on", and so on. A bare "explain" or "tell us about" would also fire on the
   guide's own intro.
2. **If no phrase matches**, `RobotInstance.classify_handoff(text)` asks the LLM
   whether the guide is speaking *to* the robot and inviting it to present. It
   answers `handoff` or `wait`.
3. **While waiting**, every input path checks for the handoff and nothing else:
   WebSocket `chat`, WebSocket `speech` (transcribed only, with no reply
   generated) and the dashboard's HTTP chat. Everything else gets no reply.

### Q&A changes
- **The "done/more" classifier.** `classify_qa_intent` used to have the LLM
  answer `continue` for "more questions", while the question it shows the LLM
  ends "…or shall we **continue** the demonstration?". So "okay thanks" came
  back `continue` and the window never closed. The labels are now `done` and
  `more`. The method still returns `done`/`continue`, and anything other than
  `done` stays in the Q&A.
- **A solo robot's answers are judged.** In the full tour, the "shall we move
  on?" judgement never looks at the guide's own answers. A solo robot is both
  guide and presenter, so `check_qa_auto_close` does judge its answers. The
  full tour is unchanged.

### Navel's knowledge and gestures
- **Persona:** `robot_role` in all three Navel configs (`navel_client/`,
  `client/test_configs/navel_01.json`, `gazebo_client/configs/navel_01_sim.json`)
  describes Project 56. It must never claim people have been shown to
  understand robots better, because the human study hasn't been done.
- **Profile:** `navel_01`'s role is "Co-speech motion researcher", and
  `navel_001` inherits it as the physical twin.
- **Facts:** migration `016_navel_project56` holds the topics, declared scope
  and 26 facts for `navel_01`, plus a note that Method B doesn't run inside
  this server. The physical **`navel_001` received them through sim-to-real
  transfer #1** (`data/transfers/navel_01__navel_001__v1.json`: 27 facts, 7
  topics). The old emotion-aware drafts are removed from `tools/seed_topic_facts.py`.
- **Gestures:** all three Navel configs list Navel's **11 gestures** as
  `allowed_tags`: DEFAULT, POINT, EMPHASISE, AGREE, LISTEN, UNCERTAIN, DENY,
  GREET, HAPPY, SURPRISE, SAD. The solo greeting asks for `[GREET]`, since
  Navel has no `[WAVE]`.

### Dashboard (`robot-dashboard/src/components/DemoTab.jsx`)
The `awaiting_cue` state gets its own colour, and Start is disabled while the
demo waits.

### Robot ids
| id | What |
|---|---|
| `navel_01` | the simulation twin (mock / Gazebo clients) |
| `navel_001` | the physical Navel, `sim_twin: navel_01` in `profiles/lab_demo.yaml` |

A solo run on `navel_001` is recorded as `mode=live, format=solo`; on `navel_01`
it's `mock` (or `gazebo`).

---

## Running it with the real Navel

The Navel side lives in **`robot-robot-interaction`**, branch
**`feat/navel-demo-client`**:
- `navel-client/` speaks this server's protocol;
- `navel-server/tools/method_b_plan_service.py` is the Method B planner.

On the robot, the client is deployed to `~/control-studio/scripts/navel-client-demo/`.

```bash
# 1. This PC: central server
cd v6.0.0/server && python3 app.py

# 2. This PC: Method B planner (wait for "Warm-up plan in …")
cd robot-robot-interaction/navel-server && python3 tools/method_b_plan_service.py --port 8001

# 3. Navel
cd ~/control-studio/scripts/navel-client-demo && python3 robot.py
```

Then **Connect** `navel_001` on the dashboard, and **Start** with only Navel
selected.

**How the pieces fit:**
- The server and Navel talk exactly as described in the main README (Navel
  registers, the server dials out to it on :8765).
- Navel runs every line it is given through **Method B**: it posts the text to
  the planner and gets back segments, each with a gesture, a predicted duration
  and a DEFAULT arm variant. It speaks and gestures each segment, and sends the
  `ack` only after the last one.
- If the planner can't be reached, Navel falls back to one gesture per reply.

**Network (current lab setup):**
- **This PC** is `192.168.0.102`: the server is on `:5000` and the planner on `:8001`.
- **Navel** is `192.168.0.10:8765`. Its config uses `ip_address: auto`, so it
  registers whatever address reaches the server.
- **If the PC's address changes**, update `server_url` and `method_b_planner.url`
  in `navel-client/client_config.json`.

**In Navel's log, look for:**
- `ID : navel_001`, `auto-detected 192.168.0.10`, `Registered with server`;
- `Method B: http://…:8001`;
- per line: `[MethodB] planned N segment(s) in 0.2Xs: [...]`, then `[MethodB] S1 …`.

---

## Tests

```bash
cd v6.0.0/server && python3 -m pytest tests/      # 1348 pass on this branch
```

| File | Tests | Covers |
|---|---|---|
| `test_navel_solo.py` | 34 | the solo script's shape, the handoff phrase list and LLM fallback, the waiting state (no reply, no interruption), the solo Q&A judgement, the grounded talk, migration 016's facts, one persona across the configs, gestures within Navel's 11 |
| `test_navel_connection.py` | 5 | the whole solo demo **over a real WebSocket** to the shared client: intro ignored, handoff, every step acked in order, a streamed answer, an interruption, completion |
| `test_qa_classifier.py` | 13 | the `done/more` labels and the `classify_handoff` parsing |

---

## Measured on the hardware and models used

| What | Result |
|---|---|
| Handoff fallback (`qwen2.5:7b`, 3 runs each) | unlisted handoffs start Navel 27/30; intro lines about Navel never start it (36/36) |
| Q&A "done" classifier | done-replies close the window 11/15 (was 1/15); real questions stay open 15/15 |
| Method B planner latency | ~0.2–0.5 s per text (warm); measured from Navel, 0.22 s round trip |
| First live run on Navel (29 Sep) | registration, connect, handoff, scripted steps + acks, STT, interruption and Q&A all worked |

## Known limitations

- **Mostly DEFAULT gestures.** GREET (3.6 s), POINT (4.4 s) and EMPHASISE
  (6.0 s) are longer than most demo sentences, so Method B's duration filter
  removes them. Where they do fit, the V1 scorer rarely picks them, which is
  the bias Project 56 documents.
- **Repetition is filtered within one reply only.** Streamed Q&A sentences are
  planned one by one.
- **A sentence already playing finishes**, because the Navel SDK has no call to
  stop speech. A barge-in drops only what's queued.
- **The default distance-follow moves Navel's base** toward and away from
  people (`gesture_config` in the Navel client). Consider turning it off with
  visitors standing close.

## Related

- **`robot-robot-interaction/lab-demo-server/`** is a standalone copy of this
  branch's Navel demo path, a byte-identical copy of 53 files from commit
  `7780ef4`, running on a JSON snapshot instead of Supabase. Changes to the
  demo, Q&A or handoff code here need copying there; its README has the steps.

## Commits (on top of `feat/server-refactor`)

| Commit | |
|---|---|
| `8d14087` | Navel solo demo with a human guide |
| `98d542f` | profile: `silbot_01` declares `embodiment: sim` |
| `503412f` | Project 56 persona and grounded project talk |
| `0aedd38` | Q&A classifier label no longer collides with "shall we continue?" |
| `f7fe6f4` | any natural handoff starts the solo robot |
| `7b80d67` | test: solo demo over a real WebSocket |
| `7780ef4` | Navel's 11 gestures as allowed tags; greet with `[GREET]` |
