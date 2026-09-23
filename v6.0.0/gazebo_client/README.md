# gazebo_client

Bridges the 4 simulated TurtleBot3 robots in `../gazebo/lab_world` into the
v6.0.0 demo server as normal robot clients, using the same `BasicClient` base
every real robot (Pepper, ChatBox, Navel, Silbot) uses — the server can't
tell the difference between a simulated and a real robot registering as
`pepper_01`.

Mapping: `robot1` (Waffle, guide) -> `pepper_01`, `robot2` (Burger) ->
`chatbox_01`, `robot3` (Burger) -> `navel_01`, `robot4` (Burger) ->
`silbot_01`.

`robot4`'s spawn pose in `gazebo/launch_demo_env.launch.py` (`x=-3.0,
y=-5.0`) is a placeholder — chosen without being able to see the running
simulation, so it isn't verified against the wall layout. Nudge the `-x`/`-y`
args on `spawn_robot4` if it lands inside a wall or somewhere awkward.

Each bridge process:
- Subscribes to `<namespace>/odom` and posts `{x, y, frame_id}` to
  `POST /robots/<client_id>/presence` once a second — feeds the server's
  proximity-based presence tracking with real simulated poses.
- "Speaks" `demo_step`/`chat_response` text via `SimTTSOutputModule`
  (console output, timed by word count — no real audio).
- Accepts typed input via `TextInputModule` as simulated chat/questions.
- Drives to a target pose on a `demo_nav` event (only `pepper_01`'s bridge
  ever receives one, since it's the only guide/mobile role in the demo
  script) — a closed-loop turn-then-drive controller using live odom, and
  reports arrival with the same `send_ack()` a robot already uses after
  finishing a spoken step. See "Navigation steps" below.

`gazebo/approach_demo.py` is a separate, standalone open-loop script for
manually smoke-testing movement outside the demo flow — it's not part of
the demo run and shouldn't be run at the same time as the bridges (both
would fight over `/robot1/cmd_vel`).

## Navigation steps

There are two ways a run gets its script, and navigation had to be wired
into both:
- The **static** `DEMO_STEPS` list in `server/demo/demo_script.py`, used
  when `/demo/start` is called with no `robot_ids`.
- The **dynamic** `build_script()`, used when `robot_ids` IS passed — which
  is what the dashboard actually does (see `robot-dashboard/src/api.js`),
  so this is the path a normal dashboard-started demo takes.

Both now insert a `DemoStep(step_kind="navigation", nav_target=...)` for
each project robot right **before** that robot's `introduce_*` hand-off
step — the guide walks to the station first, arrives, then speaks the
hand-off, rather than announcing from across the room and walking over
mid-thought. `build_script()`'s navigation steps come from
`GAZEBO_STATIONS` / `station_for()` in `demo_script.py` (the same
coordinates as the static list), threaded through via
`DemoOrchestrator(station_lookup=station_for)` in `server/app.py` — the
same dependency-injection pattern `subject_lookup` already used. A robot
with no entry in `GAZEBO_STATIONS` gets no navigation step, so a demo run
with a real (non-simulated) guide is unaffected.

When the orchestrator reaches a navigation step, it sends the target
robot's bridge a `demo_nav` event with `{x, y, frame_id}` instead of a
`demo_step` with text. The bridge drives there and, on arrival, sends the
normal `{"type": "ack", "step_id": ...}` message — the orchestrator
advances exactly as it would after a spoken step's TTS finishes. If a robot
other than a `GazeboBridge` is connected (e.g. a real Pepper), a safety-net
default handler in `BasicClient` (`client/client.py`) acks navigation steps
immediately instead of moving, so the demo never hangs waiting for a robot
that can't walk.

## Running the full lab demo against Gazebo

Four commands, one per terminal:

1. Launch Gazebo + spawn the 4 robots:
   ```
   source /opt/ros/humble/setup.bash
   ros2 launch ../gazebo/launch_demo_env.launch.py
   ```
   Then confirm the odom topic names match each config's `gazebo_config.odom_topic`:
   ```
   ros2 topic list | grep odom
   ```

2. Start the central server (in `v6.0.0/server/`):
   ```
   python3 app.py
   ```

3. Start the dashboard (as normal — unrelated to this package).

4. Start and connect all 4 bridges in one go, with `run_gazebo_demo.py` —
   the Gazebo equivalent of `client/run_lab_test.py`. ROS2 must be sourced
   in THIS shell first, since the bridge subprocesses inherit its
   environment (no need to source it again per-robot):
   ```
   source /opt/ros/humble/setup.bash
   cd v6.0.0/gazebo_client
   python3 run_gazebo_demo.py
   ```
   It spawns `pepper_01`/`chatbox_01`/`navel_01`/`silbot_01`, waits for them
   to register, then `POST`s `/connect` for each — prefixing every log line
   with the robot id so all four are readable in one terminal. Ctrl+C stops
   all four together.

   To run a bridge on its own instead (e.g. while iterating on one robot),
   the old per-process form still works:
   ```
   python3 gazebo_bridge.py configs/pepper_01_sim.json
   ```

Then, to actually run the tour:

5. Check presence is flowing:
   ```
   curl http://127.0.0.1:5000/robots/pepper_01/presence
   ```

6. Start the demo (dashboard "Start Demo", or the server's demo-start
   endpoint). Watch it play through the opening, then before ChatBox is
   introduced: the `pepper_01` bridge's log lines show
   `[Nav] 'navigate_to_chatbox_01': navigating to (9.70, 1.04)`, robot1
   turns and drives toward robot2 in Gazebo, then `[Nav] ... arrived` —
   THEN the guide speaks the hand-off and ChatBox greets. Same pattern for
   Navel and Silbot later in the tour.

7. Type at a bridge's `💬 You:` prompt to send a simulated question as that
   robot. Under `run_gazebo_demo.py` all four subprocesses share the one
   terminal's stdin, so whatever you type reaches whichever robot's input
   thread happens to be waiting on it first — not reliably the one you
   meant. To type as a specific robot, run that one bridge standalone
   instead (the per-process form from step 4, in its own terminal).

8. If a navigation step ever seems stuck: check for
   `[Nav] ... did not reach target in time` in the logs — the orchestrator
   will show `ERROR` state after `timeout_sec`; `POST /demo/next` (or the
   dashboard's manual "next") recovers it, same as a stuck TTS step.
