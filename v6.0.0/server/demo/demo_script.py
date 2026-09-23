"""
demo/demo_script.py
====================
EDIT THIS FILE to change the CARES lab demo sequence.

Flow:
  Pepper: greeting + lab intro
  ↓
  For each project:
    Pepper: introduce the project concept
    Pepper: hand off to assigned robot
    Robot:  speaks about their project (RAG-aware via system prompt)
    Pepper: Q&A window — visitors can speak to any robot for `qa_timeout` seconds
    Pepper: transition to next project
  ↓
  Pepper: wrap-up + open floor

Q&A windows:
  • After each robot speaks, Pepper invites questions.
  • The orchestrator enters a timed Q&A window (qa_window=True on a step).
  • Visitors can speak to any connected robot — normal LLM/RAG pipeline handles it.
  • The window auto-closes after `qa_timeout` seconds, or press "Move On" on dashboard.
  • At ANY time during the demo use POST /demo/qa to open an ad-hoc Q&A window.

Each step uses generate=True so the robot's LLM generates natural speech from the
instruction in `text`, building real conversation history across the demo instead of
speaking hardcoded lines. The `text` field is a concise prompt/instruction.

Edit instructions:
  • Change PEPPER/CHATBOX/NAVEL/SILBOT to match your client_config.json `client_id`.
  • Edit step `text` fields to customise the instruction given to the robot's LLM.
  • Adjust `timeout_sec` — keep generous values (60–90s) for generate steps.
  • Set `qa_timeout` on Q&A steps (how long to accept visitor questions; 0 = manual).
  • Comment out steps to skip them.
  • Restart the server after any change.
"""

import math

from demo.demo_orchestrator import DemoStep, StepRole

# Facing directions for GAZEBO_ROUTES' cardinal legs (radians, world frame:
# 0 = +x/east, increasing counter-clockwise) — named so a station's
# FINAL_YAW reads as "which way is back toward the corridor" rather than a
# bare number. Used below to turn the guide to face the visitors once it
# arrives at each station, instead of leaving it facing whatever direction
# its last leg of travel happened to be (usually straight at the project
# robot it just walked up to) — see gazebo_bridge.py's _turn_to_yaw.
FACE_EAST  = 0.0
FACE_NORTH = math.pi / 2
FACE_WEST  = math.pi
FACE_SOUTH = -math.pi / 2

# ── Robot IDs — must match client_id in each robot's client_config.json ───────

PEPPER  = "pepper_01"    # Main guide robot — full DB access
CHATBOX = "chatbox_01"   # Project A robot
NAVEL   = "navel_01"     # Project B robot
SILBOT  = "silbot_01"    # Project C robot

# ── Gazebo routes ────────────────────────────────────────────────────────────
# A straight line between two points in lab_world usually crosses a wall —
# confirmed live, repeatedly. These waypoint sequences are each an EDGE
# between two adjacent, live-tested points, confirmed clear by actually
# driving it in Gazebo (not derived from the schematic floor-plan alone —
# that gives the intended shape/order, but not real coordinates).
#
# Keyed by (from, to) so a route COMPOSES correctly for whatever order the
# operator actually picks in the dashboard's reorderable "Demo order" list
# (see robot-dashboard/src/components/DemoTab.jsx) — not just one hardcoded
# sequence. A live run got stuck when the operator picked SILBOT before
# NAVEL: the old station-per-destination-only design had no route for that
# transition and produced garbage. Now: if consecutive stops in whatever
# order was picked have an edge below, that edge is used; if not (e.g. a
# non-adjacent jump straight from CHATBOX to SILBOT), build_script() just
# skips navigation for that one leg rather than sending an unproven or
# wrong route — same fail-safe philosophy as an unknown robot_id.
#
# START is a sentinel for the guide's spawn/idle pose — not a real robot_id,
# never appears in project_ids, only ever as an edge endpoint here.
START = "__start__"

# DOOR is a second sentinel: the guide's stop at the doorway to approach and
# greet the visitors BEFORE the opening speech — not just a waypoint passed
# through en route to chatbox_01. Splitting the old single (START, CHATBOX)
# edge at the door lets build_script() insert a dedicated navigation step
# there (see "approach_visitors" below) so the guide visibly turns and walks
# up to the visitors first, then greets them, THEN continues on to the first
# project robot — rather than greeting from its spawn pose and only passing
# the visitors later, mid-transit to chatbox_01.
DOOR = "__door__"

GAZEBO_ROUTES = {
    # guide start (0.092, 5.528) -> through the doorway, where the visitors
    # wait (see lab_world's visitor_1/2/3, y=7.2) -> and back.
    (START, DOOR): [
        {"x": 0.092, "y": 6.818, "frame_id": "map"},
    ],
    (DOOR, START): [
        {"x": 0.092, "y": 5.528, "frame_id": "map"},
    ],
    # doorway -> down the corridor -> east to chatbox_01's station, and back.
    (DOOR, CHATBOX): [
        {"x": 0.092, "y": 1.058, "frame_id": "map"},
        {"x": 9.695, "y": 1.035, "frame_id": "map"},
    ],
    (CHATBOX, DOOR): [
        {"x": 0.092, "y": 1.058, "frame_id": "map"},
        {"x": 0.092, "y": 6.818, "frame_id": "map"},
    ],
    # chatbox_01's station -> back along the corridor -> south to
    # (8.128, -5.582), and back. That physical station is now silbot_01's
    # (robots were swapped — see launch_demo_env.launch.py's spawn_robot3/
    # spawn_robot4 — the walkable path itself didn't change, only which
    # robot sits at each end of it, so these edges were relabeled rather
    # than re-derived).
    (CHATBOX, SILBOT): [
        {"x": 8.132, "y": 1.058,  "frame_id": "map"},
        {"x": 8.128, "y": -5.582, "frame_id": "map"},
    ],
    (SILBOT, CHATBOX): [
        {"x": 8.132, "y": 1.058,  "frame_id": "map"},
        {"x": 9.695, "y": 1.035,  "frame_id": "map"},
    ],
    # navel_01 was moved live, in the Gazebo GUI, from (-3.0, -5.0) to
    # (0.25, -5.0) — see launch_demo_env.launch.py's spawn_robot3 — after a
    # from-the-north corridor guess for the old station got the guide stuck
    # for a full timeout (that bay, and this new one, are both walled on
    # the west/north/east — Wall_36/22/33 for the old one, Wall_38/40/41
    # for this one — so approaching from the north was never going to work
    # for either). The edges below go from the SOUTH instead: down past the
    # whole row of bay-dividing walls (they don't reach below y=-6.9), west
    # along the clear corridor at y=-7.3, then north back up into whichever
    # bay is the target. Driven live end-to-end with direct cmd_vel probing
    # (not derived from the wall XML — that's what produced the stuck
    # guess) and confirmed arriving within tolerance at every waypoint,
    # door included, all the way back to START.
    # Each of these three edges STARTS at a station the guide is currently
    # parked next to (close enough for final_yaw to turn and face it — see
    # below), and the very next waypoint after this comment used to be due
    # south of that same station: a straight line straight through the
    # parked robot's own body. Live testing confirmed the route ITSELF was
    # clear; the collision was this departure leg only. Each now opens with
    # one diagonal "step aside" waypoint (east + partway south) that clears
    # the robot's footprint before merging back onto the confirmed south
    # corridor point, rather than driving straight at it.
    # Step-aside offsets widened twice now. First from the original
    # (8.8,-6.5)/(0.9,-6.5) — live testing before the shell swap (turtlebot
    # meshes) called those "confirmed clear", but a real run still clipped
    # the departure station both times this edge was exercised
    # (SILBOT->NAVEL hit silbot_01; NAVEL->START hit navel_01). Root cause:
    # the guide's own arrival target for a station IS that station's exact
    # spawn coordinate (see build_script()'s per-project loop), so it parks
    # within final_arrival_tolerance_m (~1.0m) of the robot it just
    # presented — close enough that even a moderate step-aside distance, or
    # variance in exactly where within that tolerance it stopped, can still
    # sweep through the parked robot's body while turning to face the first
    # waypoint. Widened again after NAVEL->START clipped navel_01 a SECOND
    # time at ~1.1m/1.6-2.1m clearance (recorded on video, not just a
    # server-side timeout — the nav step itself completed within budget
    # both times, so this was purely visual clipping the timeout/arrival
    # logs never surfaced). Now ~2.5m clear of the station instead of
    # ~1.1-2.1m, at y=-7.2 — deep inside the "walls don't reach below
    # y=-6.9" safe zone rather than right at its edge. Re-test this edge
    # specifically before trusting it the way the rest of this dict's edges
    # are trusted — it has now been wrong twice at smaller margins.
    (SILBOT, NAVEL): [
        {"x": 9.0,   "y": -6.9, "frame_id": "map"},   # step aside — clear silbot_01's body
        {"x": 8.128, "y": -7.3, "frame_id": "map"},   # merge onto the confirmed corridor
        {"x": 0.25,  "y": -7.3, "frame_id": "map"},
        {"x": 0.25,  "y": -5.0, "frame_id": "map"},
    ],
    (NAVEL, SILBOT): [
        {"x": 1.6,   "y": -7.2,  "frame_id": "map"},  # step aside — clear navel_01's body
        {"x": 0.25,  "y": -7.3,  "frame_id": "map"},  # merge onto the confirmed corridor
        {"x": 8.128, "y": -7.3,  "frame_id": "map"},
        {"x": 8.128, "y": -5.582, "frame_id": "map"},
    ],
    # navel_01 straight back to start — south out of the bay (with the same
    # step-aside as above), then the same south corridor at x=0.092 (not
    # 0.25 — this is the guide's own home corridor's x, confirmed clear the
    # whole way from y=-7.3 north) up past the bay row, into the corridor,
    # through the door, to spawn.
    (NAVEL, START): [
        {"x": 1.6,   "y": -7.2,  "frame_id": "map"},  # step aside — clear navel_01's body
        {"x": 0.092, "y": -7.3,  "frame_id": "map"},
        {"x": 0.092, "y": 0.1,   "frame_id": "map"},
        {"x": 0.092, "y": 1.058, "frame_id": "map"},
        {"x": 0.092, "y": 6.818, "frame_id": "map"},
        {"x": 0.092, "y": 5.528, "frame_id": "map"},
    ],
}


def gazebo_route(from_id: str, to_id: str) -> list:
    """The waypoint list for one edge, or [] if this pair has no
    live-tested route — callers must treat [] as "skip navigation for this
    leg", never as "go there directly", since [] does not mean the path is
    clear."""
    return GAZEBO_ROUTES.get((from_id, to_id)) or []


# The known linear chain of stations, in PHYSICAL order — not necessarily
# the order any given tour visits them in. Only used to compose the
# closing "walk back to start" leg (below): however far along this chain
# the guide's last stop was, walk back one edge at a time to START.
# SILBOT before NAVEL: robots were physically swapped (see
# launch_demo_env.launch.py) — silbot_01 now sits where navel_01 used to
# (closer to chatbox_01), so it's second in physical order, not third.
# DOOR sits between START and CHATBOX now that the door is its own stop
# (see DOOR above) — the return trip walks NAVEL -> SILBOT -> CHATBOX ->
# DOOR -> START one edge at a time, same as the forward trip in reverse.
_GAZEBO_CHAIN = [START, DOOR, CHATBOX, SILBOT, NAVEL]

# Which way each station faces to look back toward the corridor/door the
# guide (and, following behind at a delay, the visitors) just came from —
# i.e. away from the project robot it just walked up to. Keyed by the
# ARRIVING robot_id, not the edge, since build_script()'s per-project loop
# builds one generically-named "navigate_to_{robot_id}" step per station
# regardless of which edge got it there.
STATION_FACE_YAW = {
    CHATBOX: FACE_WEST,   # approached from the west-corridor, chatbox sits east of it
    SILBOT:  FACE_NORTH,  # approached from the north (the corridor down to it)
    NAVEL:   FACE_SOUTH,  # approached from the south (the new corridor route)
}

# The opposite of STATION_FACE_YAW — turns the guide to look AT the project
# robot instead of back toward the visitors. Used for the silent turn-only
# steps build_script() inserts around each robot's own speech, so the guide
# is visibly looking at whoever is actually talking rather than facing the
# audience the whole block through, including while the project robot
# itself has the floor.
STATION_ROBOT_YAW = {
    CHATBOX: FACE_EAST,
    SILBOT:  FACE_SOUTH,
    NAVEL:   FACE_NORTH,
}

# Each station's own coordinate (matches ROBOT_SPAWNS in gazebo/visitor_follow.py
# and launch_demo_env.launch.py's spawn_robot2-4 args) — used as the nav_target
# for the turn-only steps below. The guide is already standing here (it
# navigated to this exact point to arrive), so this "drive" is ~0m and
# resolves almost instantly; only the final_yaw turn actually does anything.
STATION_COORD = {
    CHATBOX: {"x": 9.695, "y": 1.035, "frame_id": "map"},
    SILBOT:  {"x": 8.128, "y": -5.582, "frame_id": "map"},
    NAVEL:   {"x": 0.25,  "y": -5.0, "frame_id": "map"},
}


def _turn_step(step_id: str, guide_id: str, robot_id: str, yaw: float) -> "DemoStep":
    """A silent (text="") turn-in-place, reusing the navigation step
    mechanism purely for its final_yaw turn — see STATION_COORD. Tagged to
    the same block_robot_id as the speech around it so plan revision drops
    or keeps it along with the rest of that block, never orphaned."""
    return DemoStep(
        step_id     = step_id,
        robot_id    = guide_id,
        text        = "",
        step_kind   = "navigation",
        nav_target  = [STATION_COORD[robot_id]] if robot_id in STATION_COORD else [],
        timeout_sec = 20,
        block_robot_id = robot_id,
        role           = StepRole.TRANSITION,
        final_yaw   = yaw,
    )


def gazebo_route_to_start(current: str) -> list:
    """Route from `current` (the guide's last known stop) back to START.

    Prefers a direct (current, START) edge when one is defined — e.g.
    (NAVEL, START) skips retracing silbot_01/chatbox_01/the door as
    separate hops, matching the operator's floor-plan diagram's own
    direct-looking return path. Falls back to composing backward through
    _GAZEBO_CHAIN one edge at a time when no direct edge exists. []
    if `current` isn't on the chain, or if any link back is missing —
    same fail-safe rule as gazebo_route(): no route, not a guess."""
    direct = gazebo_route(current, START)
    if direct:
        return direct
    if current not in _GAZEBO_CHAIN:
        return []
    idx = _GAZEBO_CHAIN.index(current)
    waypoints = []
    for i in range(idx, 0, -1):
        edge = gazebo_route(_GAZEBO_CHAIN[i], _GAZEBO_CHAIN[i - 1])
        if not edge:
            return []
        waypoints += edge
    return waypoints


# ── Demo script ────────────────────────────────────────────────────────────────

DEMO_STEPS = [

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OPENING — Pepper walks to the door, greets the visitors, introduces the lab
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Same reasoning as build_script()'s "approach_visitors" step: arrive at
    # the visitors before speaking to them, not mid-transit to chatbox_01.
    # NOT "Follow me!" — nobody is following yet at this point, the guide
    # is walking TO the visitors, not leading them anywhere.
    DemoStep(
        step_id     = "approach_visitors",
        robot_id    = PEPPER,
        text        = "Let's go say hello!",
        step_kind   = "navigation",
        nav_target  = gazebo_route(START, DOOR),
        timeout_sec = 60,
        block_robot_id = None,
        role           = StepRole.OPENING,
        final_yaw   = FACE_NORTH,   # visitors wait north of the door
    ),

    DemoStep(
        step_id     = "greeting",
        robot_id    = PEPPER,
        text        = "You are opening the CARES lab demonstration for a group of visitors. "
                      "Welcome them warmly, introduce yourself as Pepper the lab guide, and let them know "
                      "you are excited to show them around today. Keep it to 2 sentences. Start with [GREETING].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = None,
        role           = StepRole.OPENING,
    ),

    DemoStep(
        step_id     = "lab_intro",
        robot_id    = PEPPER,
        text        = "Briefly explain what CARES stands for (Centre for Automation and Robotic Engineering Science) "
                      "and what the lab researches — intelligent robots that can communicate, collaborate, and assist "
                      "people in real-world environments. Keep it to 2-3 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = None,
        role           = StepRole.OPENING,
    ),

    DemoStep(
        step_id     = "overview",
        robot_id    = PEPPER,
        text        = "Set expectations for the demo: visitors will meet three research robots today, "
                      "each working on a different project. After each robot speaks, there will be time "
                      "to ask questions. Keep it to 2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = None,
        role           = StepRole.OPENING,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT A — ChatBox: Conversational AI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "intro_project_a",
        robot_id    = PEPPER,
        text        = "Introduce the first research project: conversational AI and retrieval-augmented generation. "
                      "Explain that this project focuses on how robots can hold long, contextually aware conversations "
                      "by combining language models with a knowledge base. Keep it to 2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = CHATBOX,
        role           = StepRole.INTRO,
    ),

    # Guide physically walks to ChatBox's station in the Gazebo sim BEFORE
    # introducing ChatBox — arrives, then speaks the hand-off, rather than
    # announcing from across the room. Completed by the robot reporting
    # arrival, not TTS. Waypoints are a live-tested edge (see GAZEBO_ROUTES
    # above) — a straight line here crosses a wall. From DOOR, not START:
    # the guide already made the START->DOOR leg during "approach_visitors"
    # above, so this continues from there.
    DemoStep(
        step_id     = "navigate_to_chatbox",
        robot_id    = PEPPER,
        text        = "Follow me!",
        step_kind   = "navigation",
        nav_target  = gazebo_route(DOOR, CHATBOX),
        timeout_sec = 120,
        block_robot_id = CHATBOX,
        role           = StepRole.TRANSITION,
        final_yaw   = FACE_WEST,   # back toward the corridor/visitors, not at ChatBox
    ),

    DemoStep(
        step_id     = "introduce_chatbox",
        robot_id    = PEPPER,
        text        = "Hand off to ChatBox, the robot leading the conversational AI research. "
                      "Point towards ChatBox and invite them to say hello to the visitors. "
                      "1-2 sentences. Use [POINT].",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = CHATBOX,
        role           = StepRole.HANDOFF,
    ),

    DemoStep(
        step_id     = "chatbox_greeting",
        robot_id    = CHATBOX,
        text        = "Greet the visitors warmly for the first time. You are ChatBox. "
                      "Introduce yourself and express genuine excitement about meeting the visitors. "
                      "2 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = CHATBOX,
        role           = StepRole.GREETING,
    ),

    DemoStep(
        step_id     = "chatbox_prompt",
        robot_id    = PEPPER,
        text        = "Ask ChatBox to explain their research project to the visitors. "
                      "1 sentence. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 40,
        block_robot_id = CHATBOX,
        role           = StepRole.PROMPT,
    ),

    DemoStep(
        step_id     = "chatbox_project",
        robot_id    = CHATBOX,
        text        = "Explain your research on retrieval-augmented generation (RAG) to a non-expert audience. "
                      "Cover: what RAG is, how combining language models with a searchable knowledge base helps robots "
                      "give accurate answers, and how you maintain context across a long conversation. "
                      "Make it engaging and accessible. 3-4 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 90,
        block_robot_id = CHATBOX,
        role           = StepRole.PROJECT,
    ),

    # Q&A window — visitors can speak to ChatBox or any robot
    DemoStep(
        step_id     = "qa_invite_a",
        robot_id    = PEPPER,
        text        = "Open a Q&A session after ChatBox's presentation. "
                      "Invite visitors to ask questions — they can speak directly to ChatBox or to you. "
                      "Let them know you will wait until everyone is ready to move on. "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,    # manual advance only — operator clicks Move On
        block_robot_id = CHATBOX,
        role           = StepRole.QA,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT B — Silbot: Human-Aware Navigation
    # (physically second now — robots were swapped, see GAZEBO_ROUTES and
    # launch_demo_env.launch.py's spawn_robot3/spawn_robot4)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_b",
        robot_id    = PEPPER,
        text        = "Transition from the ChatBox Q&A to the second project. "
                      "Give a brief, warm sign-off to ChatBox and announce you are moving on. "
                      "1-2 sentences. Use [DEFAULT]. Include 'let us move on' in your response.",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = CHATBOX,
        role           = StepRole.TRANSITION,
    ),

    DemoStep(
        step_id     = "intro_project_b",
        robot_id    = PEPPER,
        text        = "Introduce the second research project: human-aware navigation. "
                      "Frame it as the question: how can a robot move through a crowded space "
                      "safely, politely, and predictably — the way a person would? "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = SILBOT,
        role           = StepRole.INTRO,
    ),

    # Guide physically walks to Silbot's station in the Gazebo sim BEFORE
    # introducing Silbot — arrives, then speaks the hand-off. Waypoints are
    # a live-tested edge (see GAZEBO_ROUTES above).
    DemoStep(
        step_id     = "navigate_to_silbot",
        robot_id    = PEPPER,
        text        = "Follow me!",
        step_kind   = "navigation",
        nav_target  = gazebo_route(CHATBOX, SILBOT),
        timeout_sec = 120,
        block_robot_id = SILBOT,
        role           = StepRole.TRANSITION,
        final_yaw   = FACE_NORTH,   # back toward the corridor/visitors, not at Silbot
    ),

    DemoStep(
        step_id     = "introduce_silbot",
        robot_id    = PEPPER,
        text        = "Hand off to Silbot, the robot specialising in human-aware navigation. "
                      "Point towards Silbot and invite them to come say hello to the visitors. "
                      "1-2 sentences. Use [POINT].",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = SILBOT,
        role           = StepRole.HANDOFF,
    ),

    DemoStep(
        step_id     = "silbot_greeting",
        robot_id    = SILBOT,
        text        = "Greet the visitors warmly for the first time. You are Silbot. "
                      "Introduce yourself and briefly mention that you navigate spaces with awareness and courtesy. "
                      "2 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = SILBOT,
        role           = StepRole.GREETING,
    ),

    DemoStep(
        step_id     = "silbot_prompt",
        robot_id    = PEPPER,
        text        = "Ask Silbot to explain their navigation research to the visitors. "
                      "1 sentence. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 40,
        block_robot_id = SILBOT,
        role           = StepRole.PROMPT,
    ),

    DemoStep(
        step_id     = "silbot_project",
        robot_id    = SILBOT,
        text        = "Explain your human-aware navigation research to a non-expert audience. "
                      "Cover: that rather than just avoiding obstacles, you predict where people are moving, "
                      "plan routes that do not cut through conversations or crowd groups, "
                      "that you were trained in simulation and tested in real office corridors at CARES, "
                      "and your goal of moving through shared spaces the way a respectful colleague would. "
                      "3-4 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 90,
        block_robot_id = SILBOT,
        role           = StepRole.PROJECT,
    ),

    # Q&A window
    DemoStep(
        step_id     = "qa_invite_b",
        robot_id    = PEPPER,
        text        = "Open a Q&A session after Silbot's presentation on human-aware navigation. "
                      "Invite visitors to ask questions about navigation and social robotics — "
                      "they can speak to Silbot or to you. "
                      "Let them know you will wait until everyone is ready to continue. "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,    # manual advance only — operator clicks Move On
        block_robot_id = SILBOT,
        role           = StepRole.QA,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT C — Navel: Emotion-Aware Interaction
    # (physically third now — see note above)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_c",
        robot_id    = PEPPER,
        text        = "Transition from the Silbot Q&A to the third and final project. "
                      "Brief warm sign-off to Silbot and announce the move to the next project. "
                      "1-2 sentences. Use [DEFAULT]. Include 'let us move on' in your response.",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = SILBOT,
        role           = StepRole.TRANSITION,
    ),

    DemoStep(
        step_id     = "intro_project_c",
        robot_id    = PEPPER,
        text        = "Introduce the third research project: emotion-aware interaction. "
                      "Explain that this project studies how robots can recognise a person's emotional state "
                      "and adapt their communication style accordingly. 2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = NAVEL,
        role           = StepRole.INTRO,
    ),

    # Guide physically walks to Navel's station in the Gazebo sim BEFORE
    # introducing Navel — arrives, then speaks the hand-off. Waypoints are
    # a live-tested edge (see GAZEBO_ROUTES above).
    DemoStep(
        step_id     = "navigate_to_navel",
        robot_id    = PEPPER,
        text        = "Follow me!",
        step_kind   = "navigation",
        nav_target  = gazebo_route(SILBOT, NAVEL),
        timeout_sec = 120,
        block_robot_id = NAVEL,
        role           = StepRole.TRANSITION,
        final_yaw   = FACE_SOUTH,   # back toward the corridor/visitors, not at Navel
    ),

    DemoStep(
        step_id     = "introduce_navel",
        robot_id    = PEPPER,
        text        = "Hand off to Navel, the robot leading emotion-aware interaction research. "
                      "Point towards Navel and invite them to say hi to everyone. "
                      "1-2 sentences. Use [POINT].",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = NAVEL,
        role           = StepRole.HANDOFF,
    ),

    DemoStep(
        step_id     = "navel_greeting",
        robot_id    = NAVEL,
        text        = "Greet the visitors warmly for the first time. You are Navel. "
                      "Introduce yourself and mention that meeting new people is literally part of your research. "
                      "2 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 50,
        block_robot_id = NAVEL,
        role           = StepRole.GREETING,
    ),

    DemoStep(
        step_id     = "navel_prompt",
        robot_id    = PEPPER,
        text        = "Ask Navel to share what their research is about with the visitors. "
                      "1 sentence. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 40,
        block_robot_id = NAVEL,
        role           = StepRole.PROMPT,
    ),

    DemoStep(
        step_id     = "navel_project",
        robot_id    = NAVEL,
        text        = "Explain your emotion-aware interaction research to a non-expert audience. "
                      "Cover: that you detect facial expressions and tone of voice in real time, "
                      "how you adapt your speaking style based on what you detect "
                      "(e.g. slower when someone looks confused, warmer when someone seems upset), "
                      "and your goal of making conversation with a robot feel natural. "
                      "3-4 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 90,
        block_robot_id = NAVEL,
        role           = StepRole.PROJECT,
    ),

    # Q&A window
    DemoStep(
        step_id     = "qa_invite_c",
        robot_id    = PEPPER,
        text        = "Open a Q&A session after Navel's presentation on emotion-aware interaction. "
                      "Invite visitors to ask questions — they can speak to Navel or to you. "
                      "Let them know you will be here until everyone is ready to wrap up. "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,    # manual advance only — operator clicks Move On
        block_robot_id = NAVEL,
        role           = StepRole.QA,
    ),

    # Guide walks back to its starting position before wrapping up — the
    # reverse of the full start -> door -> chatbox -> silbot -> navel route,
    # composed by gazebo_route_to_start() from the same live-tested edges
    # (see GAZEBO_ROUTES / _GAZEBO_CHAIN above).
    DemoStep(
        step_id     = "navigate_to_start",
        robot_id    = PEPPER,
        text        = "Let's head back to wrap things up!",
        step_kind   = "navigation",
        nav_target  = gazebo_route_to_start(NAVEL),
        timeout_sec = 180,
        block_robot_id = None,
        # CLOSING, not TRANSITION: this has no block_robot_id, and
        # FlowGraph.from_script() classifies any non-block step by role —
        # CLOSING or it falls through to "opening" (see that method's
        # docstring), silently inflating the fixed opening-duration
        # baseline every eval_scenarios.py scenario is hand-calibrated
        # against. CLOSING is also semantically correct on its own: the
        # walk back to start should survive an emergency DROP_REMAINING
        # cut, same as wrap_up/open_floor, not strand the guide mid-tour.
        role           = StepRole.CLOSING,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CLOSING — Pepper wraps up
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "wrap_up",
        robot_id    = PEPPER,
        text        = "Close the main part of the demo. Thank ChatBox, Silbot, and Navel by name. "
                      "Summarise that each robot brings a unique capability and together you are all "
                      "working towards robots that can truly work alongside people. "
                      "2-3 sentences. Use [HAPPY].",
        generate    = True,
        timeout_sec = 60,
        block_robot_id = None,
        role           = StepRole.CLOSING,
    ),

    DemoStep(
        step_id     = "open_floor",
        robot_id    = PEPPER,
        text        = "Open a general Q&A — the demo is complete but the floor is open. "
                      "Invite visitors to approach any of the robots or speak to you with any remaining questions. "
                      "Thank them warmly for joining the CARES lab demonstration today. "
                      "2-3 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,    # manual close — operator ends open floor via dashboard
        block_robot_id = None,
        role           = StepRole.CLOSING,
    ),

]


# ── Dynamic script builder ─────────────────────────────────────────────────────

PROJECT_CHECKLIST = (
    ("problem",  "the problem your research is solving, in plain terms"),
    ("approach", "how your approach works"),
    ("impact",   "why it matters — what it makes possible"),
)
"""What a project talk has to cover, as separately tickable points.

These are the same three things the old single-paragraph instruction asked
for; the change is that each is its own step, so "covered" is a position in
the script rather than something that has to be inferred from what was said.
That distinction is the whole point. A visitor interrupting mid-talk used to
lose every point the robot had not reached yet, because the run loop treats
an interrupted step as finished and advances past it — one step meant one
chance. As separate steps the unreached points are simply still ahead of the
play head, and run when the Q&A closes.

Generic rather than per-robot: every project answers these three, and a
per-robot checklist is a configuration surface nobody has asked for yet. If
one is needed later it belongs beside robot_role in the robots table, and
this becomes the default for robots that do not define their own.
"""


def build_script(guide_id: str, project_ids: list, subjects: dict = None) -> list:
    """
    Build a demo script dynamically from a guide robot and an ordered list of
    project robots.  The guide introduces the lab and each robot; project robots
    present their own research (content generated by their LLM/persona).

    Args:
        guide_id    : client_id of the host/guide robot (plays the Pepper role).
        project_ids : Ordered list of project robot client_ids — whatever
                      order the operator picked in the dashboard's
                      reorderable "Demo order" list. Navigation between
                      consecutive stops (see the "one block per project
                      robot" loop below) uses GAZEBO_ROUTES, keyed by
                      (from, to): a live-tested edge if this exact
                      transition has one, otherwise no navigation step for
                      that leg — never an unproven or wrong route. A run
                      with a real (non-simulated) guide, or an order with
                      no matching edges, behaves exactly as before (no
                      navigation steps at all).
        subjects    : {robot_id: "what this robot researches"}, used in the
                      guide's introduction. Optional, and omitting it is why
                      this argument exists: the instruction used to say "give
                      a teaser of the research area" while telling the guide
                      nothing except a client_id, so the guide invented one —
                      a live run had Pepper introduce Silbot, whose subject is
                      navigation, as working on understanding emotions. The
                      caller supplies these from the same declared scope
                      routing uses, so what the guide SAYS a robot does and
                      where questions about it GO cannot drift apart.

    Returns:
        A list of DemoStep objects ready to pass to DemoOrchestrator.load_script().
    """
    steps = []
    n = len(project_ids)
    subjects = subjects or {}
    prev_stop = START  # tracks where GAZEBO_ROUTES thinks the guide is, for edge lookups

    # Guide turns and walks to the doorway, where the visitors wait, BEFORE
    # greeting them — arrives, then speaks, rather than greeting from its
    # spawn pose and only reaching them later while already walking toward
    # chatbox_01. Same guard as every other nav step: no edge, no step (a
    # real, non-simulated guide skips this silently, same as before).
    # NOT "Follow me!" — nobody is following yet at this point.
    door_route = gazebo_route(START, DOOR)
    if door_route:
        steps.append(DemoStep(
            step_id     = "approach_visitors",
            robot_id    = guide_id,
            text        = "Let's go say hello!",
            step_kind   = "navigation",
            nav_target  = door_route,
            timeout_sec = 60,
            role        = StepRole.OPENING,
            final_yaw   = FACE_NORTH,   # visitors wait north of the door
        ))
        prev_stop = DOOR

    # ── Opening ────────────────────────────────────────────────────────────────
    # No block_robot_id: these belong to no project, so plan revision never
    # touches them. They are behind the play head by the time anyone would ask.
    steps.append(DemoStep(
        step_id     = "greeting",
        robot_id    = guide_id,
        text        = "You are opening the CARES lab demonstration for a group of visitors. "
                      "Welcome them warmly, introduce yourself as the lab guide, and let them know "
                      "you are excited to show them around today. Keep it to 2 sentences. Start with [GREETING].",
        generate    = True,
        timeout_sec = 60,
        role        = StepRole.OPENING,
    ))

    steps.append(DemoStep(
        step_id     = "lab_intro",
        robot_id    = guide_id,
        text        = "Briefly explain what CARES stands for (Centre for Automation and Robotic Engineering Science) "
                      "and what the lab researches — intelligent robots that can communicate, collaborate, and assist "
                      "people in real-world environments. Keep it to 2-3 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 60,
        role        = StepRole.OPENING,
    ))

    if n > 0:
        robot_count_word = {1: "one", 2: "two", 3: "three", 4: "four"}.get(n, str(n))
        steps.append(DemoStep(
            step_id     = "overview",
            robot_id    = guide_id,
            text        = f"Set expectations for the demo: visitors will meet {robot_count_word} research robot"
                          f"{'s' if n != 1 else ''} today, each working on a different project. "
                          "After each robot speaks, there will be time to ask questions. "
                          "Keep it to 2 sentences. Use [DEFAULT].",
            generate    = True,
            timeout_sec = 60,
            role        = StepRole.OPENING,
        ))
    else:
        steps.append(DemoStep(
            step_id     = "overview",
            robot_id    = guide_id,
            text        = "Give the visitors a brief overview of the CARES lab and what they can expect today. "
                          "Invite them to look around and ask you any questions they have. "
                          "2 sentences. Use [DEFAULT].",
            generate    = True,
            timeout_sec = 60,
            role        = StepRole.OPENING,
        ))

    # ── One block per project robot ────────────────────────────────────────────
    # Every step carries block_robot_id and role. That tagging is what lets
    # DemoOrchestrator.revise_script() act on "ChatBox's part of the tour"
    # mid-demo — skip it, trim it, reorder it — without pattern-matching step_id
    # strings. Add a step here and it must be tagged, or plan revision will
    # silently leave it behind when the rest of its block moves.
    for i, robot_id in enumerate(project_ids):
        is_last = (i == n - 1)

        # Guide physically walks to this robot's station BEFORE introducing
        # it — arrives, THEN speaks the hand-off, rather than announcing
        # from across the room and walking while talking (demo_step and
        # demo_nav are never sent concurrently for the same robot, so the
        # walk is always silent either way; this order just makes the
        # visible sequence read as "arrive, then speak" instead of
        # "announce, then wander off mid-thought"). Edge lookup keyed by
        # (prev_stop, robot_id) — see GAZEBO_ROUTES — so this composes
        # correctly for whatever order the operator actually picked, not
        # just one hardcoded sequence.
        route = gazebo_route(prev_stop, robot_id)
        if route:
            steps.append(DemoStep(
                step_id     = f"navigate_to_{robot_id}",
                robot_id    = guide_id,
                text        = "Follow me!",
                step_kind   = "navigation",
                nav_target  = route,
                timeout_sec = 120,
                block_robot_id = robot_id,
                role           = StepRole.TRANSITION,
                final_yaw   = STATION_FACE_YAW.get(robot_id),
            ))
        prev_stop = robot_id

        # Teaser and hand-off in ONE utterance. They were two steps, and two
        # separate generations produced two sentences that did not follow
        # from each other — a live run had the guide give a decent teaser
        # ("Next, we have Silbot, which focuses on...") and then, as a
        # separate step, say "Great, let us move on to the next project!",
        # which reads as a non sequitur because the second generation could
        # not see the first. One instruction, one utterance, one thought.
        steps.append(DemoStep(
            step_id     = f"introduce_{robot_id}",
            robot_id    = guide_id,
            text        = (f"Introduce the next research project and hand off to {robot_id} "
                           "in one flowing turn. "
                           + (f"Their research area is: {subjects[robot_id]}. Introduce THAT "
                              "subject and nothing else — do not invent or guess a different "
                              "one. " if subjects.get(robot_id) else "")
                           + "Give a brief, intriguing teaser of it without going into detail "
                           "— that is the robot's job — "
                           f"then turn to {robot_id} and invite them to greet the visitors. "
                           "Make it one connected thought, not two announcements. "
                           "Speak ONLY your own words: do not write what the other "
                           "robot says, and never put their name followed by a "
                           "line of their dialogue — they greet the visitors "
                           "themselves, immediately after you. "
                           "2-3 sentences. Use [POINT]."),
            generate    = True,
            timeout_sec = 60,
            block_robot_id = robot_id,
            role           = StepRole.HANDOFF,
        ))

        # Guide turns to look at the project robot before it starts
        # talking — see STATION_ROBOT_YAW — rather than staying faced
        # toward the visitors (its own final_yaw from the navigate_to_X
        # step above) for the robot's entire greeting.
        steps.append(_turn_step(f"face_{robot_id}_greeting", guide_id, robot_id,
                                 STATION_ROBOT_YAW.get(robot_id)))

        steps.append(DemoStep(
            step_id     = f"{robot_id}_greeting",
            robot_id    = robot_id,
            text        = "Greet the visitors warmly for the first time. Introduce yourself and express "
                          "genuine excitement about meeting them. 2 sentences. Start with [WAVE].",
            generate    = True,
            timeout_sec = 50,
            block_robot_id = robot_id,
            role           = StepRole.GREETING,
        ))

        # Back to the visitors — the guide is the one talking again.
        steps.append(_turn_step(f"face_visitors_{robot_id}_prompt", guide_id, robot_id,
                                 STATION_FACE_YAW.get(robot_id)))

        steps.append(DemoStep(
            step_id     = f"{robot_id}_prompt",
            robot_id    = guide_id,
            text        = f"Ask {robot_id} to explain their research project to the visitors. "
                          "1 sentence. Use [DEFAULT].",
            generate    = True,
            timeout_sec = 40,
            block_robot_id = robot_id,
            role           = StepRole.PROMPT,
        ))

        # Back to the project robot for its own explanation (the three
        # PROJECT_CHECKLIST points below are consecutive robot-only speech
        # with no guide step between them, so one turn covers all three).
        steps.append(_turn_step(f"face_{robot_id}_project", guide_id, robot_id,
                                 STATION_ROBOT_YAW.get(robot_id)))

        # PROJECT and QA are never dropped by a COMPRESS — the research content
        # and the visitors' chance to ask about it are what the tour is for.
        #
        # One step per content point rather than one paragraph covering all
        # three. The old single step asked for "what problem you are solving,
        # your approach, and why it matters" in one generation, and a visitor
        # interrupting partway through lost the rest of it: the run loop
        # counts the interrupted step as done and moves on, so a robot cut off
        # after its first sentence had introduced its project and said
        # nothing else. Splitting the checklist into steps means the points
        # not yet reached are still sitting in the script — they run after the
        # Q&A closes, with no resume machinery involved, because they were
        # never skipped in the first place.
        for point_id, point_brief in PROJECT_CHECKLIST:
            steps.append(DemoStep(
                step_id     = f"{robot_id}_project_{point_id}",
                robot_id    = robot_id,
                text        = f"You are explaining your research to a non-expert audience, "
                              f"one point at a time. Cover ONLY this point now: {point_brief}. "
                              "Do not summarise the whole project and do not repeat what you "
                              "have already said. 1-2 sentences. Use [DEFAULT].",
                generate    = True,
                timeout_sec = 60,
                block_robot_id = robot_id,
                role           = StepRole.PROJECT,
            ))

        # Back to the visitors — the guide is opening the floor to them.
        steps.append(_turn_step(f"face_visitors_{robot_id}_qa", guide_id, robot_id,
                                 STATION_FACE_YAW.get(robot_id)))

        steps.append(DemoStep(
            step_id     = f"qa_invite_{robot_id}",
            robot_id    = guide_id,
            text        = f"Open a Q&A session after {robot_id}'s presentation. "
                          f"Invite visitors to ask questions — they can speak directly to {robot_id} or to you. "
                          "Let them know you will wait until everyone is ready to move on. "
                          "2 sentences. Use [DEFAULT].",
            generate    = True,
            timeout_sec = 60,
            qa_window   = True,
            qa_timeout  = 0,    # manual advance only — operator clicks Move On
            block_robot_id = robot_id,
            role           = StepRole.QA,
        ))

        if not is_last:
            next_robot = project_ids[i + 1]
            # Belongs to the block it signs off, not the one it announces —
            # skipping robot i must take its own farewell with it.
            steps.append(DemoStep(
                step_id     = f"transition_to_{next_robot}",
                robot_id    = guide_id,
                text        = f"Transition from the {robot_id} Q&A to the next project. "
                              f"Give a brief, warm sign-off to {robot_id} and announce you are moving on. "
                              "1-2 sentences. Use [DEFAULT]. Include 'let us move on' in your response.",
                generate    = True,
                timeout_sec = 50,
                block_robot_id = robot_id,
                role           = StepRole.TRANSITION,
            ))

    # ── Closing ────────────────────────────────────────────────────────────────
    if n > 0:
        robot_names = ", ".join(project_ids[:-1]) + (f", and {project_ids[-1]}" if n > 1 else project_ids[0])
        wrap_text = (
            f"Close the main part of the demo. Thank {robot_names} by name. "
            "Summarise that each robot brings a unique capability and together you are all "
            "working towards robots that can truly work alongside people. "
            "2-3 sentences. Use [HAPPY]."
        )
    else:
        wrap_text = (
            "Wrap up the demonstration. Thank the visitors for joining and let them know they are welcome "
            "to ask you more questions or explore the lab. 2 sentences. Use [HAPPY]."
        )
    # Guide walks back to its starting position before wrapping up —
    # composed by walking the known chain backward from wherever the guide
    # actually last stopped (prev_stop), not tied to one hardcoded order.
    # A non-simulated run, or a last stop with no route back, just skips
    # this — same fail-safe rule as every other navigation step here.
    return_route = gazebo_route_to_start(prev_stop)
    if return_route:
        steps.append(DemoStep(
            step_id     = "navigate_to_start",
            robot_id    = guide_id,
            text        = "Let's head back to wrap things up!",
            step_kind   = "navigation",
            nav_target  = return_route,
            timeout_sec = 180,
            # CLOSING — see the static DEMO_STEPS version of this same step
            # for why (FlowGraph's opening/closing classification, and
            # surviving DROP_REMAINING).
            role        = StepRole.CLOSING,
        ))

    # StepRole.CLOSING survives a DROP_REMAINING — a tour cut short for time
    # still gets a proper goodbye rather than stopping mid-sentence.
    steps.append(DemoStep(
        step_id     = "wrap_up",
        robot_id    = guide_id,
        text        = wrap_text,
        generate    = True,
        timeout_sec = 60,
        role        = StepRole.CLOSING,
    ))

    steps.append(DemoStep(
        step_id     = "open_floor",
        robot_id    = guide_id,
        text        = "Open a general Q&A — the demo is complete but the floor is open. "
                      "Invite visitors to approach any of the robots or speak to you with any remaining questions. "
                      "Thank them warmly for joining the CARES lab demonstration today. "
                      "2-3 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,
        role        = StepRole.CLOSING,
    ))

    return steps
