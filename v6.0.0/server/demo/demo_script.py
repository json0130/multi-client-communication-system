"""
demo/demo_script.py
====================
EDIT THIS FILE to define the CARES lab demo sequence.

Each DemoStep maps to one robot action:
  - step_id     : unique name (used for ACK matching + logs)
  - robot_id    : must match the client_id in that robot's client_config.json
  - text        : what the robot will say (emotion tags drive Arduino/animation)
  - timeout_sec : how long server waits for ACK before going to ERROR state

To add a step: copy an existing DemoStep and edit.
To skip a step: comment it out.
To change robot IDs: update the constants below.

Restart the server after any changes here.
"""

from demo.demo_orchestrator import DemoStep

# ── Robot IDs — must match client_id in each robot's client_config.json ───────

PEPPER  = "pepper_01"    # Main guide robot (Pepper)
CHATBOX = "chatbox_01"   # ChatBox robot
NAVEL   = "navel_01"     # Navel robot
SILBOT  = "silbot_01"    # Silbot robot

# ── Demo script ────────────────────────────────────────────────────────────────
#
# Flow:
#   1. Pepper greets visitors and introduces the lab
#   2. Pepper introduces each robot by name, then hands off to that robot
#   3. Each robot introduces itself and its project
#   4. Pepper wraps up and invites questions
#
# To run: POST /demo/start
# To skip a stuck step: POST /demo/next
# To stop early: POST /demo/stop

DEMO_STEPS = [

    # ── Opening ──────────────────────────────────────────────────────────────

    DemoStep(
        step_id     = "greeting",
        robot_id    = PEPPER,
        text        = "[GREETING] Welcome to the CARES lab! I am Pepper, your guide for today.",
        timeout_sec = 15,
    ),

    DemoStep(
        step_id     = "lab_intro",
        robot_id    = PEPPER,
        text        = "[WAVE] This is the Centre for Automation and Robotic Engineering Science. "
                      "We work on intelligent robots that can communicate, collaborate, and assist people.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "team_intro",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Today you will meet some of our robots. "
                      "Each one has a unique role and set of capabilities.",
        timeout_sec = 15,
    ),

    # ── ChatBox ───────────────────────────────────────────────────────────────

    DemoStep(
        step_id     = "introduce_chatbox",
        robot_id    = PEPPER,
        text        = "[POINT] First, let me introduce ChatBox. ChatBox, please say hello!",
        timeout_sec = 10,
    ),

    DemoStep(
        step_id     = "chatbox_greeting",
        robot_id    = CHATBOX,
        text        = "[WAVE] Hello everyone! I am ChatBox. "
                      "I specialise in natural language conversation and can answer questions on many topics.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "ask_chatbox_project",
        robot_id    = PEPPER,
        text        = "[DEFAULT] ChatBox, can you tell our visitors about your research project?",
        timeout_sec = 10,
    ),

    DemoStep(
        step_id     = "chatbox_project",
        robot_id    = CHATBOX,
        text        = "[DEFAULT] My project focuses on multi-turn dialogue management. "
                      "I use a large language model combined with retrieval augmented generation "
                      "to give contextually accurate answers.",
        timeout_sec = 25,
    ),

    # ── Navel ─────────────────────────────────────────────────────────────────

    DemoStep(
        step_id     = "introduce_navel",
        robot_id    = PEPPER,
        text        = "[POINT] Next is Navel. Navel, please introduce yourself!",
        timeout_sec = 10,
    ),

    DemoStep(
        step_id     = "navel_greeting",
        robot_id    = NAVEL,
        text        = "[WAVE] Hi! I am Navel. I am designed for social interaction and emotion recognition.",
        timeout_sec = 15,
    ),

    DemoStep(
        step_id     = "ask_navel_project",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Navel, what is your research about?",
        timeout_sec = 10,
    ),

    DemoStep(
        step_id     = "navel_project",
        robot_id    = NAVEL,
        text        = "[DEFAULT] I study how robots can adapt their communication style "
                      "based on the emotional state of the person they are speaking with.",
        timeout_sec = 25,
    ),

    # ── Silbot ────────────────────────────────────────────────────────────────

    DemoStep(
        step_id     = "introduce_silbot",
        robot_id    = PEPPER,
        text        = "[POINT] And finally, meet Silbot. Silbot, your turn!",
        timeout_sec = 10,
    ),

    DemoStep(
        step_id     = "silbot_greeting",
        robot_id    = SILBOT,
        text        = "[WAVE] Hello! I am Silbot. I focus on physical interaction and navigation.",
        timeout_sec = 15,
    ),

    DemoStep(
        step_id     = "ask_silbot_project",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Silbot, tell us about your project.",
        timeout_sec = 10,
    ),

    DemoStep(
        step_id     = "silbot_project",
        robot_id    = SILBOT,
        text        = "[DEFAULT] My research is on human-aware navigation. "
                      "I learn to move safely and naturally in spaces shared with people.",
        timeout_sec = 25,
    ),

    # ── Closing ───────────────────────────────────────────────────────────────

    DemoStep(
        step_id     = "pepper_wrap_up",
        robot_id    = PEPPER,
        text        = "[HAPPY] Thank you ChatBox, Navel, and Silbot! "
                      "As you can see, each robot brings a unique capability to the lab.",
        timeout_sec = 15,
    ),

    DemoStep(
        step_id     = "invite_questions",
        robot_id    = PEPPER,
        text        = "[DEFAULT] We would love to hear your questions. "
                      "Feel free to speak to any of us — we are all here to help!",
        timeout_sec = 15,
    ),

]
