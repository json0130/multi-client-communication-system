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
  • The window auto-closes after `qa_timeout` seconds, or press "Next Step" on dashboard.
  • At ANY time during the demo use POST /demo/qa to open an ad-hoc Q&A window.

Edit instructions:
  • Change PEPPER/CHATBOX/NAVEL/SILBOT to match your client_config.json `client_id`.
  • Edit step `text` fields to customise speech.
  • Adjust `timeout_sec` to match expected TTS duration + buffer.
  • Set `qa_timeout` on Q&A steps (how long to accept visitor questions).
  • Comment out steps to skip them.
  • Restart the server after any change.
"""

from demo.demo_orchestrator import DemoStep

# ── Robot IDs — must match client_id in each robot's client_config.json ───────

PEPPER  = "pepper_01"    # Main guide robot — full DB access
CHATBOX = "chatbox_01"   # Project A robot
NAVEL   = "navel_01"     # Project B robot
SILBOT  = "silbot_01"    # Project C robot

# ── Demo script ────────────────────────────────────────────────────────────────

DEMO_STEPS = [

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # OPENING — Pepper welcomes visitors and introduces the lab
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "greeting",
        robot_id    = PEPPER,
        text        = "[GREETING] Welcome to the CARES lab! "
                      "I am Pepper, your guide for today's demonstration.",
        timeout_sec = 15,
    ),

    DemoStep(
        step_id     = "lab_intro",
        robot_id    = PEPPER,
        text        = "[WAVE] CARES stands for the Centre for Automation and Robotic Engineering Science. "
                      "We research intelligent robots that can communicate, collaborate, and assist people "
                      "in real-world environments.",
        timeout_sec = 22,
    ),

    DemoStep(
        step_id     = "overview",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Today you will meet three of our research robots, "
                      "each working on a different project. "
                      "After each introduction, there will be time to ask questions directly.",
        timeout_sec = 18,
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT A — ChatBox: Conversational AI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "intro_project_a",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Our first project focuses on conversational AI — "
                      "specifically how robots can hold long, contextually aware conversations "
                      "using retrieval-augmented generation.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "introduce_chatbox",
        robot_id    = PEPPER,
        text        = "[POINT] Let me introduce ChatBox, the robot leading this research!",
        timeout_sec = 8,
    ),

    DemoStep(
        step_id     = "chatbox_intro",
        robot_id    = CHATBOX,
        text        = "[WAVE] Hello everyone! I am ChatBox. "
                      "My research is about building robots that remember context across long conversations "
                      "and can retrieve relevant knowledge on demand.",
        timeout_sec = 22,
    ),

    DemoStep(
        step_id     = "chatbox_project_detail",
        robot_id    = CHATBOX,
        text        = "[DEFAULT] I use a combination of large language models and vector search "
                      "to answer questions accurately even when the topic changes. "
                      "My researcher is working on making these responses both factually grounded and natural.",
        timeout_sec = 25,
    ),

    # Q&A window — visitors can speak to ChatBox or any robot
    DemoStep(
        step_id     = "qa_invite_a",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Does anyone have questions about ChatBox's project? "
                      "You can speak directly to ChatBox or to me. "
                      "Take your time — I will wait until we are ready to move on.",
        timeout_sec = 12,
        qa_window   = True,
        qa_timeout  = 0,      # manual advance only — operator clicks Move On
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT B — Navel: Emotion-Aware Interaction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_b",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Excellent! Now let's move on to our second project.",
        timeout_sec = 8,
    ),

    DemoStep(
        step_id     = "intro_project_b",
        robot_id    = PEPPER,
        text        = "[DEFAULT] This project explores emotion-aware interaction — "
                      "how robots can recognise a person's emotional state and adapt their "
                      "communication style accordingly.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "introduce_navel",
        robot_id    = PEPPER,
        text        = "[POINT] Meet Navel, who is working on exactly this!",
        timeout_sec = 8,
    ),

    DemoStep(
        step_id     = "navel_intro",
        robot_id    = NAVEL,
        text        = "[WAVE] Hi! I am Navel. I study how robots can become more empathetic "
                      "by detecting facial expressions and tone of voice in real time.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "navel_project_detail",
        robot_id    = NAVEL,
        text        = "[DEFAULT] When I detect that someone looks confused or upset, "
                      "I adjust my response — speaking more slowly, choosing simpler words, "
                      "or offering to explain further. My goal is to make human-robot interaction "
                      "feel more natural and supportive.",
        timeout_sec = 28,
    ),

    # Q&A window
    DemoStep(
        step_id     = "qa_invite_b",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Please feel free to ask Navel about emotion recognition, "
                      "or ask me anything about the lab. "
                      "I will wait here until we are all ready to continue.",
        timeout_sec = 12,
        qa_window   = True,
        qa_timeout  = 0,      # manual advance only — operator clicks Move On
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT C — Silbot: Human-Aware Navigation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_c",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Wonderful. And now for our third project.",
        timeout_sec = 8,
    ),

    DemoStep(
        step_id     = "intro_project_c",
        robot_id    = PEPPER,
        text        = "[DEFAULT] This research asks: how can a robot move through a crowded space "
                      "safely, politely, and predictably — the way a person would?",
        timeout_sec = 18,
    ),

    DemoStep(
        step_id     = "introduce_silbot",
        robot_id    = PEPPER,
        text        = "[POINT] That is Silbot's area of expertise. Silbot, please say hello!",
        timeout_sec = 8,
    ),

    DemoStep(
        step_id     = "silbot_intro",
        robot_id    = SILBOT,
        text        = "[WAVE] Hello! I am Silbot. I specialise in navigation that respects "
                      "personal space and social norms.",
        timeout_sec = 15,
    ),

    DemoStep(
        step_id     = "silbot_project_detail",
        robot_id    = SILBOT,
        text        = "[DEFAULT] Rather than just avoiding obstacles, I predict where people are going "
                      "and plan paths that do not interrupt conversations or crowd groups. "
                      "I was trained in simulation and tested in real office corridors here at CARES.",
        timeout_sec = 28,
    ),

    # Q&A window
    DemoStep(
        step_id     = "qa_invite_c",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Any questions for Silbot about navigation and social robotics? "
                      "Feel free to take your time — I will let you know when we are ready to wrap up.",
        timeout_sec = 12,
        qa_window   = True,
        qa_timeout  = 0,      # manual advance only — operator clicks Move On
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CLOSING — Pepper wraps up
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "wrap_up",
        robot_id    = PEPPER,
        text        = "[HAPPY] Thank you, ChatBox, Navel, and Silbot! "
                      "As you have seen, each robot brings a unique capability to the lab. "
                      "Together we are building robots that can truly work alongside people.",
        timeout_sec = 22,
    ),

    DemoStep(
        step_id     = "open_floor",
        robot_id    = PEPPER,
        text        = "[DEFAULT] We now open the floor to general questions. "
                      "You are welcome to approach any of our robots or speak to me. "
                      "Thank you for joining us today at CARES!",
        timeout_sec = 18,
        qa_window   = True,
        qa_timeout  = 0,      # manual close — operator ends open floor via dashboard
    ),

]
