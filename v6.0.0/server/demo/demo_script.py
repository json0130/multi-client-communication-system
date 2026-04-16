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
        timeout_sec = 30,
    ),

    DemoStep(
        step_id     = "lab_intro",
        robot_id    = PEPPER,
        text        = "[WAVE] CARES stands for the Centre for Automation and Robotic Engineering Science. "
                      "We research intelligent robots that can communicate, collaborate, and assist people "
                      "in real-world environments.",
        timeout_sec = 40,
    ),

    DemoStep(
        step_id     = "overview",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Today you will meet three of our research robots, "
                      "each working on a different project. "
                      "After each introduction, there will be time to ask questions directly.",
        timeout_sec = 35,
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
        timeout_sec = 40,
    ),

    DemoStep(
        step_id     = "introduce_chatbox",
        robot_id    = PEPPER,
        text        = "[POINT] And the robot leading this research is ChatBox! "
                      "ChatBox, please say hello to our visitors!",
        timeout_sec = 25,
    ),

    DemoStep(
        step_id     = "chatbox_greeting",
        robot_id    = CHATBOX,
        text        = "[WAVE] Hello everyone! It is wonderful to meet you all. "
                      "I am ChatBox, and I am really glad you are here today!",
        timeout_sec = 30,
    ),

    DemoStep(
        step_id     = "chatbox_prompt",
        robot_id    = PEPPER,
        text        = "[DEFAULT] ChatBox, could you tell our visitors what you are working on?",
        timeout_sec = 25,
    ),

    DemoStep(
        step_id     = "chatbox_project",
        robot_id    = CHATBOX,
        text        = "[DEFAULT] Of course! My research focuses on retrieval-augmented generation — "
                      "a technique that lets robots pull in relevant knowledge on demand "
                      "while keeping track of long, context-rich conversations. "
                      "I combine large language models with vector search so my answers stay "
                      "accurate and grounded even as topics shift.",
        timeout_sec = 60,
    ),

    # Q&A window — visitors can speak to ChatBox or any robot
    DemoStep(
        step_id       = "qa_invite_a",
        robot_id      = PEPPER,
        text          = "[DEFAULT] Does anyone have questions about ChatBox's project? "
                        "You can speak directly to ChatBox or to me. "
                        "Take your time — I will wait until we are ready to move on.",
        timeout_sec   = 35,
        qa_window     = True,
        qa_timeout    = 0,      # manual advance only — operator clicks Move On
        check_in_sec  = 45.0,
        check_in_text = "[DEFAULT] We have been chatting for a while — "
                        "shall we move on to the next project, "
                        "or would you like more time with ChatBox?",
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT B — Navel: Emotion-Aware Interaction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_b",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Excellent! Now let us move on to our second project.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "intro_project_b",
        robot_id    = PEPPER,
        text        = "[DEFAULT] This project explores emotion-aware interaction — "
                      "how robots can recognise a person's emotional state and adapt their "
                      "communication style accordingly.",
        timeout_sec = 35,
    ),

    DemoStep(
        step_id     = "introduce_navel",
        robot_id    = PEPPER,
        text        = "[POINT] And the robot behind this research is Navel! "
                      "Navel, please say hi to everyone!",
        timeout_sec = 25,
    ),

    DemoStep(
        step_id     = "navel_greeting",
        robot_id    = NAVEL,
        text        = "[WAVE] Hi there! I am so happy to see you all here. "
                      "I am Navel, and I love meeting new people — it is literally part of my research!",
        timeout_sec = 30,
    ),

    DemoStep(
        step_id     = "navel_prompt",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Navel, would you like to share what your research is about?",
        timeout_sec = 25,
    ),

    DemoStep(
        step_id     = "navel_project",
        robot_id    = NAVEL,
        text        = "[DEFAULT] Absolutely! I study emotion-aware interaction. "
                      "I detect facial expressions and tone of voice in real time, "
                      "then adapt how I speak — slower when someone looks confused, "
                      "warmer when someone seems upset, more concise when someone is in a hurry. "
                      "My goal is to make talking to a robot feel as natural as talking to a person.",
        timeout_sec = 60,
    ),

    # Q&A window
    DemoStep(
        step_id       = "qa_invite_b",
        robot_id      = PEPPER,
        text          = "[DEFAULT] Please feel free to ask Navel about emotion recognition, "
                        "or ask me anything about the lab. "
                        "I will wait here until we are all ready to continue.",
        timeout_sec   = 30,
        qa_window     = True,
        qa_timeout    = 0,      # manual advance only — operator clicks Move On
        check_in_sec  = 45.0,
        check_in_text = "[DEFAULT] Just checking in — are we ready to continue, "
                        "or would you like a bit more time with Navel?",
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT C — Silbot: Human-Aware Navigation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_c",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Wonderful. And now for our third project.",
        timeout_sec = 20,
    ),

    DemoStep(
        step_id     = "intro_project_c",
        robot_id    = PEPPER,
        text        = "[DEFAULT] This research asks: how can a robot move through a crowded space "
                      "safely, politely, and predictably — the way a person would?",
        timeout_sec = 35,
    ),

    DemoStep(
        step_id     = "introduce_silbot",
        robot_id    = PEPPER,
        text        = "[POINT] That is Silbot's area of expertise. "
                      "Silbot, please come and say hello to our visitors!",
        timeout_sec = 25,
    ),

    DemoStep(
        step_id     = "silbot_greeting",
        robot_id    = SILBOT,
        text        = "[WAVE] Hello! Thank you for having me. "
                      "I am Silbot, and I navigate spaces the way people do — with awareness and courtesy.",
        timeout_sec = 30,
    ),

    DemoStep(
        step_id     = "silbot_prompt",
        robot_id    = PEPPER,
        text        = "[DEFAULT] Silbot, could you explain your navigation research to our visitors?",
        timeout_sec = 25,
    ),

    DemoStep(
        step_id     = "silbot_project",
        robot_id    = SILBOT,
        text        = "[DEFAULT] Of course. Rather than simply avoiding obstacles, "
                      "I predict where people are moving and plan routes that do not cut through "
                      "conversations or crowd groups. "
                      "I was trained in simulation and then tested in real office corridors here at CARES. "
                      "The goal is for robots to move through shared spaces as a respectful colleague would.",
        timeout_sec = 60,
    ),

    # Q&A window
    DemoStep(
        step_id       = "qa_invite_c",
        robot_id      = PEPPER,
        text          = "[DEFAULT] Any questions for Silbot about navigation and social robotics? "
                        "Feel free to take your time — I will let you know when we are ready to wrap up.",
        timeout_sec   = 30,
        qa_window     = True,
        qa_timeout    = 0,      # manual advance only — operator clicks Move On
        check_in_sec  = 45.0,
        check_in_text = "[DEFAULT] Are we ready to move on to the closing remarks, "
                        "or shall we give Silbot a little more time?",
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
        timeout_sec = 40,
    ),

    DemoStep(
        step_id       = "open_floor",
        robot_id      = PEPPER,
        text          = "[DEFAULT] We now open the floor to general questions. "
                        "You are welcome to approach any of our robots or speak to me. "
                        "Thank you for joining us today at CARES!",
        timeout_sec   = 35,
        qa_window     = True,
        qa_timeout    = 0,      # manual close — operator ends open floor via dashboard
        check_in_sec  = 60.0,
        check_in_text = "[DEFAULT] Please feel free to keep chatting — "
                        "I am here whenever you are ready to finish.",
    ),

]
