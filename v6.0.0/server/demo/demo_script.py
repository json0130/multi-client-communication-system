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
        text        = "You are opening the CARES lab demonstration for a group of visitors. "
                      "Welcome them warmly, introduce yourself as Pepper the lab guide, and let them know "
                      "you are excited to show them around today. Keep it to 2 sentences. Start with [GREETING].",
        generate    = True,
        timeout_sec = 60,
    ),

    DemoStep(
        step_id     = "lab_intro",
        robot_id    = PEPPER,
        text        = "Briefly explain what CARES stands for (Centre for Automation and Robotic Engineering Science) "
                      "and what the lab researches — intelligent robots that can communicate, collaborate, and assist "
                      "people in real-world environments. Keep it to 2-3 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 60,
    ),

    DemoStep(
        step_id     = "overview",
        robot_id    = PEPPER,
        text        = "Set expectations for the demo: visitors will meet three research robots today, "
                      "each working on a different project. After each robot speaks, there will be time "
                      "to ask questions. Keep it to 2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
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
    ),

    DemoStep(
        step_id     = "introduce_chatbox",
        robot_id    = PEPPER,
        text        = "Hand off to ChatBox, the robot leading the conversational AI research. "
                      "Point towards ChatBox and invite them to say hello to the visitors. "
                      "1-2 sentences. Use [POINT].",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "chatbox_greeting",
        robot_id    = CHATBOX,
        text        = "Greet the visitors warmly for the first time. You are ChatBox. "
                      "Introduce yourself and express genuine excitement about meeting the visitors. "
                      "2 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "chatbox_prompt",
        robot_id    = PEPPER,
        text        = "Ask ChatBox to explain their research project to the visitors. "
                      "1 sentence. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 40,
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
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT B — Navel: Emotion-Aware Interaction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_b",
        robot_id    = PEPPER,
        text        = "Transition from the ChatBox Q&A to the second project. "
                      "Give a brief, warm sign-off to ChatBox and announce you are moving on. "
                      "1-2 sentences. Use [DEFAULT]. Include 'let us move on' in your response.",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "intro_project_b",
        robot_id    = PEPPER,
        text        = "Introduce the second research project: emotion-aware interaction. "
                      "Explain that this project studies how robots can recognise a person's emotional state "
                      "and adapt their communication style accordingly. 2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
    ),

    DemoStep(
        step_id     = "introduce_navel",
        robot_id    = PEPPER,
        text        = "Hand off to Navel, the robot leading emotion-aware interaction research. "
                      "Point towards Navel and invite them to say hi to everyone. "
                      "1-2 sentences. Use [POINT].",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "navel_greeting",
        robot_id    = NAVEL,
        text        = "Greet the visitors warmly for the first time. You are Navel. "
                      "Introduce yourself and mention that meeting new people is literally part of your research. "
                      "2 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "navel_prompt",
        robot_id    = PEPPER,
        text        = "Ask Navel to share what their research is about with the visitors. "
                      "1 sentence. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 40,
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
    ),

    # Q&A window
    DemoStep(
        step_id     = "qa_invite_b",
        robot_id    = PEPPER,
        text        = "Open a Q&A session after Navel's presentation on emotion-aware interaction. "
                      "Invite visitors to ask questions — they can speak to Navel or to you. "
                      "Let them know you will wait until everyone is ready to continue. "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,    # manual advance only — operator clicks Move On
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROJECT C — Silbot: Human-Aware Navigation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "transition_to_c",
        robot_id    = PEPPER,
        text        = "Transition from the Navel Q&A to the third and final project. "
                      "Brief warm sign-off to Navel and announce the move to the next project. "
                      "1-2 sentences. Use [DEFAULT]. Include 'let us move on' in your response.",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "intro_project_c",
        robot_id    = PEPPER,
        text        = "Introduce the third research project: human-aware navigation. "
                      "Frame it as the question: how can a robot move through a crowded space "
                      "safely, politely, and predictably — the way a person would? "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
    ),

    DemoStep(
        step_id     = "introduce_silbot",
        robot_id    = PEPPER,
        text        = "Hand off to Silbot, the robot specialising in human-aware navigation. "
                      "Point towards Silbot and invite them to come say hello to the visitors. "
                      "1-2 sentences. Use [POINT].",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "silbot_greeting",
        robot_id    = SILBOT,
        text        = "Greet the visitors warmly for the first time. You are Silbot. "
                      "Introduce yourself and briefly mention that you navigate spaces with awareness and courtesy. "
                      "2 sentences. Start with [WAVE].",
        generate    = True,
        timeout_sec = 50,
    ),

    DemoStep(
        step_id     = "silbot_prompt",
        robot_id    = PEPPER,
        text        = "Ask Silbot to explain their navigation research to the visitors. "
                      "1 sentence. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 40,
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
    ),

    # Q&A window
    DemoStep(
        step_id     = "qa_invite_c",
        robot_id    = PEPPER,
        text        = "Open a Q&A session after Silbot's presentation on human-aware navigation. "
                      "Invite visitors to ask questions about navigation and social robotics — "
                      "they can speak to Silbot or to you. "
                      "Let them know you will be here until everyone is ready to wrap up. "
                      "2 sentences. Use [DEFAULT].",
        generate    = True,
        timeout_sec = 60,
        qa_window   = True,
        qa_timeout  = 0,    # manual advance only — operator clicks Move On
    ),

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CLOSING — Pepper wraps up
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    DemoStep(
        step_id     = "wrap_up",
        robot_id    = PEPPER,
        text        = "Close the main part of the demo. Thank ChatBox, Navel, and Silbot by name. "
                      "Summarise that each robot brings a unique capability and together you are all "
                      "working towards robots that can truly work alongside people. "
                      "2-3 sentences. Use [HAPPY].",
        generate    = True,
        timeout_sec = 60,
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
    ),

]
