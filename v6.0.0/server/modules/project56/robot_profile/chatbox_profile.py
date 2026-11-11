"""ChatBox (Jetson + ESP32) profile: which planner actions it can perform, and how.

The planner speaks the shared action vocabulary (GREET, POINT, ...). This profile lists the
subset ChatBox can perform and CHATBOX_NATIVE_TAG gives the tag the Jetson client
(robot.py EMOTION_MAP) turns into an ESP32 expression.

Only actions with a distinct motion through the CURRENT client are listed. Left out on purpose:
AGREE/DENY (firmware has head_nod/head_shake, but EMOTION_MAP has no tag for them, so they
would fall back to "default"), and HAPPY/FEAR/SURPRISE as native tags (they alias other motions).
"""

from __future__ import annotations

from modules.project56.co_speech.profiles import (
    NAVEL_LINEAR_SPEECH_DURATION_PROFILE,
    NAVEL_SPEECH_TIMING_PROFILE,
)
from modules.project56.robot_profile.models import (
    ActionProfile,
    RobotProfile,
    RobotSpeechProfile,
)

# CALIBRATION KNOB. Derived from the firmware, not measured: every expression plays 5 steps,
# one per 900 ms (ServoControl.ino executeExpression), so ~3.6 s to issue the last step plus
# servo travel. Time a real run on the robot and replace.
EXPRESSION_SECONDS = 4.5

# PLACEHOLDER. Navel's speech timing, not measured on ChatBox (edge-tts on the Jetson).
# Time a few replies and replace; wrong values only shift when gestures are scheduled.
_SPEECH = RobotSpeechProfile(
    segmentation_timing=NAVEL_SPEECH_TIMING_PROFILE,
    runtime_duration=NAVEL_LINEAR_SPEECH_DURATION_PROFILE,
)

# planner action -> native tag sent as "[TAG] text" (Jetson EMOTION_MAP key)
CHATBOX_NATIVE_TAG: dict[str, str] = {
    "DEFAULT": "DEFAULT",
    "GREET": "GREETING",
    "POINT": "POINT",
    "UNCERTAIN": "CONFUSED",
    "SAD": "SAD",
    "HAPPY": "HANDS_CLAP",
}

_DESCRIPTIONS = {
    "DEFAULT": ("generic_speaking",
                "Return to the neutral speaking pose for ordinary statements and "
                "explanations when no more specific action fits."),
    "GREET": ("expressive_action",
              "Open, welcoming pose for a welcome, hello, "
              "introduction or goodbye."),
    "POINT": ("expressive_action",
              "Point an arm towards a person, object, display or direction being discussed."),
    "UNCERTAIN": ("expressive_action",
                  "Tilt the head from side to side with a puzzled look when unsure, "
                  "considering possibilities or unable to give a definite answer."),
    "SAD": ("expressive_action",
            "Droop the ears and lower the head for sadness, sympathy, disappointment or difficulty."),
    "HAPPY": ("expressive_action",
              "Clap both hands with raised shoulders for praise, success, excitement or good news."),
}

CHATBOX_PROFILE = RobotProfile(
    robot_id="chatbox_jetson_001",       # client_id the Jetson registers with the hub
    speech=_SPEECH,
    actions={
        tag: ActionProfile(tag=tag, description=desc, duration_seconds=EXPRESSION_SECONDS,
                           profiled=True, kind=kind)
        for tag, (kind, desc) in _DESCRIPTIONS.items()
    },
)
