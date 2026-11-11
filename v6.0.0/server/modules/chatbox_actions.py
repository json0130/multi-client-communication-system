"""What the ChatBox (Jetson + ESP32) can actually perform, as the tags its client understands.

The Jetson turns a `[TAG]` into an ESP32 expression through EMOTION_MAP (FINAL_DEMO/CHATBOX_CLIENT/
robot.py); a tag it doesn't know ("[EXCITED]", "[CURIOUS]") silently becomes the neutral pose. So the
chat prompt only offers the tags below, and `normalize_tag` maps whatever the model still writes onto
one of them. Tags here are EMOTION_MAP keys that give a DISTINCT motion (a test checks that).

Left out on purpose: ANGRY, SLEEP (the client can do them; a companion shouldn't pick them).
Firmware moves with no client tag yet (head_nod, head_shake, ears_perk, dance_*): add a tag to
EMOTION_MAP first, then list it here.
"""
from __future__ import annotations

ACTIONS = {          # tag -> when to use it (this text goes into the prompt)
    "DEFAULT": "calm neutral pose: ordinary statements, listening",
    "GREETING": "open, welcoming pose: hello or goodbye, someone just arrived or you are introducing yourself",
    "POINT": "point: showing or suggesting something, a direction, 'over there'",
    "CONFUSED": "head tilt: puzzled, unsure, curious about something",
    "SHRUG": "shrug: 'I don't know', 'maybe', not sure",
    "SAD": "droop: sad, sorry, sympathetic",
    "HANDS_CLAP": "clap: excited, happy, praise, good news, celebration",
    "EARS_WIGGLE": "ear wiggle: playful, amused, cheeky",
    "POSE": "strike a pose: showing off, proud, confident, 'ta-da'",
}

# words a model uses for a feeling -> the action that shows it
ALIASES = {
    "HAPPY": "HANDS_CLAP", "EXCITED": "HANDS_CLAP", "JOY": "HANDS_CLAP", "JOYFUL": "HANDS_CLAP",
    "CHEERFUL": "HANDS_CLAP", "DELIGHTED": "HANDS_CLAP", "PROUD": "HANDS_CLAP",
    "THRILLED": "HANDS_CLAP", "CELEBRATE": "HANDS_CLAP", "CELEBRATING": "HANDS_CLAP",
    "ENTHUSIASTIC": "HANDS_CLAP", "PRAISE": "HANDS_CLAP", "ENCOURAGING": "HANDS_CLAP",
    "HELLO": "GREETING", "HI": "GREETING", "GOODBYE": "GREETING", "BYE": "GREETING", "FAREWELL": "GREETING",
    "WAVE": "GREETING",             # no wave any more
    "WELCOME": "GREETING", "WELCOMING": "GREETING", "FRIENDLY": "GREETING", "WARM": "GREETING",
    "CURIOUS": "CONFUSED", "INTERESTED": "CONFUSED", "THINKING": "CONFUSED", "THOUGHTFUL": "CONFUSED",
    "PUZZLED": "CONFUSED", "UNSURE": "CONFUSED", "SURPRISE": "CONFUSED", "SURPRISED": "CONFUSED",
    "WONDER": "CONFUSED", "FEAR": "SAD", "SORRY": "SAD", "SYMPATHETIC": "SAD", "EMPATHETIC": "SAD",
    "CONCERNED": "SAD", "WORRIED": "SAD", "CARING": "SAD", "GENTLE": "SAD", "COMFORTING": "SAD",
    "PLAYFUL": "EARS_WIGGLE", "AMUSED": "EARS_WIGGLE", "FUNNY": "EARS_WIGGLE", "CHEEKY": "EARS_WIGGLE",
    "NEUTRAL": "DEFAULT", "CALM": "DEFAULT", "RELAXED": "DEFAULT", "SUPPORTIVE": "DEFAULT",
    "LISTENING": "DEFAULT", "ATTENTIVE": "DEFAULT",
}


def normalize_tag(tag: str) -> str:
    """Any tag text -> one of ACTIONS. Empty stays empty (plain speech); unknown -> DEFAULT."""
    key = (tag or "").strip().strip("[]").strip().upper().replace(" ", "_").replace("-", "_")
    if not key:
        return ""
    return key if key in ACTIONS else ALIASES.get(key, "DEFAULT")


def tag_rule() -> str:
    """The prompt line that tells the model which tags exist."""
    return ("• Begin every reply with exactly ONE of these action tags in square brackets, "
            "chosen to match what you say: "
            + "; ".join(f"[{t}] = {d}" for t, d in ACTIONS.items())
            + ". Use no other tag and no other emotion word.\n")
