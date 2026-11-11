"""Persona -> how the robot SOUNDS: speed, pitch, tone, volume on top of its base voice.

The base voice (gTTS on the ChatBox Jetson) is unchanged; these are offsets the client applies
with ffmpeg (FINAL_DEMO/CHATBOX_CLIENT/OutputModules/edge_tts_output.py). Sent to the robot as
`voice_config` in persona_update. All numbers are a starting point to tune by ear.

    rate    "+8%"    speed vs the client's normal pace
    pitch   "+1.5st" semitones; speed is not affected
    tone    "+2dB"   > 0 brighter (treble up, bass down), < 0 warmer
    volume  "+1dB"
"""

from __future__ import annotations

import re
from typing import Dict, Optional

# companion, storyteller and tutor are ONE voice (the client's base voice); each role and each
# OCEAN setting only nudges it. role label -> (rate %, pitch st, tone dB, volume dB), added to the OCEAN part below
ROLE_VOICE = {
    "companion":     (-4, 0.0, -1, 0),
    "storyteller":   (-10, 0.5, -1, 0),
    "tutor":         (0, 0.5, 1, 0),
}
# The other roles are different PEOPLE. The client's base voice (gTTS) is a single female voice, so
# a person is made by shifting its pitch well past what OCEAN ever does (and slowing/speeding + toning it):
# a man = pitch down, a young boy = pitch up. "gender" is sent along for the client's own use.
# role label -> (rate %, pitch st, tone dB, volume dB, gender). Tune by ear.
CHARACTER_VOICE = {
    "receptionist":  (-3, -5.0, -3, 1, "male"),     # adult man
    "reading buddy": (6, 5.0, 2, 0, "male"),        # young boy
}
LIMITS = {"rate": 25, "pitch": 4, "tone": 6, "volume": 3}   # per side; keeps ffmpeg's atempo in range
CHARACTER_LIMITS = {"rate": 30, "pitch": 7, "tone": 6, "volume": 3}


def _clamp(v: float, key: str, limits: Dict[str, float] = LIMITS) -> float:
    return max(-limits[key], min(limits[key], v))


def voice_for(role_label: str, ocean: Dict[str, float],
              adjust: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """OCEAN in -1..1. Extraverts talk faster, higher and brighter; neurotic voices a touch
    faster and higher but quieter; agreeable voices warmer; conscientious ones a little slower."""
    character = CHARACTER_VOICE.get(role_label)
    r0, p0, t0, v0 = character[:4] if character else ROLE_VOICE.get(role_label, (0, 0.0, 0, 0))
    lim = CHARACTER_LIMITS if character else LIMITS
    o = {k: float(ocean.get(k, 0.0)) for k in "OCEAN"}
    a = {k: float((adjust or {}).get(k) or 0.0) for k in ("rate", "pitch", "tone", "volume")}   # per-robot tweak
    rate = _clamp(r0 + 14 * o["E"] + 5 * o["N"] - 3 * o["C"] + a["rate"], "rate", lim)
    pitch = _clamp(p0 + 2.5 * o["E"] + 1.5 * o["N"] + a["pitch"], "pitch", lim)
    tone = _clamp(t0 + 3 * o["E"] - 2 * o["A"] + a["tone"], "tone", lim)
    volume = _clamp(v0 + 2 * o["E"] - 1 * o["N"] + a["volume"], "volume", lim)
    out = {"rate": f"{rate:+.0f}%", "pitch": f"{pitch:+.1f}st",
           "tone": f"{tone:+.0f}dB", "volume": f"{volume:+.0f}dB"}
    if character:
        out["gender"] = character[4]
    return out


def _num(value, unit: str) -> float:
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*" + unit + r"\s*", str(value), re.I)
    return float(m.group(1)) if m else 0.0


def navel_tts_tags(voice: Dict[str, str], base_volume: int = 60) -> str:
    """The same voice spec as Navel TTS tags (Navel SDK 'TTS Tags'), put in front of the text it speaks:
    <vol,N> 20..400 (Navel's client default is 60), <rpit,N> pitch as % of the default, <rspd,N> speed in %.
    Navel has no tone control, so `tone` is ignored. Its client prefixes its own defaults on every
    utterance; ours come after them in the text, so ours win."""
    rate = max(50, min(200, round(100 + _num(voice.get("rate", 0), "%"))))
    pitch = max(50, min(200, round(100 * 2 ** (_num(voice.get("pitch", 0), "st") / 12))))
    vol = max(20, min(400, round(base_volume * 10 ** (_num(voice.get("volume", 0), "dB") / 20))))
    return f"<vol,{vol}><rpit,{pitch}><rspd,{rate}>"
