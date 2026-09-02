"""
Turn a topic's (affinity, confidence) into the signed, hedged sentence the
person-memory prompt shows the robot. PURE presentation — imports nothing beyond
the scale helpers, so it is unit-testable headlessly and reused by the prompt/viz.

Two independent axes:
  * AFFINITY → the verb (positive/negative signal):
        >= 0.66  "like"      (likes)
        <= 0.34  "dislike"   (dislikes; the line also warns the robot off it)
        else     neutral     (unsure / not stated)
  * CONFIDENCE → the hedge (how sure the reading is):
        >= 0.85  "clearly"
        >= 0.60  "probably"
        else     "possibly"  (rare — the extraction gate is 0.6 — but handled)

Confidence changes ONLY the wording here; it never touches the BN clamp.
"""

from __future__ import annotations

# Bucket thresholds (documented once here).
_LIKE_AT = 0.66
_DISLIKE_AT = 0.34
_CLEARLY_AT = 0.85
_PROBABLY_AT = 0.60


def affinity_word(affinity: float) -> str:
    """'like' | 'dislike' | 'neutral' from an internal [0,1] affinity."""
    a = float(affinity)
    if a >= _LIKE_AT:
        return "like"
    if a <= _DISLIKE_AT:
        return "dislike"
    return "neutral"


def confidence_hedge(confidence: float) -> str:
    """'clearly' | 'probably' | 'possibly' from a [0,1] confidence."""
    c = float(confidence)
    if c >= _CLEARLY_AT:
        return "clearly"
    if c >= _PROBABLY_AT:
        return "probably"
    return "possibly"


def topic_memory_line(topic_label: str, affinity: float, confidence: float) -> str:
    """One signed, hedged sentence about a topic, e.g.
        like,    high conf → "They clearly like jazz."
        dislike, mid  conf → "They probably dislike baseball — avoid raising it."
        neutral            → "They may be neutral on pasta."
    A neutral reading is intentionally confidence-agnostic (there is nothing to
    hedge about 'unsure')."""
    label = str(topic_label).strip()
    bucket = affinity_word(affinity)
    hedge = confidence_hedge(confidence)
    if bucket == "like":
        return f"They {hedge} like {label}."
    if bucket == "dislike":
        return f"They {hedge} dislike {label} — avoid raising it."
    return f"They may be neutral on {label}."
