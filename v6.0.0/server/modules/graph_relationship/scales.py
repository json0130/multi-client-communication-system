"""
Affinity scale boundary helpers (pure — no imports beyond stdlib).

Two representations of the same signal live in the system:
  * HUMAN-FACING  0–10   (0 = clearly dislikes, 5 = neutral, 10 = clearly likes) —
    what the extraction LLM emits and what the viz shows.
  * INTERNAL      [0,1]  (0.0 dislike / 0.5 neutral / 1.0 like) — what is STORED on
    the about edge and read straight into the BN clamp with zero conversion.

These two functions are the ONLY place the two scales are converted. No other code
should hand-multiply/divide by 10.
"""

from __future__ import annotations


def aff01_from_10(x: float) -> float:
    """Human 0–10 sentiment → internal [0,1] affinity. Clamps out-of-range input."""
    v = max(0.0, min(10.0, float(x)))
    return v / 10.0


def aff10_from_01(x: float) -> float:
    """Internal [0,1] affinity → human 0–10 sentiment (round when displaying)."""
    v = max(0.0, min(1.0, float(x)))
    return v * 10.0
