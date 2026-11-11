"""Stable text metrics shared by runtime co-speech components."""

from __future__ import annotations

import re


# Matches the frozen Speech Segmenter and Navel calibration token rule.
WORD_PATTERN = re.compile(r"\b[\w]+(?:'[\w]+)*\b")


def count_spoken_words(text: str) -> int:
    """Count word-like spoken tokens without normalising the source text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(WORD_PATTERN.findall(text))
