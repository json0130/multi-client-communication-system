"""Data contracts shared by the Method B co-speech pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechTimingProfile:
    """Robot speech timing configuration used for duration metadata."""

    words_per_second: float

    def __post_init__(self) -> None:
        if self.words_per_second <= 0:
            raise ValueError("words_per_second must be greater than zero")


@dataclass(frozen=True)
class LinearSpeechDurationProfile:
    """Fixed robot-specific parameters for runtime duration estimation."""

    intercept_seconds: float
    seconds_per_word: float
    model: str = "linear_word_count"

    def __post_init__(self) -> None:
        if self.model != "linear_word_count":
            raise ValueError("model must be 'linear_word_count'")
        if (
            isinstance(self.intercept_seconds, bool)
            or not isinstance(self.intercept_seconds, (int, float))
            or not math.isfinite(self.intercept_seconds)
            or self.intercept_seconds < 0
        ):
            raise ValueError("intercept_seconds must be finite and non-negative")
        if (
            isinstance(self.seconds_per_word, bool)
            or not isinstance(self.seconds_per_word, (int, float))
            or not math.isfinite(self.seconds_per_word)
            or self.seconds_per_word <= 0
        ):
            raise ValueError("seconds_per_word must be finite and greater than zero")


@dataclass(frozen=True)
class AtomicSpeechUnit:
    """One deterministic candidate span supplied to the LLM segmenter."""

    unit_id: int
    start_offset: int
    end_offset: int
    text: str

    def __post_init__(self) -> None:
        if self.unit_id <= 0:
            raise ValueError("unit_id must be greater than zero")
        if self.start_offset < 0 or self.end_offset <= self.start_offset:
            raise ValueError("atomic unit offsets must define a non-empty span")
        if not self.text:
            raise ValueError("atomic unit text must be non-empty")
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("atomic unit offsets must match its text length")

    def to_dict(self) -> dict[str, int | str]:
        """Return evaluation/debug data without changing the source slice."""
        return {
            "unit_id": self.unit_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "text": self.text,
        }


@dataclass(frozen=True)
class SpeechSegment:
    """One final, ordered unit of robot speech."""

    segment_id: int
    text: str
    word_count: int
    estimated_duration: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return the wire-format representation used by the gateways."""
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "word_count": self.word_count,
            "estimated_duration": self.estimated_duration,
        }
