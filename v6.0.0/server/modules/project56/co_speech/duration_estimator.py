"""Runtime speech-duration estimation after frozen speech segmentation."""

from __future__ import annotations

from collections.abc import Sequence

from modules.project56.co_speech.models import (
    LinearSpeechDurationProfile,
    SpeechSegment,
)
from modules.project56.co_speech.profiles import NAVEL_LINEAR_SPEECH_DURATION_PROFILE
from modules.project56.co_speech.text_metrics import count_spoken_words


class SpeechDurationEstimator:
    """Apply a fixed calibrated linear word-count model at runtime."""

    def __init__(
        self,
        profile: LinearSpeechDurationProfile = (
            NAVEL_LINEAR_SPEECH_DURATION_PROFILE
        ),
    ) -> None:
        if not isinstance(profile, LinearSpeechDurationProfile):
            raise TypeError("profile must be a LinearSpeechDurationProfile")
        self.profile = profile

    def estimate(self, text: str) -> float:
        """Estimate seconds for non-empty speech without fitting or I/O."""
        word_count = count_spoken_words(text)
        if word_count == 0:
            raise ValueError("speech text must contain at least one word")
        return (
            self.profile.intercept_seconds
            + self.profile.seconds_per_word * word_count
        )

    def annotate_segments(
        self,
        segments: Sequence[SpeechSegment],
    ) -> list[SpeechSegment]:
        """Return new duration-annotated segments without mutating input."""
        annotated: list[SpeechSegment] = []
        for segment in segments:
            if not isinstance(segment, SpeechSegment):
                raise TypeError("segments must contain SpeechSegment objects")
            word_count = count_spoken_words(segment.text)
            if word_count == 0:
                raise ValueError("segment text must contain at least one word")
            annotated.append(SpeechSegment(
                segment_id=segment.segment_id,
                text=segment.text,
                word_count=word_count,
                estimated_duration=(
                    self.profile.intercept_seconds
                    + self.profile.seconds_per_word * word_count
                ),
            ))
        return annotated
