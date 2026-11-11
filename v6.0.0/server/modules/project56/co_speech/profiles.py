"""Robot-specific speech timing profiles for co-speech processing."""

from modules.project56.co_speech.models import (
    LinearSpeechDurationProfile,
    SpeechTimingProfile,
)


# Median of the real-robot lab-tour calibration dataset. This profile estimates
# duration after segmentation; it does not decide segment boundaries.
NAVEL_SPEECH_TIMING_PROFILE = SpeechTimingProfile(
    words_per_second=2.556228887,
)


# Selected from the all-lab fit recorded by the duration-estimator comparison.
# Temporary source until robot-specific timing metadata is loaded from the DB.
NAVEL_LINEAR_SPEECH_DURATION_PROFILE = LinearSpeechDurationProfile(
    intercept_seconds=0.5379889576211432,
    seconds_per_word=0.32360131854586377,
)
