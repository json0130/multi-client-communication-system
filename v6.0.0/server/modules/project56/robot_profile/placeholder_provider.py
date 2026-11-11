"""Fixed Navel profile used until a database-backed provider is available."""

from __future__ import annotations

from types import MappingProxyType

from modules.project56.co_speech.profiles import (
    NAVEL_LINEAR_SPEECH_DURATION_PROFILE,
    NAVEL_SPEECH_TIMING_PROFILE,
)
from modules.project56.robot_profile.chatbox_profile import CHATBOX_PROFILE
from modules.project56.robot_profile.models import (
    ActionProfile,
    RobotProfile,
    RobotSpeechProfile,
)


def _expressive(
    tag: str,
    description: str,
    duration_seconds: float,
) -> ActionProfile:
    return ActionProfile(
        tag=tag,
        description=description,
        duration_seconds=duration_seconds,
        profiled=True,
        kind="expressive_action",
    )


_NAVEL_001_PROFILE = RobotProfile(
    robot_id="Navel_001",
    speech=RobotSpeechProfile(
        segmentation_timing=NAVEL_SPEECH_TIMING_PROFILE,
        runtime_duration=NAVEL_LINEAR_SPEECH_DURATION_PROFILE,
    ),
    actions={
        # Source: evaluation/action_durations_2026-10-08.json (robot measurement,
        # 2026-10-08). Values are proposed median profile durations.
        "DEFAULT": ActionProfile(
            tag="DEFAULT",
            description=(
                "Use a generic co-speech speaking gesture for neutral or "
                "explanatory speech when no more specific expressive action "
                "is appropriate."
            ),
            duration_seconds=2.402,
            profiled=True,
            kind="generic_speaking",
        ),
        "POINT": _expressive(
            "POINT",
            "Point towards a person, object, display, or direction "
            "being discussed.",
            4.424062207999668,
        ),
        "EMPHASISE": _expressive(
            "EMPHASISE",
            "Use an expressive hand or arm gesture to reinforce an "
            "important statement.",
            5.965826047999144,
        ),
        "AGREE": _expressive(
            "AGREE",
            "Use an affirmative gesture, such as nodding, when confirming "
            "information or expressing agreement.",
            1.208040928000628,
        ),
        "LISTEN": _expressive(
            "LISTEN",
            "Adopt an attentive pose when inviting the user to continue "
            "speaking.",
            2.008,
        ),
        "UNCERTAIN": _expressive(
            "UNCERTAIN",
            "Use a hesitant gesture when unsure, considering possibilities, "
            "or unable to give a definite answer.",
            1.7096621760010748,
        ),
        "DENY": _expressive(
            "DENY",
            "Use a negative gesture when refusing, disagreeing, or "
            "correcting information.",
            1.5059152319990972,
        ),
        "GREET": _expressive(
            "GREET",
            "Raise one hand in a friendly greeting for a welcome, "
            "introduction, hello, or goodbye.",
            3.6204421760012337,
        ),
        "HAPPY": _expressive(
            "HAPPY",
            "Use a cheerful or celebratory gesture for praise, success, "
            "excitement, or positive news.",
            1.004123583999899,
        ),
        "SURPRISE": _expressive(
            "SURPRISE",
            "React with a surprised gesture when responding to unexpected "
            "information.",
            1.0749532480003836,
        ),
        "SAD": _expressive(
            "SAD",
            "Use a subdued or sympathetic gesture when responding to sadness, "
            "disappointment, or difficulty.",
            1.004316991999076,
        ),
    },
)

_PLACEHOLDER_PROFILES = MappingProxyType({
    _NAVEL_001_PROFILE.robot_id: _NAVEL_001_PROFILE,
    CHATBOX_PROFILE.robot_id: CHATBOX_PROFILE,
})


class PlaceholderRobotProfileProvider:
    """Return immutable fixed profiles without database or calibration I/O."""

    def get_profile(self, robot_id: str) -> RobotProfile:
        try:
            return _PLACEHOLDER_PROFILES[robot_id]
        except KeyError as exc:
            raise KeyError(
                f"No robot profile configured for {robot_id!r}"
            ) from exc
