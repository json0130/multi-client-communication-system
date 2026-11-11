"""Filter Method B action candidates using annotated speech duration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from modules.project56.co_speech.models import SpeechSegment
from modules.project56.robot_profile.models import ActionProfile


@dataclass(frozen=True)
class DurationFilterResult:
    """Eligible canonical tags and reasons for rejected candidates."""

    eligible_tags: tuple[str, ...]
    rejected: Mapping[str, str]


def _valid_seconds(value: object, *, allow_zero: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (value >= 0 if allow_zero else value > 0)
    )


class DurationFilter:
    """Keep actions whose profiled duration fits an annotated segment.

    The caller supplies an already segmented and runtime-duration-annotated
    ``SpeechSegment`` plus the resolved ``RobotProfile.actions`` mapping. The
    filter does not estimate speech duration, load calibration data, insert a
    fallback action, or mutate its inputs.
    """

    def __init__(
        self,
        margin_seconds: float = 0.0,
        overrun_allowance_seconds: float = 0.0,
    ):
        if not _valid_seconds(margin_seconds, allow_zero=True):
            raise ValueError("margin_seconds must be finite and non-negative")
        if not _valid_seconds(overrun_allowance_seconds, allow_zero=True):
            raise ValueError(
                "overrun_allowance_seconds must be finite and non-negative"
            )
        self.margin_seconds = margin_seconds
        self.overrun_allowance_seconds = overrun_allowance_seconds

    def filter(
        self,
        segment: SpeechSegment,
        allowed_tags: Sequence[str],
        actions: Mapping[str, ActionProfile],
    ) -> DurationFilterResult:
        """Return candidates whose duration fits speech plus allowed overrun."""
        if not _valid_seconds(segment.estimated_duration):
            raise ValueError("Segment duration must be finite and positive")

        tags = dict.fromkeys(
            normalised
            for tag in allowed_tags
            if (normalised := tag.strip().strip("[]").strip().upper())
        )
        eligible: list[str] = []
        rejected: dict[str, str] = {}

        for tag in tags:
            action = actions.get(tag)
            if action is None:
                rejected[tag] = "missing_action_profile"
            elif not action.profiled or action.duration_seconds is None:
                rejected[tag] = "unknown_duration"
            elif not _valid_seconds(action.duration_seconds):
                rejected[tag] = "invalid_action_duration"
            elif (
                action.duration_seconds + self.margin_seconds
                > segment.estimated_duration + self.overrun_allowance_seconds
            ):
                rejected[tag] = "exceeds_segment_duration"
            else:
                eligible.append(tag)

        return DurationFilterResult(
            eligible_tags=tuple(eligible),
            rejected=MappingProxyType(rejected),
        )
