"""Immutable robot speech and action metadata contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from modules.project56.co_speech.models import (
    LinearSpeechDurationProfile,
    SpeechTimingProfile,
)


ActionKind = Literal[
    "expressive_action",
    "generic_speaking",
]

_SUPPORTED_ACTION_KINDS = frozenset({
    "expressive_action",
    "generic_speaking",
})
_CANONICAL_TAG_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class RobotSpeechProfile:
    """Separate frozen segmentation timing from runtime duration timing."""

    segmentation_timing: SpeechTimingProfile
    runtime_duration: LinearSpeechDurationProfile

    def __post_init__(self) -> None:
        if not isinstance(self.segmentation_timing, SpeechTimingProfile):
            raise TypeError("segmentation_timing must be a SpeechTimingProfile")
        if not isinstance(self.runtime_duration, LinearSpeechDurationProfile):
            raise TypeError(
                "runtime_duration must be a LinearSpeechDurationProfile"
            )


@dataclass(frozen=True)
class ActionProfile:
    """Semantic and calibrated timing metadata for one canonical action."""

    tag: str
    description: str
    duration_seconds: float | None
    profiled: bool
    kind: ActionKind

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tag, str)
            or not _CANONICAL_TAG_PATTERN.fullmatch(self.tag)
        ):
            raise ValueError(
                "tag must be a non-empty canonical uppercase action tag"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if self.kind not in _SUPPORTED_ACTION_KINDS:
            raise ValueError(f"Unsupported action kind: {self.kind!r}")
        if type(self.profiled) is not bool:
            raise TypeError("profiled must be a bool")

        if self.profiled:
            if (
                isinstance(self.duration_seconds, bool)
                or not isinstance(self.duration_seconds, (int, float))
                or not math.isfinite(self.duration_seconds)
                or self.duration_seconds <= 0
            ):
                raise ValueError(
                    "A profiled action requires a finite positive duration"
                )
        elif self.duration_seconds is not None:
            raise ValueError(
                "An unprofiled action must use duration_seconds=None"
            )


@dataclass(frozen=True)
class RobotProfile:
    """Complete shared metadata for one physical robot configuration."""

    robot_id: str
    speech: RobotSpeechProfile
    actions: Mapping[str, ActionProfile]

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id.strip():
            raise ValueError("robot_id must be a non-empty string")
        if not isinstance(self.speech, RobotSpeechProfile):
            raise TypeError("speech must be a RobotSpeechProfile")
        if not isinstance(self.actions, Mapping):
            raise TypeError("actions must be a mapping")

        immutable_actions: dict[str, ActionProfile] = {}
        for tag, action in self.actions.items():
            if not isinstance(action, ActionProfile):
                raise TypeError("actions must contain ActionProfile values")
            if tag != action.tag:
                raise ValueError(
                    f"Action mapping key {tag!r} does not match tag {action.tag!r}"
                )
            immutable_actions[tag] = action

        object.__setattr__(
            self,
            "actions",
            MappingProxyType(immutable_actions),
        )
