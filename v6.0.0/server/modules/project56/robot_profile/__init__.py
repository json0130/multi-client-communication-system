"""Shared robot-specific metadata contracts and profile providers."""

from modules.project56.robot_profile.models import (
    ActionKind,
    ActionProfile,
    RobotProfile,
    RobotSpeechProfile,
)
from modules.project56.robot_profile.placeholder_provider import (
    PlaceholderRobotProfileProvider,
)
from modules.project56.robot_profile.provider import RobotProfileProvider

__all__ = [
    "ActionKind",
    "ActionProfile",
    "PlaceholderRobotProfileProvider",
    "RobotProfile",
    "RobotProfileProvider",
    "RobotSpeechProfile",
]
