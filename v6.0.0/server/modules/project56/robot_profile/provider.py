"""Storage-independent robot-profile provider interface."""

from __future__ import annotations

from typing import Protocol

from modules.project56.robot_profile.models import RobotProfile


class RobotProfileProvider(Protocol):
    """Supply robot metadata without exposing its storage implementation."""

    def get_profile(self, robot_id: str) -> RobotProfile:
        """Return the configured profile or raise KeyError when absent."""
        ...
