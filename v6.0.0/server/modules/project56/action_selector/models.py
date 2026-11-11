"""
modules/action_selector/models.py
=================================
Shared data contracts for action selection.

The models keep the selector independent of RobotInstance, gateways, database
repositories, and robot-client implementations. Callers provide plain dialogue
and robot-state data through ActionContext and receive the selected optional
action plus a complete scoring trace in ActionSelection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from modules.project56.action_selector.duration_filter import DurationFilterResult


@dataclass(frozen=True)
class ActionContext:
    """
    Information that may influence action selection.

    This is deliberately generic so the action selector does not need to
    import RobotInstance or know about the central server.
    """

    user_message: str
    dialogue: str
    robot_id: str = ""
    user_emotion: str = "unknown"
    robot_state: dict[str, Any] = field(default_factory=dict)
    previous_dialogue_segments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionScore:
    """Component and combined scores for one candidate action."""

    tag: str
    language_score: float
    affordance_score: float
    combined_score: float


@dataclass(frozen=True)
class ActionSelection:
    """Selected action and scores; None explicitly means no gesture."""

    selected_tag: str | None
    scores: tuple[ActionScore, ...]
    used_fallback: bool = False
    duration_filter_result: "DurationFilterResult | None" = None
