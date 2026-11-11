"""
modules/action_selector/affordance_provider.py
==============================================
Rule-based physical-feasibility scorer.

What it does:
  - Reads the current robot state supplied through ActionContext
  - Returns a feasibility score from 0.0 to 1.0 for one candidate action
  - Prevents unavailable or unsafe actions from winning final selection
  - Keeps hardware rules separate from dialogue-relevance scoring

This class does not connect to the robot or execute gestures. The integration
layer supplies state such as gesture readiness, disabled tags, arm occupancy,
and obstacle detection. Robot-specific capability and safety rules can be
extended here without changing the language scorer or gateway code.
"""

from __future__ import annotations

from modules.project56.action_selector.models import ActionContext


class AffordanceProvider:
    """
    Returns a value between 0.0 and 1.0.

    0.0 means the action cannot or must not execute.
    1.0 means the action is fully available.
    """

    def score(self, tag: str, context: ActionContext) -> float:
        """
        Return the current physical-feasibility score for one action.

        DEFAULT is always feasible because it represents speech without a
        physical action. Other tags are evaluated against context.robot_state.
        """
        tag = self._normalise(tag)
        state = context.robot_state

        # DEFAULT means speaking without performing a physical action.
        if tag == "DEFAULT":
            return 1.0

        # Do not choose physical gestures if the gesture output is unavailable.
        if not state.get("gesture_module_ready", True):
            return 0.0

        disabled_tags = {
            self._normalise(value)
            for value in state.get("disabled_tags", [])
        }

        if tag in disabled_tags:
            return 0.0

        # Examples of action-specific rules.
        if tag == "MOVE_FORWARD":
            if state.get("obstacle_detected", False):
                return 0.0
            return 0.9

        # Available gesture with no additional restrictions.
        return 1.0

    @staticmethod
    def _normalise(tag: str) -> str:
        """Convert a configured tag to its canonical unbracketed form."""
        return tag.strip().strip("[]").upper()
