"""Stateless consecutive-action filtering for Method B candidates."""

from __future__ import annotations

from collections.abc import Sequence


class RepetitionFilter:
    """Apply the current V1 consecutive-action repetition policy.

    Non-DEFAULT candidates matching ``previous_tag`` are removed. DEFAULT is
    deliberately retained because it is a generic-speaking action whose
    physical executor can vary between left, right, and both-arm variants.
    This policy can be revisited when per-segment execution is integrated.

    The caller owns and explicitly supplies any previous-selection state.
    Instances of this class store no history and never mutate their inputs.
    """

    def filter(
        self,
        candidate_tags: Sequence[str],
        previous_tag: str | None = None,
    ) -> list[str]:
        """Return ordered canonical candidates allowed by the V1 policy."""
        previous = self._normalise(previous_tag) if previous_tag else None
        tags = dict.fromkeys(
            normalised
            for tag in candidate_tags
            if tag and (normalised := self._normalise(tag))
        )
        return [tag for tag in tags if tag == "DEFAULT" or tag != previous]

    @staticmethod
    def _normalise(tag: str) -> str:
        """Match the project's uppercase, unbracketed action-tag format."""
        return tag.strip().strip("[]").strip().upper()
