"""Sequentially populate one temporary Method B co-speech plan."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from modules.project56.co_speech.models import SpeechSegment
from modules.project56.co_speech.plan import CoSpeechPlan, CoSpeechPlanBuilder


if TYPE_CHECKING:
    from modules.project56.action_selector import (
        ActionContext,
        ActionSelection,
        ActionSelectorModule,
    )


class _MethodBSelector(Protocol):
    def select_method_b(
        self,
        allowed_tags: list[str],
        context: "ActionContext",
        segment: SpeechSegment,
        *,
        previous_tag: str | None = None,
    ) -> "ActionSelection": ...


class CoSpeechPlanOrchestrator:
    """Build a selected plan sequentially without executing robot behaviour."""

    def __init__(self, selector: "ActionSelectorModule | _MethodBSelector"):
        self._selector = selector

    def build_selected_plan(
        self,
        segments: Sequence[SpeechSegment],
        *,
        allowed_tags: Sequence[str],
        context: "ActionContext",
        on_segment=None,
    ) -> CoSpeechPlan:
        """Select once per ordered segment and return a complete plan.

        `on_segment(text, selected_tag)` is called as each segment is resolved (ChatBox addition).
        Selection exceptions propagate unchanged. Already resolved segments
        remain represented by the response-local builder, and no later segment
        is attempted after an error.
        """
        builder = CoSpeechPlanBuilder(segments)
        candidate_tags = list(allowed_tags)

        for planned_segment in builder.plan.segments:
            speech_segment = planned_segment.speech_segment
            current_plan = builder.plan
            previous_tag = current_plan.previous_selected_tag(
                speech_segment.segment_id
            )
            segment_context = replace(
                context,
                dialogue=speech_segment.text,
            )
            selection = self._selector.select_method_b(
                allowed_tags=list(candidate_tags),
                context=segment_context,
                segment=speech_segment,
                previous_tag=previous_tag,
            )
            builder.record_selection(
                segment_id=speech_segment.segment_id,
                selected_tag=selection.selected_tag,
            )
            if on_segment is not None:      # lets the caller start speaking before the rest is planned
                on_segment(speech_segment.text, selection.selected_tag)

        plan = builder.plan
        if not plan.is_complete:
            raise RuntimeError("Co-speech plan selection finished incomplete")
        return plan
