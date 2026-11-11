"""Temporary response-scoped state for Method B co-speech planning."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from modules.project56.co_speech.models import SpeechSegment


_CANONICAL_TAG_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SelectionStatus(str, Enum):
    """Selection state for one planned speech segment."""

    PENDING = "pending"
    SELECTED = "selected"
    NO_ACTION = "no_action"


@dataclass(frozen=True)
class PlannedSegment:
    """An immutable speech segment plus its action-selection state."""

    speech_segment: SpeechSegment
    selection_status: SelectionStatus = SelectionStatus.PENDING
    selected_tag: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.speech_segment, SpeechSegment):
            raise TypeError("speech_segment must be a SpeechSegment")
        if not isinstance(self.selection_status, SelectionStatus):
            raise TypeError("selection_status must be a SelectionStatus")

        if self.selection_status is SelectionStatus.SELECTED:
            if (
                not isinstance(self.selected_tag, str)
                or not _CANONICAL_TAG_PATTERN.fullmatch(self.selected_tag)
                or self.selected_tag == "NO_ACTION"
            ):
                raise ValueError(
                    "SELECTED requires a canonical uppercase action tag"
                )
        elif self.selected_tag is not None:
            raise ValueError(
                f"{self.selection_status.value.upper()} cannot contain an action tag"
            )

    @property
    def is_resolved(self) -> bool:
        return self.selection_status is not SelectionStatus.PENDING

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Return a flat debug representation without duplicating source data."""
        return {
            **self.speech_segment.to_dict(),
            "selection_status": self.selection_status.value,
            "selected_tag": self.selected_tag,
        }


@dataclass(frozen=True)
class CoSpeechPlan:
    """Immutable ordered snapshot of one response's co-speech state."""

    segments: tuple[PlannedSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple):
            raise TypeError("segments must be a tuple of PlannedSegment objects")

        seen_ids: set[int] = set()
        for segment in self.segments:
            if not isinstance(segment, PlannedSegment):
                raise TypeError("segments must contain PlannedSegment objects")
            segment_id = segment.speech_segment.segment_id
            if segment_id in seen_ids:
                raise ValueError(f"Duplicate speech segment ID: {segment_id}")
            seen_ids.add(segment_id)

    @property
    def is_complete(self) -> bool:
        """Return whether every segment has a resolved selection.

        An empty response plan is complete because it has nothing to process.
        """
        return all(segment.is_resolved for segment in self.segments)

    def previous_selected_tag(self, segment_id: int) -> str | None:
        """Return the closest resolved preceding selection for orchestration.

        Both no resolved predecessor and a preceding NO_ACTION result return
        ``None``, which is the value expected by ``select_method_b``.
        """
        target_index = self._index_for(segment_id)
        for segment in reversed(self.segments[:target_index]):
            if segment.is_resolved:
                return segment.selected_tag
        return None

    def to_dict(self) -> dict[str, list[dict[str, int | float | str | None]]]:
        """Return an ordered debug representation; no persistence is performed."""
        return {"segments": [segment.to_dict() for segment in self.segments]}

    def _index_for(self, segment_id: int) -> int:
        for index, segment in enumerate(self.segments):
            if segment.speech_segment.segment_id == segment_id:
                return index
        raise KeyError(f"Unknown speech segment ID: {segment_id}")


class CoSpeechPlanBuilder:
    """Response-scoped builder for recording final per-segment selections."""

    def __init__(self, segments: Sequence[SpeechSegment]):
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
            raise TypeError("segments must be a sequence of SpeechSegment objects")

        planned_segments: list[PlannedSegment] = []
        seen_ids: set[int] = set()
        for segment in segments:
            if not isinstance(segment, SpeechSegment):
                raise TypeError("segments must contain SpeechSegment objects")
            if segment.segment_id in seen_ids:
                raise ValueError(f"Duplicate speech segment ID: {segment.segment_id}")
            seen_ids.add(segment.segment_id)
            planned_segments.append(PlannedSegment(speech_segment=segment))

        self._segments = planned_segments

    @property
    def plan(self) -> CoSpeechPlan:
        """Return an immutable snapshot of the current response state."""
        return CoSpeechPlan(tuple(self._segments))

    def record_selection(
        self,
        segment_id: int,
        selected_tag: str | None,
    ) -> PlannedSegment:
        """Record one supplied selector result; accidental repeats fail."""
        index = self._index_for(segment_id)
        current = self._segments[index]
        if current.selection_status is not SelectionStatus.PENDING:
            raise RuntimeError(
                f"Selection already recorded for speech segment ID: {segment_id}"
            )

        if selected_tag is None:
            resolved = PlannedSegment(
                speech_segment=current.speech_segment,
                selection_status=SelectionStatus.NO_ACTION,
                selected_tag=None,
            )
        else:
            resolved = PlannedSegment(
                speech_segment=current.speech_segment,
                selection_status=SelectionStatus.SELECTED,
                selected_tag=selected_tag,
            )

        self._segments[index] = resolved
        return resolved

    def _index_for(self, segment_id: int) -> int:
        for index, segment in enumerate(self._segments):
            if segment.speech_segment.segment_id == segment_id:
                return index
        raise KeyError(f"Unknown speech segment ID: {segment_id}")
