"""Optional immutable traces for observing existing Method B decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from modules.project56.action_selector.duration_filter import DurationFilterResult
from modules.project56.action_selector.models import ActionScore
from modules.project56.co_speech.models import SpeechSegment


@dataclass(frozen=True)
class CandidateTimingTrace:
    """Profile timing metadata observed for one initial candidate."""

    tag: str
    action_duration_seconds: float | None
    profiled: bool | None
    kind: str | None


@dataclass(frozen=True)
class DurationPreparationTrace:
    """Exact inputs and output of one production duration-filter call."""

    segment: SpeechSegment
    initial_candidates: tuple[str, ...]
    candidate_timings: tuple[CandidateTimingTrace, ...]
    margin_seconds: float
    overrun_allowance_seconds: float
    result: DurationFilterResult


@dataclass(frozen=True)
class LanguageScoringTrace:
    """Input and same-call output observed inside LanguageScorer."""

    semantic_candidates: tuple[str, ...]
    candidate_descriptions: Mapping[str, str]
    robot_id: str
    user_message: str
    dialogue: str
    user_emotion: str
    semantic_scores: Mapping[str, float]
    raw_response: str | None
    fallback_reason: str | None
    scorer_prompt: str | None
    scoring_mode: str
    raw_responses: tuple[str, ...]
    action_description_set: str
    scorer_latency_seconds: float


@dataclass(frozen=True)
class DependentSelectionTrace:
    """Repetition, scoring, and final selection for prepared candidates."""

    previous_tag: str | None
    candidates_before_repetition: tuple[str, ...]
    candidates_after_repetition: tuple[str, ...]
    removed_by_repetition: tuple[str, ...]
    language_scoring: LanguageScoringTrace | None
    action_scores: tuple[ActionScore, ...]
    selected_tag: str | None
    used_fallback: bool
