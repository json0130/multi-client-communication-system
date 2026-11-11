"""
modules/action_selector/action_selector_module.py
================================================
Server-side action selection module.

What it does:
  - Normalises the action tags configured for a robot
  - Gets dialogue-relevance scores from LanguageScorer
  - Gets physical-feasibility scores from AffordanceProvider
  - Multiplies both scores and selects the highest-scoring action
  - Preserves legacy DEFAULT fallback for the existing Method A entry point
  - Represents Method B no-action safely as None

Scoring rule:
  combined_score = language_score * affordance_score

The selector runs on the central server and does not execute robot hardware.
The caller supplies the generated dialogue, allowed tags, and current robot
state, then forwards the returned tag to the robot client for execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from modules.project56.base import BaseModule
from modules.project56.action_selector.affordance_provider import AffordanceProvider
from modules.project56.action_selector.duration_filter import (
    DurationFilter,
    DurationFilterResult,
)
from modules.project56.action_selector.diagnostics import (
    CandidateTimingTrace,
    DependentSelectionTrace,
    DurationPreparationTrace,
    LanguageScoringTrace,
)
from modules.project56.action_selector.language_scorer import LanguageScorer
from modules.project56.action_selector.repetition_filter import RepetitionFilter
from modules.project56.action_selector.models import (
    ActionContext,
    ActionScore,
    ActionSelection,
)
from modules.project56.co_speech.models import SpeechSegment
from modules.project56.robot_profile import RobotProfile


class ActionSelectorModule(BaseModule):
    """Coordinate language and feasibility scoring for robot actions."""

    def __init__(
        self,
        llm,
        robot_profile: RobotProfile | None = None,
        language_scoring_mode: str = "legacy",
        action_description_set: str = "v1",
        duration_overrun_allowance_seconds: float = 0.0,
        scorer_context: str = "none",
    ):
        """Create a selector using an already-initialised LLM module."""
        self.robot_profile = robot_profile
        self._duration_overrun_allowance_seconds = duration_overrun_allowance_seconds
        self._language_scorer = LanguageScorer(
            llm,
            action_profiles=(
                robot_profile.actions if robot_profile is not None else None
            ),
            scoring_mode=language_scoring_mode,
            action_description_set=action_description_set,
            scorer_context=scorer_context,
        )
        self._affordance_provider = AffordanceProvider()
        self._available = False

    def initialize(self) -> bool:
        """Mark the selector as ready after its dependencies are attached."""
        self._available = True
        return True

    def is_available(self) -> bool:
        """Return True after initialize() has completed successfully."""
        return self._available

    def get_status(self) -> dict:
        """Return module health information for the robot health endpoint."""
        return {
            "module": "action_selector",
            "available": self._available,
        }

    def select(
        self,
        allowed_tags: list[str],
        context: ActionContext,
    ) -> ActionSelection:
        """
        Score all allowed tags and return the best feasible action.

        Tags are normalised and de-duplicated before scoring. DEFAULT is added
        when absent. The returned ActionSelection includes every component
        score so callers can log and inspect the decision. If all combined
        scores are zero, DEFAULT is returned with used_fallback set to True.
        """
        tags = self._prepare_tags(allowed_tags)
        return self._select_candidates(
            tags=tags,
            context=context,
            no_action_on_failure=False,
        )

    def select_method_b(
        self,
        allowed_tags: list[str],
        context: ActionContext,
        segment: SpeechSegment,
        *,
        previous_tag: str | None = None,
    ) -> ActionSelection:
        """Filter and select a Method B gesture without owning history state."""
        duration_result = self.prepare_method_b_candidates(
            allowed_tags=allowed_tags,
            segment=segment,
        )
        return self.select_method_b_prepared(
            duration_result=duration_result,
            context=context,
            previous_tag=previous_tag,
        )

    def prepare_method_b_candidates(
        self,
        allowed_tags: list[str],
        segment: SpeechSegment,
        *,
        diagnostic_callback: (
            Callable[[DurationPreparationTrace], None] | None
        ) = None,
    ) -> DurationFilterResult:
        """Apply only duration-independent Method B candidate preparation."""
        if self.robot_profile is None:
            raise RuntimeError(
                "Method B action selection requires a resolved RobotProfile"
            )

        tags = list(dict.fromkeys(
            self._normalise(tag)
            for tag in allowed_tags
            if tag and tag.strip()
        ))
        duration_filter = DurationFilter(
            overrun_allowance_seconds=self._duration_overrun_allowance_seconds,
        )
        result = duration_filter.filter(
            segment=segment,
            allowed_tags=tags,
            actions=self.robot_profile.actions,
        )
        if diagnostic_callback is not None:
            diagnostic_callback(DurationPreparationTrace(
                segment=segment,
                initial_candidates=tuple(tags),
                candidate_timings=tuple(
                    CandidateTimingTrace(
                        tag=tag,
                        action_duration_seconds=(
                            action.duration_seconds if action is not None else None
                        ),
                        profiled=(action.profiled if action is not None else None),
                        kind=(action.kind if action is not None else None),
                    )
                    for tag in tags
                    for action in (self.robot_profile.actions.get(tag),)
                ),
                margin_seconds=duration_filter.margin_seconds,
                overrun_allowance_seconds=duration_filter.overrun_allowance_seconds,
                result=result,
            ))
        return result

    def select_method_b_prepared(
        self,
        duration_result: DurationFilterResult,
        context: ActionContext,
        *,
        previous_tag: str | None = None,
        temperature: float | None = None,
        diagnostic_callback: (
            Callable[[DependentSelectionTrace], None] | None
        ) = None,
    ) -> ActionSelection:
        """Apply previous-dependent filtering and scoring to prepared tags."""
        repetition_eligible_tags = RepetitionFilter().filter(
            candidate_tags=duration_result.eligible_tags,
            previous_tag=previous_tag,
        )
        language_traces: list[LanguageScoringTrace] = []
        selection = self._select_candidates(
            tags=repetition_eligible_tags,
            context=context,
            no_action_on_failure=True,
            temperature=temperature,
            language_diagnostic_callback=(
                language_traces.append if diagnostic_callback is not None else None
            ),
        )
        result = replace(
            selection,
            duration_filter_result=duration_result,
        )
        if diagnostic_callback is not None:
            repetition_set = set(repetition_eligible_tags)
            diagnostic_callback(DependentSelectionTrace(
                previous_tag=previous_tag,
                candidates_before_repetition=duration_result.eligible_tags,
                candidates_after_repetition=tuple(repetition_eligible_tags),
                removed_by_repetition=tuple(
                    tag
                    for tag in duration_result.eligible_tags
                    if tag not in repetition_set
                ),
                language_scoring=(language_traces[0] if language_traces else None),
                action_scores=result.scores,
                selected_tag=result.selected_tag,
                used_fallback=result.used_fallback,
            ))
        return result

    def _select_candidates(
        self,
        tags: list[str],
        context: ActionContext,
        *,
        no_action_on_failure: bool,
        temperature: float | None = None,
        language_diagnostic_callback: (
            Callable[[LanguageScoringTrace], None] | None
        ) = None,
    ) -> ActionSelection:
        """Score an already-normalised candidate set under one fallback mode."""
        if not tags:
            return ActionSelection(
                selected_tag=None,
                scores=(),
                used_fallback=True,
            )

        scorer_kwargs = {"no_action_on_failure": no_action_on_failure}
        if language_diagnostic_callback is not None:
            scorer_kwargs["diagnostic_callback"] = language_diagnostic_callback
        # Preserve existing scorer calls and compatible injected test doubles
        # unless an evaluation explicitly requests a temperature override.
        if temperature is not None:
            scorer_kwargs["temperature"] = temperature
        language_scores = self._language_scorer.score_actions(
            tags, context, **scorer_kwargs,
        )

        results: list[ActionScore] = []

        for tag in tags:
            language_score = language_scores.get(tag, 0.0)
            affordance_score = self._affordance_provider.score(tag, context)
            combined_score = language_score * affordance_score

            results.append(
                ActionScore(
                    tag=tag,
                    language_score=language_score,
                    affordance_score=affordance_score,
                    combined_score=combined_score,
                )
            )

        best = max(
            results,
            key=lambda result: (
                result.combined_score,
                result.tag == "DEFAULT",
            ),
        )

        all_zero = all(result.combined_score <= 0.0 for result in results)

        return ActionSelection(
            selected_tag=(
                None
                if all_zero and no_action_on_failure
                else ("DEFAULT" if all_zero else best.tag)
            ),
            scores=tuple(results),
            used_fallback=all_zero,
        )

    @classmethod
    def _prepare_tags(cls, tags: list[str]) -> list[str]:
        """Normalise, de-duplicate, and add the DEFAULT fallback tag."""
        normalised = list(dict.fromkeys(
            cls._normalise(tag)
            for tag in tags
            if tag and tag.strip()
        ))

        # DEFAULT must always be an available safe fallback.
        if "DEFAULT" not in normalised:
            normalised.append("DEFAULT")

        return normalised

    @staticmethod
    def _normalise(tag: str) -> str:
        """Convert a tag such as '[wave]' to the canonical 'WAVE' form."""
        return tag.strip().strip("[]").upper()
