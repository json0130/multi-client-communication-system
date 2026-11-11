"""One-segment-lookahead scheduling for Method B co-speech plans."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from modules.project56.co_speech.models import SpeechSegment
from modules.project56.co_speech.plan import CoSpeechPlan, CoSpeechPlanBuilder, PlannedSegment


if TYPE_CHECKING:
    from modules.project56.action_selector import ActionContext, ActionSelection
    from modules.project56.action_selector.duration_filter import DurationFilterResult


class _PreparedMethodBSelector(Protocol):
    def prepare_method_b_candidates(
        self,
        allowed_tags: list[str],
        segment: SpeechSegment,
    ) -> "DurationFilterResult": ...

    def select_method_b_prepared(
        self,
        duration_result: DurationFilterResult,
        context: "ActionContext",
        *,
        previous_tag: str | None = None,
    ) -> "ActionSelection | Awaitable[ActionSelection]": ...


class SegmentExecutor(Protocol):
    """Injected execution boundary; real Navel execution is intentionally absent."""

    async def execute(self, segment: PlannedSegment) -> None: ...


@dataclass(frozen=True)
class PreparedPlanSegment:
    """A speech segment and its already computed duration-filter result."""

    speech_segment: SpeechSegment
    duration_filter_result: "DurationFilterResult"


@dataclass(frozen=True)
class LookaheadPlanningTiming:
    """Monotonic timing diagnostics for one lookahead selection."""

    segment_id: int
    planning_start: float
    planning_end: float
    planning_latency_seconds: float
    available_lookahead_seconds: float
    ready_before_current_finished: bool
    overrun_seconds: float


@dataclass(frozen=True)
class LookaheadRunResult:
    """Completed plan plus compact runtime-planning diagnostics."""

    plan: CoSpeechPlan
    prepared_segments: tuple[PreparedPlanSegment, ...]
    duration_preparation_latency_seconds: float
    initial_planning_latency_seconds: float
    planning_timings: tuple[LookaheadPlanningTiming, ...]

    @property
    def lookahead_success_count(self) -> int:
        return sum(
            timing.ready_before_current_finished
            for timing in self.planning_timings
        )

    @property
    def lookahead_success_rate(self) -> float:
        if not self.planning_timings:
            return 0.0
        return self.lookahead_success_count / len(self.planning_timings)

    @property
    def total_overrun_seconds(self) -> float:
        return sum(timing.overrun_seconds for timing in self.planning_timings)

    @property
    def max_overrun_seconds(self) -> float:
        return max(
            (timing.overrun_seconds for timing in self.planning_timings),
            default=0.0,
        )


class OneSegmentLookaheadCoordinator:
    """Overlap current fake execution with selection of exactly one next segment."""

    def __init__(
        self,
        selector: _PreparedMethodBSelector,
        executor: SegmentExecutor,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._selector = selector
        self._executor = executor
        self._clock = clock

    async def run_with_lookahead(
        self,
        segments: Sequence[SpeechSegment],
        *,
        allowed_tags: Sequence[str],
        context: "ActionContext",
        include_previous_segments_in_context: bool = False,
    ) -> LookaheadRunResult:
        """Prepare durations up front, then select/execute with one lookahead.

        ``include_previous_segments_in_context`` is for a streaming response:
        each selected segment sees up to six earlier segments in addition to
        the context supplied by the caller. It is opt-in to preserve legacy
        Method B behavior.
        """
        builder = CoSpeechPlanBuilder(segments)
        candidate_tags = list(allowed_tags)
        preparation_start = self._clock()
        prepared_segments = tuple(
            PreparedPlanSegment(
                speech_segment=planned.speech_segment,
                duration_filter_result=self._selector.prepare_method_b_candidates(
                    allowed_tags=list(candidate_tags),
                    segment=planned.speech_segment,
                ),
            )
            for planned in builder.plan.segments
        )
        preparation_latency = self._clock() - preparation_start

        if not prepared_segments:
            return LookaheadRunResult(
                plan=builder.plan,
                prepared_segments=(),
                duration_preparation_latency_seconds=preparation_latency,
                initial_planning_latency_seconds=0.0,
                planning_timings=(),
            )

        segment_history = (
            list(context.previous_dialogue_segments[-6:])
            if include_previous_segments_in_context else None
        )
        first_selection = await self._select_prepared(
            prepared_segments[0],
            context=context,
            previous_tag=None,
            segment_history=segment_history,
        )
        initial_latency = self._clock() - preparation_start
        builder.record_selection(
            prepared_segments[0].speech_segment.segment_id,
            first_selection.selected_tag,
        )

        timings: list[LookaheadPlanningTiming] = []
        for index, prepared in enumerate(prepared_segments):
            current = self._planned_segment(
                builder,
                prepared.speech_segment.segment_id,
            )
            execution_start = self._clock()
            execution_task = asyncio.create_task(self._executor.execute(current))
            planning_task: asyncio.Task[tuple[float, float]] | None = None

            if index + 1 < len(prepared_segments):
                next_prepared = prepared_segments[index + 1]
                previous_tag = current.selected_tag
                planning_task = asyncio.create_task(
                    self._plan_next(
                        builder,
                        next_prepared,
                        context=context,
                        previous_tag=previous_tag,
                        segment_history=segment_history,
                    )
                )

            try:
                await execution_task
            except BaseException:
                if planning_task is not None:
                    planning_task.cancel()
                    await asyncio.gather(planning_task, return_exceptions=True)
                raise

            execution_end = self._clock()
            if planning_task is not None:
                planning_start, planning_end = await planning_task
                planning_latency = planning_end - planning_start
                timings.append(
                    LookaheadPlanningTiming(
                        segment_id=(
                            prepared_segments[index + 1].speech_segment.segment_id
                        ),
                        planning_start=planning_start,
                        planning_end=planning_end,
                        planning_latency_seconds=planning_latency,
                        available_lookahead_seconds=max(
                            0.0,
                            execution_end - max(execution_start, planning_start),
                        ),
                        ready_before_current_finished=planning_end <= execution_end,
                        overrun_seconds=max(0.0, planning_end - execution_end),
                    )
                )

        plan = builder.plan
        if not plan.is_complete:
            raise RuntimeError("Lookahead execution finished with an incomplete plan")
        return LookaheadRunResult(
            plan=plan,
            prepared_segments=prepared_segments,
            duration_preparation_latency_seconds=preparation_latency,
            initial_planning_latency_seconds=initial_latency,
            planning_timings=tuple(timings),
        )

    async def _plan_next(
        self,
        builder: CoSpeechPlanBuilder,
        prepared: PreparedPlanSegment,
        *,
        context: "ActionContext",
        previous_tag: str | None,
        segment_history: list[str] | None,
    ) -> tuple[float, float]:
        planning_start = self._clock()
        selection = await self._select_prepared(
            prepared,
            context=context,
            previous_tag=previous_tag,
            segment_history=segment_history,
        )
        planning_end = self._clock()
        builder.record_selection(
            prepared.speech_segment.segment_id,
            selection.selected_tag,
        )
        return planning_start, planning_end

    async def _select_prepared(
        self,
        prepared: PreparedPlanSegment,
        *,
        context: "ActionContext",
        previous_tag: str | None,
        segment_history: list[str] | None,
    ) -> "ActionSelection":
        segment_context = replace(
            context,
            dialogue=prepared.speech_segment.text,
            previous_dialogue_segments=(
                tuple(segment_history[-6:])
                if segment_history is not None
                else context.previous_dialogue_segments
            ),
        )
        method = self._selector.select_method_b_prepared
        if inspect.iscoroutinefunction(method):
            selection = await method(
                duration_result=prepared.duration_filter_result,
                context=segment_context,
                previous_tag=previous_tag,
            )
        else:
            selection = await asyncio.to_thread(
                method,
                duration_result=prepared.duration_filter_result,
                context=segment_context,
                previous_tag=previous_tag,
            )
        if segment_history is not None:
            segment_history.append(prepared.speech_segment.text)
            del segment_history[:-6]
        return selection

    @staticmethod
    def _planned_segment(
        builder: CoSpeechPlanBuilder,
        segment_id: int,
    ) -> PlannedSegment:
        return next(
            planned
            for planned in builder.plan.segments
            if planned.speech_segment.segment_id == segment_id
        )
