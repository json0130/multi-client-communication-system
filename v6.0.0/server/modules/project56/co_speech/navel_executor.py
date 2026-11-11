"""Real Navel transport adapter for resolved Method B speech segments."""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from modules.project56.co_speech.plan import PlannedSegment, SelectionStatus


DEFAULT_VARIANTS = ("left", "right", "both")


class DefaultVariantSelector:
    """Choose DEFAULT physical variants from non-repeating shuffled bags."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._bag: list[str] = []
        self._last_variant: str | None = None

    def next_variant(self) -> str:
        """Return one variant, refilling after every complete three-item bag."""
        if not self._bag:
            self._bag = list(DEFAULT_VARIANTS)
            self._rng.shuffle(self._bag)
            if self._last_variant == self._bag[0]:
                swap_index = next(
                    index
                    for index, variant in enumerate(self._bag[1:], start=1)
                    if variant != self._last_variant
                )
                self._bag[0], self._bag[swap_index] = (
                    self._bag[swap_index],
                    self._bag[0],
                )

        selected = self._bag.pop(0)
        self._last_variant = selected
        return selected


class NavelSegmentTransport(Protocol):
    """Transport boundary used by the server-side segment executor."""

    async def execute_segment(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NavelSegmentExecutionRecord:
    """Server-observed request/ack timing plus robot-reported measurements."""

    segment_id: int
    selected_tag: str | None
    default_variant: str | None
    predicted_speech_duration_seconds: float
    execution_requested: float
    execution_completed: float
    robot_metrics: Mapping[str, Any]

    @property
    def round_trip_execution_seconds(self) -> float:
        return self.execution_completed - self.execution_requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "selected_tag": self.selected_tag,
            "default_variant": self.default_variant,
            "predicted_speech_duration_seconds": (
                self.predicted_speech_duration_seconds
            ),
            "execution_requested": self.execution_requested,
            "execution_completed": self.execution_completed,
            "round_trip_execution_seconds": self.round_trip_execution_seconds,
            "robot_metrics": dict(self.robot_metrics),
        }


class NavelSegmentExecutor:
    """Adapt a resolved PlannedSegment to the robot's segment protocol."""

    def __init__(
        self,
        transport: NavelSegmentTransport,
        *,
        default_variant: str = "auto",
        default_variant_selector: DefaultVariantSelector | None = None,
        timeout_seconds: float = 120.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if default_variant not in {"auto", *DEFAULT_VARIANTS}:
            raise ValueError(
                "DEFAULT variant mode must be one of: auto, left, right, both"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._transport = transport
        self._default_variant = default_variant
        self._default_variant_selector = (
            default_variant_selector
            if default_variant_selector is not None
            else DefaultVariantSelector()
        )
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self.records: list[NavelSegmentExecutionRecord] = []

    async def execute(self, segment: PlannedSegment) -> None:
        if not isinstance(segment, PlannedSegment):
            raise TypeError("segment must be a PlannedSegment")
        if segment.selection_status is SelectionStatus.PENDING:
            raise ValueError("Cannot execute a PENDING Method B segment")

        speech = segment.speech_segment
        request_id = uuid.uuid4().hex
        resolved_default_variant = self._resolve_default_variant(segment.selected_tag)
        payload = {
            "event": "method_b_segment",
            "request_id": request_id,
            "segment_id": speech.segment_id,
            "text": speech.text,
            "selected_tag": segment.selected_tag,
            "predicted_speech_duration_seconds": speech.estimated_duration,
        }
        if resolved_default_variant is not None:
            payload["default_variant"] = resolved_default_variant
        requested = self._clock()
        response = await self._transport.execute_segment(
            payload,
            timeout_seconds=self._timeout_seconds,
        )
        completed = self._clock()

        if response.get("request_id") != request_id:
            raise RuntimeError("Navel returned a mismatched segment request_id")
        if response.get("segment_id") != speech.segment_id:
            raise RuntimeError("Navel returned a mismatched speech segment ID")
        if response.get("selected_tag") != segment.selected_tag:
            raise RuntimeError("Navel acknowledged a different selected action")
        if not response.get("ok"):
            error_type = response.get("error_type", "RuntimeError")
            message = response.get("error", "unknown Navel execution failure")
            raise RuntimeError(f"Navel {error_type}: {message}")
        if response.get("default_variant") != resolved_default_variant:
            raise RuntimeError("Navel acknowledged a different DEFAULT variant")

        metrics = response.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise RuntimeError("Navel returned invalid segment timing metrics")
        self.records.append(NavelSegmentExecutionRecord(
            segment_id=speech.segment_id,
            selected_tag=segment.selected_tag,
            default_variant=resolved_default_variant,
            predicted_speech_duration_seconds=speech.estimated_duration,
            execution_requested=requested,
            execution_completed=completed,
            robot_metrics=dict(metrics),
        ))

    def _resolve_default_variant(self, selected_tag: str | None) -> str | None:
        """Resolve physical variation only after semantic DEFAULT is selected."""
        if selected_tag != "DEFAULT":
            return None
        if self._default_variant == "auto":
            selected = self._default_variant_selector.next_variant()
            if selected not in DEFAULT_VARIANTS:
                raise ValueError(
                    "DEFAULT variant selector returned an unsupported variant"
                )
            return selected
        return self._default_variant


class DirectNavelWebSocketTransport:
    """Development transport connecting directly to the robot WebSocket."""

    def __init__(self, url: str, *, open_timeout_seconds: float = 10.0) -> None:
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("Navel WebSocket URL must start with ws:// or wss://")
        if open_timeout_seconds <= 0:
            raise ValueError("open_timeout_seconds must be positive")
        self.url = url
        self._open_timeout_seconds = open_timeout_seconds
        self._websocket = None
        self._execution_lock = asyncio.Lock()

    async def __aenter__(self) -> "DirectNavelWebSocketTransport":
        import websockets

        self._websocket = await websockets.connect(
            self.url,
            open_timeout=self._open_timeout_seconds,
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

    async def execute_segment(
        self,
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        if self._websocket is None:
            raise RuntimeError("Direct Navel WebSocket is not connected")
        request_id = payload.get("request_id")
        async with self._execution_lock:
            await self._websocket.send(json.dumps(dict(payload)))
            deadline = asyncio.get_running_loop().time() + timeout_seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for Navel segment {request_id}"
                    )
                raw = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=remaining,
                )
                response = json.loads(raw)
                if (
                    response.get("type") == "method_b_segment_result"
                    and response.get("request_id") == request_id
                ):
                    return response
