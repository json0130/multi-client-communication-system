"""LLM boundary-ID speech segmentation for Method B evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

from modules.project56.co_speech.atomic_speech_unitizer import AtomicSpeechUnitizer
from modules.project56.co_speech.models import (
    AtomicSpeechUnit,
    SpeechSegment,
    SpeechTimingProfile,
)
from modules.project56.co_speech.profiles import NAVEL_SPEECH_TIMING_PROFILE
from modules.project56.co_speech.speech_segmenter import (
    SegmentationTimingPolicy,
    SpeechSegmenter,
)
from modules.project56.llm.llm_provider import LLMResponse


logger = logging.getLogger(__name__)

OUTPUT_CONTRACT_VERSION = "boundary_ids_v2"
SEGMENTATION_POLICY_VERSION = "semantic_timing_v2"

SEGMENTATION_SYSTEM_PROMPT = """
You are a speech segmentation component for a social robot.

The response has already been divided into numbered atomic speech units. These
units are only candidate boundaries. They are NOT the final segmentation.

A segment is a coherent communicative unit that is also practically suitable
for one co-speech gesture.

Segmentation rules:
- The normal target duration is approximately 2 to 6 seconds.
- The duration target is not a strict limit. A segment may exceed 6 seconds
  when no clean boundary preserves complete, natural communicative units.
- Short standalone acts such as greetings, confirmations, and brief responses
  may remain below 2 seconds.
- Create a boundary when the communicative function or topic clearly changes,
  including statement or explanation to question, explanation to offer or
  invitation, explanation to instruction, confirmation to a new explanation,
  greeting to explanation, or a clear topic transition.
- Do not create a boundary solely because of and, but, so, because, however,
  commas, semicolons, or the separation of atomic units.
- Keep semantically continuous material together when it remains practical for
  one gesture. If it becomes substantially longer than the practical timing
  window, split at a natural sentence or clause boundary.
- Avoid boundaries that create meaningless, dependent, or unnatural short
  fragments.
- Combine adjacent units only when they retain one communicative function and
  their combined duration remains practical for one gesture opportunity.

The supplied durations are deterministic estimates based on the calibrated
Navel speech rate. Use them only for practical grouping; do not reproduce or
recalculate them.

Choose only which boundaries between adjacent atomic units should remain.
Return exactly one JSON object in this form:
{"boundaries_after": [1, 3]}

A number N means that a final communicative segment ends after atomic unit N.

Output rules:
- Only use unit IDs supplied in the input.
- IDs must be unique and strictly increasing.
- Never include the final atomic-unit ID because the response end is not a
  segmentation boundary.
- Use an empty list when the complete response should remain one segment.
- Return JSON only.
- Do not return speech text, explanations, Markdown, scores, action tags, or
  additional keys.
""".strip()


class SegmentationLLM(Protocol):
    """Subset of LLMModule/LLMProvider used by this segmenter."""

    def is_available(self) -> bool: ...

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse: ...


@dataclass
class LLMSegmentationDiagnostics:
    """Debug data for the most recent V2 segmentation request."""

    output_contract_version: str = OUTPUT_CONTRACT_VERSION
    segmentation_policy_version: str = SEGMENTATION_POLICY_VERSION
    atomic_units: list[AtomicSpeechUnit] = field(default_factory=list)
    raw_llm_response: str | None = None
    parsed_boundaries_after: list | None = None
    validation_error: str | None = None
    reconstructed_segments: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "output_contract_version": self.output_contract_version,
            "segmentation_policy_version": self.segmentation_policy_version,
            "atomic_units": [unit.to_dict() for unit in self.atomic_units],
            "raw_llm_response": self.raw_llm_response,
            "parsed_boundaries_after": self.parsed_boundaries_after,
            "validation_error": self.validation_error,
            "reconstructed_segments": self.reconstructed_segments,
        }


class LLMSpeechSegmenter:
    """Select semantic boundaries between deterministic source spans."""

    output_contract_version = OUTPUT_CONTRACT_VERSION

    def __init__(
        self,
        llm: SegmentationLLM,
        profile: SpeechTimingProfile = NAVEL_SPEECH_TIMING_PROFILE,
        unitizer: AtomicSpeechUnitizer | None = None,
        timing_policy: SegmentationTimingPolicy = SegmentationTimingPolicy(),
    ) -> None:
        self._llm = llm
        self.profile = profile
        self._unitizer = unitizer or AtomicSpeechUnitizer()
        # Reuse the baseline's public calibrated word/duration calculations.
        # Its rule-based segment() method is never called here.
        self._metrics = SpeechSegmenter(profile, timing_policy)
        self.last_error: str | None = None
        self.last_diagnostics = LLMSegmentationDiagnostics()

    def segment(self, text: str) -> list[SpeechSegment]:
        """Return semantic segments, or one lossless segment on LLM failure."""
        self.last_error = None
        self.last_diagnostics = LLMSegmentationDiagnostics()
        if not text or not text.strip():
            return []

        source = text.strip()
        try:
            units = self._unitizer.unitize(source)
        except Exception as exc:
            return self._safe_fallback(source, str(exc))

        self.last_diagnostics.atomic_units = units
        if not self._llm or not self._llm.is_available():
            return self._safe_fallback(source, "LLM provider is not available")

        user_prompt = json.dumps(
            {
                "atomic_units": [
                    {
                        "unit_id": unit.unit_id,
                        "text": unit.text,
                        "estimated_duration_seconds": round(
                            self._metrics.estimate_duration(unit.text),
                            3,
                        ),
                    }
                    for unit in units
                ],
                "timing_policy": {
                    "target_min_seconds": (
                        self._metrics.timing_policy.target_min_seconds
                    ),
                    "target_max_seconds": (
                        self._metrics.timing_policy.target_max_seconds
                    ),
                    "substantially_long_seconds": (
                        self._metrics.timing_policy.substantially_long_seconds
                    ),
                    "speech_rate_wps": self.profile.words_per_second,
                },
            },
            ensure_ascii=False,
        )

        try:
            response = self._llm.generate(
                SEGMENTATION_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.0,
                response_format=self._boundary_response_format(len(units)),
            )
            raw = response.text if isinstance(response.text, str) else ""
            if not raw.strip() and isinstance(response.clean_text, str):
                raw = response.clean_text
            self.last_diagnostics.raw_llm_response = raw

            payload = self._parse_json(raw)
            candidate = (
                payload.get("boundaries_after")
                if isinstance(payload, dict)
                else None
            )
            if isinstance(candidate, list):
                self.last_diagnostics.parsed_boundaries_after = list(candidate)
            boundaries = self._validate_boundary_ids(payload, len(units))
            segment_texts = self._reconstruct(source, units, boundaries)
            self.last_diagnostics.reconstructed_segments = list(segment_texts)
        except Exception as exc:
            return self._safe_fallback(source, str(exc))

        return self._build_segments(segment_texts)

    @staticmethod
    def _boundary_response_format(unit_count: int) -> dict:
        """Constrain native structured output to non-final atomic-unit IDs."""
        array_schema: dict = {
            "type": "array",
            "items": {"type": "integer"},
            "uniqueItems": True,
            "maxItems": max(0, unit_count - 1),
        }
        if unit_count > 1:
            array_schema["items"].update({
                "minimum": 1,
                "maximum": unit_count - 1,
            })

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "speech_boundary_selection",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "boundaries_after": array_schema,
                    },
                    "required": ["boundaries_after"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _parse_json(raw: str) -> dict:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("LLM returned empty segmentation output")
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned malformed segmentation JSON") from exc
        return payload

    @staticmethod
    def _validate_boundary_ids(payload: dict, unit_count: int) -> list[int]:
        if not isinstance(payload, dict) or set(payload) != {"boundaries_after"}:
            raise ValueError(
                "Segmentation output must contain only 'boundaries_after'"
            )

        boundaries = payload["boundaries_after"]
        if not isinstance(boundaries, list):
            raise ValueError("'boundaries_after' must be a list")
        if any(type(boundary) is not int for boundary in boundaries):
            raise ValueError("Every boundary ID must be an integer")
        if len(boundaries) != len(set(boundaries)):
            raise ValueError("Boundary IDs must be unique")
        if boundaries != sorted(boundaries):
            raise ValueError("Boundary IDs must be strictly increasing")
        if unit_count in boundaries:
            raise ValueError("The final atomic-unit ID cannot be a boundary")
        if any(boundary < 1 or boundary >= unit_count for boundary in boundaries):
            raise ValueError("Boundary ID is outside the available unit range")
        return boundaries

    @staticmethod
    def _reconstruct(
        source: str,
        units: list[AtomicSpeechUnit],
        boundaries_after: list[int],
    ) -> list[str]:
        """
        Slice only the immutable source; never use model-generated speech text.

        Atomic separator whitespace belongs to the preceding unit span. At a
        selected final boundary it is removed from that segment's right edge.
        Internal whitespace inside grouped units remains byte-for-byte intact.
        """
        final_unit_ids = [*boundaries_after, units[-1].unit_id]
        texts: list[str] = []
        start_offset = units[0].start_offset

        for final_unit_id in final_unit_ids:
            end_offset = units[final_unit_id - 1].end_offset
            segment = source[start_offset:end_offset].strip()
            if not segment:
                raise ValueError("Boundary IDs produced an empty segment")
            texts.append(segment)
            start_offset = end_offset

        if start_offset != len(source):
            raise ValueError("Reconstructed segments do not cover the source")
        return texts

    def _safe_fallback(self, source: str, reason: str) -> list[SpeechSegment]:
        self.last_error = reason
        self.last_diagnostics.validation_error = reason
        logger.warning(
            "[LLMSpeechSegmenter] Invalid output; preserving full response: %s",
            reason,
        )
        return self._build_segments([source])

    def _build_segments(self, texts: list[str]) -> list[SpeechSegment]:
        segments: list[SpeechSegment] = []
        for segment_id, text in enumerate(texts, start=1):
            word_count = self._metrics.count_words(text)
            segments.append(
                SpeechSegment(
                    segment_id=segment_id,
                    text=text,
                    word_count=word_count,
                    estimated_duration=round(
                        self._metrics.estimate_duration(text),
                        3,
                    ),
                )
            )
        return segments
