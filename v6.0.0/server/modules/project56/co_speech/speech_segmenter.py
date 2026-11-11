"""Deterministic speech segmentation for Method B co-speech planning."""

from __future__ import annotations

import re
from dataclasses import dataclass

from modules.project56.co_speech.atomic_speech_unitizer import AtomicSpeechUnitizer
from modules.project56.co_speech.models import SpeechSegment, SpeechTimingProfile
from modules.project56.co_speech.profiles import NAVEL_SPEECH_TIMING_PROFILE


WORD_PATTERN = re.compile(r"\b[\w]+(?:'[\w]+)*\b")
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'’”]))\s+"
)
DEPENDENT_FRAGMENT_PATTERN = re.compile(
    r"^(?:because|although|unless|whereas|even though|provided that)\b",
    re.IGNORECASE,
)
GREETING_PATTERN = re.compile(
    r"^(?:hello|hi|hey|welcome\b|thanks?\b)",
    re.IGNORECASE,
)
CONFIRMATION_PATTERN = re.compile(
    r"^(?:yes\b|exactly\b|of course\b|that is correct\b)",
    re.IGNORECASE,
)
PROBLEM_PATTERN = re.compile(
    r"^(?:unfortunately\b|one limitation\b|there (?:is|are) .*limitation\b)",
    re.IGNORECASE,
)
TRANSITION_PATTERN = re.compile(
    r"^(?:by the way\b|for now\b|next\b|now let(?:'s| us)\b)",
    re.IGNORECASE,
)
INSTRUCTION_PATTERN = re.compile(
    r"^(?:please\b|(?:you|the visitor)\s+(?:should|must|need to)\b|"
    r"stand\b|move\b|wait\b|come\b)",
    re.IGNORECASE,
)
OFFER_PATTERN = re.compile(
    r"^(?:let me\b|I can\b|I (?:can|could|would) (?:show|explain|demonstrate)\b)",
    re.IGNORECASE,
)
COHESIVE_CONTINUATION_PATTERN = re.compile(
    r"^(?:it|this|these|those|they|the (?:result|results|effect|effects)|"
    r"as a result|also|in addition)\b",
    re.IGNORECASE,
)
LEADING_CONNECTOR_PATTERN = re.compile(
    r"^(?:and|but|so|however|therefore|yet|nevertheless)\b[\s,]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SegmentationTimingPolicy:
    """Controlled timing interpretation for semantic/timing V2."""

    target_min_seconds: float = 2.0
    target_max_seconds: float = 6.0
    substantially_long_seconds: float = 8.0

    def __post_init__(self) -> None:
        if not 0 < self.target_min_seconds <= self.target_max_seconds:
            raise ValueError("Segment duration targets must be positive and ordered")
        if self.substantially_long_seconds <= self.target_max_seconds:
            raise ValueError("Long-segment trigger must exceed the target maximum")


class SpeechSegmenter:
    """
    Split generated speech into structurally conservative, action-free units.

    Sentence boundaries expose normal gesture opportunities. Communicative
    function, local cohesion, and the configured practical timing window then
    determine whether adjacent units merge or long clauses split.
    """

    def __init__(
        self,
        profile: SpeechTimingProfile = NAVEL_SPEECH_TIMING_PROFILE,
        timing_policy: SegmentationTimingPolicy = SegmentationTimingPolicy(),
    ) -> None:
        self.profile = profile
        self.timing_policy = timing_policy
        self._unitizer = AtomicSpeechUnitizer()

    def segment(self, text: str) -> list[SpeechSegment]:
        """Return final ordered segments for a full generated response."""
        if not text or not text.strip():
            return []

        initial = self._split_sentences(text.strip())
        split_segments: list[str] = []
        for sentence in initial:
            split_segments.extend(self._split_sentence(sentence))

        merged_segments = self._merge_related_units(split_segments)
        return [
            self._make_segment(index, segment_text)
            for index, segment_text in enumerate(merged_segments, start=1)
        ]

    def count_words(self, text: str) -> int:
        """Count words using the same token rule as Navel calibration."""
        return len(WORD_PATTERN.findall(text))

    def estimate_duration(self, text: str) -> float:
        """Estimate speech time from the configured calibrated rate."""
        return self.count_words(text) / self.profile.words_per_second

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(text) if part.strip()]

    def _split_sentence(self, text: str) -> list[str]:
        """Split for a clear function change or a substantially long unit."""
        units = self._unitizer.unitize(text)
        if len(units) < 2:
            return [text]

        candidates = [unit.end_offset for unit in units[:-1]]
        for boundary in candidates:
            left = text[:boundary].strip()
            right = text[boundary:].strip()
            if self._is_clear_function_change(left, right):
                return self._split_sentence(left) + self._split_sentence(right)

        if self.estimate_duration(text) <= self.timing_policy.substantially_long_seconds:
            return [text]

        viable: list[tuple[float, int, str, str]] = []
        for boundary in candidates:
            left = text[:boundary].strip()
            right = text[boundary:].strip()
            left_duration = self.estimate_duration(left)
            right_duration = self.estimate_duration(right)
            if min(left_duration, right_duration) < self.timing_policy.target_min_seconds:
                continue
            viable.append((abs(left_duration - right_duration), boundary, left, right))

        if not viable:
            return [text]

        _, _, left, right = min(viable)
        return self._split_sentence(left) + self._split_sentence(right)

    def _merge_related_units(self, segments: list[str]) -> list[str]:
        """Merge dependent or short, cohesive same-function neighbours."""
        merged: list[str] = []
        index = 0

        while index < len(segments):
            current = segments[index]
            has_next = index + 1 < len(segments)
            is_dependent = (
                DEPENDENT_FRAGMENT_PATTERN.match(current)
                and "," not in current
            )
            if has_next and is_dependent:
                merged.append(self._join(current, segments[index + 1]))
                index += 2
                continue

            if has_next:
                following = segments[index + 1]
                combined = self._join(current, following)
                same_flow = (
                    self._communicative_function(current)
                    == self._communicative_function(following)
                    and COHESIVE_CONTINUATION_PATTERN.match(following)
                )
                if (
                    same_flow
                    and self.estimate_duration(combined)
                    <= self.timing_policy.target_max_seconds
                ):
                    merged.append(combined)
                    index += 2
                    continue

            merged.append(current)
            index += 1

        return merged

    def _is_clear_function_change(self, left: str, right: str) -> bool:
        return self._communicative_function(left) != self._communicative_function(right)

    @staticmethod
    def _communicative_function(text: str) -> str:
        candidate = LEADING_CONNECTOR_PATTERN.sub("", text.strip())
        if candidate.endswith("?"):
            return "question"
        for function, pattern in (
            ("greeting", GREETING_PATTERN),
            ("confirmation", CONFIRMATION_PATTERN),
            ("problem", PROBLEM_PATTERN),
            ("transition", TRANSITION_PATTERN),
            ("instruction", INSTRUCTION_PATTERN),
            ("offer", OFFER_PATTERN),
        ):
            if pattern.match(candidate):
                return function
        return "explanation"

    @staticmethod
    def _join(left: str, right: str) -> str:
        return f"{left.rstrip()} {right.lstrip()}"

    def _make_segment(self, segment_id: int, text: str) -> SpeechSegment:
        word_count = self.count_words(text)
        return SpeechSegment(
            segment_id=segment_id,
            text=text,
            word_count=word_count,
            estimated_duration=round(
                word_count / self.profile.words_per_second,
                3,
            ),
        )
