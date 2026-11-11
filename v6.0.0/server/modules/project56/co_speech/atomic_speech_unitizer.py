"""Lossless deterministic candidate units for LLM speech segmentation."""

from __future__ import annotations

import re

from modules.project56.co_speech.models import AtomicSpeechUnit


WORD_PATTERN = re.compile(r"\b[\w]+(?:'[\w]+)*\b")
ATOMIC_UNITIZER_VERSION = "v2.1"
FINITE_AUXILIARIES = frozenset({
    "am", "is", "are", "was", "were", "can", "cannot", "could", "will", "would",
    "shall", "should", "may", "might", "must", "do", "does", "did",
    "have", "has", "had",
})
EXPLICIT_SUBJECTS = frozenset({
    "i", "we", "you", "he", "she", "it", "they", "this", "that",
    "these", "those", "there",
})
CLAUSE_ADVERBS = frozenset({
    "also", "already", "currently", "generally", "normally", "often",
    "probably", "still", "then", "usually",
})
NON_FINITE_AFTER_SUBJECT = frozenset({
    "a", "an", "because", "if", "that", "the", "to", "unless", "when",
    "while",
})
SENTENCE_END_PATTERN = re.compile(
    r"[.!?]+(?:[\"'’”])?(?:\s+|$)"
)
SEMICOLON_PATTERN = re.compile(r";(?:\s+|$)")
COLON_PATTERN = re.compile(
    r":\s+(?=(?:I|we|you|he|she|it|they|this|that|these|those|there|the|a|an)\b)",
    re.IGNORECASE,
)
CONNECTOR_PATTERN = re.compile(
    r"(?P<separator>,\s+|\s+)(?=(?P<connector>but|so|however|therefore|yet|"
    r"nevertheless|and|though)\b)",
    re.IGNORECASE,
)
STRONG_LOCAL_BOUNDARY_PATTERN = re.compile(r"[.!?;:]")
DISCOURSE_ADVERBS = frozenset({"however", "therefore", "nevertheless", "though"})


class AtomicSpeechUnitizer:
    """
    Expose a fine-grained superset of plausible communicative boundaries.

    Atomic spans partition the stripped clean source exactly. Whitespace after
    a candidate boundary belongs to the preceding unit. These candidates are
    structural only; the LLM decides which ones become final boundaries.
    """

    def unitize(self, text: str) -> list[AtomicSpeechUnit]:
        if not text or not text.strip():
            return []

        source = text.strip()
        boundaries = self._candidate_boundaries(source)
        units: list[AtomicSpeechUnit] = []
        start = 0

        for end in [*boundaries, len(source)]:
            span = source[start:end]
            if not span.strip():
                start = end
                continue
            units.append(
                AtomicSpeechUnit(
                    unit_id=len(units) + 1,
                    start_offset=start,
                    end_offset=end,
                    text=span,
                )
            )
            start = end

        self._validate_partition(source, units)
        return units

    def _candidate_boundaries(self, source: str) -> list[int]:
        boundaries = {
            match.end()
            for pattern in (SENTENCE_END_PATTERN, SEMICOLON_PATTERN, COLON_PATTERN)
            for match in pattern.finditer(source)
            if match.end() < len(source)
        }

        for match in CONNECTOR_PATTERN.finditer(source):
            boundary = match.end()
            if boundary >= len(source):
                continue
            if self._is_plausible_connector_boundary(
                source,
                boundary,
                match.group("connector").lower(),
                match.group("separator"),
            ):
                boundaries.add(boundary)

        return self._remove_empty_fragments(source, sorted(boundaries))

    @classmethod
    def _is_plausible_connector_boundary(
        cls,
        source: str,
        boundary: int,
        connector: str,
        separator: str,
    ) -> bool:
        """Require an independent-looking clause immediately after a connector."""
        local_left_start = max(
            source.rfind(".", 0, boundary),
            source.rfind("?", 0, boundary),
            source.rfind("!", 0, boundary),
            source.rfind(";", 0, boundary),
            source.rfind(":", 0, boundary),
        ) + 1
        if len(WORD_PATTERN.findall(source[local_left_start:boundary])) < 2:
            return False

        # A sentence-medial discourse adverb is not a boundary merely because
        # it follows whitespace. Punctuation must introduce it.
        if connector in DISCOURSE_ADVERBS and not separator.startswith(","):
            return False

        remaining = source[boundary:]
        strong_end = STRONG_LOCAL_BOUNDARY_PATTERN.search(remaining)
        local_clause = remaining[:strong_end.start()] if strong_end else remaining
        connector_match = re.match(
            rf"{re.escape(connector)}\b[\s,]*",
            local_clause,
            re.IGNORECASE,
        )
        if not connector_match:
            return False

        clause_body = local_clause[connector_match.end():].strip()
        if connector == "so" and re.match(r"that\b", clause_body, re.IGNORECASE):
            return False
        return cls._has_local_independent_clause(clause_body)

    @staticmethod
    def _has_local_independent_clause(clause_body: str) -> bool:
        """Conservatively recognise a local explicit subject plus finite verb."""
        words = [match.group(0) for match in WORD_PATTERN.finditer(clause_body)]
        if len(words) < 2:
            return False

        lowered = [word.lower() for word in words]
        subject = lowered[0]
        if subject in EXPLICIT_SUBJECTS:
            verb_index = 1
            while verb_index < len(lowered) and lowered[verb_index] in CLAUSE_ADVERBS:
                verb_index += 1
            if verb_index >= len(lowered):
                return False
            verb = lowered[verb_index]
            return (
                verb in FINITE_AUXILIARIES
                or (
                    verb not in NON_FINITE_AFTER_SUBJECT
                    and not verb.endswith("ing")
                )
            )

        # Determiner-led noun subjects are accepted only when a nearby explicit
        # auxiliary/modal makes the clause structure unambiguous.
        if subject in {"a", "an", "the"}:
            return any(word in FINITE_AUXILIARIES for word in lowered[2:6])

        return False

    @staticmethod
    def _remove_empty_fragments(source: str, boundaries: list[int]) -> list[int]:
        accepted: list[int] = []
        start = 0
        for boundary in boundaries:
            if source[start:boundary].strip() and source[boundary:].strip():
                accepted.append(boundary)
                start = boundary
        return accepted

    @staticmethod
    def _validate_partition(
        source: str,
        units: list[AtomicSpeechUnit],
    ) -> None:
        if not units:
            raise ValueError("Atomic unitizer produced no units for non-empty text")

        cursor = 0
        for expected_id, unit in enumerate(units, start=1):
            if unit.unit_id != expected_id or unit.start_offset != cursor:
                raise ValueError("Atomic units do not form an ordered source partition")
            if source[unit.start_offset:unit.end_offset] != unit.text:
                raise ValueError("Atomic unit text does not match its source span")
            cursor = unit.end_offset

        if cursor != len(source) or "".join(unit.text for unit in units) != source:
            raise ValueError("Atomic units do not reconstruct the source exactly")
