"""
Self-declared culture detection (app layer).

A person's `belongs_to_culture` tag must come from what THEY explicitly say about
their own background — never inferred from their name, face, language, or the fact
that they like a cuisine/music. `detect_self_declared_culture` reads the person's
own turns and returns a culture label ONLY when they clearly self-identify
("I'm Korean", "I was born in Korea", "my family is Korean"); otherwise None.

This is deliberately conservative — liking kimchi, watching K-dramas, or speaking
about Korea is NOT a declaration of being Korean.

LLM-gated (the extraction LLM is injected); no PAD, no embeddings.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

LLMFn = Callable[[str, str], str]

_SYSTEM = (
    "You decide whether the PERSON explicitly states their OWN cultural, ethnic, or "
    "national background in what they said.\n"
    "Reply with ONLY the background as a single short label (e.g. 'Korean', "
    "'Japanese', 'Nigerian', 'Mexican') — and ONLY when the person clearly says they "
    "ARE that, are FROM there, were born there, or that their family/heritage is that.\n"
    "Reply exactly 'NONE' if they merely like a cuisine/music/show, mention a place, "
    "speak a language, or if it is at all ambiguous.\n"
    "Never guess from names, food, language, or preferences — only an explicit "
    "self-statement about who they are.\n"
    "Examples:\n"
    "  'I'm Korean' -> Korean\n"
    "  'my parents are from Japan' -> Japanese\n"
    "  'I love kimchi and k-dramas' -> NONE\n"
    "  'I visited Korea last year' -> NONE\n"
    "  'I'm learning Korean' -> NONE"
)


def _person_transcript(turns: List[dict]) -> str:
    """Only the PERSON's own words — never the robot's replies (which may mention a
    culture and must not trigger a self-declaration)."""
    lines = [str(t.get("child", "")).strip()
             for t in turns if t.get("child")]
    return "\n".join(l for l in lines if l)


def detect_self_declared_culture(turns: List[dict], llm_fn: LLMFn) -> Optional[str]:
    """Return a self-declared culture label (e.g. 'Korean') or None.

    Conservative: only an EXPLICIT self-statement counts. Returns None on any
    ambiguity, on 'NONE', on an empty transcript, or on LLM failure.
    """
    transcript = _person_transcript(turns)
    if not transcript:
        return None
    try:
        raw = llm_fn(_SYSTEM, transcript)
    except Exception:  # noqa: BLE001 — detection is best-effort
        return None

    label = (raw or "").strip().strip('".\'').splitlines()[0].strip() if raw else ""
    # Reject empties / explicit NONE / sentences / tags.
    if not label or label.upper() == "NONE":
        return None
    if label.startswith("[") or len(label.split()) > 3 or len(label) > 30:
        return None
    # Must be a plausible demonym word (letters/spaces/hyphen only).
    if not re.fullmatch(r"[A-Za-z][A-Za-z \-]*", label):
        return None
    return label.title()
