"""
decision/visitor_profile.py
============================
What is known about a visitor BEFORE the tour starts, and what that produces.

Two effects, both the SAME mechanism applied in two places — this is the
design the handoff brief settled on rather than something new:

  STYLE     A framing directive appended to every generated step's instruction
            text. Every robot shifts the same way for the same audience — a
            technical visitor gets more precise language from ChatBox AND
            Navel AND Silbot alike. That makes it a property of the VISITOR,
            not of any one robot, so it is injected once, uniformly, at
            generation time — never stored per robot.

  INTEREST  Resolved to topic ids ONCE, when the profile is set, not re-resolved
            on every turn. Used as the STANDING layer-2 baseline for block
            importance for the rest of the run, until a fresh utterance
            supplies a more specific, more recent signal — see
            decision.planner.resolve_emphasis, which is the explicit ordering
            between "what the visitor said just now" and "what they told us at
            the start".

Immutable once set: a mid-tour change of interest is a fresh utterance, handled
separately, not a mutation of this profile. Re-profiling mid-run would also
retroactively change earlier PLAN_REVISE decisions' interpretation if this were
mutable shared state, which is exactly the kind of drift Observation is built to
prevent elsewhere.
"""

from __future__ import annotations
from dataclasses import dataclass


# Suffixes are deliberately short — appended to an instruction that already
# tells the robot what to say; this only steers HOW, not WHAT.
STYLE_FRAMING = {
    "technical": (
        " Frame this for a technical audience: use precise terminology, name "
        "the underlying methods or algorithms, and do not shy away from "
        "implementation detail."
    ),
    "business": (
        " Frame this for a business audience: lead with practical value and "
        "real-world application, avoid technical jargon, and keep the "
        "emphasis on outcomes rather than mechanism. State it; do not offer "
        "a demonstration or invite the visitor to try anything — closing "
        "with \"interested in seeing this in action?\" belongs to the "
        "interactive framing and blurs the two."
    ),
    "interactive": (
        " Frame this for a hands-on audience: invite the visitor to ask a "
        "question or try something, and keep the tone conversational rather "
        "than a monologue."
    ),
    "general": "",   # no directive — the script's own wording stands as-is
}

DEFAULT_STYLE = "general"


@dataclass(frozen=True)
class VisitorProfile:
    """Set once, before a demo starts. `topics` must already be resolved —
    this module does not import decision.kg_policy, matching the rest of
    decision/'s no-I/O discipline; resolution happens at the call site that
    does have the graph (the gateway that receives the pre-demo form)."""

    interest_text: str = ""
    style: str = DEFAULT_STYLE
    topics: tuple = ()

    @property
    def framing(self) -> str:
        return STYLE_FRAMING.get(self.style, "")

    def as_dict(self) -> dict:
        return {"interest_text": self.interest_text, "style": self.style,
               "topics": list(self.topics)}
