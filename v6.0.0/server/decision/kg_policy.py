"""
decision/kg_policy.py
=====================
QA_ROUTE backed by the competence graph.

The baseline routes to whoever heard the question — no decision at all. This
replaces that with: work out what the question is ABOUT, ask the graph which
robot handles that subject, and route there.

Everything it needs is injected. The graph arrives as a snapshot rather than a
live store, so a routing decision costs no database round-trip on the critical
path and the same snapshot can be replayed offline against a different policy.

FAILS OPEN, ALWAYS
Every path that cannot produce a confident answer returns None, and the caller
falls back to the receiver. An unresolvable topic, an empty graph, a fleet of
one — all of them mean "no opinion", not "error". A routing layer that can
strand a visitor's question is worse than no routing layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from decision.kg import RobotTopicEdge
from decision.kg_infer import route

MIN_TOPIC_OVERLAP = 1
"""Content words a question must share with a topic label to resolve to it."""

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "is",
    "are", "was", "how", "what", "why", "when", "who", "which", "can", "could",
    "do", "does", "did", "you", "your", "it", "that", "this", "about", "tell",
    "me", "us", "more", "explain", "describe", "work", "works", "like",
}


# Stripped repeatedly, longest match first, until nothing applies. Enough
# morphology for a fourteen-topic technical vocabulary and nothing more — this
# is a matcher, not a linguistics project.
#
# Note what is NOT here: -ation and friends. Stripping "navigation" straight to
# "navig" while "navigate" only reaches "navigat" leaves them further apart than
# before. Letting -ion and -e apply in separate passes lands both on "navigat".
# The same fix carries "coordinate"/"coordination" and "conversation"/
# "conversational".
_SUFFIXES = (
    "ities", "ions", "ing", "ies", "ion", "ers", "est", "ely",
    "al", "ed", "es", "er", "ly", "s", "e",
)
_MIN_STEM = 4


def _stem(word: str) -> str:
    """Crude iterative suffix stripper, with one undoubling rule.

    Exists because corpus questions failed to resolve on pure morphology: "how
    do you navigate around people" shares no token with "social robot
    navigation", "how do you build a map of the lab" none with "mapping and
    localisation". Every one is a suffix away, and a person reads them instantly.

    Stemming was tried before embeddings deliberately: an embedding model
    measured on this vocabulary matched surface lexical overlap anyway — scoring
    "emotion recognition" against "speech recognition" at 0.76 while missing
    "retrieval augmented generation" against "large language models" entirely.
    Cheaper fix first.

    Applied to BOTH sides, so it can only merge tokens that were already close.
    _MIN_STEM is 4 rather than 3 because at 3 the stripping runs away — "cues"
    reaches "cu" and starts colliding with unrelated words.
    """
    w = word
    for _ in range(3):                      # bounded; three passes is plenty
        for suffix in _SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= _MIN_STEM:
                w = w[: -len(suffix)]
                break
        else:
            break
    # mapping -> mapp -> map. Undouble the final repeated consonant English
    # inserts before -ing/-ed, which otherwise blocks the match.
    if len(w) > _MIN_STEM - 1 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def _words(text: str) -> set:
    """Content words of a phrase, stemmed. Stopwords are removed BEFORE
    stemming, since the stopword list is written in surface forms."""
    raw = {w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
           if w and w not in STOPWORDS and len(w) > 2}
    return {_stem(w) for w in raw}


@dataclass(frozen=True)
class RoutingDecision:
    """What the graph decided, and enough to explain it in the log."""

    robot_id: str
    topic_id: str
    topic_label: str
    reason: str          # from kg_infer.route — argmax, or which explore rule
    score: float


class KGRouter:
    """
    Resolve a question to a topic, then a topic to a robot.

    Topic resolution is word overlap against topic labels, which is crude and
    deliberately so: a smarter resolver would be a second thing to defend, and
    the claim under test is about the GRAPH, not about matching. Ambiguity
    resolves to no topic rather than a guess — routing on a misread subject is
    worse than not routing.
    """

    def __init__(
        self,
        edges: Iterable[RobotTopicEdge],
        links: Iterable[tuple],
        topics: Iterable[dict],
        explore: bool = True,
    ):
        self._edges = list(edges)
        self._links = list(links)
        self._topics = {t["id"]: t.get("label", t["id"]) for t in topics}
        self._words = {tid: _words(label) for tid, label in self._topics.items()}
        self._explore = explore

    # ── Topic resolution ──────────────────────────────────────────────────────

    def resolve_topic(self, utterance: str) -> Optional[str]:
        """Best-matching topic id, or None when nothing clearly matches.

        Returns None on a TIE as well as on no match. Two topics matching a
        question equally well means the question did not identify one, and
        picking either would route on a coin flip.
        """
        qw = _words(utterance)
        if not qw:
            return None
        scored = [(tid, len(qw & tw)) for tid, tw in self._words.items()]
        scored = [(t, n) for t, n in scored if n >= MIN_TOPIC_OVERLAP]
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[1], x[0]))
        if len(scored) > 1 and scored[0][1] == scored[1][1]:
            return None
        return scored[0][0]

    # ── Routing ───────────────────────────────────────────────────────────────

    def decide(self, utterance: str, robot_ids: Iterable[str]) -> Optional[RoutingDecision]:
        """Who should answer? None means the graph has no opinion."""
        robot_ids = list(robot_ids)
        if len(robot_ids) < 2:
            return None          # nothing to choose between
        topic_id = self.resolve_topic(utterance)
        if topic_id is None:
            return None

        picked, reason = route(self._edges, self._links, topic_id,
                               robot_ids, explore=self._explore)
        if picked is None:
            return None

        from decision.kg_infer import rank_robots
        ranked = dict(rank_robots(self._edges, self._links, topic_id, robot_ids))
        return RoutingDecision(
            robot_id=picked, topic_id=topic_id,
            topic_label=self._topics.get(topic_id, topic_id),
            reason=reason, score=round(ranked.get(picked, 0.5), 4),
        )
