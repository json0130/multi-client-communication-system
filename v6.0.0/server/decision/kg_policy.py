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

MIN_OVERLAP_TO_LEAVE_CONTEXT = 2
"""Content words needed to resolve a question ONTO A SUBJECT THE CURRENT
SPEAKER DOES NOT OWN.

A single shared word is enough to identify a topic when there is no context
to contradict it, and far too little to justify taking a question away from
the robot the visitor is standing in front of. A real run: mid-way through
Silbot's navigation talk a visitor asked "which technique or model do you
use". `which`, `do` and `you` are stopwords, so the whole utterance reduced
to {hmm, model, techniqu, then, use} — and `model` alone matched "large
language models", so the question went to ChatBox, which answered about
retrieval-augmented generation. Silbot then had to contradict it: "No, we
use SLAM and social robotics techniques. RAG is not part of our approach."

Note what was lost: `you` is the word that made it a question ABOUT SILBOT,
and stopword removal discards it. Rather than special-casing pronouns —
which would need a second matcher and a second thing to defend — the
context does that work. Inside a robot's own block its declared topics are
the assumed subject, and leaving them takes more than one generic noun.

The cost is real and worth stating: "what about emotions?" asked during
Silbot's talk now resolves to nothing and Silbot answers it, rather than
routing to Navel on the strength of one word. That is the safer failure —
the robot in front of the visitor replying, possibly to say whose area it
is, beats a confident handover built on a coincidence."""

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


ABSENT_ROBOT_POLICY = "defer"
"""What to do when the robot the graph would have picked is not in the
conversation. A named flag rather than a hardcoded branch because it is a
behavioural choice worth measuring, not an implementation detail:

  "defer"         the guide says the topic will be covered when the group
                  reaches that robot's station. Preserves the routing
                  decision — the right robot still answers, just later —
                  and costs the visitor a wait.
  "guide_answers" the guide answers now, from what it knows. Costs the
                  visitor the specialist's answer, saves them the wait.

Defaults to "defer" because the specialist's answer at their own station is
the better answer, and the tour is going there anyway. It is only the wrong
call when the group will NEVER reach that station — which is why deferring
checks the remaining plan first and falls back to guide_answers when the
block has already been cut. See KGRouter.decide.
"""

VALID_ABSENT_POLICIES = ("defer", "guide_answers")


@dataclass(frozen=True)
class RoutingDecision:
    """What the graph decided, and enough to explain it in the log."""

    robot_id: Optional[str]   # who answers; None when the question was deferred
    topic_id: str
    topic_label: str
    reason: str          # from kg_infer.route — argmax, or which explore rule
    score: float
    # How many candidates the decision actually chose BETWEEN, after
    # eligibility, presence and declared scope had narrowed the field. 1
    # means there was no choice to make, which is what tells the outcome
    # path that a clean segment here carries no information about quality —
    # see kg_feedback.Segment.note_routed.
    candidates_considered: int = 0

    # Set only when reason is a defer: the absent robot whose own station
    # will cover this topic later. The caller uses it to name that robot in
    # what the guide says, and to know that NOTHING was routed — a deferred
    # question is not an observation about anybody. See decision/kg_feedback.py's
    # Segment silence rule.
    deferred_to: Optional[str] = None

    @property
    def is_deferred(self) -> bool:
        return self.deferred_to is not None


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
        absent_robot_ids: Optional[Iterable[str]] = None,
        absent_policy: str = ABSENT_ROBOT_POLICY,
    ):
        self._edges = list(edges)
        self._links = list(links)
        self._topics = {t["id"]: t.get("label", t["id"]) for t in topics}
        self._words = {tid: _words(label) for tid, label in self._topics.items()}
        self._explore = explore
        self._absent = set(absent_robot_ids or ())
        if absent_policy not in VALID_ABSENT_POLICIES:
            raise ValueError(
                f"absent_policy must be one of {VALID_ABSENT_POLICIES}, "
                f"got {absent_policy!r}"
            )
        self._absent_policy = absent_policy

    # ── Topic resolution ──────────────────────────────────────────────────────

    def declared_topics(self, robot_id: Optional[str]) -> set:
        """The topics this robot's project area covers, from `specialised`.

        Empty for an unknown robot, or before migration 010 has been applied
        — which is what makes every context-sensitive path below degrade to
        the behaviour it had before declared scope existed.
        """
        if not robot_id:
            return set()
        return {e.topic_id for e in self._edges
                if e.robot_id == robot_id and e.specialised}

    def resolve_topic(self, utterance: str,
                      context_topics: Optional[Iterable[str]] = None) -> Optional[str]:
        """Best-matching topic id, or None when nothing clearly matches.

        Returns None on a TIE as well as on no match. Two topics matching a
        question equally well means the question did not identify one, and
        picking either would route on a coin flip.

        `context_topics` are the subjects currently being presented — the
        declared area of whoever is speaking. A match that would leave them
        needs MIN_OVERLAP_TO_LEAVE_CONTEXT words rather than one; see that
        constant for the run this comes from. With no context supplied the
        rule cannot fire and resolution behaves exactly as it always did.
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

        topic, overlap = scored[0]
        context = set(context_topics or ())
        if (context and topic not in context
                and overlap < MIN_OVERLAP_TO_LEAVE_CONTEXT):
            return None
        return topic

    # ── Routing ───────────────────────────────────────────────────────────────

    def decide(
        self,
        utterance: str,
        robot_ids: Iterable[str],
        remaining_block_ids: Optional[Iterable[str]] = None,
        guide_robot_id: Optional[str] = None,
        absent_robot_ids: Optional[Iterable[str]] = None,
        context_robot_id: Optional[str] = None,
    ) -> Optional[RoutingDecision]:
        """
        Who should answer? None means the graph has no opinion.

        `absent_robot_ids` overrides the constructor's set for this call.
        It has to be per-call rather than fixed at construction: the router
        is cached with a TTL and reused across many turns, while who is in
        the conversation changes as the group moves between stations. A set
        frozen at construction would be stale by the second question.

        `context_robot_id` is whoever is presenting — its declared topics
        are treated as the subject under discussion, so an ambiguous
        follow-up stays with the robot the visitor is actually talking to.

        `remaining_block_ids` and `guide_robot_id` are only consulted when
        the robot the graph would have picked is absent — see
        ABSENT_ROBOT_POLICY. Both optional, and absent handling degrades to
        "no opinion" without them rather than promising something it cannot
        check.
        """
        robot_ids = list(robot_ids)
        if len(robot_ids) < 2:
            return None          # nothing to choose between
        # The subject currently on the floor. A weak match that would leave
        # it is not enough to move the question elsewhere — see
        # MIN_OVERLAP_TO_LEAVE_CONTEXT.
        topic_id = self.resolve_topic(
            utterance, context_topics=self.declared_topics(context_robot_id))
        if topic_id is None:
            return None

        absent = (set(absent_robot_ids) if absent_robot_ids is not None
                  else self._absent)

        picked, reason = route(self._edges, self._links, topic_id,
                               robot_ids, explore=self._explore,
                               absent=absent)

        # Who WOULD have answered if everyone were present. Computed only to
        # detect the absent-best case: if presence changed the answer, that
        # is a different situation from ordinary routing and gets its own
        # policy rather than silently handing the question to a runner-up
        # the visitor never asked about.
        if absent:
            picked_ignoring_absence, _ = route(
                self._edges, self._links, topic_id, robot_ids,
                explore=self._explore,
            )
            if (picked_ignoring_absence is not None
                    and picked_ignoring_absence in absent):
                return self._handle_absent(
                    topic_id, picked_ignoring_absence,
                    remaining_block_ids, guide_robot_id,
                )

        if picked is None:
            return None

        # The field the decision actually chose between, after every
        # structural filter. One candidate means there was no choice, which
        # the outcome path needs to know — see Segment.note_routed.
        from decision.kg_infer import surviving_candidates
        considered = len(surviving_candidates(self._edges, topic_id,
                                              robot_ids, absent))
        return self._decision(topic_id, picked, reason, robot_ids, considered)

    def _handle_absent(
        self,
        topic_id: str,
        absent_pick: str,
        remaining_block_ids: Optional[Iterable[str]],
        guide_robot_id: Optional[str],
    ) -> Optional[RoutingDecision]:
        """
        The best robot for this topic is not in the conversation.

        Deferring promises the visitor that the topic gets covered at that
        robot's station, so it is only honest while that station is still on
        the itinerary. PLAN_REVISE can have cut the block already — under
        time pressure that is exactly when it would have — and a promise
        about a station the group will never reach is worse than simply
        answering now. So a defer that cannot be kept becomes a
        guide_answers, and says so in the reason.
        """
        label = self._topics.get(topic_id, topic_id)

        if self._absent_policy == "defer":
            still_coming = (remaining_block_ids is None
                            or absent_pick in set(remaining_block_ids))
            if still_coming:
                return RoutingDecision(
                    robot_id=None, topic_id=topic_id, topic_label=label,
                    reason=f"defer: {absent_pick} absent, covered at their station",
                    score=0.0, deferred_to=absent_pick,
                )
            # Fall through to guide_answers — the block is gone, so there is
            # no station left to defer to.
            if guide_robot_id:
                return RoutingDecision(
                    robot_id=guide_robot_id, topic_id=topic_id, topic_label=label,
                    reason=f"guide answers: {absent_pick} absent and its block "
                           f"was already cut, nothing left to defer to",
                    score=0.0,
                )

        if self._absent_policy == "guide_answers" and guide_robot_id:
            return RoutingDecision(
                robot_id=guide_robot_id, topic_id=topic_id, topic_label=label,
                reason=f"guide answers: {absent_pick} absent",
                score=0.0,
            )

        # No guide to fall back on. "No opinion" hands the turn to whoever
        # received it, which is the same degradation every other unresolvable
        # path in this module takes.
        return None

    def _decision(self, topic_id: str, picked: str, reason: str,
                  robot_ids: list, considered: int = 0) -> RoutingDecision:
        from decision.kg_infer import rank_robots
        ranked = dict(rank_robots(self._edges, self._links, topic_id, robot_ids))
        return RoutingDecision(
            robot_id=picked, topic_id=topic_id,
            topic_label=self._topics.get(topic_id, topic_id),
            reason=reason, score=round(ranked.get(picked, 0.5), 4),
            candidates_considered=considered,
        )
