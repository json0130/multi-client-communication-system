"""
decision/kg_infer.py
====================
Reading the competence graph: what does it say about a robot and a topic it has
never been observed on?

Compiled ON READ, never persisted — the same discipline as
modules/preference_model.py in the CHATBOX tree. Only the learned edges and the
topic links are stored; everything here is derived, so a change to the inference
rule never requires a migration or invalidates stored weights.

The walk is deliberately the same shape as preference_model.rank_suggestions:
observed nodes are clamped and never move, unobserved nodes are seeded at a
prior, and evidence spreads over weighted links for a fixed number of rounds
with damping. Keeping the two recognisably identical means a finding about one
transfers to the other.

Two differences, both forced by the setting:

  * The clamp is confidence-weighted from an OBSERVATION COUNT rather than an
    LLM's self-report. An edge seen once contributes almost nothing no matter
    how extreme its weight.
  * Propagation is per-robot. Competence at `emotion recognition` says something
    about the same robot at `facial expression analysis`, and nothing at all
    about a different robot at either.

  * Influence is a SYMMETRIC pull toward the neighbour, not preference_model's
    upward noisy-OR floor. That rule reads `p[dst] = max(p[dst], p[src]*0.8*w)`,
    which can only raise a neighbour ABOVE the unobserved prior when
    p[src] > prior / (0.8*w). CHATBOX's unobserved prior is 0.30, so that fires
    easily. Competence starts at 0.5 — "no opinion" — and with a strong link
    (w=0.8) a source would need to exceed 0.78 to move anything. A single
    correction clamps to 0.5625, so ported unchanged the rule made POSITIVE
    corrections propagate to nothing at all while negative ones spread fine.
    Measured: 0 topics reached vs 4.7. Symmetric pull removes the asymmetry,
    and is the right semantics here anyway — a neighbour of a topic a robot
    handles well should be pulled up in proportion, exactly as a neighbour of
    one it handles badly is pulled down.

WHETHER THIS HELPS IS AN EMPIRICAL QUESTION, not a design claim. With a sparse
link set most corrections reach nothing. tools/kg_reach.py measures it.
"""

from __future__ import annotations

from typing import Iterable, Optional

from decision.kg import NEUTRAL, PRIOR, RobotTopicEdge

DAMPING = 0.80
"""Per-hop attenuation. Same value as preference_model._DAMPING."""

ROUNDS = 2
"""Propagation rounds. Same as preference_model._ROUNDS — two hops is enough to
reach a neighbour's neighbour, and more turns the graph into a smoother."""


def infer(
    edges: Iterable[RobotTopicEdge],
    links: Iterable[tuple],
    robot_id: str,
    topics: Optional[Iterable[str]] = None,
) -> dict:
    """
    Posterior competence for one robot over every topic.

    `links` is [(topic_a, topic_b, weight), ...], undirected.
    Returns {topic_id: posterior}, posteriors in [0,1].

    Observed topics report their clamped weight and are pinned. Unobserved
    topics start at the neutral prior and may be pulled up or down by observed
    neighbours — symmetric, so a robot known to be BAD at one topic is inferred
    to be worse at its neighbours, not merely unknown.
    """
    observed: dict = {}
    for e in edges:
        if e.robot_id == robot_id and e.n_obs > 0:
            observed[e.topic_id] = e.clamped

    universe = set(topics or [])
    universe |= set(observed)
    for a, b, _w in links:
        universe.add(a)
        universe.add(b)
    if not universe:
        return {}

    p = {t: observed.get(t, PRIOR) for t in universe}

    def influence(src: str, dst: str, w: float) -> None:
        if dst in observed:
            return                      # pinned: observations are the evidence
        # Symmetric partial move toward the neighbour, in whichever direction it
        # sits. Only a source that has actually departed from neutral pulls at
        # all, so an unobserved neighbour-of-a-neighbour stays put rather than
        # dragging the whole graph toward the prior it already holds.
        if abs(p[src] - NEUTRAL) < 1e-9:
            return
        p[dst] += DAMPING * w * (p[src] - p[dst])

    for _ in range(ROUNDS):
        for a, b, w in links:
            if a in p and b in p:
                influence(a, b, float(w))
                influence(b, a, float(w))
    return p


def rank_robots(
    edges: Iterable[RobotTopicEdge],
    links: Iterable[tuple],
    topic_id: str,
    robot_ids: Iterable[str],
) -> list:
    """
    [(robot_id, posterior)] for one topic, best first — the QA_ROUTE read.

    Ties are broken by robot_id so the ordering is deterministic; with an empty
    graph every robot scores the prior and the caller gets a stable, arbitrary
    order rather than a different one each call.
    """
    edges = list(edges)
    links = list(links)
    scored = [(r, infer(edges, links, r).get(topic_id, PRIOR)) for r in robot_ids]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


# ── Routing, with exploration ─────────────────────────────────────────────────

MIN_EVIDENCE_MARGIN = 0.05
"""How far apart the best and worst candidate must be before the graph
claims an opinion on a topic NOBODY declares.

Only applies with exploration OFF, to undeclared topics, and only with more
than one candidate. A near-tie is exactly what the exploration rules exist
for, so a campaign still spends that turn learning; live, the same closeness
means the graph cannot tell the candidates apart. A
declared specialist wins on configuration and needs no evidence at all —
that is what declaring means — and a single remaining candidate is the
answer whatever its score.

Deliberately a SPREAD and not a distance from the prior. A robot the graph
is confident is bad at something is a real opinion about ordering, and
gating on "above the prior" would discard it and strand a question that had
a defensible answer.

Set equal to EXPLORE_MARGIN, which already encodes "closer than this and the
graph cannot tell two robots apart". Measured against the live graph, the
orphan topics separated their top two by 0.003 — two orders of magnitude
inside this."""

EXPLORE_MARGIN = 0.05
"""Posterior gap below which two robots are treated as indistinguishable."""

EXPLORE_MIN_OBS = 2
"""Observations below which an edge counts as unexplored.

Was 3, which made the ignorance rule unreachable. The leader's clamped weight
against an unobserved rival grows 0.0625 / 0.1417 / 0.2031 at one, two and three
observations, so by the time the leader cleared a minimum of 3 the gap had
already passed the 0.20 cap. The two conditions never held at once and
exploration fired 7 times in 128 decisions. At 2 the windows overlap."""

EXPLORE_MAX_GAP = 0.20
"""Posterior gap above which exploration stops firing entirely.

Without this bound the ignorance rule sends every question on a topic to the
least-observed robot until it clears EXPLORE_MIN_OBS — even when another robot
is demonstrably excellent at it. Across 14 topics and a fleet that is a lot of
deliberately poor answers in front of visitors. Exploration is for cases where
the graph genuinely cannot tell; a 0.37 gap is not one of those."""


def _specialists(edges: list, topic_id: str) -> set:
    """
    Robots whose declared project area covers `topic_id`.

    Configuration, not evidence — see RobotTopicEdge.specialised. Empty when
    nothing has been declared, which is what keeps the whole mechanism inert
    on a graph that has not opted in.
    """
    return {e.robot_id for e in edges
            if e.topic_id == topic_id and e.specialised}


def surviving_candidates(
    edges: list,
    topic_id: str,
    robot_ids: Iterable[str],
    absent: Optional[Iterable[str]] = None,
) -> list:
    """
    The robots a routing decision is actually choosing between, in order.

    Applies the three STRUCTURAL filters, all before competence is consulted:
    eligibility (may this robot answer at all), presence (can it right now),
    and declared scope (is this its subject).

    Extracted because two callers need the same answer for different
    reasons — route() needs the list, and KGRouter needs its LENGTH, since a
    field of one means the decision had nothing to choose between and a
    clean outcome there says nothing about quality (see
    kg_feedback.Segment.note_routed). Computing it twice would let them
    disagree.

    Scope is EXCLUSIVE where it is declared. If a topic has specialists,
    only they may answer it, and the field does not reopen when they are
    unavailable — an absent specialist produces a defer (see
    ABSENT_ROBOT_POLICY), which is the honest outcome, rather than a silent
    handover to a robot whose project it is not.

    It was a preference, and the reopening was the problem: it meant routing
    ALWAYS had an opinion, so the LLM delegation fallback could never run and
    sat in the system untested. Exclusivity gives the two mechanisms
    genuinely separate jobs — the graph answers what it was configured to
    answer, and anything outside every declared area is handed to delegation
    rather than resolved on noise.

    Returns [] when scope excludes everyone. That is "no candidate", which
    route() reports as no opinion, not an error.
    """
    excluded = _ineligible(edges, topic_id) | set(absent or ())
    candidates = [r for r in robot_ids if r not in excluded]

    specialists = _specialists(edges, topic_id)
    if specialists:
        return [r for r in candidates if r in specialists]
    return candidates


def _ineligible(edges: list, topic_id: str) -> set:
    """
    Robot ids explicitly excluded from `topic_id`.

    Opt-out, not opt-in: absence of an edge, or an edge nobody has marked
    ineligible, keeps a robot in the running. Only an edge that exists AND is
    explicitly eligible=False excludes it — a hard capability fact, never a
    weight of 0.0, which is a competence claim and a very different thing.
    """
    return {e.robot_id for e in edges if e.topic_id == topic_id and not e.eligible}


def route(
    edges: Iterable[RobotTopicEdge],
    links: Iterable[tuple],
    topic_id: str,
    robot_ids: Iterable[str],
    explore: bool = True,
    absent: Optional[Iterable[str]] = None,
) -> tuple:
    """
    Pick a robot for a question about `topic_id`. Returns (robot_id, reason).

    Ineligible robots are filtered out of the candidate list BEFORE ranking,
    not after — they never enter rank_robots/infer at all, so an excluded
    robot cannot win by score and cannot be picked by the exploration rules
    below either. A candidate list of fewer than two after filtering behaves
    exactly like any other case with nothing to choose between: "no robots"
    or a single-candidate argmax.

    `absent` — robots not currently in the conversation, from
    decision.presence — is filtered at the SAME point and for the same
    reason. A robot that has walked away cannot answer no matter how well
    the graph rates it, so letting it rank and then discarding the winner
    would mean the runner-up gets picked by an exploration rule written for
    a completely different situation. Absence is a fact about the room, not
    a competence claim, so it is never written back to the graph — see
    decision/presence.py.

    WHY NOT ARGMAX
    Routing to the best-scoring robot every time starves the graph. Whichever
    robot starts marginally ahead on a topic takes every question on it, the
    alternatives are never observed, and their edges sit at the prior forever.
    The only thing that could correct this is a supervisor intervening — which
    is exactly the expensive, sparse signal the graph exists to economise on. So
    the accumulated-supervision story would hold only for topics where a human
    happened to step in early, which is not a property you want to depend on.

    The exploration rule is deliberately crude, because a subtle one would be a
    second contribution to defend:

      * if the top two are within EXPLORE_MARGIN, prefer the LESS observed one —
        the graph cannot tell them apart, so spend the turn learning
      * if the runner-up has fewer than EXPLORE_MIN_OBS observations on this
        topic AND the gap is at most EXPLORE_MAX_GAP, prefer it — an unobserved
        robot is unknown, not bad, and the prior deliberately says so

    `explore` IS AN EXPLICIT PARAMETER AND MUST STAY ONE. It is set by the
    caller — the rollout harness turns it on, a live demo turns it off. Never
    infer it from state (whether a run looks like a rehearsal, whether the graph
    looks sparse): a flag that defaults wrong means deliberately routing a real
    visitor's question to a robot the graph knows is worse.

    Both are ties-and-ignorance rules only, and both are bounded by the gap. A
    robot the graph is CONFIDENT is worse is never chosen, so exploration never
    knowingly gives visitors a bad answer — it only spends turns the graph
    cannot already call.
    """
    edges = list(edges)
    links = list(links)
    robot_ids = surviving_candidates(edges, topic_id, robot_ids, absent)

    ranked = rank_robots(edges, links, topic_id, robot_ids)
    if not ranked:
        return None, "no robots"

    # A FIELD THE GRAPH CANNOT TELL APART. Ranking here can be ordering
    # robots on propagated fractions of a single distant observation: the
    # live graph separated the top two by 0.003 on all three orphan topics,
    # which is noise presented as a decision. When the candidates are that
    # close the honest answer is that the graph has no opinion, which hands
    # the turn to the receiver or to LLM delegation — the mechanism that
    # exists for exactly the questions declared scope does not cover.
    #
    # This used to exempt topics that HAVE specialists, on the reasoning that
    # declared scope had already decided. That holds only while every topic
    # has at most one declarer, which is true of the seeded vocabulary and is
    # not a property of the design: the moment two robots declare the same
    # subject — the natural way to give competence something to decide — the
    # field is two specialists at the identical prior, and rank_robots breaks
    # that tie lexicographically. Measured before this change: a contested
    # topic with zero observations returned ("navel_01", "argmax"). A robot
    # picked out of the alphabet, reported as an argmax, with no decline and
    # nothing in the log to say the graph had no idea. The exemption made an
    # untested path the default for anyone who widened the scope.
    #
    # The test is the SPREAD between candidates, not the distance from the
    # prior. A robot the graph is confident is BAD at something is still a
    # real opinion about ordering, and with one candidate left there is
    # nothing to be uncertain between — both of those must still route.
    # Only when NOT exploring. A near-tie is the exploration rules' whole
    # purpose — a campaign should spend that turn learning which robot is
    # better. Live, with exploration off, the same closeness means the graph
    # cannot tell them apart and should say so rather than pick one.
    if (not explore and len(ranked) > 1
            and ranked[0][1] - ranked[-1][1] < MIN_EVIDENCE_MARGIN):
        return None, "no confident opinion"
    if len(ranked) == 1 or not explore:
        return ranked[0][0], "argmax"

    obs = {e.robot_id: e.n_obs for e in edges if e.topic_id == topic_id}
    (top_id, top_p), (next_id, next_p) = ranked[0], ranked[1]

    if top_p - next_p < EXPLORE_MARGIN:
        if obs.get(next_id, 0) < obs.get(top_id, 0):
            return next_id, "explore: tied, runner-up less observed"

        # Everything tied AND equally observed — the cold-start case, and the
        # one that matters most. rank_robots breaks ties lexicographically, so
        # without this the alphabetically-first robot takes every question on
        # every topic, accumulates all the evidence, and the others are never
        # tried. Exploration exists precisely to stop that, and it was the case
        # it handled worst.
        #
        # Spread by TOTAL observations across all topics rather than per-topic:
        # per-topic counts are all zero here and carry no information, while the
        # global count says which robot the graph knows least about overall.
        if obs.get(next_id, 0) == obs.get(top_id, 0):
            total = {}
            for e in edges:
                total[e.robot_id] = total.get(e.robot_id, 0) + e.n_obs
            tied = [r for r, p in ranked if top_p - p < EXPLORE_MARGIN]
            least = min(tied, key=lambda r: (total.get(r, 0), r))
            if least != top_id:
                return least, "explore: cold tie, least observed overall"
        return top_id, "argmax: tied, already the least observed"

    if (top_p - next_p <= EXPLORE_MAX_GAP
            and obs.get(next_id, 0) < EXPLORE_MIN_OBS <= obs.get(top_id, 0)):
        return next_id, "explore: runner-up barely observed"

    return top_id, "argmax"
