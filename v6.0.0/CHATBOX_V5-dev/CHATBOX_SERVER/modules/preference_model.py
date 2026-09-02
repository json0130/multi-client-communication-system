"""
Bayesian preference overlay (Command B) — compiled ON READ, NEVER persisted.

`rank_suggestions(store, person_id)` ranks topics the robot could tentatively bring
up with a person, blending:
  * the person's CULTURE priors  (base rates — the robot's prior knowledge)
  * OBSERVED interests            (topics they actually engage with → strong evidence)
  * one-hop `related_topic` links (semantic neighbours → soft propagation)

This mirrors the computed-on-read `_tier_from_edges` pattern:
  * READ-ONLY over the store — zero writes, zero node/edge creation.
  * No live embeddings — consumes only the STORED `related_topic` edge weights.
  * No LLM, no PAD. Does not modify graph_relationship/ (imports its pure reads only).

Namespace join (post-redesign): culture priors live on `culture_topic` nodes
(`ck:<culture>:<slug>`) which are DISTINCT from a person's shared `topic:<slug>`
nodes. So the model works in a unified SLUG space — a culture prior for "kimchi"
and a person topic "kimchi" are the same concept, joined by normalized label.

Cross-namespace bridges (Step 2): a bare label-join is not traversable — a person's
OBSERVED interest could never propagate to a culturally-adjacent culture topic, so
the BN degraded to a base-rate lookup. `related_topic` BRIDGE edges (built by
kg_extraction.link_cross_namespace_bridges) now connect `topic:` ↔ `ck:` nodes, and
this read simply lets the SAME 2-round noisy-OR follow them: when gathering a
person topic's `related_topic` neighbours we now accept `culture_topic` neighbours
too (they resolve into the same slug space). No change to rounds, damping, floor, or
the signed clamp — only which neighbours the existing walk is allowed to visit.

Signed evidence (Approach-1 Step 1): an observed topic is clamped at its stored
`affinity` (internal [0,1]: 0 dislike / 0.5 neutral / 1 like) rather than a flat
positive constant. A LIKED topic clamps HIGH and pushes neighbours up (noisy-OR, as
before); a DISLIKED topic clamps LOW and pulls related neighbours DOWN (symmetric).
Observed topics — liked OR disliked — are still excluded from returned suggestions.

Confidence-weighted clamp (Approach-1 Step 3): the clamp is now
`0.5 + (affinity - 0.5) * confidence`, so an UNCERTAIN observation is pulled toward
neutral and moves the posterior less — a shaky signal can't stomp a strong culture
prior, and the BN agrees with the prompt's Step-1 hedge word. confidence 1.0 →
clamp == affinity (identical to Step 2). This only changes the clamp VALUE; rounds
(2), damping (0.8), floor, bridge traversal, and the candidate set are untouched.
"""

from __future__ import annotations

from typing import List, Tuple

from modules.graph_relationship.cultures import person_culture, culture_priors
from modules.graph_relationship.topics import (
    person_topic_affinity, topic_related, normalize_label,
)

# Inference constants (deliberately simple — no external BN library).
_DEFAULT_PRIOR = 0.30   # unobserved concept with no culture prior
_NEUTRAL       = 0.50   # affinity midpoint — below this an observation pulls DOWN
_DAMPING       = 0.80   # noisy-OR edge damping w
_ROUNDS        = 2      # propagation rounds over related_topic edges


def clamp_from(affinity: float, confidence: float) -> float:
    """Confidence-weighted observed-topic clamp (Approach-1 Step 3):

        clamp = 0.5 + (affinity - 0.5) * confidence

    Pull the affinity toward neutral (0.5) by how UNSURE the reading is:
      * confidence 1.0 → clamp == affinity (identical to Step 2);
      * confidence 0   → clamp == 0.5 (the observation barely moves the posterior);
      * neutral (0.5) is a fixed point at any confidence.
    Symmetric — a confident dislike still pins low; an unsure one sits nearer 0.5."""
    return _NEUTRAL + (affinity - _NEUTRAL) * confidence


_USE_PERSON_CULTURE = object()   # sentinel: default = derive culture from the person


def rank_suggestions(
    store, person_id: str, k: int = 3, floor: float = 0.35,
    *, culture_id=_USE_PERSON_CULTURE,
) -> List[Tuple[str, float]]:
    """Top-k UNOBSERVED topics to tentatively suggest, as [(node_id, posterior)].

    Deterministic. Degrades gracefully to prior-only ranking when there are no
    `related_topic` links. Returns [] when there is nothing to suggest.

    `culture_id` selects which culture's priors to use: default → the person's own
    `belongs_to_culture`; pass an explicit id to use that culture instead (the
    active-culture override), or None to use NO culture priors (generic).
    """
    # ── base rates: culture priors, keyed by slug ─────────────────────────────
    prior_by_slug: dict = {}
    ck_id_by_slug: dict = {}
    cid = person_culture(store, person_id) if culture_id is _USE_PERSON_CULTURE else culture_id
    if cid:
        for ck_id, label, prior in culture_priors(store, cid):
            s = normalize_label(label)
            prior_by_slug[s] = prior
            ck_id_by_slug.setdefault(s, ck_id)

    # ── observed evidence: the person's own interest topics (clamped) ─────────
    # Step 3 — CONFIDENCE-WEIGHTED clamp:  clamp = 0.5 + (affinity - 0.5) * confidence
    # The 0.6 extraction gate is binary (once past it, 0.61 and 0.99 look alike), so
    # a shaky "jazz is okay I guess" (affinity 1.0, conf 0.62) used to clamp exactly
    # like a rock-solid "I love jazz" (conf 0.95). Folding confidence in pulls an
    # uncertain observation toward neutral (0.5), shrinking how far it moves the
    # posterior — so a weak signal can't stomp a strong culture prior, and the BN's
    # number now agrees with the prompt's Step-1 hedge word. Calibration, not a new
    # model: confidence 1.0 → clamp == affinity (identical to Step 2). Signed &
    # symmetric: a confident dislike still pins low; an unsure one sits nearer 0.5.
    observed_aff: dict = {}          # slug -> confidence-weighted affinity clamp
    topic_id_by_slug: dict = {}
    for topic, aff, conf in person_topic_affinity(store, person_id):
        s = normalize_label(topic.label)
        observed_aff[s] = clamp_from(aff, conf)   # last-wins on duplicate slugs
        topic_id_by_slug.setdefault(s, topic.id)
    observed: set = set(observed_aff)

    # ── candidate expansion: one-hop related neighbours of the person's topics ─
    for s, tid in list(topic_id_by_slug.items()):
        for rt in topic_related(store, tid):
            topic_id_by_slug.setdefault(normalize_label(rt.label), rt.id)

    concepts: set = set(prior_by_slug) | set(topic_id_by_slug)
    if not concepts:
        return []

    # ── initialise posteriors ─────────────────────────────────────────────────
    # Observed → its signed affinity clamp; unobserved → culture prior (or default).
    p: dict = {
        s: (observed_aff[s] if s in observed else prior_by_slug.get(s, _DEFAULT_PRIOR))
        for s in concepts
    }

    # ── gather related_topic edges (with STORED weights) in slug space ────────
    slug_by_id = {tid: s for s, tid in topic_id_by_slug.items()}
    edges: List[Tuple[str, str, float]] = []
    seen: set = set()
    for s, tid in list(topic_id_by_slug.items()):
        for edge, nb in store.query_neighbors(tid, "related_topic"):
            # Accept culture_topic neighbours too — this is how a person interest's
            # evidence crosses a Step-2 bridge into a `ck:` culture topic. Both node
            # types resolve into the shared slug space below.
            if nb.node_type not in ("topic", "culture_topic"):
                continue
            bs = slug_by_id.get(nb.id) or normalize_label(nb.label)
            if bs not in p:                    # neighbour-of-neighbour: add candidate
                p[bs] = prior_by_slug.get(bs, _DEFAULT_PRIOR)
                concepts.add(bs)
                topic_id_by_slug.setdefault(bs, nb.id)
                slug_by_id.setdefault(nb.id, bs)
            key = tuple(sorted((s, bs)))
            if s == bs or key in seen:
                continue
            seen.add(key)
            edges.append((s, bs, float(edge.weight)))

    # ── signed propagation (observed stay clamped; unobserved may move) ───────
    # Each edge (a,b,w) lets each endpoint influence the OTHER (if unobserved):
    #   * a HIGH neighbour raises b via noisy-OR  → p[b] = max(p[b], p[a]·0.8·w)
    #   * a LOW  neighbour (p[a] < p[b]) pulls b DOWN toward p[a], scaled by 0.8·w:
    #       p[b] += (0.8·w)·(p[a] − p[b])   (a move partway toward the low clamp)
    # This is symmetric to the upward push, so a disliked observed topic drags its
    # related neighbours down while a liked one lifts them up. Observed nodes never
    # move (they are the clamps).
    def _influence(src_slug: str, dst_slug: str, w: float) -> None:
        if dst_slug in observed:
            return
        w_eff = _DAMPING * w
        # upward noisy-OR floor
        p[dst_slug] = max(p[dst_slug], p[src_slug] * w_eff)
        # downward pull: only a genuine DISLIKE (below neutral) that is lower than
        # the neighbour drags it toward its low clamp, scaled by 0.8·w.
        if p[src_slug] < _NEUTRAL and p[src_slug] < p[dst_slug]:
            p[dst_slug] += w_eff * (p[src_slug] - p[dst_slug])

    for _ in range(_ROUNDS):
        for a, b, w in edges:
            _influence(a, b, w)
            _influence(b, a, w)

    # ── rank UNOBSERVED concepts with posterior ≥ floor ───────────────────────
    ranked: List[Tuple[str, float, float]] = []
    for s in concepts:
        if s in observed:
            continue
        post = p[s]
        if post + 1e-12 < floor:
            continue
        node_id = topic_id_by_slug.get(s) or ck_id_by_slug.get(s)
        if node_id is None:
            continue
        ranked.append((node_id, post, prior_by_slug.get(s, _DEFAULT_PRIOR)))
    # posterior desc, then higher prior, then lexicographic id
    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return [(node_id, round(post, 4)) for node_id, post, _prior in ranked[:k]]
