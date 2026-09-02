"""
Culture layer helpers — the ROBOT's prior cultural knowledge.

  robot  --knows_culture-------> Culture("Korean")
  Culture --culture_prior[0.8]-> CultureTopic("kimchi")      (robot-owned knowledge)
  person --belongs_to_culture--> Culture("Korean")           (manual background tag)

Culture topics are their OWN nodes (id `ck:<culture>:<topic>`), deliberately SEPARATE
from the shared person-interest TopicNodes. A person is only *tagged* with a culture;
the culture layer never writes person→topic edges, so tagging one person Korean never
couples another person who happens to share a topic. A person connects to a real topic
only by actually discussing it (extraction, over time).

Design contract: imports ONLY schema.py + store.py (+ the pure topics slug) — no LLM,
no PAD, no embeddings, nothing in modules/. Stateless functions only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from .schema import (
    BelongsToCultureEdge, CultureNode, CulturePriorEdge, CultureTopicNode,
    KnowsCultureEdge, Provenance, TopicCategory,
)
from .store import GraphStore
from .topics import normalize_label


def _prov(source: Optional[str], confidence: float = 1.0) -> Provenance:
    return Provenance(source=source or "cultures",
                      confidence=max(0.0, min(1.0, float(confidence))),
                      timestamp=datetime.now(timezone.utc))


def _cat_value(category) -> str:
    if category is None:
        return "other"
    return category.value if hasattr(category, "value") else str(category)


# ── ids ───────────────────────────────────────────────────────────────────────

def culture_id(label: str) -> str:
    """Deterministic id `culture:<slug>` (same normalization as topic_id)."""
    return f"culture:{normalize_label(label)}"


def culture_topic_id(culture_id: str, label: str) -> str:
    """Deterministic id `ck:<culture-slug>:<topic-slug>` — namespaced under the
    culture so it never collides with a shared person `topic:<slug>` node."""
    cslug = culture_id.split(":", 1)[1] if ":" in culture_id else culture_id
    return f"ck:{cslug}:{normalize_label(label)}"


# ── nodes / edges ─────────────────────────────────────────────────────────────

def ensure_culture(store: GraphStore, label: str) -> CultureNode:
    """Get-or-create the CultureNode for `label` (deterministic id). Idempotent."""
    cid = culture_id(label)
    existing = store.get_node(cid)
    if existing is not None and existing.node_type == "culture":
        return existing
    node = CultureNode(id=cid, label=str(label).strip())
    store.upsert_node(node)
    return node


def ensure_culture_topic(store: GraphStore, culture_id: str, label: str,
                         category=None, facts=None) -> CultureTopicNode:
    """Get-or-create a CultureTopicNode under `culture_id` (deterministic id).
    Idempotent; fills `category` only when still 'other' (first non-other wins) and
    replaces `facts` when a non-empty list is provided (re-seeding refreshes them)."""
    tid = culture_topic_id(culture_id, label)
    facts = [str(f).strip() for f in (facts or []) if str(f).strip()]
    existing = store.get_node(tid)
    if existing is not None and existing.node_type == "culture_topic":
        upd: dict = {}
        new_cat = _cat_value(category)
        if new_cat != "other" and _cat_value(existing.category) == "other":
            upd["category"] = TopicCategory(new_cat)
        if facts:
            upd["facts"] = facts
        if upd:
            existing = existing.model_copy(update=upd)
            store.upsert_node(existing)
        return existing
    node = CultureTopicNode(id=tid, label=str(label).strip(),
                            category=_cat_value(category), facts=facts)
    store.upsert_node(node)
    return node


def knows_culture(store: GraphStore, robot_id: str, culture_id: str,
                  *, source: Optional[str] = None) -> None:
    """Link a robot to a culture it holds as prior knowledge (knows_culture).
    Idempotent — one edge per robot-culture pair."""
    store.upsert_edge(KnowsCultureEdge(
        source_id=robot_id, target_id=culture_id, provenance=_prov(source)))


def assign_culture(store: GraphStore, person_id: str, culture_id: str,
                   *, source: Optional[str] = None) -> None:
    """Tag a person with a culture (belongs_to_culture). Idempotent — one edge per
    person-culture pair. Does NOT link the person to any culture topics."""
    store.upsert_edge(BelongsToCultureEdge(
        source_id=person_id, target_id=culture_id, provenance=_prov(source)))


def set_culture_prior(store: GraphStore, culture_id: str, culture_topic_id: str,
                      prior: float, *, source: Optional[str] = None) -> None:
    """Upsert a culture→culture_topic prior, clamped to [0,1]. Re-setting replaces
    the stored value (one edge per culture-topic pair)."""
    p = max(0.0, min(1.0, float(prior)))
    store.upsert_edge(CulturePriorEdge(
        source_id=culture_id, target_id=culture_topic_id, prior=p,
        provenance=_prov(source)))


# ── reads ─────────────────────────────────────────────────────────────────────

def culture_priors(store: GraphStore, culture_id: str) -> List[Tuple[str, str, float]]:
    """[(culture_topic_id, label, prior), ...] for one culture, sorted by prior
    DESC (stable lexicographic tie-break on id). Index-based read."""
    out: List[Tuple[str, str, float]] = []
    for edge, neighbor in store.query_neighbors(culture_id, "culture_prior"):
        if neighbor.node_type == "culture_topic":
            out.append((neighbor.id, neighbor.label, float(edge.prior)))
    out.sort(key=lambda x: (-x[2], x[0]))
    return out


def person_culture(store: GraphStore, person_id: str) -> Optional[str]:
    """The culture_id this person is tagged with, or None. If more than one is
    (unexpectedly) assigned, returns the lexicographically-first for determinism."""
    cids = [n.id for _e, n in store.query_neighbors(person_id, "belongs_to_culture")
            if n.node_type == "culture"]
    return sorted(cids)[0] if cids else None


def person_culture_source(store: GraphStore, person_id: str) -> Optional[str]:
    """Provenance `source` of the person's belongs_to_culture edge, or None if
    untagged. Lets a consumer distinguish a background the person STATED themselves
    (source starts 'self-declared:') from one that was manually/seed-assigned — the
    former is a recallable fact, the latter only a tentative hint."""
    cid = person_culture(store, person_id)
    if cid is None:
        return None
    edge = store.get_edge(person_id, cid, "belongs_to_culture")
    return edge.provenance.source if edge is not None else None


def person_culture_self_declared(store: GraphStore, person_id: str) -> bool:
    """True iff the person's culture tag came from their OWN explicit statement
    (not a manual/seed assignment or any inference)."""
    src = person_culture_source(store, person_id)
    return bool(src) and src.startswith("self-declared")


def person_culture_style_hint(store: GraphStore, person_id: str) -> str:
    """The STATIC manner/'how to talk' hint of the culture this person is tagged with,
    or "" if untagged / the culture has no hint. Same string for every interaction —
    no tier/affect/situation variation (that is Approach 2)."""
    cid = person_culture(store, person_id)
    if cid is None:
        return ""
    node = store.get_node(cid)
    return getattr(node, "style_hint", "") or "" if node is not None else ""


def culture_knowers(store: GraphStore, culture_id: str) -> List[str]:
    """Robot ids that hold this culture as prior knowledge (knows_culture)."""
    return sorted(n.id for _e, n in store.query_neighbors(culture_id, "knows_culture")
                  if n.node_type == "robot")
