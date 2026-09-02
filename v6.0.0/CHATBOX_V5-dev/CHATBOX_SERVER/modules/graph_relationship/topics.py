"""
Shared-TopicNode helpers: a single TopicNode is reached from both sides.

  robot  --knows-------------------------> Topic("jazz")
  person --has_interest--> Interest("music") --about--> Topic("jazz")

resolve_topic() gives every topic a deterministic id ("topic:" + slug), so the
robot's "jazz" and a human interest's "jazz" resolve to ONE node. Read helpers
traverse index-based (O(neighbours)); no full-graph scans.

Design contract: imports ONLY schema.py + store.py — no PAD, no kg_bridge.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from .schema import (
    AboutEdge, ConversationNode, HasConversationEdge, HasInterestEdge,
    InterestNode, Provenance, RelatedTopicEdge, TopicCategory, TopicNode,
)
from .store import GraphStore

# A matcher SELECTS the best capability item (e.g. 'knows jazz') that covers a
# topic label (e.g. 'jazz'), or None. keyword_matcher is the default; an
# embedding-based matcher can be injected without changing any call site.
Matcher = Callable[[List[str], str], Optional[str]]


def _prov(source: Optional[str], confidence: float = 1.0) -> Provenance:
    return Provenance(source=source or "topics",
                      confidence=max(0.0, min(1.0, float(confidence))),
                      timestamp=datetime.now(timezone.utc))


def normalize_label(label: str) -> str:
    """Stable slug: lowercase alnum, dashes for the rest."""
    return re.sub(r"[^a-z0-9]+", "-", str(label).strip().lower()).strip("-")


def topic_id(label: str) -> str:
    return f"topic:{normalize_label(label)}"


def interest_id(person_id: str, label: str) -> str:
    return f"interest:{person_id}:{normalize_label(label)}"


def add_person_interest(
    store: GraphStore, person_id: str, interest_label: str,
    topic_labels: Optional[List[str]] = None, *, summary: Optional[str] = None,
    source: Optional[str] = None, affinity: float = 0.5, confidence: float = 1.0,
) -> Optional[InterestNode]:
    """Upsert a person's Interest (deterministic id) with has_interest, and an
    about-edge to each shared Topic. Idempotent — re-adding does not duplicate.

    If `summary` is given, it is attached as a per-person note on each of the
    interest's Topic nodes. The person node must already exist. `affinity`
    (internal [0,1]) and `confidence` are written on each about edge (defaults =
    neutral / fully-trusted).
    """
    label = str(interest_label).strip()
    if not label:
        return None
    aff = max(0.0, min(1.0, float(affinity)))
    conf = max(0.0, min(1.0, float(confidence)))
    inode = InterestNode(id=interest_id(person_id, label), label=label)
    store.upsert_node(inode)
    store.upsert_edge(HasInterestEdge(
        source_id=person_id, target_id=inode.id, provenance=_prov(source, conf)))
    for t in topic_labels or []:
        t = str(t).strip()
        if not t:
            continue
        topic = resolve_topic(store, t)
        store.upsert_edge(AboutEdge(
            source_id=inode.id, target_id=topic.id, affinity=aff, confidence=conf,
            provenance=_prov(source, conf)))
        if summary:
            add_topic_note(store, topic, person_id, summary)
    return inode


def _cat_value(category) -> str:
    """Category as a plain string ('music'), tolerant of enum or str input."""
    if category is None:
        return "other"
    return category.value if hasattr(category, "value") else str(category)


def resolve_topic(store: GraphStore, label: str, category=None) -> TopicNode:
    """Get-or-create the shared TopicNode for `label` (deterministic id).

    The id is derived from the NORMALIZED LABEL only (topic_id) — `category` is an
    attribute, never part of identity — so re-resolving the same label returns the
    SAME node even if a later extraction disagrees on category, and existing topic
    notes are preserved.

    Category update rule (conservative, no embeddings/merge): only fill in a real
    category when the node is still the "other" fallback. If the node already has a
    non-"other" category, keep it (first non-other wins); a conflicting new category
    is ignored. TopicNode has no provenance field, so the conflict is not persisted.
    """
    tid = topic_id(label)
    existing = store.get_node(tid)
    if existing is not None and existing.node_type == "topic":
        new_cat = _cat_value(category)
        if new_cat != "other" and _cat_value(existing.category) == "other":
            # model_copy does NOT re-validate → coerce to the enum explicitly.
            existing = existing.model_copy(update={"category": TopicCategory(new_cat)})
            store.upsert_node(existing)
        return existing
    node = TopicNode(id=tid, label=str(label), category=_cat_value(category))
    store.upsert_node(node)
    return node


def add_topic_note(
    store: GraphStore, topic_ref, person_id: str, text: str,
) -> Optional[TopicNode]:
    """Append a per-person conversation summary to a Topic's `notes` list.

    Idempotent: an identical (person, text) note is not appended twice.
    `topic_ref` may be a TopicNode or a topic label.
    """
    text = str(text).strip()
    if not text:
        return None
    tid = topic_ref.id if hasattr(topic_ref, "id") else topic_id(topic_ref)
    node = store.get_node(tid)
    if node is None or node.node_type != "topic":
        return None
    notes = list(node.notes)
    if any(n.get("person") == person_id and n.get("text") == text for n in notes):
        return node
    notes.append({
        "person": person_id, "text": text,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    node = node.model_copy(update={"notes": notes})
    store.upsert_node(node)
    return node


def conversation_id(person_id: str, robot_id: str) -> str:
    return f"conversation:{person_id}:{robot_id}"


def update_conversation(
    store: GraphStore, person_id: str, robot_id: str, *,
    topic: Optional[str] = None, mood: Optional[float] = None,
    emotion: Optional[str] = None, max_topics: int = 3,
    create: bool = True, source: Optional[str] = None,
) -> Optional[ConversationNode]:
    """Update the live conversation-status node for a person↔robot IN PLACE.

    `topic`   — a keyword to push onto the rolling list (deduped, most-recent
                last, capped at `max_topics`). None leaves the list unchanged.
    `mood`    — current valence (-1..+1); `emotion` — current label.
    `create`  — when False, only update an existing node (never create one), so
                a plain mood tick before any conversation is a no-op.

    This is a dedicated node (NOT a shared TopicNode): it is never matched to
    capabilities/interests, so it cannot pick up spurious about-edges. The
    person and robot are linked to it once, via FAST has_conversation edges.
    """
    cid = conversation_id(person_id, robot_id)
    node = store.get_node(cid)
    if node is None or node.node_type != "conversation":
        if not create:
            return None
        node = ConversationNode(id=cid)
        store.apply_delta(
            nodes=[node],
            edges=[
                HasConversationEdge(source_id=person_id, target_id=cid,
                                    provenance=_prov(source)),
                HasConversationEdge(source_id=robot_id, target_id=cid,
                                    provenance=_prov(source)),
            ],
        )
        node = store.get_node(cid)

    update: dict = {}
    if topic:
        t = str(topic).strip()
        if t:
            topics = [x for x in node.topics if x.lower() != t.lower()]
            topics.append(t)
            update["topics"] = topics[-max_topics:]
    if mood is not None:
        update["mood"] = max(-1.0, min(1.0, float(mood)))
    if emotion is not None:
        update["emotion"] = emotion
    if update:
        node = node.model_copy(update=update)
        store.upsert_node(node)
    return node


def get_conversation(store: GraphStore, person_id: str, robot_id: str) -> Optional[ConversationNode]:
    node = store.get_node(conversation_id(person_id, robot_id))
    return node if node is not None and node.node_type == "conversation" else None


def topic_degree(store: GraphStore, topic_id_: str) -> int:
    """Number of edges incident to a topic node — its 'establishedness'. Used to
    pick the canonical node when consolidating duplicates. Pure O(neighbors)."""
    return len(store.query_neighbors(topic_id_))


def merge_topics(
    store: GraphStore, canonical_id: str, duplicate_id: str, *,
    source: Optional[str] = None,
) -> Optional[TopicNode]:
    """PURE graph surgery: fold `duplicate` topic into `canonical`.

    Redirects every edge touching the duplicate onto the canonical (preserving
    edge type / label / provenance; collisions merge under the store's normal
    rule), unions the two nodes' notes (deduped) plus a `merged_from` marker,
    upgrades the canonical category only if it was still 'other', then deletes the
    duplicate node. No embeddings, no LLM. Idempotent; `canonical==duplicate` is a
    no-op. Returns the canonical TopicNode, or None if either id is not a topic.
    """
    if canonical_id == duplicate_id:
        return store.get_node(canonical_id)
    canon = store.get_node(canonical_id)
    dup = store.get_node(duplicate_id)
    if (canon is None or canon.node_type != "topic"
            or dup is None or dup.node_type != "topic"):
        return None

    # Redirect every edge incident to the duplicate onto the canonical.
    for edge, _nbr in list(store.query_neighbors(duplicate_id)):
        new_src = canonical_id if edge.source_id == duplicate_id else edge.source_id
        new_dst = canonical_id if edge.target_id == duplicate_id else edge.target_id
        store.delete_edge(edge.source_id, edge.target_id, edge.edge_type)
        if new_src == new_dst:
            continue  # never create a self-loop on the canonical
        store.upsert_edge(edge.model_copy(update={
            "source_id": new_src, "target_id": new_dst}))

    # Union notes (dedup by person+text) + record the merge for provenance.
    notes = list(canon.notes)
    seen = {(n.get("person"), n.get("text")) for n in notes}
    for n in dup.notes:
        key = (n.get("person"), n.get("text"))
        if key not in seen:
            seen.add(key)
            notes.append(n)
    notes.append({"person": None, "text": f"merged_from: {dup.label}",
                  "ts": datetime.now(timezone.utc).isoformat()})
    update: dict = {"notes": notes}
    if _cat_value(canon.category) == "other" and _cat_value(dup.category) != "other":
        update["category"] = TopicCategory(_cat_value(dup.category))  # coerce to enum
    store.upsert_node(canon.model_copy(update=update))

    store.delete_node(duplicate_id)
    return store.get_node(canonical_id)


def link_related_topic(
    store: GraphStore, topic_a_id: str, topic_b_id: str, weight: float, *,
    source: Optional[str] = None,
) -> bool:
    """Add an undirected Topic↔Topic `related_topic` edge (stored once, endpoints
    sorted). Idempotent: returns False if it already exists or either id isn't a
    topic; True if a new edge was written. Pure — no embeddings/LLM."""
    if topic_a_id == topic_b_id:
        return False
    a, b = sorted((topic_a_id, topic_b_id))
    na, nb = store.get_node(a), store.get_node(b)
    if na is None or na.node_type != "topic" or nb is None or nb.node_type != "topic":
        return False
    if store.get_edge(a, b, "related_topic") is not None:
        return False
    store.upsert_edge(RelatedTopicEdge(
        source_id=a, target_id=b, weight=max(0.0, min(1.0, float(weight))),
        provenance=_prov(source)))
    return True


def link_related_cross(
    store: GraphStore, node_a_id: str, node_b_id: str, weight: float, *,
    source: Optional[str] = None,
) -> bool:
    """Undirected `related_topic` edge that MAY span namespaces — a person
    `topic:<slug>` and a culture `ck:<culture>:<slug>` CultureTopic node.

    Same edge type / storage as link_related_topic (endpoints sorted, weight is the
    similarity, idempotent), but each endpoint may be a `topic` OR a `culture_topic`
    node. This is a relatedness LINK only: it NEVER merges or changes node identity,
    so a `ck:` node and a `topic:` node with the same slug stay two DISTINCT nodes,
    now traversable by the preference BN's existing propagation. Pure — no
    embeddings/LLM. Returns True iff a new edge was written."""
    _CROSS = ("topic", "culture_topic")
    if node_a_id == node_b_id:
        return False
    a, b = sorted((node_a_id, node_b_id))
    na, nb = store.get_node(a), store.get_node(b)
    if (na is None or na.node_type not in _CROSS
            or nb is None or nb.node_type not in _CROSS):
        return False
    if store.get_edge(a, b, "related_topic") is not None:
        return False
    store.upsert_edge(RelatedTopicEdge(
        source_id=a, target_id=b, weight=max(0.0, min(1.0, float(weight))),
        provenance=_prov(source)))
    return True


def merge_interests(
    store: GraphStore, canonical_id: str, duplicate_id: str, *,
    source: Optional[str] = None,
) -> Optional[InterestNode]:
    """PURE graph surgery: fold a duplicate Interest into the canonical one.

    Redirects every edge touching the duplicate (person --has_interest--> and
    --about--> Topic) onto the canonical, then deletes the duplicate. Used to
    merge near-duplicate interests (e.g. "sports" vs "sport" from old LLM labels
    vs the new category-named interests). No embeddings. Idempotent.
    """
    if canonical_id == duplicate_id:
        return store.get_node(canonical_id)
    canon = store.get_node(canonical_id)
    dup = store.get_node(duplicate_id)
    if (canon is None or canon.node_type != "interest"
            or dup is None or dup.node_type != "interest"):
        return None
    for edge, _nbr in list(store.query_neighbors(duplicate_id)):
        new_src = canonical_id if edge.source_id == duplicate_id else edge.source_id
        new_dst = canonical_id if edge.target_id == duplicate_id else edge.target_id
        store.delete_edge(edge.source_id, edge.target_id, edge.edge_type)
        if new_src == new_dst:
            continue
        store.upsert_edge(edge.model_copy(update={
            "source_id": new_src, "target_id": new_dst}))
    store.delete_node(duplicate_id)
    return store.get_node(canonical_id)


def _neighbors_of_type(store: GraphStore, node_id: str, edge_type: str, node_type: str):
    return [
        n for _e, n in store.query_neighbors(node_id, edge_type)
        if n.node_type == node_type
    ]


def person_interests(store: GraphStore, person_id: str) -> List[Tuple[InterestNode, List[TopicNode]]]:
    """[(InterestNode, [TopicNode it is about]), ...] for one person (index-based)."""
    out: List[Tuple[InterestNode, List[TopicNode]]] = []
    for interest in _neighbors_of_type(store, person_id, "has_interest", "interest"):
        topics = _neighbors_of_type(store, interest.id, "about", "topic")
        out.append((interest, topics))
    return out


def person_topic_affinity(
    store: GraphStore, person_id: str,
) -> List[Tuple[TopicNode, float, float]]:
    """[(TopicNode, affinity, confidence), ...] for one person — the "observed"
    evidence read by BOTH the preference BN (affinity → signed clamp) and the
    person-memory prompt (affinity → like/neutral/dislike word, confidence →
    hedge). Reads affinity/confidence straight off the person Interest --about-->
    Topic edge; edges written before this feature report the schema defaults
    (0.5 neutral / 1.0 confident). Pure O(neighbours) read."""
    out: List[Tuple[TopicNode, float, float]] = []
    for interest in _neighbors_of_type(store, person_id, "has_interest", "interest"):
        for edge, topic in store.query_neighbors(interest.id, "about"):
            if topic.node_type != "topic":
                continue
            aff = float(getattr(edge, "affinity", 0.5))
            conf = float(getattr(edge, "confidence", 1.0))
            out.append((topic, aff, conf))
    return out


def person_topics(store: GraphStore, person_id: str) -> List[Tuple[str, str]]:
    """[(topic_label, category), ...] for one person — the distinct topics they
    reach via any interest. Pure store read (no LLM); used to condition the
    graph-aware extraction prompt so the LLM reuses established topics."""
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for _interest, topics in person_interests(store, person_id):
        for t in topics:
            if t.id in seen:
                continue
            seen.add(t.id)
            out.append((t.label, _cat_value(t.category)))
    return out


def add_person_topic(
    store: GraphStore, person_id: str, label: str, category=None, *,
    source: Optional[str] = None, confidence: float = 1.0,
    affinity: float = 0.5, summary: Optional[str] = None,
) -> Optional[TopicNode]:
    """Wire a NEW typed topic into the person's Interest layer:
        person --has_interest--> Interest(category) --about--> Topic(label, category)
    The interest node is the topic's category (e.g. 'music'), so typed topics group
    under their category. Idempotent (deterministic ids + upsert). Optional `summary`
    is attached as a per-person topic note.

    `affinity` (internal [0,1]: 0 dislike / 0.5 neutral / 1 like) and `confidence`
    are written onto the about edge — the "observed" evidence the BN and the memory
    prompt read. Re-adding OVERWRITES both with the new reading (EWMA blending of
    successive readings is future work)."""
    label = str(label).strip()
    if not label:
        return None
    conf = max(0.0, min(1.0, float(confidence)))
    aff = max(0.0, min(1.0, float(affinity)))
    topic = resolve_topic(store, label, category=category)
    interest_label = _cat_value(category)
    inode = InterestNode(id=interest_id(person_id, interest_label), label=interest_label)
    store.upsert_node(inode)
    store.upsert_edge(HasInterestEdge(
        source_id=person_id, target_id=inode.id, provenance=_prov(source, conf)))
    store.upsert_edge(AboutEdge(
        source_id=inode.id, target_id=topic.id, affinity=aff, confidence=conf,
        provenance=_prov(source, conf)))
    if summary:
        add_topic_note(store, topic, person_id, summary)
    return topic


def reinforce_person_topic(
    store: GraphStore, person_id: str, label: str, *,
    source: Optional[str] = None, confidence: float = 1.0,
    affinity: float = 0.5, summary: Optional[str] = None,
) -> Optional[TopicNode]:
    """Refresh an EXISTING person→interest→topic path (re-stamp provenance) without
    creating any new node. Returns the topic if a path was found, else None. Used
    for topics the LLM reports as already-known (existing_topics_discussed).

    Overwrites the about edge's `affinity`/`confidence` with the new reading (simple
    overwrite for now; EWMA blending is future work)."""
    tid = topic_id(label)
    topic = store.get_node(tid)
    if topic is None or topic.node_type != "topic":
        return None
    conf = max(0.0, min(1.0, float(confidence)))
    aff = max(0.0, min(1.0, float(affinity)))
    refreshed = False
    for interest, topics in person_interests(store, person_id):
        if any(t.id == tid for t in topics):
            store.upsert_edge(HasInterestEdge(
                source_id=person_id, target_id=interest.id, provenance=_prov(source, conf)))
            store.upsert_edge(AboutEdge(
                source_id=interest.id, target_id=tid, affinity=aff, confidence=conf,
                provenance=_prov(source, conf)))
            refreshed = True
    if summary:
        add_topic_note(store, topic, person_id, summary)
    return topic if refreshed else None


def _person_topic_ids(store: GraphStore, person_id: str) -> set:
    """Topic ids the person reaches via any interest -> about -> topic path."""
    ids = set()
    for _interest, topics in person_interests(store, person_id):
        ids.update(t.id for t in topics)
    return ids


def robot_capability(store: GraphStore, robot_id: str):
    """The robot's CapabilityNode (holds the items list), or None."""
    caps = _neighbors_of_type(store, robot_id, "has_capability", "capability")
    return caps[0] if caps else None


def robot_topics(store: GraphStore, robot_id: str) -> List[TopicNode]:
    """Topics the robot reaches via has_capability -> capability -> about -> topic."""
    cap = robot_capability(store, robot_id)
    if cap is None:
        return []
    seen: dict = {t.id: t for t in _neighbors_of_type(store, cap.id, "about", "topic")}
    return list(seen.values())


# --- capability ↔ topic matching + linking ---------------------------------

def keyword_match(item: str, topic_label: str) -> bool:
    """True if capability item and topic share a word, or one contains the other
    (normalized). e.g. 'good at math' ~ 'math', 'knows jazz' ~ 'jazz'."""
    a, b = normalize_label(item), normalize_label(topic_label)
    if not a or not b:
        return False
    if a == b:
        return True
    if set(a.split("-")) & set(b.split("-")):
        return True
    return b in a or a in b


def keyword_matcher(items: List[str], topic_label: str) -> Optional[str]:
    """Default selector: the first capability item that keyword-matches `topic`."""
    for item in items:
        if keyword_match(item, topic_label):
            return item
    return None


def link_capability_to_topic(
    store: GraphStore, robot_id: str, topic, *,
    matcher: Optional[Matcher] = None, source: Optional[str] = None,
) -> Optional[str]:
    """If the matcher selects a capability item for `topic`, add a labeled
    about-edge Capability --about[label=<item>]--> Topic. Returns the item or None.

    Idempotent: if an about-edge already exists it is left as-is (not relabeled).
    `topic` may be a TopicNode or a topic label.
    """
    matcher = matcher or keyword_matcher
    cap = robot_capability(store, robot_id)
    if cap is None:
        return None
    tid = topic.id if hasattr(topic, "id") else topic_id(topic)
    tnode = store.get_node(tid)
    if tnode is None or tnode.node_type != "topic":
        return None
    if store.get_edge(cap.id, tid, "about") is not None:
        return None
    item = matcher(cap.items, tnode.label)
    if item is not None:
        store.upsert_edge(AboutEdge(
            source_id=cap.id, target_id=tid, label=item, provenance=_prov(source)))
        return item
    return None


def relink_capability_topics(
    store: GraphStore, robot_id: str, *, matcher: Optional[Matcher] = None,
) -> List[Tuple[str, str]]:
    """Re-run capability→topic matching over ALL topic nodes. Useful after
    seeding or after swapping the matcher (e.g. to embeddings). Returns the
    (item, topic_label) pairs newly linked."""
    nodes = getattr(store, "_nodes", {}) or {}
    linked: List[Tuple[str, str]] = []
    for node in list(nodes.values()):
        if node.node_type != "topic":
            continue
        item = link_capability_to_topic(store, robot_id, node, matcher=matcher)
        if item:
            linked.append((item, node.label))
    return linked


def shared_topics(store: GraphStore, person_id: str, robot_id: str) -> List[str]:
    """Labels of topics BOTH sides reach — the robot via its capability and the
    person via an interest — i.e. the shared TopicNodes.

    Traverses robot -> capability -> about -> topic intersected with
    person -> interest -> about -> topic. Index-based; O(neighbours), no scan.
    """
    robot_by_id = {t.id: t for t in robot_topics(store, robot_id)}
    person_ids = _person_topic_ids(store, person_id)
    shared_ids = person_ids & set(robot_by_id)
    return sorted(robot_by_id[tid].label for tid in shared_ids)


def topic_related(store: GraphStore, topic_id_: str) -> List[TopicNode]:
    """TopicNodes linked to this one by a `related_topic` edge (one hop)."""
    return _neighbors_of_type(store, topic_id_, "related_topic", "topic")


def _person_topics_by_id(store: GraphStore, person_id: str) -> dict:
    out: dict = {}
    for _interest, topics in person_interests(store, person_id):
        for t in topics:
            out[t.id] = t.label
    return out


def related_common_ground(store: GraphStore, person_id: str, robot_id: str) -> dict:
    """Common ground including RELATED bridges (Feature-2c).

    Returns {"direct": [labels both sides reach], "bridges": [(person_topic_label,
    robot_topic_label), ...]} where a bridge is a person topic that is
    `related_topic`-linked to a robot capability topic (indirect common ground,
    e.g. person 'multiplication' ~ robot 'math problems')."""
    robot_by_id = {t.id: t.label for t in robot_topics(store, robot_id)}
    person_by_id = _person_topics_by_id(store, person_id)
    direct_ids = set(person_by_id) & set(robot_by_id)
    direct = sorted(robot_by_id[i] for i in direct_ids)
    bridges: List[Tuple[str, str]] = []
    seen: set = set()
    for pid, plabel in person_by_id.items():
        if pid in robot_by_id:
            continue                      # already direct common ground
        for r in topic_related(store, pid):
            if r.id in robot_by_id and (pid, r.id) not in seen:
                seen.add((pid, r.id))
                bridges.append((plabel, robot_by_id[r.id]))
    return {"direct": direct, "bridges": bridges}


def person_related_pairs(store: GraphStore, person_id: str) -> List[Tuple[str, str]]:
    """[(label_a, label_b), ...] related-topic pairs among the person's own topics —
    so the robot knows which of their interests connect (rap ~ hiphop)."""
    person_by_id = _person_topics_by_id(store, person_id)
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for pid, plabel in person_by_id.items():
        for r in topic_related(store, pid):
            if r.id in person_by_id:
                key = tuple(sorted((pid, r.id)))
                if key not in seen:
                    seen.add(key)
                    out.append((plabel, person_by_id[r.id]))
    return out
