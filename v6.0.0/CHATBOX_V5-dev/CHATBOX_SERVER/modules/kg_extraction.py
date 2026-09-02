"""
App-layer graph-aware topic extraction (Feature: fine-grained typing + reuse).

Lives OUTSIDE `graph_relationship/` on purpose: all LLM prompt/parse/guard logic
stays in the APP layer. It is handed an `llm_fn(system, user) -> str` (the harness
`LLMClient.respond`) and imports ONLY pure schema/store helpers from
`graph_relationship`. `graph_relationship/` never imports this module, so it stays
copy-pasteable with zero LLM/PAD dependencies.

Flow per session:
  1. read the person's EXISTING topics (label, category) from the graph (pure read)
  2. prompt the LLM, conditioned on that list, for two buckets:
        existing_topics_discussed | new_topics     (+ optional per-topic summary)
  3. deterministic guards (JSON parse, enum, hallucination, confidence, normalize)
  4. apply via pure store helpers (reinforce existing path / add new typed topic)

Nothing is written on a JSON parse failure (never partially apply). Per-item
failures drop only that item. Re-running on an already-extracted session is a
no-op (deterministic ids + upsert; caller marks the session extracted).
"""
from __future__ import annotations

import json
from typing import Callable, List, Optional, Tuple

import math

from modules.graph_relationship.schema import TOPIC_CATEGORIES
from modules.graph_relationship.scales import aff01_from_10
from modules.graph_relationship.topics import (
    add_person_topic,
    link_related_cross,
    link_related_topic,
    merge_topics,
    normalize_label,
    person_topics,
    reinforce_person_topic,
    topic_degree,
    topic_id,
)
from modules.graph_relationship.extraction import format_transcript  # pure transcript renderer

LLMFn = Callable[[str, str], str]

# Drop any extracted item below this confidence (constant — easy to tune).
CONFIDENCE_MIN = 0.6
# Weight for an LLM-asserted specific↔broader relation (e.g. twice ~ kpop). The
# embedding matcher can't see this (proper noun vs genre), so world knowledge from
# the LLM supplies it. Sits in the related band, feeds Command B propagation.
RELATION_WEIGHT = 0.7


def build_system_prompt(existing: List[Tuple[str, str]]) -> str:
    """Graph-aware extraction prompt, conditioned on the person's known topics."""
    cats = ", ".join(sorted(TOPIC_CATEGORIES))
    listing = "\n".join(f'  - "{label}" [{cat}]' for label, cat in existing) or "  (none yet)"
    return (
        "You extract the TOPICS the CHILD talked about, from a conversation between "
        "a CHILD and a companion ROBOT. Output ONLY one JSON object — no prose, no "
        "code fences.\n\n"
        "The child ALREADY has these known topics (reuse them; do NOT invent "
        "near-duplicates):\n"
        f"{listing}\n\n"
        "Return exactly this JSON:\n"
        '{"existing_topics_discussed": ['
        '{"label": "<COPY VERBATIM from the list above>", '
        '"sentiment": <0-10>, '
        '"confidence": <0.0-1.0>, "summary": "<optional one short sentence>"}], '
        '"new_topics": ['
        '{"label": "<short canonical noun phrase, lowercase>", '
        f'"category": "<one of: {cats}>", '
        '"sentiment": <0-10>, '
        '"confidence": <0.0-1.0>, "summary": "<optional one short sentence>"}], '
        '"relations": [{"a": "<specific topic label>", "b": "<broader topic label>"}]}\n\n'
        "Rules:\n"
        "- For each topic, rate the person's sentiment 0-10 (0 = clearly dislikes, "
        "5 = neutral or unclear, 10 = clearly likes). Use 5 when it isn't stated. "
        "Use the FULL range — anchors: \"hate it\"/\"can't stand it\" → 0-1; "
        "\"don't really like it\"/\"not a fan\" → 2-3; \"it's okay\"/no opinion → 5; "
        "\"used to like it\"/\"liked it\"/\"pretty good\" → 7-8; \"love it\"/"
        "\"my favourite\" → 9-10.\n"
        "- Prefer reusing an existing label over a near-duplicate: if \"jazz\" is "
        "known and the child says \"jazz music\", REUSE \"jazz\".\n"
        "- Put a topic in new_topics ONLY if it is genuinely distinct from EVERY "
        "existing topic listed above.\n"
        "- CRITICAL: only use existing_topics_discussed for a label copied VERBATIM "
        "from the known list. If a topic is NOT in that list, it is NEW — put it in "
        "new_topics WITH a category (e.g. a K-pop group → music). Never place an "
        "unlisted topic in existing_topics_discussed.\n"
        f"- category MUST be one of: {cats}. Use \"other\" if unsure — never invent one.\n"
        "- GROUPING: if the child mentions specific examples of a broader topic "
        "(K-pop groups like Twice or NewJeans → the umbrella genre \"kpop\"; a specific "
        "athlete → their sport), ALSO include the broader topic, and add each pair to "
        "\"relations\" as {\"a\": specific, \"b\": broader}. Both labels MUST appear in "
        "your existing/new topics above. Use [] when there is nothing to relate.\n"
        "- Include ONLY topics the CHILD actually expressed; use empty arrays if none.\n"
        "- Return ONLY the JSON object."
    )


def _parse_json_object(raw: str) -> Optional[dict]:
    """First JSON object in a possibly-noisy response, or None."""
    if not raw:
        return None
    s = raw.strip()
    i, j = s.find("{"), s.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        obj = json.loads(s[i:j + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _conf(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _affinity(v) -> float:
    """Map an LLM `sentiment` (human 0-10 scale) to internal [0,1] affinity.

    Missing or out-of-range sentiment is treated as 5 (neutral) → 0.5 affinity;
    the item is NOT dropped for that alone (a bad sentiment is not a bad topic)."""
    try:
        s = float(v)
    except (TypeError, ValueError):
        return 0.5
    if not (0.0 <= s <= 10.0):
        return 0.5
    return aff01_from_10(s)


def _summary(v) -> Optional[str]:
    if not v:
        return None
    s = str(v).strip()[:280]
    return s or None


def extract_and_apply_topics(
    store, person_id: str, robot_id: str, turns: list, llm_fn: LLMFn,
    *, session_id: Optional[str] = None,
) -> dict:
    """Graph-aware, typed topic extraction for one session. Returns a summary dict:
       {applied, reinforced:[(label,conf)], added:[(label,cat,conf)], dropped:[...]}"""
    source = f"extraction:{session_id}" if session_id else "extraction"

    existing = person_topics(store, person_id)                       # [(label, category)]
    existing_norm = {normalize_label(l): (l, c) for l, c in existing}

    raw = llm_fn(build_system_prompt(existing), format_transcript(turns))
    obj = _parse_json_object(raw)
    if obj is None:
        # Parse failure → write NOTHING (never partially apply).
        return {"applied": False, "reason": "json_parse_failed",
                "reinforced": [], "added": [], "dropped": []}

    reinforced: list = []
    added: list = []
    dropped: list = []

    def _reinforce(canon_label, conf, affinity, summary):
        node = reinforce_person_topic(store, person_id, canon_label, source=source,
                                      confidence=conf, affinity=affinity, summary=summary)
        if node:
            reinforced.append((canon_label, conf))
        else:
            dropped.append(("existing", canon_label, "no_path"))

    # ── existing_topics_discussed ─────────────────────────────────────────────
    for item in (obj.get("existing_topics_discussed") or []):
        if not isinstance(item, dict):
            dropped.append(("existing", item, "not_object")); continue
        label = str(item.get("label", "")).strip()
        conf = _conf(item.get("confidence"))
        affinity = _affinity(item.get("sentiment"))   # missing/oob → 0.5, item kept
        summary = _summary(item.get("summary"))
        norm = normalize_label(label)
        if not norm:
            dropped.append(("existing", label, "empty")); continue
        if conf < CONFIDENCE_MIN:
            dropped.append(("existing", label, f"low_conf<{CONFIDENCE_MIN}")); continue
        if norm not in existing_norm:
            # Hallucinated 'existing' topic (not in the list we provided). It has no
            # category, so it cannot become a valid new topic → drop.
            dropped.append(("existing", label, "not_in_provided_list")); continue
        _reinforce(existing_norm[norm][0], conf, affinity, summary)  # reuse canonical label

    # ── new_topics ────────────────────────────────────────────────────────────
    for item in (obj.get("new_topics") or []):
        if not isinstance(item, dict):
            dropped.append(("new", item, "not_object")); continue
        label = str(item.get("label", "")).strip()
        conf = _conf(item.get("confidence"))
        affinity = _affinity(item.get("sentiment"))   # missing/oob → 0.5, item kept
        summary = _summary(item.get("summary"))
        cat = str(item.get("category", "")).strip().lower()
        norm = normalize_label(label)
        if not norm:
            dropped.append(("new", label, "empty")); continue
        if conf < CONFIDENCE_MIN:
            dropped.append(("new", label, f"low_conf<{CONFIDENCE_MIN}")); continue
        if cat not in TOPIC_CATEGORIES:
            # Category outside the closed taxonomy → DROP the item (write nothing).
            # The prompt tells the LLM to use "other" when unsure, so a value outside
            # the enum is a malformed response, not a real topic.
            dropped.append(("new", label, f"bad_category:{cat}")); continue
        if norm in existing_norm:
            # LLM put a known topic under new_topics → reinforce, do not duplicate.
            _reinforce(existing_norm[norm][0], conf, affinity, summary); continue
        node = add_person_topic(store, person_id, label, cat, source=source,
                                confidence=conf, affinity=affinity, summary=summary)
        if node:
            added.append((label, cat, conf))

    # ── relations (specific ↔ broader, e.g. twice ~ kpop) ─────────────────────
    # Only link topic nodes that exist (created above or already present); the pure
    # link_related_topic no-ops on a non-topic id, so bad labels are harmless.
    related: list = []
    for item in (obj.get("relations") or []):
        if not isinstance(item, dict):
            continue
        a, b = normalize_label(item.get("a", "")), normalize_label(item.get("b", ""))
        if not a or not b or a == b:
            continue
        if link_related_topic(store, topic_id(a), topic_id(b),
                               RELATION_WEIGHT, source=source):
            related.append((a, b))

    return {"applied": True, "reinforced": reinforced, "added": added,
            "dropped": dropped, "related": related}


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2 — semantic topic consolidation (app layer; injected embed_fn)
# ─────────────────────────────────────────────────────────────────────────────
# Merges near-duplicate topic nodes (e.g. "hiphop" / "hip hop") that Feature-1's
# exact-label reuse cannot catch. Embedding + pairing decisions live HERE; the
# graph surgery is the pure topics.merge_topics(). NOT run during live extraction
# — invoked on demand (dry-run first), so merges are deterministic and reviewable.

EmbedFn = Callable[[str], List[float]]

CONSOLIDATE_FLOOR = 0.86        # cosine >= this → MERGE (near-identical labels)
RELATED_FLOOR = 0.60           # [RELATED_FLOOR, CONSOLIDATE_FLOOR) → LINK (related, distinct)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _topic_nodes(store) -> List[tuple]:
    """[(id, label, category_str), ...] for every TopicNode (app-layer read)."""
    out = []
    for n in getattr(store, "_nodes", {}).values():
        if getattr(n, "node_type", None) == "topic":
            cat = n.category.value if hasattr(n.category, "value") else str(n.category)
            out.append((n.id, n.label, cat))
    return out


def _culture_topic_nodes(store) -> List[tuple]:
    """[(id, label, category_str), ...] for every CultureTopicNode (`ck:…`)."""
    out = []
    for n in getattr(store, "_nodes", {}).values():
        if getattr(n, "node_type", None) == "culture_topic":
            cat = n.category.value if hasattr(n.category, "value") else str(n.category)
            out.append((n.id, n.label, cat))
    return out


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _interest_nodes(store) -> List[tuple]:
    """[(id, person_id, label), ...] for every InterestNode (id = interest:person:slug)."""
    out = []
    for n in getattr(store, "_nodes", {}).values():
        if getattr(n, "node_type", None) == "interest":
            parts = n.id.split(":", 2)            # ['interest', person, slug]
            person = parts[1] if len(parts) == 3 else n.id
            out.append((n.id, person, n.label))
    return out


# ── shared consolidation core (topics + interests) ──────────────────────────

def _embed(items, embed_fn: EmbedFn) -> dict:
    """items: iterable of (id, label). Returns {id: vec}, skipping embed failures."""
    vecs: dict = {}
    for id_, label in items:
        try:
            v = embed_fn(label)
        except Exception:  # noqa: BLE001 — backend down → skip
            v = None
        if v:
            vecs[id_] = v
    return vecs


def _pairs(ids, vecs, *, lo, hi, ok):
    """Yield (a, b, round(sim,3)) for id pairs with lo <= sim < hi and ok(a, b)."""
    for i in range(len(ids)):
        a = ids[i]
        if a not in vecs:
            continue
        for j in range(i + 1, len(ids)):
            b = ids[j]
            if b in vecs and ok(a, b):
                sim = _cosine(vecs[a], vecs[b])
                if lo <= sim < hi:
                    yield a, b, round(sim, 3)


def _merge_by_similarity(store, items, embed_fn, floor, merge_fn, *, ok, dry_run, source):
    """Merge groups of near-duplicate nodes. items: [(id, label)]; `ok(a,b)` filters
    pairs; canonical = highest degree, ties → shortest then lexicographic. Shared by
    consolidate_topics and consolidate_interests."""
    label_of = {i: l for i, l in items}
    ids = list(label_of)
    vecs = _embed(items, embed_fn)
    uf = _UnionFind(ids)
    pairs = []
    for a, b, sim in _pairs(ids, vecs, lo=floor, hi=float("inf"), ok=ok):
        pairs.append((label_of[a], label_of[b], sim))
        uf.union(a, b)
    groups: dict = {}
    for i in ids:
        groups.setdefault(uf.find(i), []).append(i)
    merges, n_groups = [], 0
    for members in groups.values():
        if len(members) < 2:
            continue
        n_groups += 1
        canonical = min(members,
                        key=lambda t: (-topic_degree(store, t), len(label_of[t]), label_of[t]))
        for m in members:
            if m != canonical:
                merges.append((label_of[canonical], label_of[m]))
                if not dry_run:
                    merge_fn(store, canonical, m, source=source)
    return {"pairs": pairs, "merges": merges, "groups": n_groups}


def _same_category(cat_of, enabled):
    return (lambda a, b: cat_of[a] == cat_of[b]) if enabled else (lambda a, b: True)


def consolidate_topics(
    store, embed_fn: EmbedFn, *,
    floor: float = CONSOLIDATE_FLOOR, same_category_only: bool = True,
    dry_run: bool = False, source: str = "consolidate",
) -> dict:
    """Merge near-duplicate topics (cosine >= floor) into one canonical node.
    same_category_only never merges across categories. Returns
    {"pairs","merges","groups","dry_run"}."""
    from modules.graph_relationship.topics import merge_topics
    topics = _topic_nodes(store)                    # (id, label, category)
    cat_of = {tid: cat for tid, _l, cat in topics}
    r = _merge_by_similarity(
        store, [(tid, lbl) for tid, lbl, _c in topics], embed_fn, floor, merge_topics,
        ok=_same_category(cat_of, same_category_only), dry_run=dry_run, source=source)
    r["dry_run"] = dry_run
    return r


def consolidate_interests(
    store, embed_fn: EmbedFn, *,
    floor: float = CONSOLIDATE_FLOOR, dry_run: bool = False,
    source: str = "consolidate",
) -> dict:
    """Merge near-duplicate Interest nodes PER PERSON — e.g. "sports" (old LLM
    label) vs "sport" (new category-named interest)."""
    from modules.graph_relationship.topics import merge_interests
    by_person: dict = {}
    for iid, person, label in _interest_nodes(store):
        by_person.setdefault(person, []).append((iid, label))
    out = {"pairs": [], "merges": [], "groups": 0, "dry_run": dry_run}
    for items in by_person.values():
        if len(items) < 2:
            continue
        r = _merge_by_similarity(
            store, items, embed_fn, floor, merge_interests,
            ok=lambda a, b: True, dry_run=dry_run, source=source)
        out["pairs"] += r["pairs"]; out["merges"] += r["merges"]; out["groups"] += r["groups"]
    return out


def link_related_topics(
    store, embed_fn: EmbedFn, *,
    related_floor: float = RELATED_FLOOR, merge_floor: float = CONSOLIDATE_FLOOR,
    same_category_only: bool = True, dry_run: bool = False, source: str = "related",
) -> dict:
    """Add Topic↔Topic `related_topic` links for pairs that are related but NOT
    near-duplicates: cosine in [related_floor, merge_floor), same-category only by
    default (rap~hiphop, tennis~basketball). Returns {"links","existing","dry_run"}."""
    from modules.graph_relationship.topics import link_related_topic
    topics = _topic_nodes(store)
    cat_of = {tid: cat for tid, _l, cat in topics}
    label_of = {tid: lbl for tid, lbl, _c in topics}
    vecs = _embed([(tid, lbl) for tid, lbl, _c in topics], embed_fn)
    report = {"links": [], "existing": 0, "dry_run": dry_run}
    for a, b, sim in _pairs(list(label_of), vecs, lo=related_floor, hi=merge_floor,
                            ok=_same_category(cat_of, same_category_only)):
        if store.get_edge(*sorted((a, b)), "related_topic") is not None:
            report["existing"] += 1
            continue
        report["links"].append((label_of[a], label_of[b], sim))
        if not dry_run:
            link_related_topic(store, a, b, sim, source=source)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — cross-namespace culture↔person bridges (app layer; injected embed_fn)
# ─────────────────────────────────────────────────────────────────────────────
# Culture priors live on `ck:<culture>:<slug>` CultureTopic nodes; person interests
# live on `topic:<slug>` nodes. Today these two namespaces are disconnected islands
# — they only label-join, with no edge between them — so a person's OBSERVED interest
# can never propagate to a culturally-adjacent culture topic, and the BN degrades to
# a base-rate lookup. This pass adds `related_topic` bridge edges ACROSS the
# namespaces so the existing 2-round noisy-OR can carry person evidence into culture
# topics (e.g. observed `topic:jazz` lifts `ck:korean:kpop` above its raw prior).
#
# Bridges are the SAME related_topic edge type (no new type, no merge): a `ck:` node
# and a same-slug `topic:` node remain two DISTINCT nodes, now linked. Two idempotent
# sources, embedding-only or exact-string (NO LLM):
#   1. exact-slug (weight 1.0): topic:<slug> whose slug == a ck:<culture>:<slug>;
#   2. embedding band [0.60, 0.86): same "related but not near-dup" band as the
#      within-namespace related links, same_category_only, exact-slug pairs excluded.

def link_cross_namespace_bridges(
    store, embed_fn: EmbedFn, *,
    related_floor: float = RELATED_FLOOR, merge_floor: float = CONSOLIDATE_FLOOR,
    same_category_only: bool = True, dry_run: bool = False, source: str = "bridge",
) -> dict:
    """Link person `topic:` nodes to culture `ck:` CultureTopic nodes so the BN can
    propagate across the namespaces. Returns
    {"exact": [(topic_label, ck_label, 1.0)], "links": [(topic_label, ck_label, sim)],
     "existing": int, "dry_run": bool}. Idempotent — re-running adds nothing."""
    topics = _topic_nodes(store)                    # (id, label, category)
    ck = _culture_topic_nodes(store)                # (id, label, category)
    if not topics or not ck:
        return {"exact": [], "links": [], "existing": 0, "dry_run": dry_run}

    label_of = {i: l for i, l, _c in topics}
    label_of.update({i: l for i, l, _c in ck})
    exact: List[Tuple[str, str, float]] = []
    existing = 0

    # ── 1. exact-slug bridges (weight 1.0) ────────────────────────────────────
    topic_by_slug = {normalize_label(l): i for i, l, _c in topics}
    for ck_id, ck_lbl, _c in ck:
        tid = topic_by_slug.get(normalize_label(ck_lbl))
        if tid is None:
            continue
        if store.get_edge(*sorted((tid, ck_id)), "related_topic") is not None:
            existing += 1
            continue
        exact.append((label_of[tid], label_of[ck_id], 1.0))
        if not dry_run:
            link_related_cross(store, tid, ck_id, 1.0, source=source)

    # ── 2. embedding bridges in [related_floor, merge_floor) ──────────────────
    # Reuse the existing _embed / _pairs / _same_category machinery, pointed at the
    # combined id set with an `ok` that keeps ONLY cross-namespace pairs (one topic,
    # one ck) that are not already linked (exact-slug from step 1 wins).
    cat_of = {i: c for i, _l, c in topics}
    cat_of.update({i: c for i, _l, c in ck})
    is_ck = {i for i, _l, _c in ck}
    all_items = [(i, l) for i, l, _c in topics] + [(i, l) for i, l, _c in ck]
    ids = [i for i, _l in all_items]
    vecs = _embed(all_items, embed_fn)
    same_cat = _same_category(cat_of, same_category_only)

    def _ok(a: str, b: str) -> bool:
        if (a in is_ck) == (b in is_ck):        # both topic or both ck → not a bridge
            return False
        if not same_cat(a, b):
            return False
        return store.get_edge(*sorted((a, b)), "related_topic") is None

    links: List[Tuple[str, str, float]] = []
    for a, b, sim in _pairs(ids, vecs, lo=related_floor, hi=merge_floor, ok=_ok):
        links.append((label_of[a], label_of[b], sim))
        if not dry_run:
            link_related_cross(store, a, b, sim, source=source)

    return {"exact": exact, "links": links, "existing": existing, "dry_run": dry_run}
