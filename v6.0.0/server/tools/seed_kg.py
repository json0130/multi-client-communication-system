"""
tools/seed_kg.py
================
Build the demo topic vocabulary, and the similarity links over it.

    python3 tools/seed_kg.py --dry-run     # show what it would write
    python3 tools/seed_kg.py               # write it

WHY THIS IS NEEDED AT ALL
The demo system has no topic namespace. The CHATBOX KG has one — 28 topics —
but they are a child's interests (jazz, baseball, kpop) and transfer nothing to
a research tour. Its `projects` table was designed with a `keywords TEXT[]`
column for exactly this, but migration 001 was never applied, so there is no
seed data either. The vocabulary is built from scratch here.

WHERE TOPICS COME FROM, in priority order:
  1. projects.keywords   — the designed-in source, if migration 001 is applied
  2. SEED_TOPICS         — a curated CARES vocabulary, used when 1 is empty

robots.robot_role was tried as a source and abandoned. Splitting role sentences
into content words produced 159 "topics" including `about`, `all`, `always` and
`answer`, and zero usable links. A role is a sentence about a persona, not a
list of subjects, and no amount of stopword filtering turns one into the other.

HOW LINKS ARE BUILT, in priority order:
  1. EMBEDDINGS  — cosine similarity between topic labels, banded exactly as
     CHATBOX's kg_extraction.link_related_topics does it: a pair is LINKED when
     cosine falls in [RELATED_FLOOR, MERGE_FLOOR). Below the lower bound they
     are unrelated; at or above the upper bound they are near-duplicates that
     should be merged rather than linked. This is the default and it matters for
     more than convenience: a topology that is COMPUTED is not open to the
     objection that the author drew the edges that made the result work.
  2. CURATED     — the hand-authored SEED_LINKS, used only with --no-embed or
     when no embedding backend is reachable.

Token overlap was tried and abandoned. It scores `emotion recognition` against
`facial expression analysis` at zero — they share no words — and produced no
links at all across the seed vocabulary.

Expect the link set to stay SPARSE either way; CHATBOX has 7 links over 28
topics. Whether that sparsity leaves enough structure for a correction to
generalise is an empirical question — tools/kg_reach.py measures it, and
--compare reports how far the derived topology differs from the authored one.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.kg import TopicEdge   # noqa: E402

# Cosine bands, taken from CHATBOX kg_extraction (RELATED_FLOOR /
# CONSOLIDATE_FLOOR) so the two systems band similarity identically.
RELATED_FLOOR = 0.60   # below this: unrelated, no edge
MERGE_FLOOR = 0.86     # at or above: near-duplicate labels, should be merged

# Jaccard fallback for keyword harvesting only. Deliberately not tuned.
LINK_FLOOR = 0.34

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "you",
    "are", "is", "am", "your", "robot", "research", "project", "using", "based",
    "helpful", "assistant", "lab", "work", "works", "working",
}

# The seed vocabulary: subjects a visitor might actually ask a CARES robot about.
SEED_TOPICS = [
    ("retrieval augmented generation", "ai"),
    ("large language models", "ai"),
    ("conversational memory", "ai"),
    ("knowledge graphs", "ai"),
    ("emotion recognition", "hri"),
    ("facial expression analysis", "hri"),
    ("social signals", "hri"),
    ("human robot trust", "hri"),
    ("social robot navigation", "navigation"),
    ("mapping and localisation", "navigation"),
    ("multi robot coordination", "systems"),
    ("robot hardware", "systems"),
    ("speech recognition", "speech"),
    ("text to speech", "speech"),
]

# Hand-authored neighbours, with a rough strength. These are the edges an expert
# would draw; see the module docstring for why they are not computed.
SEED_LINKS = [
    ("retrieval augmented generation", "large language models", 0.80),
    ("retrieval augmented generation", "knowledge graphs", 0.65),
    ("retrieval augmented generation", "conversational memory", 0.70),
    ("large language models", "conversational memory", 0.60),
    ("knowledge graphs", "conversational memory", 0.55),
    ("emotion recognition", "facial expression analysis", 0.85),
    ("emotion recognition", "social signals", 0.70),
    ("social signals", "human robot trust", 0.60),
    ("emotion recognition", "human robot trust", 0.50),
    ("social robot navigation", "mapping and localisation", 0.75),
    ("social robot navigation", "social signals", 0.55),
    ("multi robot coordination", "social robot navigation", 0.50),
    ("multi robot coordination", "robot hardware", 0.45),
    ("speech recognition", "text to speech", 0.70),
    ("speech recognition", "large language models", 0.50),
    ("text to speech", "robot hardware", 0.40),
]


def slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(label).strip().lower()).strip("-")
    return f"topic:{s}"


def tokens(label: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", label.lower())
            if w and w not in STOPWORDS and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def collect_topics() -> tuple[list[dict], bool]:
    """(topics, from_projects). Prefers project keywords, falls back to the seed."""
    from data.connection import get_client
    out: dict = {}

    try:
        rows = get_client().table("projects").select("name,keywords").execute().data or []
        for row in rows:
            for kw in (row.get("keywords") or []):
                if kw and kw.strip():
                    out.setdefault(slug(kw), {
                        "id": slug(kw), "label": kw.strip().lower(),
                        "category": "project", "source": "projects.keywords"})
    except Exception:
        pass   # migration 001 not applied — expected, not an error

    if out:
        return sorted(out.values(), key=lambda t: t["label"]), True

    for label, cat in SEED_TOPICS:
        out[slug(label)] = {"id": slug(label), "label": label,
                            "category": cat, "source": "seed"}
    return sorted(out.values(), key=lambda t: t["label"]), False


def embed_links(topics: list[dict]) -> tuple[list, list]:
    """(links, merge_candidates) from embedding cosine. Raises if unreachable.

    Same banding as CHATBOX: [RELATED_FLOOR, MERGE_FLOOR) links, >= MERGE_FLOOR
    is flagged as a near-duplicate rather than linked — an edge between two names
    for the same thing is not a relation, it is a vocabulary bug.
    """
    import json
    import urllib.request
    from core.config import cfg

    url = f"http://{cfg.llm.ollama_host}:{cfg.llm.ollama_port}/api/embeddings"

    def embed(text: str) -> list:
        req = urllib.request.Request(
            url, data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp).get("embedding", [])

    def norm(v):
        m = sum(x * x for x in v) ** 0.5
        return [x / m for x in v] if m else v

    vecs = {t["id"]: norm(embed(t["label"])) for t in topics}
    links, merges = [], []
    for a, b in combinations(sorted(vecs), 2):
        if not vecs[a] or not vecs[b]:
            continue
        cos = sum(x * y for x, y in zip(vecs[a], vecs[b]))
        if cos >= MERGE_FLOOR:
            merges.append((a, b, round(cos, 4)))
        elif cos >= RELATED_FLOOR:
            links.append(TopicEdge(a, b, round(cos, 4), source="embedding"))
    return links, merges


def curated_links(topics: list[dict]) -> list[TopicEdge]:
    known = {t["id"] for t in topics}
    return [TopicEdge(slug(a), slug(b), w, source="curated")
            for a, b, w in SEED_LINKS
            if slug(a) in known and slug(b) in known]


def build_links(topics: list[dict], from_projects: bool,
                floor: float = LINK_FLOOR) -> list[TopicEdge]:
    """Curated links for the seed vocabulary; token overlap for harvested keywords."""
    known = {t["id"] for t in topics}

    if not from_projects:
        return curated_links(topics)

    tok = {t["id"]: tokens(t["label"]) for t in topics}
    links = []
    for a, b in combinations(sorted(tok), 2):
        w = jaccard(tok[a], tok[b])
        if w >= floor:
            links.append(TopicEdge(a, b, round(w, 4), source="token-overlap"))
    return links


def seed_from_db(dry_run: bool = False, use_embeddings: bool = True) -> dict:
    """Seed the VOCABULARY and its links. Never the robot→topic weights.

    NEVER SEED robot→topic FROM PROJECT ASSIGNMENTS.
    It is the obvious convenience — the projects table already says which robot
    owns which subject — and it would silently destroy the result this graph
    exists to produce. The topic clusters already map close to 1:1 onto project
    assignments (12 of 16 authored links are within-category). Seeding
    competence from the same assignments would make propagation re-derive a
    partition that was typed in by hand, and every generalisation number becomes
    circular: the graph would be confirming its own input.

    robot→topic edges must start at the 0.5 prior and move only on observation.
    An edge seeded away from 0.5 is a capability fact wearing a competence
    weight; if a capability genuinely constrains routing, filter the candidate
    list before route() sees it rather than encoding it here.
    """
    from data import demo_kg_repo as repo
    topics, from_projects = collect_topics()

    merges: list = []
    link_source = "curated"
    links = None
    if use_embeddings:
        try:
            links, merges = embed_links(topics)
            link_source = "embedding"
        except Exception as e:
            print(f"  ! embeddings unavailable ({str(e)[:50]}) — using curated links")
    if links is None:
        links = build_links(topics, from_projects)
    result = {
        "topics": len(topics), "links": len(links), "dry_run": dry_run,
        "source": "projects.keywords" if from_projects else "seed vocabulary",
        "link_source": link_source,
        "merge_candidates": merges,
        "sample_topics": [t["label"] for t in topics[:12]],
        "sample_links": [(l.topic_a, l.topic_b, l.weight) for l in links[:10]],
        # The number that decides whether propagation is worth anything.
        "links_per_topic": round(len(links) * 2 / len(topics), 2) if topics else 0.0,
    }
    if not dry_run:
        repo.upsert_topics(topics)
        repo.upsert_links(links)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Seed the demo topic vocabulary.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-embed", action="store_true",
                    help="use the hand-authored links instead of deriving them")
    ap.add_argument("--compare", action="store_true",
                    help="report how far the derived topology differs from the authored one")
    args = ap.parse_args()

    if args.compare:
        return compare()

    r = seed_from_db(dry_run=args.dry_run, use_embeddings=not args.no_embed)
    print("=" * 58)
    print("  Topic vocabulary" + ("  (DRY RUN — nothing written)" if args.dry_run else ""))
    print("=" * 58)
    print(f"  source          : {r['source']}")
    print(f"  link source     : {r['link_source']}")
    print(f"  topics          : {r['topics']}")
    print(f"  topic links     : {r['links']}")
    print(f"  links per topic : {r['links_per_topic']}")
    print()
    print("  sample topics:", ", ".join(r["sample_topics"]))
    if r["sample_links"]:
        print()
        print("  sample links:")
        for a, b, w in r["sample_links"]:
            print(f"      {w:.2f}  {a} ~ {b}")
    else:
        print()
        print("  NO LINKS — every correction will stay on the exact topic it was")
        print("  made on. Propagation cannot help until the vocabulary has")
        print("  genuinely related entries or a real embedding matcher is wired in.")
    if r.get("merge_candidates"):
        print()
        print("  NEAR-DUPLICATES (cosine >= %.2f) — these are two names for one" % MERGE_FLOOR)
        print("  concept and should be merged, not linked:")
        for a, b, c in r["merge_candidates"]:
            print(f"      {c:.2f}  {a} == {b}")
    return 0


def compare() -> int:
    """How far does the derived topology differ from the authored one?

    The authored links are the author's belief about the domain. If the derived
    set agrees closely, the hand-authoring objection is moot either way; if it
    disagrees, the derived set is the defensible one to report.
    """
    topics, _ = collect_topics()
    hand = {(l.topic_a, l.topic_b): l.weight for l in curated_links(topics)}
    try:
        derived_links, _ = embed_links(topics)
    except Exception as e:
        print(f"Embeddings unavailable: {e}")
        return 1
    derived = {(l.topic_a, l.topic_b): l.weight for l in derived_links}

    both = sorted(set(hand) & set(derived))
    print("=" * 66)
    print("  Authored vs derived topology")
    print("=" * 66)
    print(f"  authored : {len(hand)} links")
    print(f"  derived  : {len(derived)} links")
    print(f"  shared   : {len(both)}")
    print(f"  authored only : {len(set(hand) - set(derived))}")
    print(f"  derived only  : {len(set(derived) - set(hand))}")
    if both:
        diffs = [abs(hand[k] - derived[k]) for k in both]
        print(f"  weight gap on shared links: mean {sum(diffs)/len(diffs):.3f}, "
              f"max {max(diffs):.3f}")
    print()
    for k in sorted(set(hand) - set(derived)):
        print(f"    authored only  {hand[k]:.2f}  {k[0]} ~ {k[1]}")
    for k in sorted(set(derived) - set(hand))[:12]:
        print(f"    derived only   {derived[k]:.2f}  {k[0]} ~ {k[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
