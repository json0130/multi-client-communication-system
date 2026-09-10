"""
data/demo_kg_repo.py
====================
Persistence for the robot→topic competence graph.

Schema in data/migrations/006_demo_kg.sql. The arithmetic lives in
decision/kg.py and is never duplicated here — this module reads an edge, hands
it to the pure updater, and writes the result back.

Read-modify-write is the shape, which means two corrections applied to the same
edge at the same instant can lose one. That is accepted rather than locked
around: corrections arrive at human speed, seconds apart, and the alternative is
a stored procedure that would put the update arithmetic in SQL where it cannot
be unit-tested. If concurrent supervisors ever become real, move the increment
into a Postgres function and delete apply_observation.
"""

from __future__ import annotations
import time
from typing import Callable, Iterable, Optional, Sequence

from data.connection import get_client
from decision.kg import Evidence, RobotTopicEdge, TopicEdge

TOPICS = "demo_topics"
LINKS = "demo_topic_links"
EDGES = "demo_robot_topic"
VIEW = "demo_kg_edges"


# ── Vocabulary ────────────────────────────────────────────────────────────────

def upsert_topics(rows: Sequence[dict]) -> int:
    """Insert or update topics. `rows` need id/label; category and source optional."""
    if not rows:
        return 0
    payload = [{
        "id": r["id"], "label": r["label"],
        "category": r.get("category") or "other",
        "source": r.get("source") or "",
    } for r in rows]
    get_client().table(TOPICS).upsert(payload).execute()
    return len(payload)


VOCAB_TTL_SEC = 60.0
"""How long the topic vocabulary and link graph are reused before re-reading.

These two tables are CONFIGURATION — a topic list and the semantic links
between topics — edited by hand through the dashboard, not by the running
demo. Re-reading them per answer cost 160ms of the ~225ms that topic-link
grounding added to every visitor question, for data that had not changed
since the server booted.

A minute, rather than caching for the process lifetime, so an edit made in
the dashboard during a demo still takes effect within one; the writers below
clear it immediately, so an edit made through this module is visible at once.
The learned edges (demo_robot_topic) are NOT cached — those change while the
demo runs, which is the entire point of them."""

_vocab_cache: dict = {}


def _cached(key: str, fetch: Callable):
    hit = _vocab_cache.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < VOCAB_TTL_SEC:
        return hit[1]
    value = fetch()
    _vocab_cache[key] = (time.monotonic(), value)
    return value


def invalidate_vocab_cache() -> None:
    """Drop the cached vocabulary. Called by every writer in this module."""
    _vocab_cache.clear()


def all_topics() -> list[dict]:
    def fetch():
        try:
            return get_client().table(TOPICS).select("*").order("label").execute().data or []
        except Exception as e:
            print(f"[demo_kg_repo] all_topics error: {e}")
            return []
    return _cached("topics", fetch)


def topic_labels() -> dict:
    """{topic_id: label} — what a caller naming a topic in a prompt needs.

    Separate from all_topics so the mapping is built once per cache fill
    rather than once per answer.
    """
    return _cached("labels",
                   lambda: {t["id"]: (t.get("label") or t["id"])
                            for t in all_topics()})


# ── Topic ↔ topic links ───────────────────────────────────────────────────────

def upsert_links(links: Iterable[TopicEdge]) -> int:
    """Insert or update topic↔topic edges. TopicEdge sorts its own endpoints,
    which the table's CHECK constraint also enforces — a link written the other
    way round would be a second row for the same pair."""
    rows = [l.as_row() for l in links]
    if not rows:
        return 0
    get_client().table(LINKS).upsert(rows).execute()
    invalidate_vocab_cache()
    invalidate_vocab_cache()
    return len(rows)


def all_links() -> list[dict]:
    def fetch():
        try:
            return get_client().table(LINKS).select("*").execute().data or []
        except Exception as e:
            print(f"[demo_kg_repo] all_links error: {e}")
            return []
    return _cached("links", fetch)


def neighbours(topic_id: str) -> list[tuple[str, float]]:
    """[(neighbour_topic_id, weight)] for one topic, both directions.

    Filtered from the cached link set rather than queried. This ran two
    round trips of its own — one per direction — on every grounded answer,
    against a table of eighteen rows that changes when a researcher edits it.
    """
    out = []
    for r in all_links():
        if r.get("topic_a") == topic_id:
            out.append((r["topic_b"], float(r["weight"])))
        elif r.get("topic_b") == topic_id:
            out.append((r["topic_a"], float(r["weight"])))
    return out


# ── The learned edge ──────────────────────────────────────────────────────────

def get_edge(robot_id: str, topic_id: str) -> RobotTopicEdge:
    """The stored edge, or a fresh one at the prior. Never None: an absent edge
    and an unobserved edge mean the same thing, and returning None would push
    that decision onto every caller."""
    try:
        rows = (get_client().table(EDGES).select("*")
                .eq("robot_id", robot_id).eq("topic_id", topic_id)
                .limit(1).execute().data or [])
        if rows:
            return RobotTopicEdge.from_row(rows[0])
    except Exception as e:
        print(f"[demo_kg_repo] get_edge error: {e}")
    return RobotTopicEdge(robot_id=robot_id, topic_id=topic_id)


def put_edge(edge: RobotTopicEdge) -> None:
    """Write an edge. Raises — callers decide whether losing it is tolerable."""
    get_client().table(EDGES).upsert([edge.as_row()]).execute()


def apply_observation(
    robot_id: str, topic_id: str, target: float,
    kind: Evidence = Evidence.SUPERVISOR,
) -> Optional[RobotTopicEdge]:
    """Fold one observation into an edge and persist it. Returns the new edge,
    or None if the write failed.

    Swallows write failures for the same reason the decision sink does: a
    correction that cannot be stored must not take a live demo down with it.
    """
    try:
        updated = get_edge(robot_id, topic_id).update(target, kind)
        put_edge(updated)
        return updated
    except Exception as e:
        print(f"[demo_kg_repo] apply_observation failed for "
              f"{robot_id}/{topic_id}: {e}")
        return None


# ── Reads for the dashboard ───────────────────────────────────────────────────

def declared_topics_by_robot() -> dict:
    """{robot_id: [topic label, ...]} for every DECLARED edge.

    Read by robot/robot_instance.py when it lists teammates, so the model can
    see who owns what instead of inferring it from a prose role. Without it a
    robot asked about a peer's subject answered anyway: Silbot, asked how it
    would recall something a visitor said an hour ago, explained long-context
    memory — ChatBox's declared area — rather than handing over.

    Empty on any failure, which returns the peer list to roles alone.
    """
    try:
        rows = (get_client().table(VIEW).select("robot_id,topic_label,specialised")
                .eq("specialised", True).execute().data or [])
    except Exception as e:
        print(f"[demo_kg_repo] declared_topics_by_robot error: {e}")
        return {}
    out: dict = {}
    for r in rows:
        out.setdefault(r["robot_id"], []).append(r.get("topic_label") or "")
    return {k: sorted(v for v in vs if v) for k, vs in out.items()}


def graph(robot_id: Optional[str] = None) -> list[dict]:
    """Edges with confidence/clamped/human_share precomputed by the view, so the
    UI never reimplements the arithmetic."""
    try:
        q = get_client().table(VIEW).select("*")
        if robot_id:
            q = q.eq("robot_id", robot_id)
        return q.order("clamped", desc=True).execute().data or []
    except Exception as e:
        print(f"[demo_kg_repo] graph error: {e}")
        return []


def summary() -> dict:
    """Headline numbers: size of the graph and how much of it came from people."""
    edges = graph()
    observed = [e for e in edges if e["n_obs"] > 0]
    sup = sum(e["n_supervisor"] for e in edges)
    out = sum(e["n_outcome"] for e in edges)
    return {
        "topics": len(all_topics()),
        "topic_links": len(all_links()),
        "edges": len(edges),
        "observed_edges": len(observed),
        "n_supervisor": sup,
        "n_outcome": out,
        # The number worth reporting in the paper.
        "human_share": (sup / (sup + out)) if (sup + out) else 0.0,
    }
