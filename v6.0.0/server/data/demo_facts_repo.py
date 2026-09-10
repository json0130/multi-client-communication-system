"""
data/demo_facts_repo.py
=======================
Grounded detail for the topics the robots present.

Read failures degrade to "no facts", which puts the robot back exactly where
it was before this table existed — able to talk about the topic in general,
and instructed to decline specifics. A grounding lookup that cannot reach the
database must never stop a robot answering.
"""

from __future__ import annotations

from typing import Optional

from data.connection import get_client

TABLE = "demo_topic_facts"
COVERAGE = "demo_topic_fact_coverage"

# Ordered by how much a listener gains from hearing it. A limitation is the
# most valuable thing a research demo can offer and the least likely to be
# invented, so it leads; a publication is a pointer rather than an answer.
KIND_ORDER = ("limitation", "method", "model", "metric",
              "dataset", "hardware", "publication")


def _spread(rows: list[dict], limit: int) -> list[dict]:
    """The `limit` most useful rows, with a SPREAD of kinds rather than a run
    of the first one.

    Sorting flat by KIND_ORDER and truncating looks right until a topic has
    more material than fits. Limitations lead, so a topic with five of them
    and a limit of eight spends five slots before the first method — and the
    named model, the dataset, the hardware, the measured number are exactly
    what got cut. Those are the rows that stopped the robots inventing
    "GraphSLAM and FastSLAM" and "a retrieval model like BM25"; losing them
    to an abundance of honest caveats would reintroduce the original bug by
    way of the fix for it.

    So fill by rounds: the best limitation, then the best method, the best
    model, and so on through KIND_ORDER, then round again. A limitation still
    comes first, and a topic that HAS a named model can no longer fail to
    mention it.
    """
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(r.get("kind") or "", []).append(r)
    order = list(KIND_ORDER) + sorted(k for k in by_kind if k not in KIND_ORDER)
    out: list[dict] = []
    while len(out) < limit:
        took = False
        for kind in order:
            group = by_kind.get(kind)
            if group:
                out.append(group.pop(0))
                took = True
                if len(out) >= limit:
                    break
        if not took:
            break
    return out


def facts_for(topic_id: str, robot_id: Optional[str] = None,
              limit: int = 6) -> list[dict]:
    """Facts to ground one answer: this robot's own, plus topic-general ones.

    Another robot's facts about the same topic are deliberately excluded —
    a robot must not narrate a colleague's results as its own work.
    """
    if not topic_id:
        return []
    try:
        rows = (get_client().table(TABLE).select("*")
                .eq("topic_id", topic_id).execute().data or [])
    except Exception as e:
        print(f"[demo_facts_repo] facts_for error: {e}")
        return []
    mine = [r for r in rows
            if not r.get("robot_id") or r.get("robot_id") == robot_id]
    mine.sort(key=lambda r: (
        KIND_ORDER.index(r["kind"]) if r["kind"] in KIND_ORDER else 99,
        not r.get("verified"),          # confirmed facts first
        r.get("id", 0),
    ))
    return _spread(mine, limit)


def facts_for_topics(topic_ids: list[str], robot_id: Optional[str] = None,
                     per_topic: int = 6) -> dict[str, list[dict]]:
    """facts_for over several topics in ONE query, keyed by topic id.

    Used for topic-link expansion, where the caller has three or four
    neighbouring topics to read. A query each would put four Supabase round
    trips between a visitor finishing their question and the robot starting
    to answer, which is a live demonstration's most expensive currency.

    Same robot scoping as facts_for, and it matters more here: a neighbouring
    topic usually belongs to a DIFFERENT robot — social robot navigation sits
    next to social signals, which is Navel's — and following a link must not
    become a way for one robot to narrate another's results. Only this
    robot's own rows and topic-general ones cross the hop.
    """
    ids = [t for t in (topic_ids or []) if t]
    if not ids:
        return {}
    try:
        rows = (get_client().table(TABLE).select("*")
                .in_("topic_id", ids).execute().data or [])
    except Exception as e:
        print(f"[demo_facts_repo] facts_for_topics error: {e}")
        return {}
    out: dict[str, list[dict]] = {}
    for tid in ids:
        mine = [r for r in rows
                if r.get("topic_id") == tid
                and (not r.get("robot_id") or r.get("robot_id") == robot_id)]
        mine.sort(key=lambda r: (
            KIND_ORDER.index(r["kind"]) if r["kind"] in KIND_ORDER else 99,
            not r.get("verified"),
            r.get("id", 0),
        ))
        out[tid] = _spread(mine, per_topic)
    return out


def facts_for_robot(robot_id: str, limit: int = 10) -> list[dict]:
    """Everything this robot may state, across all of its own topics.

    Used when the utterance resolved to NO topic. A question asked during a
    robot's own block is almost certainly about its work, and the
    alternative is handing the model zero grounding — which is exactly when
    a live run had Silbot invent "GraphSLAM and FastSLAM". Broad grounding
    beats none.
    """
    if not robot_id:
        return []
    try:
        rows = (get_client().table(TABLE).select("*")
                .eq("robot_id", robot_id).execute().data or [])
    except Exception as e:
        print(f"[demo_facts_repo] facts_for_robot error: {e}")
        return []
    rows.sort(key=lambda r: (
        KIND_ORDER.index(r["kind"]) if r["kind"] in KIND_ORDER else 99,
        not r.get("verified"), r.get("id", 0)))
    return _spread(rows, limit)


def coverage() -> list[dict]:
    """Per-topic fact counts, for tools/check_facts.py and the dashboard."""
    try:
        return (get_client().table(COVERAGE).select("*")
                .order("topic_id").execute().data or [])
    except Exception as e:
        print(f"[demo_facts_repo] coverage error: {e}")
        return []


def add(topic_id: str, kind: str, fact: str, robot_id: Optional[str] = None,
        verified: bool = False, source: str = "") -> Optional[dict]:
    try:
        row = {"topic_id": topic_id, "kind": kind, "fact": fact,
               "robot_id": robot_id, "verified": verified, "source": source}
        res = get_client().table(TABLE).insert([row]).execute()
        return (res.data or [None])[0]
    except Exception as e:
        print(f"[demo_facts_repo] add failed: {e}")
        return None


def set_verified(fact_id: int, verified: bool = True) -> bool:
    try:
        get_client().table(TABLE).update({"verified": verified}).eq(
            "id", fact_id).execute()
        return True
    except Exception as e:
        print(f"[demo_facts_repo] set_verified failed: {e}")
        return False


def unverified_count() -> int:
    try:
        r = (get_client().table(TABLE).select("id", count="exact")
             .eq("verified", False).limit(1).execute())
        return r.count or 0
    except Exception:
        return 0
