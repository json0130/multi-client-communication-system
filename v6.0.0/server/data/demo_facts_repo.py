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


def facts_for(topic_id: str, robot_id: Optional[str] = None,
              limit: int = 8) -> list[dict]:
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
    return mine[:limit]


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
