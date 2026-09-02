"""
data/demo_duration_repo.py
==========================
How long the tour takes, recorded in two streams that never merge.

Schema in data/migrations/008_demo_durations.sql, where the reasoning lives.
The short version: scripted step duration is a property of the content and
averages usefully; Q&A duration is a property of the operator and the group and
does not. Averaging them together makes every step's estimate worse as more data
arrives.

Writes are batched through the same BatchingDecisionSink machinery used for
decisions, for the same reason — a demo must never wait on a logging write, and
a failed write must degrade to a warning rather than interrupt a tour.
"""

from __future__ import annotations
from typing import Optional, Sequence

from data.connection import get_client

STEPS = "demo_step_durations"
QA = "demo_qa_durations"


# ── Writes ────────────────────────────────────────────────────────────────────

def write_step_durations(rows: Sequence[dict]) -> None:
    """Insert scripted-step timings. Raises; the caller's sink warns."""
    if not rows:
        return
    get_client().table(STEPS).insert(list(rows)).execute()


def write_qa_durations(rows: Sequence[dict]) -> None:
    """Insert Q&A window timings. Raises; the caller's sink warns."""
    if not rows:
        return
    get_client().table(QA).insert(list(rows)).execute()


# ── Reads ─────────────────────────────────────────────────────────────────────

def step_stats(block_robot_id: Optional[str] = None) -> list[dict]:
    """[{step_id, role, runs, mean_sec, sd_sec, ...}] — what the planner predicts with."""
    try:
        q = get_client().table("demo_step_duration_stats").select("*")
        if block_robot_id:
            q = q.eq("block_robot_id", block_robot_id)
        return q.execute().data or []
    except Exception as e:
        print(f"[demo_duration_repo] step_stats error: {e}")
        return []


def qa_stats() -> list[dict]:
    """[{step_id, windows, mean_sec, sd_sec, mean_turns, overruns, budgeted}].

    Read to CHOOSE a default Q&A budget, never to predict one. The planner sets
    Q&A length; this says what a defensible setting is and how often operators
    run past it.
    """
    try:
        return get_client().table("demo_qa_duration_stats").select("*").execute().data or []
    except Exception as e:
        print(f"[demo_duration_repo] qa_stats error: {e}")
        return []


def block_estimate(block_robot_id: str) -> dict:
    """
    Predicted seconds for one project block, EXCLUDING its Q&A.

    Q&A is deliberately absent: it is allocated, not predicted. A caller wanting
    a whole-block figure adds whatever budget it intends to grant, which keeps
    the two visibly separate at the call site instead of hiding an assumed Q&A
    length inside an "estimate".
    """
    rows = [r for r in step_stats(block_robot_id) if r.get("role") != "qa"]
    total = sum(float(r.get("mean_sec") or 0) for r in rows)
    runs = min([int(r.get("runs") or 0) for r in rows], default=0)
    return {
        "block_robot_id": block_robot_id,
        "scripted_sec": round(total, 1),
        "steps": len(rows),
        # The weakest evidence behind any step in the block. An estimate built
        # from one observation of one step is not an estimate.
        "min_runs": runs,
        "qa_sec": None,   # allocated by the planner, never predicted here
    }


def suggested_qa_budget(default_sec: float = 90.0) -> dict:
    """
    A defensible default Q&A allocation, from observed windows.

    Uses the MEDIAN-ish centre rather than the mean: Q&A length is
    operator-driven and long tails are common, so a mean is dragged upward by
    the one group that would not stop asking. Falls back to `default_sec` while
    there is too little data, and says so — a budget invented from three windows
    should not be presented as measured.
    """
    rows = qa_stats()
    windows = sum(int(r.get("windows") or 0) for r in rows)
    if windows < 10:
        return {"budget_sec": default_sec, "basis": "default", "windows": windows}
    means = sorted(float(r["mean_sec"]) for r in rows if r.get("mean_sec"))
    centre = means[len(means) // 2] if means else default_sec
    return {"budget_sec": round(centre, 0), "basis": "observed", "windows": windows}
