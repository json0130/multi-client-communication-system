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


def qa_windows(limit: int = 5000) -> list[dict]:
    """Individual Q&A windows, not the aggregate view.

    suggested_qa_budget needs `closed_by` to tell a window a person ended from
    one that ran out its own allocation, and the view groups that away.
    """
    try:
        return (get_client().table(QA).select("*")
                .limit(limit).execute().data or [])
    except Exception as e:
        print(f"[demo_duration_repo] qa_windows error: {e}")
        return []


MIN_WINDOWS_TO_TRUST = 20
"""Usable windows before an observed budget replaces the constant.

Was 10, and counted every window rather than the usable ones. Bootstrapped
over the real corpus, the median's 95% interval runs 4.3-52.7s at n=10,
9.7-42.6s at n=20, and 10.4-42.0s at n=30 — a third of the spread bought
between 10 and 20, and almost nothing after. The remaining width is not
sampling error: visitor Q&A genuinely runs from 2s to 165s, so no threshold
makes ONE pooled number good for any particular group. Twenty is where more
data stops helping a global estimate, which is also the point at which the
honest next move is a per-block or per-audience one."""


def suggested_qa_budget(default_sec: float = 90.0) -> dict:
    """
    A defensible default Q&A allocation, from observed windows.

    Uses the MEDIAN rather than the mean: Q&A length is operator-driven and
    long tails are common, so a mean is dragged upward by the one group that
    would not stop asking. Falls back to `default_sec` while there is too
    little data, and says so — a budget invented from three windows should not
    be presented as measured.

    A WINDOW THAT RAN OUT ITS OWN ALLOCATION IS NOT EVIDENCE ABOUT VISITORS.
    It is a measurement of the allocation. Half the live corpus was exactly
    that: 23 of 46 windows closed by timeout, 19 of them at exactly 5.0s
    (ALREADY_ENGAGED_QA_SEC) and 3 at exactly 60.0s. Feeding those back in
    made the estimator a closed loop — shrink a window to five seconds,
    record five seconds, pull the median down, shrink more windows. The
    pooled median over everything was 5.0s: the system had learned its own
    constant. Timeout closures are right-censored (the visitor may well have
    wanted longer), so they are excluded and counted, never averaged in.

    And it now medians the WINDOWS, not the per-step means. Medianing six
    per-step rows put the answer on whichever row happened to sit in the
    middle regardless of how many windows stood behind it: on the live
    corpus that was `open_floor`, n=1 — a single window, of a step that is
    not a project Q&A at all, returned as "observed" on the strength of 46.
    Only windows belonging to a project block count, for the same reason.
    """
    rows = qa_windows()
    usable, censored = [], 0
    for r in rows:
        if not r.get("block_robot_id"):
            continue                    # open floor, not a project Q&A window
        if (r.get("closed_by") or "") == "timeout":
            censored += 1
            continue
        try:
            usable.append(float(r["seconds"]))
        except (TypeError, ValueError, KeyError):
            continue

    out = {"windows": len(usable), "self_terminated": censored}
    if len(usable) < MIN_WINDOWS_TO_TRUST:
        return {**out, "budget_sec": default_sec, "basis": "default"}
    usable.sort()
    mid = len(usable) // 2
    centre = (usable[mid] if len(usable) % 2
              else (usable[mid - 1] + usable[mid]) / 2)
    return {**out, "budget_sec": round(centre, 0), "basis": "observed"}
