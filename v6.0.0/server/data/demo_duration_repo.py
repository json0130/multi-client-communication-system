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


def _windows_by_block(rows) -> dict:
    """{block_robot_id: [(seconds, ended_by_visitor)]} — the one place the rule
    for reading a Q&A window is written down.

    A window closed by `timeout` ran out its own allocation. It is not a
    measurement of what the visitor wanted — they may well have wanted longer
    — but it is not nothing either: the visitor wanted AT LEAST that long. It
    is right-censored, and kept as such (ended_by_visitor=False) rather than
    dropped. See censored_median for why dropping stopped being safe.

    The open-floor Q&A has no block and is not a project window, so it is
    left out.
    """
    out: dict = {}
    for r in rows or ():
        block = r.get("block_robot_id")
        if not block:
            continue
        try:
            secs = float(r["seconds"])
        except (TypeError, ValueError, KeyError):
            continue
        ended = (r.get("closed_by") or "") != "timeout"
        out.setdefault(block, []).append((secs, ended))
    return out


def censored_median(observations) -> tuple:
    """Kaplan-Meier median of window lengths: (seconds, is_lower_bound).

    `observations` are (seconds, ended_by_visitor) pairs. A window the visitor
    ended is an observed length; one that hit its time limit only says the
    visitor wanted at least that long.

    Dropping the limit-hit windows was the earlier treatment, and it was safe
    only while few windows had a limit. Once every window gets one at the
    start of a tour, the windows that hit it are the long ones, for exactly
    the projects visitors want more of. Dropping them biases those projects'
    medians DOWN — on the first allocated live runs, Navel's median came out
    39s with them dropped against 57s here — and a lower median gives that
    project less time next tour, so it hits the limit sooner. Kaplan-Meier
    keeps them in the risk set up to the moment they were cut off, which is
    all they are evidence of.

    The median is the first observed length at which the estimated share of
    visitors still wanting more falls to one half or below; exactly one half
    takes the midpoint with the next observed length, so uncensored data gives
    the ordinary median. If it never falls that far — the longest windows were
    all cut off — the median is only known to be at least the longest recorded
    window, which is returned with is_lower_bound=True.
    """
    obs = sorted((float(t), bool(e)) for t, e in observations)
    if not obs:
        return None, False
    at_risk, surviving, i = len(obs), 1.0, 0
    while i < len(obs):
        t = obs[i][0]
        ended = cut = 0
        while i < len(obs) and obs[i][0] == t:
            if obs[i][1]:
                ended += 1
            else:
                cut += 1
            i += 1
        if ended:
            surviving *= 1.0 - ended / at_risk
            if abs(surviving - 0.5) < 1e-12:
                nxt = next((u for u, e in obs[i:] if e), None)
                return ((t + nxt) / 2 if nxt is not None else t), nxt is None
            if surviving < 0.5:
                return t, False
        at_risk -= ended + cut
    return obs[-1][0], True


MIN_BLOCK_WINDOWS = 5
"""Visitor-ended windows for ONE block before its own median is used.

Counted on windows the visitor ended, not all windows: a pile of windows cut
off after a few seconds says almost nothing past those few seconds, and must
not make a thin estimate look trusted. Lower than MIN_WINDOWS_TO_TRUST
because the question is easier: a per-block figure only has to beat the
pooled one, not stand alone, and a block with no median of its own simply
takes the pool's flat share."""


def qa_median_by_block(rows=None) -> dict:
    """{block_robot_id: median window seconds} — the BASE that
    decision.planner.allocate_qa distributes on.

    Censored median (censored_median) over every window of that block, so
    windows that hit their limit count as "at least this long". Only blocks
    with MIN_BLOCK_WINDOWS visitor-ended windows appear; the rest are absent
    rather than defaulted, so the planner can tell "this block runs long" from
    "nothing is known about this block". Where the median is only a lower
    bound, the bound is used: it is still the best available base, and it
    errs toward giving the block more time, not less.
    """
    out = {}
    for block, obs in _windows_by_block(
            rows if rows is not None else qa_windows()).items():
        if sum(1 for _t, e in obs if e) < MIN_BLOCK_WINDOWS:
            continue
        median, _lower = censored_median(obs)
        if median is not None:
            out[block] = round(median, 1)
    return out


MIN_WINDOWS_TO_TRUST = 20
"""Visitor-ended windows before an observed budget replaces the constant.

Bootstrapped over the real corpus, the median's 95% interval runs 4.3-52.7s
at n=10, 9.7-42.6s at n=20, and 10.4-42.0s at n=30 — a third of the spread
bought between 10 and 20, and almost nothing after. The remaining width is
not sampling error: visitor Q&A genuinely runs from 2s to 165s, so no
threshold makes ONE pooled number good for any particular group."""


def suggested_qa_budget(default_sec: float = 90.0) -> dict:
    """
    A defensible default Q&A allocation, from observed windows.

    The censored median over every project window (see censored_median), so a
    long tail of one group that would not stop asking cannot drag it, and a
    window that hit its own limit counts as "at least this long" instead of
    as either its limit or nothing. Falls back to `default_sec` until there
    are MIN_WINDOWS_TO_TRUST windows the visitor ended, and says so.

    History worth keeping: half the first corpus was windows closed by their
    own limit, 19 of them at exactly 5.0s, and taking a plain median over
    everything returned 5.0s — the system's own constant. Treating those as
    censored at 5.0s makes them harmless: they say only "longer than five
    seconds". Only project windows count, and the median is over windows, not
    over per-step means.
    """
    obs = [o for block in _windows_by_block(qa_windows()).values() for o in block]
    ended = sum(1 for _t, e in obs if e)
    out = {"windows": ended, "self_terminated": len(obs) - ended}
    if ended < MIN_WINDOWS_TO_TRUST:
        return {**out, "budget_sec": default_sec, "basis": "default"}
    median, lower = censored_median(obs)
    return {**out, "budget_sec": round(median, 0),
            "basis": "lower_bound" if lower else "observed"}
