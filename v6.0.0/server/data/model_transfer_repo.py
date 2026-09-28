"""
data/model_transfer_repo.py
===========================
Reads and writes for sim-to-real model transfers (migration 017), used only by
tools/model_transfer.py.

Reads gather what a simulation twin has; writes apply one planned transfer.
Supabase has no multi-statement transaction over PostgREST, so apply_transfer
writes in an order that can be undone, and undoes it if any step fails: the
model_transfers row first (facts and duration priors cascade from it), then the
copies, and the target's previous scope rows are restored explicitly.
"""

from __future__ import annotations

from typing import Optional

from data.connection import get_client

NAV_PREFIXES = ("navigate_", "face_")
NAV_STEPS = ("approach_visitors",)


def _all(query, page: int = 1000) -> list:
    """Every row of a PostgREST query, paging past its row limit."""
    out, start = [], 0
    while True:
        rows = query.range(start, start + page - 1).execute().data or []
        out.extend(rows)
        if len(rows) < page:
            return out
        start += page


def source_snapshot(from_id: str, to_id: str) -> dict:
    """Everything a transfer from `from_id` to `to_id` reads, in one dict."""
    db = get_client()
    return {
        "facts": _all(db.table("demo_topic_facts").select(
            "id, topic_id, kind, fact, verified, source")
            .eq("robot_id", from_id).order("id")),
        "edges": _all(db.table("demo_robot_topic").select("*").eq("robot_id", from_id)),
        "styles": _all(db.table("demo_robot_style").select("*").eq("robot_id", from_id)),
        "step_rows": _all(db.table("demo_step_durations").select(
            "run_id, step_id, robot_id, block_robot_id, role, seconds")
            .eq("robot_id", from_id)),
        "qa_rows": _all(db.table("demo_qa_durations").select(
            "run_id, step_id, block_robot_id, seconds, closed_by")
            .eq("block_robot_id", from_id)),
        "runs": {r["id"]: r for r in _all(db.table("demo_runs").select("id, mode, format"))},
        "target_edges": _all(db.table("demo_robot_topic").select("*").eq("robot_id", to_id)),
        "target_transferred_facts": _all(db.table("demo_topic_facts").select("id")
            .eq("robot_id", to_id).not_.is_("transfer_id", "null")),
        "robots": [r["client_id"] for r in _all(db.table("robots").select("client_id"))],
    }


def next_version(from_id: str, to_id: str) -> int:
    rows = (get_client().table("model_transfers").select("version")
            .eq("from_robot_id", from_id).eq("to_robot_id", to_id)
            .order("version", desc=True).limit(1).execute().data or [])
    return (rows[0]["version"] + 1) if rows else 1


def apply_transfer(plan: dict, repo_record: str, notes: Optional[str]) -> int:
    """Write one planned transfer. Returns the model_transfers id.

    On any failure everything this call wrote is removed again and the
    target's scope rows are put back as they were; then the error re-raises.
    """
    db = get_client()
    row = db.table("model_transfers").insert({
        "version": plan["version"], "from_robot_id": plan["from"],
        "to_robot_id": plan["to"], "items": {**plan["items"], "status": "writing"},
        "notes": notes, "repo_record": repo_record,
    }).execute().data[0]
    tid = row["id"]
    try:
        if plan["facts"]:
            db.table("demo_topic_facts").insert(
                [{**f, "transfer_id": tid} for f in plan["facts"]]).execute()
        if plan["scope"]:
            # Only the listed columns are written, so a physical robot's own
            # live weight and counts on an existing edge are never touched.
            db.table("demo_robot_topic").upsert(
                [{**e, "transfer_id": tid} for e in plan["scope"]],
                on_conflict="robot_id,topic_id").execute()
        if plan["style_priors"]:
            db.table("demo_robot_style").upsert(
                [{**s, "transfer_id": tid} for s in plan["style_priors"]],
                on_conflict="robot_id,style").execute()
        if plan["duration_priors"]:
            db.table("duration_priors").insert(
                [{**p, "transfer_id": tid} for p in plan["duration_priors"]]).execute()
        # Only now, with this version's copies in place, do the previous
        # version's copies of facts give way. Facts written for the physical
        # robot directly (transfer_id NULL) always stay.
        (db.table("demo_topic_facts").delete()
           .eq("robot_id", plan["to"]).not_.is_("transfer_id", "null")
           .neq("transfer_id", tid).execute())
        db.table("model_transfers").update(
            {"items": {**plan["items"], "status": "done"}}).eq("id", tid).execute()
        return tid
    except Exception:
        _undo(db, plan, tid)
        raise


def _undo(db, plan: dict, tid: int) -> None:
    """Remove a half-written transfer and restore the target's scope."""
    try:
        before = {e["topic_id"]: e for e in plan.get("target_edges_before", [])}
        written = [e["topic_id"] for e in plan["scope"]]
        for topic in written:
            if topic in before:
                old = before[topic]
                (db.table("demo_robot_topic").update({
                    k: old.get(k) for k in ("eligible", "specialised", "prior_weight",
                                            "prior_n", "transfer_id")})
                   .eq("robot_id", plan["to"]).eq("topic_id", topic).execute())
            else:
                (db.table("demo_robot_topic").delete()
                   .eq("robot_id", plan["to"]).eq("topic_id", topic).execute())
        # Facts and duration priors carrying this id cascade with the row.
        db.table("model_transfers").delete().eq("id", tid).execute()
    except Exception as e:
        print(f"[model_transfer_repo] UNDO INCOMPLETE for transfer {tid}: {e} — "
              f"clean up by hand (017_sim_to_real_transfer_down.sql shows how).")
