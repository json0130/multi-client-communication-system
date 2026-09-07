"""
data/demo_style_repo.py
=======================
Persistence for robot→audience style fit.

Same shape as data/demo_kg_repo.py: read an existing row, hand it to the pure
updater in decision/style_fit.py, write the result back. No arithmetic here.

Read failures degrade to "nothing known", which makes generation fall back to
the plain style directive — exactly what it did before this table existed. A
style-fit lookup that cannot reach the database must never stop a robot
speaking.
"""

from __future__ import annotations

from typing import Optional

from data.connection import get_client
from decision.style_fit import StyleFit

TABLE = "demo_robot_style"
VIEW = "demo_style_fit"


def get_fit(robot_id: str, style: str) -> StyleFit:
    """The stored fit, or a fresh one at the prior. Never None — an absent row
    and an unobserved one mean the same thing, and returning None would push
    that decision onto every caller."""
    try:
        rows = (get_client().table(TABLE).select("*")
                .eq("robot_id", robot_id).eq("style", style)
                .limit(1).execute().data or [])
        if rows:
            return StyleFit.from_row(rows[0])
    except Exception as e:
        print(f"[demo_style_repo] get_fit error: {e}")
    return StyleFit(robot_id=robot_id, style=style)


def all_fits() -> list[dict]:
    """Every row with confidence/clamped precomputed by the view."""
    try:
        return (get_client().table(VIEW).select("*")
                .order("robot_id").execute().data or [])
    except Exception as e:
        print(f"[demo_style_repo] all_fits error: {e}")
        return []


def record(robot_id: str, style: str, target: float) -> Optional[StyleFit]:
    """Fold one operator judgement in and persist it.

    Swallows write failures for the same reason the decision sink does: a
    rating that cannot be stored must not take a live demo down with it.
    """
    try:
        updated = get_fit(robot_id, style).record(target)
        get_client().table(TABLE).upsert([updated.as_row()]).execute()
        return updated
    except Exception as e:
        print(f"[demo_style_repo] record failed for {robot_id}/{style}: {e}")
        return None


def snapshot() -> dict:
    """{(robot_id, style): StyleFit} for the whole table.

    Read once per generation window rather than per step — the generation
    path needs a lookup, not a round-trip for every utterance.
    """
    out: dict = {}
    try:
        rows = get_client().table(TABLE).select("*").execute().data or []
        for r in rows:
            fit = StyleFit.from_row(r)
            out[(fit.robot_id, fit.style)] = fit
    except Exception as e:
        print(f"[demo_style_repo] snapshot error: {e}")
    return out
