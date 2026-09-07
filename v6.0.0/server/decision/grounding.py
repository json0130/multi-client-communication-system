"""
decision/grounding.py
=====================
Turning stored facts into the lines a robot is allowed to state.

Pure, like the rest of decision/: it formats rows someone else fetched. The
repository read lives in data/demo_facts_repo.py and the topic resolution in
decision/kg_policy.py, so this module can be unit-tested without a database
and replayed over a stored fact set.

WHY THE UNVERIFIED MARKER IS LOUD
A fact nobody has checked and a fact a researcher confirmed must not read the
same way in a prompt. The seeded rows are drafted from robots.robot_role and
start unverified, so without the marker the draft would be spoken with the
same confidence as reviewed content — which is the failure this whole
mechanism exists to remove, reintroduced one layer up.
"""

from __future__ import annotations

from typing import Iterable

# Prefix on a fact nobody has confirmed. Read by the prompt's own instruction
# ("anything marked UNVERIFIED is a draft"), so the two must stay in step —
# see robot/prompt_builder.py::build_delegation_prompt.
UNVERIFIED = "UNVERIFIED"


def format_facts(rows: Iterable[dict]) -> list[str]:
    """Fact rows -> prompt lines, most useful first.

    Ordering is the repository's; this only renders. A row missing its text
    is dropped rather than rendered blank — an empty bullet in a prompt reads
    as a fact the robot could not recall, which is worse than one fewer line.
    """
    out: list[str] = []
    for r in rows or ():
        text = (r.get("fact") or "").strip()
        if not text:
            continue
        kind = (r.get("kind") or "").strip()
        mark = "" if r.get("verified") else f" [{UNVERIFIED}]"
        scope = "" if r.get("robot_id") else " (background, not your own result)"
        out.append(f"{kind}: {text}{scope}{mark}")
    return out
