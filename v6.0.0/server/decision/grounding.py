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
        # Reached by following a topic link rather than by being about the
        # subject asked. Named, because a fact about large language models
        # offered in answer to a question about retrieval is useful context
        # and a wrong answer to the question — the robot has to be able to
        # tell the difference, and so does the visitor.
        related = (r.get("related_topic") or "").strip()
        rel = f" (related topic: {related})" if related else ""
        out.append(f"{kind}: {text}{rel}{scope}{mark}")
    return out


NEIGHBOUR_MIN_WEIGHT = 0.5
"""How strongly two topics must be linked before one grounds an answer about
the other.

The seeded link graph spans 0.40 to 0.85. At 0.5 a question about retrieval
reaches large language models (0.80), conversational memory (0.70) and
knowledge graphs (0.65) — the neighbours a visitor would consider part of the
same answer — while robot hardware / text-to-speech (0.40) stays out."""

NEIGHBOUR_FACT_LIMIT = 3
"""How many related-topic facts may join an answer.

Small on purpose. The topic asked about supplies up to ten lines already, and
these are additive: the point is to reach a fact the neighbouring node holds,
not to hand the model a second topic's worth of material to drift into."""


def with_related(own: list[dict], related: list[tuple[str, list[dict]]],
                 limit: int = NEIGHBOUR_FACT_LIMIT) -> list[dict]:
    """Own-topic rows, then up to `limit` rows reached through topic links.

    `related` is [(topic_label, rows)] in descending link weight — strongest
    neighbour first, and one row taken from each before any neighbour offers
    a second, so a single well-connected topic cannot fill the whole
    allowance.

    Own rows are never displaced. They are what the visitor asked about; a
    neighbour's fact earns its place only in space the answer was not already
    using.
    """
    out = list(own)
    seen = {r.get("id") for r in own}
    queues = [(label, [r for r in rows if r.get("id") not in seen])
              for label, rows in related]
    taken = 0
    while taken < limit:
        moved = False
        for label, queue in queues:
            if not queue:
                continue
            row = queue.pop(0)
            if row.get("id") in seen:
                continue
            seen.add(row.get("id"))
            out.append({**row, "related_topic": label})
            taken += 1
            moved = True
            if taken >= limit:
                break
        if not moved:
            break
    return out
