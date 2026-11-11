"""
Role + persona layer — what the robot is FOR, how it talks, and what it knows for it.

  robot --has_role-----> Role("tutor", brief)                   (attached role)
  Role  --role_knows---> RoleTopic(kind, items)                  (how to talk in it)
                           kind: goal | manner | do | avoid | phrase | knowledge
  robot --has_persona--> Persona(descriptor, ocean)              (assigned persona)

Library roles (`role:<slug>`) all live in the graph; a robot PLAYS the one its
has_role edge points at. Attach / detach = moving that edge, the same way a culture
lens attaches to whoever is in front of the camera: the other roles and their
knowledge stay in the graph, unattached, and contribute nothing to the prompt.

Persona is Big-Five (OCEAN, each in [-1, 1], 0 = population average) mapped
straight to interaction-style instructions — no PAD, no affect dynamics.

Design contract (same as cultures.py): imports ONLY schema.py + store.py (+ the pure
topics slug). No LLM, no PAD, nothing in modules/. Stateless functions only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .schema import (
    HasPersonaEdge, HasRoleEdge, PersonaNode, Provenance, RoleKnowsEdge, RoleNode,
    RolePart, RoleTopicNode,
)
from .store import GraphStore
from .topics import normalize_label

TRAITS = ("O", "C", "E", "A", "N")


def _prov(source: Optional[str]) -> Provenance:
    return Provenance(source=source or "roles", confidence=1.0,
                      timestamp=datetime.now(timezone.utc))


# ── ids ───────────────────────────────────────────────────────────────────────

def role_id(label: str) -> str:
    return f"role:{normalize_label(label)}"


def role_topic_id(rid: str, label: str) -> str:
    return f"rk:{rid.split(':', 1)[-1]}:{normalize_label(label)}"


def persona_id(robot_id: str) -> str:
    return f"{robot_id}:persona:assigned"


# ── library (seeded) ──────────────────────────────────────────────────────────

def ensure_role(store: GraphStore, label: str, brief: str = "") -> RoleNode:
    """Upsert a library role. Re-seeding refreshes the brief."""
    node = RoleNode(id=role_id(label), descriptor=str(label).strip(),
                    brief=brief.strip())
    store.upsert_node(node)
    return node


def ensure_role_part(store: GraphStore, rid: str, kind: str, items: List[str],
                     label: Optional[str] = None) -> RoleTopicNode:
    """Upsert one part of a role's subgraph and link it (role_knows). Speaking
    parts (goal/manner/do/avoid/phrase) are one node each, labelled by kind; a
    KNOWLEDGE part is one node per topic, labelled by the topic."""
    kind = RolePart(kind)
    label = str(label or kind.value).strip()
    node = RoleTopicNode(id=role_topic_id(rid, label), label=label, kind=kind,
                         items=[str(i).strip() for i in items if str(i).strip()])
    store.upsert_node(node)
    store.upsert_edge(RoleKnowsEdge(source_id=rid, target_id=node.id,
                                    provenance=_prov("role-seed")))
    return node


def library_roles(store: GraphStore) -> List[RoleNode]:
    return sorted((n for n in list(store._nodes.values())   # noqa: SLF001
                   if n.node_type == "role" and n.id.startswith("role:")),
                  key=lambda n: n.id)


def role_parts(store: GraphStore, rid: str) -> Dict[str, List[Tuple[str, List[str]]]]:
    """{kind: [(label, items), ...]} for one role, labels sorted within a kind."""
    out: Dict[str, List[Tuple[str, List[str]]]] = {}
    for _e, n in store.query_neighbors(rid, "role_knows"):
        if n.node_type == "role_topic":
            out.setdefault(n.kind.value, []).append((n.label, list(n.items)))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def role_lines(store: GraphStore, rid: str, kind: str) -> List[str]:
    """All items of one speaking part, flattened (e.g. every 'avoid' rule)."""
    return [i for _l, items in role_parts(store, rid).get(kind, []) for i in items]


# ── attach / detach ───────────────────────────────────────────────────────────

def _detach(store: GraphStore, robot_id: str, edge_type: str) -> None:
    for _e, n in store.query_neighbors(robot_id, edge_type):
        store.delete_edge(robot_id, n.id, edge_type)


def attached_role(store: GraphStore, robot_id: str) -> Optional[RoleNode]:
    roles = sorted((n for _e, n in store.query_neighbors(robot_id, "has_role")
                    if n.node_type == "role"), key=lambda n: n.id)
    return roles[0] if roles else None


def attach_role(store: GraphStore, robot_id: str, rid: str,
                *, source: Optional[str] = None) -> RoleNode:
    """Detach whatever role the robot plays and attach `rid` (a library role)."""
    node = store.get_node(rid)
    if node is None or node.node_type != "role":
        raise ValueError(f"unknown role {rid!r}")
    _detach(store, robot_id, "has_role")
    store.upsert_edge(HasRoleEdge(source_id=robot_id, target_id=rid,
                                  provenance=_prov(source)))
    return node


# ── persona: OCEAN -> words + interaction style ───────────────────────────────

# A trait within ±EDGE of the average says nothing distinctive: no word, no rule.
EDGE = 0.25

# (word below average, word above average)
_WORDS = {
    "O": ("practical", "curious"),
    "C": ("easygoing", "organised"),
    "E": ("quiet", "outgoing"),
    "A": ("frank", "warm"),
    "N": ("calm", "sensitive"),
}

# Written as things to DO, not adjectives to be: models apply a trait label loosely,
# but a behaviour ("ask one question at a time") is followed and can be checked
# against a transcript.
_STYLE = {
    "O": ("Stick to concrete, everyday examples; don't wander into hypotheticals.",
          "Be imaginative: ask 'what if…' questions and make playful connections."),
    "C": ("Keep it loose and go with the flow; it's fine to follow a tangent.",
          "Be structured: one point or step at a time, and finish what you started."),
    "E": ("Keep replies short and low-key, let them lead, and don't push new topics.",
          "Bring energy: sound enthusiastic, share your own ideas, suggest what to do next."),
    "A": ("Be direct and honest; you may politely disagree or point out a mistake.",
          "Be encouraging: acknowledge what they said before adding anything, and praise effort."),
    "N": ("Stay steady and reassuring, even when something goes wrong.",
          "Show gentle caution: check they're okay and that things make sense before moving on."),
}


def clean_ocean(ocean: Dict) -> Dict[str, float]:
    """Validate an OCEAN dict → each trait a float clamped to [-1, 1]."""
    try:
        return {k: max(-1.0, min(1.0, float(ocean[k]))) for k in TRAITS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"ocean needs numeric {'/'.join(TRAITS)}") from exc


def ocean_words(ocean: Dict[str, float]) -> str:
    words = [_WORDS[k][ocean[k] > 0] for k in TRAITS if abs(ocean[k]) >= EDGE]
    return ", ".join(words) or "even-tempered"


def ocean_style(ocean: Dict[str, float]) -> List[str]:
    """Interaction-style instructions for the prompt, strongest trait first."""
    ranked = sorted((k for k in TRAITS if abs(ocean[k]) >= EDGE),
                    key=lambda k: -abs(ocean[k]))
    return [_STYLE[k][ocean[k] > 0] for k in ranked]


def assign_persona(store: GraphStore, robot_id: str, ocean: Dict,
                   *, source: Optional[str] = None) -> PersonaNode:
    """Detach the robot's persona and attach one built from `ocean`."""
    ocean = clean_ocean(ocean)
    node = PersonaNode(id=persona_id(robot_id), descriptor=ocean_words(ocean),
                       ocean=ocean)
    _detach(store, robot_id, "has_persona")
    store.upsert_node(node)
    store.upsert_edge(HasPersonaEdge(source_id=robot_id, target_id=node.id,
                                     provenance=_prov(source)))
    return node


def attached_persona(store: GraphStore, robot_id: str) -> Optional[PersonaNode]:
    ps = sorted((n for _e, n in store.query_neighbors(robot_id, "has_persona")
                 if n.node_type == "persona"), key=lambda n: n.id)
    return ps[0] if ps else None


def apply_assignment(store: GraphStore, robot_id: str, rid: str, ocean: Dict,
                     *, source: Optional[str] = None) -> Tuple[RoleNode, PersonaNode]:
    """Attach role `rid` and the OCEAN persona to the robot (one assignment)."""
    return (attach_role(store, robot_id, rid, source=source),
            assign_persona(store, robot_id, ocean, source=source))


# ── prompt rendering ───────────────────────────────────────────────────────────

def role_persona_blocks(store: GraphStore, role: Optional[RoleNode],
                        persona: Optional[PersonaNode]) -> List[str]:
    """HOW YOU TALK + WHAT YOU KNOW, from the robot's ATTACHED role subgraph and
    OCEAN persona only — nothing here depends on who the robot is talking to.
    Placed after the HOW TO REPLY rules so it colours HOW the robot replies,
    never WHAT it may claim. Shared by the webcam loop and rolesim's B0 robot."""
    parts = role_parts(store, role.id) if role else {}
    lines = lambda kind: role_lines(store, role.id, kind) if role else []
    style = (ocean_style(persona.ocean) if persona and persona.ocean else []) \
        + lines("manner") + lines("do")
    talk = []
    if lines("goal"):
        talk.append("Goal in every conversation: " + " ".join(lines("goal")))
    if style:
        talk.append("Style — keep to it every turn:\n"
                    + "\n".join(f"  – {x}" for x in style))
    if lines("avoid"):
        talk.append("Avoid:\n" + "\n".join(f"  – {x}" for x in lines("avoid")))
    if lines("phrase"):
        talk.append("Lines in your voice (for tone only — never repeat them word "
                    "for word):\n" + "\n".join(f'  – "{x}"' for x in lines("phrase")))
    out = []
    if talk:
        who = role.descriptor if role else "yourself"
        out.append(f"━━━ HOW YOU TALK AS A {who.upper()} ━━━\n" + "\n".join(talk))
    # ponytail: whole knowledge subgraph injected; rank by the message (like the
    # culture offers) once a role carries more than a handful of topics.
    knowledge = parts.get("knowledge", [])
    if knowledge:
        out.append(
            f"━━━ WHAT YOU KNOW AS A {role.descriptor.upper()} ━━━\n" + "\n".join(
                f"  – {label}: {' '.join(items)}" if items else f"  – {label}"
                for label, items in knowledge))
    return out


def role_identity_block(store: GraphStore, robot_id: str, display_name: str,
                        role: Optional[RoleNode], persona: Optional[PersonaNode]) -> str:
    """The IDENTITY block: who the robot is, its job in the role, its personality and
    its skills. Shared by the webcam loop and the persona chat (which add no
    information about the person on top of it)."""
    from .topics import robot_capability
    cap = robot_capability(store, robot_id)
    role_word = role.descriptor if role else "friendly companion"
    ident = ["━━━ IDENTITY ━━━",
             f"You are {display_name}, a {role_word} robot chatting "
             f"with someone through a webcam."]
    if role and role.brief:
        ident.append(f"Your job in this role: {role.brief}.")
    if persona:
        ident.append(f"Personality: {persona.descriptor}.")
    if cap and cap.items:
        ident.append(f"You can: {', '.join(cap.items)}.")
    return "\n".join(ident)
