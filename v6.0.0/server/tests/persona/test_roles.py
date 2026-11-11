"""Role/persona layer: attach/detach keeps one role, parts stay with their role,
OCEAN maps to words + style, junk rejected, old graphs still load."""

import pytest

from modules.graph_relationship.roles import (
    apply_assignment, attach_role, attached_persona, attached_role, ensure_role,
    ensure_role_part, library_roles, ocean_style, ocean_words, role_lines, role_parts,
)
from modules.graph_relationship.schema import Embodiment, HasRoleEdge, Provenance, RobotNode, RoleNode
from modules.graph_relationship.store import InMemoryGraphStore

SHY_TUTOR = {"O": 0.0, "C": 0.8, "E": -0.9, "A": 0.6, "N": 0.1}


@pytest.fixture
def store():
    s = InMemoryGraphStore()
    s.upsert_node(RobotNode(id="chatbox", name="ChatBox", embodiment=Embodiment.CAT))
    # what the spec seed leaves behind: an anchor-scoped role
    s.upsert_node(RoleNode(id="chatbox:role:companion", descriptor="companion"))
    s.upsert_edge(HasRoleEdge(source_id="chatbox", target_id="chatbox:role:companion",
                              provenance=Provenance(source="spec", confidence=1.0)))
    tutor = ensure_role(s, "tutor", "help with maths")
    ensure_role_part(s, tutor.id, "avoid", ["Giving the answer straight away."])
    ensure_role_part(s, tutor.id, "knowledge", ["One half is two quarters."], label="fractions")
    ensure_role(s, "storyteller")
    return s


def test_attach_detach_moves_one_edge(store):
    attach_role(store, "chatbox", "role:tutor")
    assert attached_role(store, "chatbox").id == "role:tutor"
    attach_role(store, "chatbox", "role:storyteller")
    assert [n.id for _e, n in store.query_neighbors("chatbox", "has_role")] == ["role:storyteller"]
    # detached role keeps its subgraph, ready to re-attach
    assert role_parts(store, "role:tutor") == {
        "avoid": [("avoid", ["Giving the answer straight away."])],
        "knowledge": [("fractions", ["One half is two quarters."])]}
    assert role_lines(store, "role:tutor", "avoid") == ["Giving the answer straight away."]
    assert store.get_node("chatbox:role:companion") is not None
    assert [r.id for r in library_roles(store)] == ["role:storyteller", "role:tutor"]


def test_persona_words_and_style(store):
    _role, persona = apply_assignment(store, "chatbox", "role:tutor", SHY_TUTOR)
    assert persona.descriptor == "organised, quiet, warm"
    assert attached_persona(store, "chatbox").ocean == SHY_TUTOR
    style = ocean_style(SHY_TUTOR)
    assert len(style) == 3 and "short and low-key" in style[0]     # E is strongest
    assert ocean_words({k: 0.0 for k in "OCEAN"}) == "even-tempered"
    apply_assignment(store, "chatbox", "role:tutor", {**SHY_TUTOR, "E": 0.9})
    assert len(store.query_neighbors("chatbox", "has_persona")) == 1


def test_unknown_part_rejected(store):
    with pytest.raises(ValueError):
        ensure_role_part(store, "role:tutor", "vibes", ["x"])


@pytest.mark.parametrize("rid,ocean", [("role:nope", SHY_TUTOR),
                                       ("role:tutor", {"O": 1}),
                                       ("role:tutor", {**SHY_TUTOR, "E": "loud"})])
def test_rejects_bad_input(store, rid, ocean):
    with pytest.raises(ValueError):
        apply_assignment(store, "chatbox", rid, ocean)


def test_round_trip(store, tmp_path):
    apply_assignment(store, "chatbox", "role:tutor", SHY_TUTOR)
    p = str(tmp_path / "kg.json")
    store.save(p)
    s2 = InMemoryGraphStore()
    s2.load(p)
    assert attached_role(s2, "chatbox").brief == "help with maths"
    assert role_parts(s2, "role:tutor")["knowledge"][0][0] == "fractions"
