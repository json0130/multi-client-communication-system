import json

from modules.graph_relationship.viz.server import seed_robot_graph
from modules.navel_profile import ACTIONS, CONNECTION


def _nodes(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["nodes"]


def test_navel_graph_is_the_robot_and_its_topics_only(tmp_path):
    p = str(tmp_path / "kg_navel.json")
    seed_robot_graph(p, "navel")
    nodes = _nodes(p)
    assert not any(n["node_type"] == "capability" for n in nodes)        # no capability / gesture / connection nodes
    assert sum(n["node_type"] == "topic" for n in nodes) == 7            # its declared topics
    assert not any(n["node_type"] == "person" for n in nodes)            # nothing of the ChatBox's Jay graph
    assert len(ACTIONS) == 11 and "websocket" in CONNECTION              # still available as data, not nodes
    with open(p, encoding="utf-8") as fh:
        edges = json.load(fh)["edges"]
    assert sum(e["edge_type"] == "about" and e["source_id"] == "navel" for e in edges) == 7   # robot --about--> topic


def test_seeding_twice_changes_nothing_and_an_old_file_loses_its_capability_nodes(tmp_path):
    p = str(tmp_path / "kg_navel.json")
    seed_robot_graph(p, "navel"); n = len(_nodes(p))
    seed_robot_graph(p, "navel")
    assert len(_nodes(p)) == n
    from modules.graph_relationship.schema import CapabilityNode, HasCapabilityEdge, Provenance
    from modules.graph_relationship.store import InMemoryGraphStore
    old = InMemoryGraphStore(); old.load(p, quiet=True)                   # a Navel file from the capability-node version
    for cid in ("navel:capability", "navel:action:greet", "navel:connection:websocket"):
        old.upsert_node(CapabilityNode(id=cid, items=["x"]))
        old.upsert_edge(HasCapabilityEdge(source_id="navel", target_id=cid,
                                          provenance=Provenance(source="t", confidence=1.0)))
    old.save(p)
    assert any(x["node_type"] == "capability" for x in _nodes(p))
    seed_robot_graph(p, "navel")
    assert not any(x["node_type"] == "capability" for x in _nodes(p))
    assert sum(x["node_type"] == "topic" for x in _nodes(p)) == 7
