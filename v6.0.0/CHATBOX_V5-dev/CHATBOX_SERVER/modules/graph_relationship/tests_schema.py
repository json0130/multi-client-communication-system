"""Model-level validation tests for the graph_relationship schema.

Graph-container behaviour (edge source validation, save/load round-trip) is
covered against the real InMemoryGraphStore in tests_store.py.
"""

import pytest
from pydantic import ValidationError

from .schema import MoodEdge, PreferenceEdge, Provenance, RapportEdge, Timescale


def _prov(confidence: float = 0.9) -> Provenance:
    return Provenance(source="robot:cat", confidence=confidence)


def test_edge_carries_type_and_provenance():
    edge = RapportEdge(source_id="r", target_id="p", provenance=_prov(0.8), weight=0.6)
    assert edge.edge_type == "rapport"
    assert edge.weight == 0.6
    assert edge.provenance.source == "robot:cat"
    assert edge.provenance.confidence == 0.8


def test_timescale_defaults():
    assert MoodEdge(source_id="p", target_id="p", provenance=_prov(), value=0.3).timescale == Timescale.FAST
    assert PreferenceEdge(source_id="p", target_id="t", provenance=_prov(), weight=0.7).timescale == Timescale.SLOW


@pytest.mark.parametrize("confidence", [1.5, -0.1])
def test_provenance_confidence_out_of_range_rejected(confidence):
    with pytest.raises(ValidationError):
        Provenance(source="robot:cat", confidence=confidence)


def test_weight_out_of_range_rejected():
    with pytest.raises(ValidationError):
        RapportEdge(source_id="a", target_id="b", provenance=_prov(), weight=1.5)
