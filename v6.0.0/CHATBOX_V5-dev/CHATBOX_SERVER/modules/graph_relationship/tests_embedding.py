"""
Tests for the embedding capability↔topic matcher, with an injected fake
embed_fn (no model server needed).
"""

from .embedding import capability_core, make_embedding_matcher

# Toy 4-d embedding space: math / jazz / space / (other) axes.
_VECS = {
    "math":      [1.0, 0.0, 0.0, 0.0],
    "addition":  [0.9, 0.1, 0.0, 0.0],   # ~ math
    "jazz":      [0.0, 1.0, 0.0, 0.0],
    "space":     [0.0, 0.0, 1.0, 0.0],
    "planets":   [0.0, 0.0, 0.9, 0.1],   # ~ space
    "dinosaurs": [0.0, 0.0, 0.0, 1.0],   # unrelated
}


def _fake_embed(text):
    return _VECS.get(text, [0.0, 0.0, 0.0, 1.0])


def test_capability_core_strips_filler():
    assert capability_core("good at math") == "math"
    assert capability_core("knows about space") == "space"
    assert capability_core("knows jazz") == "jazz"
    assert capability_core("tells stories") == "tells stories"


def test_embedding_matcher_selects_best_over_floor():
    m = make_embedding_matcher(_fake_embed, floor=0.5)
    items = ["good at math", "knows jazz", "knows about space"]
    assert m(items, "addition") == "good at math"       # near-match bridges
    assert m(items, "planets") == "knows about space"
    assert m(items, "jazz") == "knows jazz"              # exact
    assert m(items, "dinosaurs") is None                # below floor → no link


def test_embedding_matcher_empty_and_bad_embed():
    assert make_embedding_matcher(_fake_embed)([], "x") is None
    boom = make_embedding_matcher(lambda t: (_ for _ in ()).throw(RuntimeError()))
    assert boom(["good at math"], "addition") is None   # never raises
