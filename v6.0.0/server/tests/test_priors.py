"""
tests/test_priors.py
====================
A sim-to-real prior (migration 017) stands in for a physical robot's own
evidence only until it has LIVE_MIN_OBS observations, counts for at most
SIM_PRIOR_CAP, and is never learned into or written back.
"""

from __future__ import annotations

from decision.kg import LIVE_MIN_OBS, SIM_PRIOR_CAP, RobotTopicEdge
from decision.style_fit import StyleFit


def _edge(**kw):
    return RobotTopicEdge.from_row({"robot_id": "navel_001", "topic_id": "t", **kw})


def test_n_matches_the_agreed_thresholds():
    assert LIVE_MIN_OBS == 3 and SIM_PRIOR_CAP == 3


def test_no_prior_means_the_edge_is_its_own():
    e = _edge(weight=0.5)
    assert e.for_routing() is e and e.for_routing().evidence == "live"


def test_a_prior_stands_in_with_capped_confidence():
    r = _edge(weight=0.5, prior_weight=0.9, prior_n=40).for_routing()
    assert r.evidence == "prior" and r.weight == 0.9
    assert r.n_obs == SIM_PRIOR_CAP                  # 40 sim observations count as 3
    assert r.confidence == 0.5                       # never more than half-believed


def test_enough_live_evidence_replaces_the_prior():
    e = _edge(weight=0.2, n_supervisor=2, n_outcome=1, prior_weight=0.9, prior_n=40)
    assert e.for_routing() is e and e.for_routing().weight == 0.2


def test_learning_never_writes_into_the_prior():
    e = _edge(weight=0.5, prior_weight=0.9, prior_n=40)
    learned = e.update(1.0)
    assert learned.prior_weight == 0.9 and learned.prior_n == 40
    assert learned.n_obs == 1                        # live count, not the prior's
    assert "prior_weight" not in learned.as_row() and "prior_n" not in learned.as_row()


def test_style_fit_follows_the_same_rule():
    fit = StyleFit.from_row({"robot_id": "navel_001", "style": "technical",
                             "weight": 0.5, "n_supervisor": 0,
                             "prior_weight": 0.8, "prior_n": 10})
    g = fit.for_generation()
    assert g.weight == 0.8 and g.n_obs == SIM_PRIOR_CAP
    rated = fit.record(1.0)
    assert rated.prior_weight == 0.8 and "prior_weight" not in rated.as_row()
    seasoned = StyleFit(robot_id="x", style="s", weight=0.3, n_supervisor=3,
                        prior_weight=0.8, prior_n=10)
    assert seasoned.for_generation() is seasoned
