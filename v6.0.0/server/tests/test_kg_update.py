"""
tests/test_kg_update.py
=======================
The update rule, and the properties it exists to guarantee.

Three of these answer questions a reviewer will ask directly:

  * does repeated conflicting supervision converge, or oscillate?
    (the overwrite rule fails this, which is why it was rejected)
  * does a single correction settle an edge, or stay hedged?
  * how much of the resulting graph came from humans?

The rest guard the arithmetic: self-capping without a clamp, a decaying rate,
and a confidence that starts at zero rather than at certainty.
"""

from __future__ import annotations

import pytest

from decision.kg import (
    BASE_LR,
    Evidence,
    PRIOR,
    RobotTopicEdge,
    TopicEdge,
    blend,
    clamp_from,
    learning_rate,
)

ROBOT = "chatbox_jetson_001"
TOPIC = "topic:rag"


def edge(**kw) -> RobotTopicEdge:
    return RobotTopicEdge(robot_id=ROBOT, topic_id=TOPIC, **kw)


# ── Convergence: the reason this is not an overwrite ──────────────────────────

class TestConvergence:

    def test_conflicting_supervision_converges_rather_than_oscillating(self):
        # Two supervisors who disagree every single time. Under overwrite the
        # weight would flip between 0 and 1 forever and the rollouts would buy
        # nothing. Here it must settle near the middle.
        e = edge()
        for i in range(40):
            e = e.update(1.0 if i % 2 == 0 else 0.0, Evidence.SUPERVISOR)
        assert 0.35 < e.weight < 0.65

    def test_the_swing_between_updates_shrinks(self):
        e = edge()
        swings = []
        for i in range(30):
            before = e.weight
            e = e.update(1.0 if i % 2 == 0 else 0.0, Evidence.SUPERVISOR)
            swings.append(abs(e.weight - before))
        # Late disagreement moves the edge far less than early disagreement.
        assert swings[-1] < swings[0] / 4

    def test_consistent_supervision_converges_toward_its_target(self):
        e = edge()
        for _ in range(30):
            e = e.update(1.0, Evidence.SUPERVISOR)
        assert e.weight > 0.9

    def test_consistent_negative_supervision_converges_down(self):
        e = edge()
        for _ in range(30):
            e = e.update(0.0, Evidence.SUPERVISOR)
        assert e.weight < 0.1

    def test_recent_evidence_outweighs_old_evidence(self):
        """
        This rule is recency-weighted, and that is a choice rather than an
        accident. Ten positives then ten negatives lands LOW; the reverse lands
        HIGH — same multiset, opposite conclusions.

        The alternative, a plain running mean, is order-independent but cannot
        track a robot whose competence actually changed: a platform that was
        retuned between demos would stay pinned by its old evidence forever.
        The cost is that the graph reports recent competence, not lifetime
        competence, and a long unbroken run of one-sided evidence dominates.
        """
        obs = [1.0] * 10 + [0.0] * 10
        rising, falling = edge(), edge()
        for t in obs:
            falling = falling.update(t, Evidence.SUPERVISOR)
        for t in reversed(obs):
            rising = rising.update(t, Evidence.SUPERVISOR)
        assert falling.weight < 0.4 < 0.6 < rising.weight

    def test_alternating_disagreement_still_converges(self):
        # Recency-weighting must not reintroduce oscillation when supervisors
        # disagree turn by turn, which is the realistic contested case.
        e = edge()
        tail = []
        for i in range(60):
            e = e.update(1.0 if i % 2 == 0 else 0.0, Evidence.SUPERVISOR)
            if i >= 50:
                tail.append(e.weight)
        assert max(tail) - min(tail) < 0.1


# ── Self-capping ──────────────────────────────────────────────────────────────

class TestSelfCapping:

    @pytest.mark.parametrize("target", [0.0, 0.25, 0.5, 1.0])
    @pytest.mark.parametrize("kind", list(Evidence))
    def test_weight_stays_in_range(self, target, kind):
        e = edge()
        for _ in range(50):
            e = e.update(target, kind)
            assert 0.0 <= e.weight <= 1.0

    def test_out_of_range_targets_are_clamped_not_rejected(self):
        # A caller passing 1.4 means "strongly yes". Dropping the observation
        # over it would discard a real label.
        e = edge().update(1.4, Evidence.SUPERVISOR)
        assert 0.0 <= e.weight <= 1.0
        assert e.n_supervisor == 1

    def test_an_extreme_target_cannot_reach_the_boundary_in_one_step(self):
        # The property that makes this not an overwrite.
        e = edge().update(1.0, Evidence.SUPERVISOR)
        assert e.weight < 1.0
        assert e.weight == pytest.approx(PRIOR + BASE_LR[Evidence.SUPERVISOR] * 0.5)


# ── Learning rate ─────────────────────────────────────────────────────────────

class TestLearningRate:

    def test_supervisor_outweighs_outcome(self):
        assert learning_rate(Evidence.SUPERVISOR, 0) > learning_rate(Evidence.OUTCOME, 0)

    def test_rate_decays_with_evidence(self):
        rates = [learning_rate(Evidence.SUPERVISOR, n) for n in (0, 5, 25, 100)]
        assert rates == sorted(rates, reverse=True)

    def test_rate_never_reaches_one(self):
        # At lr == 1 the update degenerates to overwrite.
        assert all(0 < learning_rate(k, n) < 1.0
                   for k in Evidence for n in (0, 1, 10, 1000))

    def test_an_outcome_barely_moves_a_well_evidenced_edge(self):
        e = edge(weight=0.9, n_supervisor=20)
        moved = e.update(0.0, Evidence.OUTCOME)
        assert abs(moved.weight - 0.9) < 0.03


# ── Confidence and the read-time clamp ────────────────────────────────────────

class TestConfidence:

    def test_a_fresh_edge_has_zero_confidence(self):
        assert edge().confidence == 0.0

    def test_a_fresh_edge_clamps_to_exactly_neutral(self):
        # However extreme its weight, an unobserved edge must not influence
        # anything — this is the single-observation guard.
        assert edge(weight=1.0).clamped == 0.5

    def test_one_observation_stays_heavily_hedged(self):
        e = edge().update(1.0, Evidence.SUPERVISOR)
        assert e.confidence == pytest.approx(0.25)
        assert e.clamped < 0.60

    def test_confidence_rises_with_observations(self):
        confs = []
        e = edge()
        for _ in range(20):
            e = e.update(1.0, Evidence.SUPERVISOR)
            confs.append(e.confidence)
        assert confs == sorted(confs)
        assert confs[-1] > 0.85

    def test_neutral_is_a_fixed_point_of_the_clamp(self):
        assert clamp_from(0.5, 0.0) == 0.5
        assert clamp_from(0.5, 1.0) == 0.5

    def test_full_confidence_returns_the_weight_unchanged(self):
        assert clamp_from(0.83, 1.0) == pytest.approx(0.83)

    def test_the_clamp_is_symmetric(self):
        assert clamp_from(0.9, 0.5) - 0.5 == pytest.approx(0.5 - clamp_from(0.1, 0.5))


# ── Provenance ────────────────────────────────────────────────────────────────

class TestProvenance:

    def test_counts_are_tracked_separately(self):
        e = edge()
        for _ in range(3):
            e = e.update(1.0, Evidence.SUPERVISOR)
        for _ in range(7):
            e = e.update(1.0, Evidence.OUTCOME)
        assert (e.n_supervisor, e.n_outcome, e.n_obs) == (3, 7, 10)

    def test_human_share_is_reportable(self):
        e = edge(n_supervisor=3, n_outcome=9)
        assert e.human_share == pytest.approx(0.25)

    def test_human_share_of_an_unobserved_edge_is_zero_not_an_error(self):
        assert edge().human_share == 0.0

    def test_an_update_never_half_applies(self):
        # Frozen dataclass: weight and counts move together or not at all.
        before = edge()
        after = before.update(1.0, Evidence.SUPERVISOR)
        assert before.n_obs == 0 and before.weight == PRIOR
        assert after.n_obs == 1 and after.weight != PRIOR


# ── Blending and topic edges ──────────────────────────────────────────────────

class TestBlend:

    def test_no_edges_falls_back_to_the_prior(self):
        assert blend([]) == PRIOR

    def test_unobserved_edges_fall_back_to_the_prior(self):
        assert blend([edge(weight=1.0), edge(weight=0.0)]) == PRIOR

    def test_a_well_evidenced_edge_dominates_a_fresh_one(self):
        strong = edge(weight=0.95, n_supervisor=30)
        weak = RobotTopicEdge(robot_id=ROBOT, topic_id="topic:other",
                              weight=0.05, n_supervisor=1)
        assert blend([strong, weak]) > 0.7


class TestTopicEdge:

    def test_endpoints_are_stored_sorted(self):
        e = TopicEdge("topic:z", "topic:a", 0.7)
        assert (e.topic_a, e.topic_b) == ("topic:a", "topic:z")

    def test_swapping_preserves_both_endpoints(self):
        # A naive in-place swap assigns topic_a then reads it back for topic_b,
        # collapsing the edge into a self-loop that still carries a plausible
        # weight. Every seeded link came out as `X ~ X` before this was caught.
        for a, b in [("topic:z", "topic:a"), ("topic:a", "topic:z")]:
            e = TopicEdge(a, b, 0.7)
            assert e.topic_a != e.topic_b
            assert {e.topic_a, e.topic_b} == {"topic:a", "topic:z"}

    def test_already_sorted_endpoints_are_untouched(self):
        e = TopicEdge("topic:a", "topic:z", 0.7)
        assert (e.topic_a, e.topic_b) == ("topic:a", "topic:z")

    def test_other_returns_the_far_end(self):
        e = TopicEdge("topic:a", "topic:b", 0.7)
        assert e.other("topic:a") == "topic:b"
        assert e.other("topic:b") == "topic:a"
        assert e.other("topic:c") is None


# ── Round-tripping ────────────────────────────────────────────────────────────

class TestSerialization:

    def test_edge_survives_a_round_trip(self):
        e = edge(weight=0.73, n_supervisor=4, n_outcome=2).update(1.0, Evidence.OUTCOME)
        back = RobotTopicEdge.from_row(e.as_row())
        assert back.weight == pytest.approx(e.weight, abs=1e-6)
        assert (back.n_supervisor, back.n_outcome) == (e.n_supervisor, e.n_outcome)

    def test_missing_counts_default_to_zero(self):
        back = RobotTopicEdge.from_row({"robot_id": ROBOT, "topic_id": TOPIC})
        assert back.n_obs == 0 and back.weight == PRIOR


class TestTimestampParsing:
    """
    Real Postgres timestamps, not hand-made ones.

    Every test above builds its own datetime, so all of them passed while
    reading a stored edge crashed: Postgres emits however many fractional-second
    digits it needs — five in practice — and Python 3.10's fromisoformat accepts
    only three or six. Caught by the first live round-trip, not by the suite.
    """

    @pytest.mark.parametrize("raw", [
        "2026-09-02T00:01:07.73877+00:00",     # 5 digits — what Postgres sent
        "2026-09-02T00:01:07.7+00:00",         # 1
        "2026-09-02T00:01:07.73+00:00",        # 2
        "2026-09-02T00:01:07.738+00:00",       # 3
        "2026-09-02T00:01:07.7387+00:00",      # 4
        "2026-09-02T00:01:07.738770+00:00",    # 6
        "2026-09-02T00:01:07+00:00",           # none
        "2026-09-02T00:01:07.73877Z",          # Z instead of an offset
    ])
    def test_every_fractional_precision_parses(self, raw):
        from decision.kg import _parse_ts
        parsed = _parse_ts(raw)
        assert parsed is not None
        assert (parsed.year, parsed.month, parsed.day) == (2026, 9, 2)

    def test_an_unparseable_timestamp_does_not_break_the_edge(self):
        # last_updated is metadata. Losing it must not make an edge unreadable.
        edge = RobotTopicEdge.from_row({
            "robot_id": ROBOT, "topic_id": TOPIC, "weight": 0.8,
            "n_supervisor": 3, "last_updated": "not a timestamp",
        })
        assert edge.weight == 0.8 and edge.n_supervisor == 3
        assert edge.last_updated is None

    def test_a_real_row_shape_round_trips(self):
        # The exact column set demo_kg_edges returns.
        edge = RobotTopicEdge.from_row({
            "robot_id": ROBOT, "topic_id": TOPIC, "weight": 0.9055,
            "n_supervisor": 3, "n_outcome": 0, "n_displaced": 0,
            "last_updated": "2026-09-02T00:01:07.73877+00:00",
        })
        assert edge.n_obs == 3
        assert edge.last_updated.tzinfo is not None
