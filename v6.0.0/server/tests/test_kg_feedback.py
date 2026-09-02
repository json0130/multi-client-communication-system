"""
tests/test_kg_feedback.py
=========================
Closing the loop: an operator's override becoming something the graph learns.

Before this path existed, corrections landed in demo_correction_log and stopped
there — a campaign of rollouts moved no weight at all. These tests pin the
translation, and especially the asymmetry that keeps one preference for B from
eroding A everywhere the two overlap.
"""

from __future__ import annotations

import pytest

from decision.kg import Evidence, PRIOR, RobotTopicEdge
from decision.kg_feedback import apply, from_reroute, from_segment_outcome

A, B = "chatbox_001", "navel_001"
T = "topic:rag"


class FakeRepo:
    """In-memory stand-in for demo_kg_repo — the injection point that lets this
    be tested without a database."""

    def __init__(self, fail_on=None):
        self.edges = {}
        self.fail_on = fail_on

    def apply_observation(self, robot_id, topic_id, target, kind):
        if self.fail_on == robot_id:
            return None
        key = (robot_id, topic_id)
        edge = self.edges.get(key) or RobotTopicEdge(robot_id=robot_id, topic_id=topic_id)
        edge = edge.update(target, kind)
        self.edges[key] = edge
        return edge


class TestReroute:

    def test_both_robots_are_written(self):
        obs = from_reroute(T, B, A)
        assert {o.robot_id for o in obs} == {A, B}

    def test_the_chosen_robot_gets_supervisor_evidence(self):
        chosen = next(o for o in from_reroute(T, B, A) if o.robot_id == B)
        assert chosen.evidence is Evidence.SUPERVISOR
        assert chosen.target == 1.0

    def test_the_displaced_robot_gets_the_weakest_evidence(self):
        displaced = next(o for o in from_reroute(T, B, A) if o.robot_id == A)
        assert displaced.evidence is Evidence.DISPLACED
        assert displaced.target == 0.0

    def test_being_displaced_costs_far_less_than_being_chosen_gains(self):
        # The asymmetry that matters: an operator wanting B is not a judgement
        # that A is incompetent, so one reroute must not erode A the way it
        # promotes B.
        repo = FakeRepo()
        apply(from_reroute(T, B, A), repo)
        gain = repo.edges[(B, T)].weight - PRIOR
        loss = PRIOR - repo.edges[(A, T)].weight
        assert gain > 5 * loss

    def test_repeated_displacement_still_erodes_eventually(self):
        # Weak must not mean inert — twenty reroutes away from A is a signal.
        repo = FakeRepo()
        for _ in range(20):
            apply(from_reroute(T, B, A), repo)
        assert repo.edges[(A, T)].weight < 0.3

    def test_a_self_reroute_is_only_positive(self):
        # Re-confirming the robot that already had the question says nothing
        # against anyone.
        obs = from_reroute(T, A, A)
        assert len(obs) == 1 and obs[0].evidence is Evidence.SUPERVISOR

    def test_no_displaced_robot_is_fine(self):
        assert len(from_reroute(T, B, None)) == 1

    def test_an_unresolved_topic_produces_nothing(self):
        # Filing an observation against the wrong topic is worse than filing
        # none: afterwards it is indistinguishable from a real one.
        assert from_reroute(None, B, A) == []
        assert from_reroute("", B, A) == []

    def test_a_missing_chosen_robot_produces_nothing(self):
        assert from_reroute(T, "", A) == []


class TestSegmentOutcome:

    def test_an_uncorrected_segment_is_weak_positive_evidence(self):
        obs = from_segment_outcome([T], A, corrected=False)
        assert len(obs) == 1 and obs[0].evidence is Evidence.OUTCOME

    def test_a_corrected_segment_produces_nothing(self):
        # The correction already describes the event. Recording an outcome too
        # would double-count it, inflating n_obs and with it the confidence the
        # read-time clamp grants the edge.
        assert from_segment_outcome([T], A, corrected=True) == []

    def test_every_topic_touched_is_credited(self):
        obs = from_segment_outcome([T, "topic:llm"], A, corrected=False)
        assert len(obs) == 2

    def test_an_outcome_moves_an_edge_far_less_than_a_correction(self):
        by_outcome, by_correction = FakeRepo(), FakeRepo()
        apply(from_segment_outcome([T], A, corrected=False), by_outcome)
        apply(from_reroute(T, A), by_correction)
        assert (by_correction.edges[(A, T)].weight
                > by_outcome.edges[(A, T)].weight)


class TestProvenanceDecomposition:

    def test_counts_stay_separable_by_source(self):
        repo = FakeRepo()
        apply(from_reroute(T, B, A), repo)
        apply(from_segment_outcome([T], B, corrected=False), repo)
        b = repo.edges[(B, T)]
        assert (b.n_supervisor, b.n_outcome, b.n_displaced) == (1, 1, 0)
        assert repo.edges[(A, T)].n_displaced == 1

    def test_a_displacement_counts_as_human_evidence(self):
        # The operator did act — they acted by choosing someone else. Excluding
        # displacements would understate how much of the graph a person shaped.
        repo = FakeRepo()
        apply(from_reroute(T, B, A), repo)
        assert repo.edges[(A, T)].human_share == 1.0


class TestApply:

    def test_one_unwritable_edge_does_not_discard_the_rest(self):
        repo = FakeRepo(fail_on=A)
        applied = apply(from_reroute(T, B, A), repo)
        assert [e.robot_id for e in applied] == [B]

    def test_applying_nothing_is_not_an_error(self):
        assert apply([], FakeRepo()) == []


class TestSilenceRule:
    """
    A Q&A window where nobody asked anything must teach the graph nothing.

    It closes exactly like a window where three questions were answered well,
    and "no correction fired" is true of both. Crediting the silent one records
    that a robot handled questions during a period when it handled zero.

    The weight is not the problem — n_obs is. That count drives the confidence
    clamp, the mechanism that stops one correction from teaching a cluster. Built
    from silence, the clamp grants confidence to numbers with nothing behind
    them, and the hollow rows are indistinguishable from real ones afterwards.
    """

    def test_a_silent_window_emits_nothing(self):
        from decision.kg_feedback import Segment
        assert Segment().observations() == []

    def test_a_window_with_one_question_emits_one_observation(self):
        from decision.kg_feedback import Segment
        seg = Segment()
        seg.note_routed(A, T)
        assert len(seg.observations()) == 1

    def test_a_corrected_window_emits_nothing(self):
        from decision.kg_feedback import Segment
        seg = Segment()
        seg.note_routed(A, T)
        seg.note_correction()
        assert seg.observations() == []

    def test_one_segment_credits_an_edge_once(self):
        # Three questions on the same topic to the same robot is one segment's
        # worth of evidence, not three. Otherwise a chatty window inflates
        # confidence faster than a quiet one for the same quality of routing.
        from decision.kg_feedback import Segment
        seg = Segment()
        for _ in range(3):
            seg.note_routed(A, T)
        assert len(seg.observations()) == 1

    def test_distinct_edges_are_each_credited(self):
        from decision.kg_feedback import Segment
        seg = Segment()
        seg.note_routed(A, T)
        seg.note_routed(B, "topic:llm")
        assert len(seg.observations()) == 2

    def test_a_turn_whose_topic_never_resolved_is_not_recorded(self):
        # A real question, but no edge it belongs to. Attributing it to a guess
        # is worse than dropping it.
        from decision.kg_feedback import Segment
        seg = Segment()
        seg.note_routed(A, None)
        seg.note_routed(None, T)
        assert seg.observations() == []
        assert not seg.answered_anything

    def test_reset_clears_both_routing_and_correction_state(self):
        from decision.kg_feedback import Segment
        seg = Segment()
        seg.note_routed(A, T)
        seg.note_correction()
        seg.reset()
        assert seg.routed == [] and seg.corrected is False

    def test_the_observations_are_weak_evidence(self):
        from decision.kg_feedback import Segment
        seg = Segment()
        seg.note_routed(A, T)
        assert seg.observations()[0].evidence is Evidence.OUTCOME
