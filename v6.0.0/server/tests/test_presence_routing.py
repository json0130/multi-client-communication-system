"""
tests/test_presence_routing.py
===============================
Robot position as a routing filter: decision/presence.py, the presence half
of decision/kg_infer.py::route, and ABSENT_ROBOT_POLICY in
decision/kg_policy.py.

The properties that matter here are the ones that are silent when wrong:

  * an absent robot must be unreachable, not merely outranked — a robot that
    walked away cannot answer no matter how good the graph says it is
  * unknown location must mean PRESENT, matching every other fallback in the
    router; failing closed would silently drop robots whose pose has not
    arrived yet, with no visible reason
  * a defer must not promise a station the group will never reach
  * a deferred question must write NOTHING — nobody answered and the absent
    robot was never judged, so any observation would be manufactured from an
    event that did not happen (see kg_feedback.Segment's silence rule)
  * in_conversation must never disagree with location unattributably — the
    only path that can set it independently logs its source
"""

from __future__ import annotations

import logging

import pytest

from decision.kg import Evidence, RobotTopicEdge
from decision.kg_infer import route
from decision.kg_feedback import Segment
from decision.kg_policy import KGRouter, RoutingDecision
from decision.presence import DEFAULT_PROXIMITY_M, Pose, PresenceTracker

R = "robot_a"
OTHER = "robot_b"
GUIDE = "pepper_01"
TOPIC = "topic:a"
TOPICS = [{"id": TOPIC, "label": "emotion recognition"}]
QUESTION = "how does emotion recognition work"


def observed(topic, target, n=1, robot=R):
    e = RobotTopicEdge(robot_id=robot, topic_id=topic)
    for _ in range(n):
        e = e.update(target, Evidence.SUPERVISOR)
    return e


# ── The filter itself ─────────────────────────────────────────────────────────

class TestAbsentFilter:

    def test_an_absent_robot_never_wins_even_with_a_perfect_weight(self):
        # The whole point: unreachable, not outranked. R is rated 1.0 with
        # real evidence behind it and still cannot be picked.
        edges = [
            observed(TOPIC, 1.0, 5, robot=R),
            observed(TOPIC, 0.0, 5, robot=OTHER),
        ]
        picked, _ = route(edges, [], TOPIC, [R, OTHER],
                          explore=False, absent=[R])
        assert picked == OTHER

    def test_the_only_candidate_being_absent_leaves_no_robots(self):
        edges = [observed(TOPIC, 1.0, 5, robot=R)]
        picked, reason = route(edges, [], TOPIC, [R], explore=False, absent=[R])
        assert picked is None
        assert reason == "no robots"

    def test_absence_and_ineligibility_compose(self):
        third = "robot_c"
        edges = [
            observed(TOPIC, 1.0, 5, robot=R),
            RobotTopicEdge(robot_id=OTHER, topic_id=TOPIC, weight=1.0,
                           n_supervisor=5, eligible=False),
            observed(TOPIC, 0.2, 5, robot=third),
        ]
        picked, _ = route(edges, [], TOPIC, [R, OTHER, third],
                          explore=False, absent=[R])
        assert picked == third      # R absent, OTHER ineligible

    def test_no_absent_set_changes_nothing(self):
        edges = [observed(TOPIC, 1.0, 5, robot=R),
                 observed(TOPIC, 0.0, 5, robot=OTHER)]
        assert route(edges, [], TOPIC, [R, OTHER], explore=False)[0] == R
        assert route(edges, [], TOPIC, [R, OTHER], explore=False, absent=[])[0] == R


# ── Deriving presence from location ───────────────────────────────────────────

class TestPresenceDerivation:

    def test_unknown_location_is_treated_as_present(self):
        # Fails OPEN — matches every other fallback in the router. A robot
        # standing right there whose pose has not arrived must not vanish.
        t = PresenceTracker()
        assert t.in_conversation(R, reference_robot_id=OTHER) is True
        assert t.absent([R, OTHER], OTHER) == set()

    def test_a_known_pose_with_an_unknown_reference_is_present(self):
        t = PresenceTracker()
        t.set_location(R, Pose(x=100.0, y=100.0))
        # Nothing says where the group is, so nothing says R is away from it.
        assert t.in_conversation(R, reference_robot_id=OTHER) is True

    def test_a_nearby_robot_is_in_conversation(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=1.0, y=1.0))
        assert t.in_conversation(R, reference_robot_id=OTHER) is True

    def test_a_distant_robot_is_absent(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=50.0, y=0.0))
        assert t.in_conversation(R, reference_robot_id=OTHER) is False
        assert t.absent([R, OTHER], OTHER) == {R}

    def test_the_boundary_is_inclusive(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=DEFAULT_PROXIMITY_M, y=0.0))
        assert t.in_conversation(R, reference_robot_id=OTHER) is True

    def test_different_frames_are_not_comparable_so_present(self):
        # 2m apart in `map` and 2m apart in `pepper/odom` are different
        # claims. Refusing to compare them is the honest answer, and the
        # honest answer degrades to present.
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0, frame_id="map"))
        t.set_location(R, Pose(x=50.0, y=0.0, frame_id="pepper/odom"))
        assert t.in_conversation(R, reference_robot_id=OTHER) is True

    def test_a_robot_is_always_present_relative_to_itself(self):
        t = PresenceTracker()
        t.set_location(R, Pose(x=10.0, y=10.0))
        assert t.in_conversation(R, reference_robot_id=R) is True

    def test_clearing_a_location_returns_it_to_unknown(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=50.0, y=0.0))
        assert t.in_conversation(R, OTHER) is False
        t.set_location(R, None)
        assert t.in_conversation(R, OTHER) is True


class TestOverrideIsAlwaysAttributable:
    """in_conversation is derived from location. The ONE path that can make
    them disagree logs its source, so "why was this robot excluded" is always
    answerable from the log without guessing."""

    def test_an_override_beats_a_derived_absence(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=50.0, y=0.0))
        assert t.in_conversation(R, OTHER) is False
        t.set_in_conversation(R, True, source="harness")
        assert t.in_conversation(R, OTHER) is True

    def test_an_override_beats_a_derived_presence(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=0.5, y=0.0))
        t.set_in_conversation(R, False, source="harness")
        assert t.in_conversation(R, OTHER) is False

    def test_setting_an_override_is_logged_with_its_source(self, caplog):
        t = PresenceTracker()
        with caplog.at_level(logging.INFO, logger="decision.presence"):
            t.set_in_conversation(R, False, source="sim-run-7")
        assert any("sim-run-7" in rec.message and R in rec.message
                   for rec in caplog.records)

    def test_clearing_an_override_is_logged_too(self, caplog):
        t = PresenceTracker()
        t.set_in_conversation(R, False, source="sim")
        with caplog.at_level(logging.INFO, logger="decision.presence"):
            t.set_in_conversation(R, None, source="sim-cleanup")
        assert any("sim-cleanup" in rec.message for rec in caplog.records)

    def test_clearing_returns_the_robot_to_derivation(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=50.0, y=0.0))
        t.set_in_conversation(R, True, source="sim")
        assert t.in_conversation(R, OTHER) is True
        t.set_in_conversation(R, None, source="sim")
        assert t.in_conversation(R, OTHER) is False   # derived again

    def test_setting_a_location_is_not_an_override(self, caplog):
        # A pose is the normal path and must not look like a manual override
        # in the log, or the attribution the override log exists for is noise.
        t = PresenceTracker()
        with caplog.at_level(logging.INFO, logger="decision.presence"):
            t.set_location(R, Pose(x=1.0, y=1.0))
        assert not [rec for rec in caplog.records if "OVERRIDE" in rec.message]
        assert t.has_override(R) is False

    def test_snapshot_reports_which_source_decided(self):
        t = PresenceTracker()
        t.set_location(OTHER, Pose(x=0.0, y=0.0))
        t.set_location(R, Pose(x=1.0, y=0.0))
        t.set_in_conversation(OTHER, False, source="sim")
        snap = t.snapshot([R, OTHER, "robot_c"], reference_robot_id=OTHER)
        assert snap[R]["source"] == "derived"
        assert snap[OTHER]["source"] == "override"
        assert snap["robot_c"]["source"] == "unknown"


# ── ABSENT_ROBOT_POLICY ───────────────────────────────────────────────────────

class TestAbsentRobotPolicy:

    def _router(self, policy="defer", absent=()):
        # R is clearly the best robot for this topic; OTHER is clearly worse.
        edges = [observed(TOPIC, 1.0, 8, robot=R),
                 observed(TOPIC, 0.0, 8, robot=OTHER)]
        return KGRouter(edges, [], TOPICS, explore=False,
                        absent_robot_ids=absent, absent_policy=policy)

    def test_an_invalid_policy_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            KGRouter([], [], TOPICS, absent_policy="improvise")

    def test_the_default_policy_is_defer(self):
        from decision.kg_policy import ABSENT_ROBOT_POLICY
        assert ABSENT_ROBOT_POLICY == "defer"

    def test_nobody_absent_routes_normally(self):
        d = self._router().decide(QUESTION, [R, OTHER],
                                  remaining_block_ids={R, OTHER},
                                  guide_robot_id=GUIDE)
        assert d.robot_id == R
        assert d.is_deferred is False

    def test_defer_when_the_best_robots_block_is_still_ahead(self):
        d = self._router(absent=[R]).decide(
            QUESTION, [R, OTHER],
            remaining_block_ids={R, OTHER}, guide_robot_id=GUIDE)
        assert d.is_deferred is True
        assert d.deferred_to == R
        assert d.robot_id is None       # nobody answers
        assert "defer" in d.reason

    def test_defer_falls_back_to_guide_answers_when_the_block_was_cut(self):
        # PLAN_REVISE already dropped R's block, so promising the visitor
        # they will hear it at R's station promises something that will
        # never happen.
        d = self._router(absent=[R]).decide(
            QUESTION, [R, OTHER],
            remaining_block_ids={OTHER}, guide_robot_id=GUIDE)
        assert d.is_deferred is False
        assert d.robot_id == GUIDE
        assert "already cut" in d.reason

    def test_guide_answers_policy_never_defers(self):
        d = self._router(policy="guide_answers", absent=[R]).decide(
            QUESTION, [R, OTHER],
            remaining_block_ids={R, OTHER}, guide_robot_id=GUIDE)
        assert d.is_deferred is False
        assert d.robot_id == GUIDE

    def test_no_guide_to_fall_back_on_yields_no_opinion(self):
        # Degrades to "the receiver answers", the same as every other
        # unresolvable path in this module.
        d = self._router(absent=[R]).decide(
            QUESTION, [R, OTHER],
            remaining_block_ids={OTHER}, guide_robot_id=None)
        assert d is None

    def test_an_absent_runner_up_does_not_trigger_the_policy(self):
        # Only the robot that WOULD have won matters. OTHER being away
        # changes nothing, because R was going to answer anyway.
        d = self._router(absent=[OTHER]).decide(
            QUESTION, [R, OTHER],
            remaining_block_ids={R, OTHER}, guide_robot_id=GUIDE)
        assert d.robot_id == R
        assert d.is_deferred is False

    def test_absent_ids_passed_per_call_override_the_constructor(self):
        # The router is cached with a TTL and reused across turns while the
        # group moves, so a set frozen at construction would be stale.
        router = self._router(absent=[])
        d = router.decide(QUESTION, [R, OTHER], remaining_block_ids={R, OTHER},
                          guide_robot_id=GUIDE, absent_robot_ids=[R])
        assert d.is_deferred is True

    def test_unknown_remaining_plan_still_defers(self):
        # No plan information is not evidence the block was cut.
        d = self._router(absent=[R]).decide(
            QUESTION, [R, OTHER], remaining_block_ids=None,
            guide_robot_id=GUIDE)
        assert d.is_deferred is True


# ── The silence rule ──────────────────────────────────────────────────────────

class TestDeferredQuestionsWriteNothing:
    """A deferred question is not an observation about anybody: nobody
    answered, and the absent robot was never judged. Recording one would
    inflate n_obs from an event that did not happen — the exact failure
    kg_feedback.Segment's silence rule exists to prevent."""

    def test_a_deferred_turn_records_no_routing(self):
        segment = Segment()
        d = KGRouter([observed(TOPIC, 1.0, 8, robot=R)], [], TOPICS,
                     explore=False, absent_robot_ids=[R]).decide(
            QUESTION, [R, OTHER], remaining_block_ids={R, OTHER},
            guide_robot_id=GUIDE)
        assert d.is_deferred

        # This is what the gateway does NOT do on a defer.
        if not d.is_deferred:
            segment.note_routed(d.robot_id, d.topic_id)

        assert segment.observations() == []
        assert segment.answered_anything is False

    def test_a_deferred_turn_cannot_be_recorded_even_if_a_caller_tries(self):
        # Defence in depth: robot_id is None on a defer, and note_routed
        # already refuses a None robot. A caller that forgot the is_deferred
        # check still cannot manufacture an observation.
        segment = Segment()
        segment.note_routed(None, TOPIC)
        assert segment.observations() == []

    def test_a_normal_route_still_records(self):
        # The control: the silence rule must not be so eager it swallows
        # real outcomes.
        segment = Segment()
        d = KGRouter([observed(TOPIC, 1.0, 8, robot=R)], [], TOPICS,
                     explore=False).decide(
            QUESTION, [R, OTHER], remaining_block_ids={R, OTHER},
            guide_robot_id=GUIDE)
        assert not d.is_deferred
        segment.note_routed(d.robot_id, d.topic_id)
        assert len(segment.observations()) == 1

    def test_a_guide_answers_turn_is_a_real_outcome(self):
        # Distinct from defer: the guide genuinely answered a real question
        # about a real topic, so recording it is honest. The guide is
        # excluded from peer routing anyway, so this can never feed back
        # into who gets picked.
        segment = Segment()
        d = KGRouter([observed(TOPIC, 1.0, 8, robot=R)], [], TOPICS,
                     explore=False, absent_robot_ids=[R],
                     absent_policy="guide_answers").decide(
            QUESTION, [R, OTHER], remaining_block_ids={R, OTHER},
            guide_robot_id=GUIDE)
        assert d.robot_id == GUIDE and not d.is_deferred
        segment.note_routed(d.robot_id, d.topic_id)
        assert [o.robot_id for o in segment.observations()] == [GUIDE]
