"""
tests/test_kg_infer.py
======================
Propagation: does a correction on one topic reach its neighbours?

The robot→topic edge type was chosen over robot→robot and context→action on the
grounds that it is the only one where a correction generalises. These tests pin
the properties that claim depends on, including the one that was broken on the
first attempt: propagation must be SYMMETRIC. Ported unchanged from CHATBOX's
preference model, the upward rule was a noisy-OR floor tuned for an unobserved
prior of 0.30, and against a competence prior of 0.5 it fired never — positive
corrections reached zero topics while negative ones reached 4.7.
"""

from __future__ import annotations

import pytest

from decision.kg import Evidence, PRIOR, RobotTopicEdge
from decision.kg_infer import infer, rank_robots

R = "robot_a"
OTHER = "robot_b"
A, B, C, FAR = "topic:a", "topic:b", "topic:c", "topic:far"

# a—b strong, b—c weaker, `far` unconnected.
LINKS = [(A, B, 0.8), (B, C, 0.5)]
TOPICS = [A, B, C, FAR]


def observed(topic, target, n=1, robot=R):
    e = RobotTopicEdge(robot_id=robot, topic_id=topic)
    for _ in range(n):
        e = e.update(target, Evidence.SUPERVISOR)
    return e


class TestSymmetry:

    def test_positive_and_negative_corrections_reach_equally_far(self):
        pos = infer([observed(A, 1.0, 5)], LINKS, R, TOPICS)
        neg = infer([observed(A, 0.0, 5)], LINKS, R, TOPICS)
        n_pos = sum(1 for t in TOPICS if t != A and abs(pos[t] - PRIOR) >= 0.01)
        n_neg = sum(1 for t in TOPICS if t != A and abs(neg[t] - PRIOR) >= 0.01)
        assert n_pos == n_neg > 0

    def test_the_shifts_mirror_each_other(self):
        pos = infer([observed(A, 1.0, 5)], LINKS, R, TOPICS)
        neg = infer([observed(A, 0.0, 5)], LINKS, R, TOPICS)
        assert pos[B] - PRIOR == pytest.approx(PRIOR - neg[B], abs=1e-9)

    def test_a_positive_correction_raises_its_neighbour(self):
        # The case the ported noisy-OR rule failed outright.
        p = infer([observed(A, 1.0, 3)], LINKS, R, TOPICS)
        assert p[B] > PRIOR

    def test_a_negative_correction_lowers_its_neighbour(self):
        p = infer([observed(A, 0.0, 3)], LINKS, R, TOPICS)
        assert p[B] < PRIOR


class TestReach:

    def test_an_unlinked_topic_is_never_reached(self):
        p = infer([observed(A, 1.0, 10)], LINKS, R, TOPICS)
        assert p[FAR] == pytest.approx(PRIOR)

    def test_a_stronger_link_carries_more(self):
        p = infer([observed(A, 1.0, 10)], LINKS, R, TOPICS)
        # a—b is 0.8, b—c is 0.5 and a hop further out.
        assert p[B] - PRIOR > p[C] - PRIOR > 0

    def test_reach_grows_with_evidence(self):
        shifts = [infer([observed(A, 1.0, n)], LINKS, R, TOPICS)[B] - PRIOR
                  for n in (1, 3, 10)]
        assert shifts == sorted(shifts)

    def test_a_single_correction_barely_moves_a_neighbour(self):
        # The guard that matters: one click must not teach the whole cluster.
        p = infer([observed(A, 1.0, 1)], LINKS, R, TOPICS)
        assert 0 < p[B] - PRIOR < 0.10

    def test_an_observed_topic_is_pinned(self):
        edges = [observed(A, 1.0, 10), observed(B, 0.0, 10)]
        p = infer(edges, LINKS, R, TOPICS)
        # B was observed as poor; A's strong positive must not drag it up.
        assert p[B] < PRIOR


class TestIsolation:

    def test_another_robots_evidence_does_not_leak(self):
        p = infer([observed(A, 1.0, 10, robot=OTHER)], LINKS, R, TOPICS)
        assert all(p[t] == pytest.approx(PRIOR) for t in TOPICS)

    def test_an_unobserved_edge_contributes_nothing(self):
        # n_obs == 0 → confidence 0 → clamped exactly neutral.
        p = infer([RobotTopicEdge(robot_id=R, topic_id=A, weight=1.0)],
                  LINKS, R, TOPICS)
        assert all(p[t] == pytest.approx(PRIOR) for t in TOPICS)

    def test_an_empty_graph_returns_nothing(self):
        assert infer([], [], R, []) == {}


class TestRanking:

    def test_the_better_robot_ranks_first(self):
        edges = [observed(A, 1.0, 8, robot=R), observed(A, 0.0, 8, robot=OTHER)]
        ranked = rank_robots(edges, LINKS, A, [R, OTHER])
        assert ranked[0][0] == R

    def test_ranking_uses_propagated_evidence(self):
        # Nobody has been observed on B; R is good at its strong neighbour A.
        edges = [observed(A, 1.0, 8, robot=R)]
        ranked = rank_robots(edges, LINKS, B, [R, OTHER])
        assert ranked[0][0] == R
        assert ranked[0][1] > ranked[1][1]

    def test_an_empty_graph_ranks_deterministically(self):
        ranked = rank_robots([], LINKS, A, [OTHER, R])
        assert [r for r, _ in ranked] == sorted([R, OTHER])
        assert all(s == pytest.approx(PRIOR) for _, s in ranked)


class TestNoWriteBack:
    """
    Propagation is DERIVED, never persisted, and the derivation is bounded.

    The rule this module ports was a monotone noisy-OR floor — idempotent for
    free, so re-running it could not compound. A proportional pull gives that up,
    which matters because the seeded vocabulary contains cycles
    (emotion-recognition → social-signals → human-robot-trust → emotion-
    recognition). If a propagation pass ever wrote its output back into
    neighbour edges, each pass around a cycle would add a little and the graph
    would ratchet upward with no evidence behind it.

    Two things keep that from happening and both are asserted here: nothing in
    this module writes, and the pull is a contraction toward pinned values, so
    its fixed point is bounded by the observed clamps no matter how many rounds
    run.
    """

    CYCLE = [("t:a", "t:b", 0.9), ("t:b", "t:c", 0.9), ("t:c", "t:a", 0.9)]
    NODES = ["t:a", "t:b", "t:c"]

    def _strong(self):
        e = RobotTopicEdge(robot_id=R, topic_id="t:a")
        for _ in range(10):
            e = e.update(1.0, Evidence.SUPERVISOR)
        return e

    def test_infer_is_idempotent_across_calls(self):
        e = self._strong()
        assert infer([e], self.CYCLE, R, self.NODES) == \
               infer([e], self.CYCLE, R, self.NODES)

    def test_a_cycle_never_exceeds_the_observed_clamp(self):
        e = self._strong()
        p = infer([e], self.CYCLE, R, self.NODES)
        assert max(p.values()) <= e.clamped + 1e-9

    def test_extra_rounds_converge_rather_than_compound(self):
        import decision.kg_infer as ki
        e = self._strong()
        original = ki.ROUNDS
        try:
            seen = []
            for rounds in (2, 8, 64, 256):
                ki.ROUNDS = rounds
                seen.append(max(infer([e], self.CYCLE, R, self.NODES).values()))
        finally:
            ki.ROUNDS = original
        assert seen == sorted(seen)                    # monotone
        assert seen[-1] <= e.clamped + 1e-9            # and bounded
        assert seen[-1] - seen[-2] < 1e-6              # converged

    def test_the_module_performs_no_writes(self):
        # A structural guard: an import of the repo here would be the first step
        # toward a write-back path, and this is where it would be noticed.
        import inspect
        import decision.kg_infer as ki
        src = inspect.getsource(ki)
        for forbidden in ("put_edge", "upsert", "apply_observation", "get_client"):
            assert forbidden not in src, f"kg_infer must not reference {forbidden}"


class TestClampIsLoadBearing:
    """
    Propagation reads the CONFIDENCE-CLAMPED weight, not the raw one, and that
    is the mechanism preventing a single correction from teaching a cluster.

    One correction takes the raw weight to 0.75 — a 0.25 departure from neutral.
    Clamped, it is 0.5625, a 0.0625 departure. Neighbours move proportionally to
    that departure, so reading the raw weight instead would quadruple the reach
    of every first correction. Anyone "simplifying" this later needs the test to
    say no.
    """

    def test_a_single_correction_propagates_from_the_clamped_value(self):
        e = observed(A, 1.0, 1)
        assert e.weight == pytest.approx(0.75)      # raw: a 0.25 departure
        assert e.clamped == pytest.approx(0.5625)   # clamped: 0.0625

        p = infer([e], LINKS, R, TOPICS)
        # A neighbour can never move further than the source it is pulled toward.
        # Reading the raw weight would put that ceiling four times higher.
        assert 0 < p[B] - PRIOR <= (e.clamped - PRIOR) + 1e-9

    def test_reading_the_raw_weight_would_reach_much_further(self):
        # Same single correction, but with enough observations that the clamp
        # stops biting — i.e. what propagation from the raw weight would look
        # like. The neighbour moves several times as far.
        one = observed(A, 1.0, 1)
        saturated = RobotTopicEdge(robot_id=R, topic_id=A,
                                   weight=one.weight, n_supervisor=200)
        hedged = infer([one], LINKS, R, TOPICS)[B] - PRIOR
        unhedged = infer([saturated], LINKS, R, TOPICS)[B] - PRIOR
        assert unhedged > 3 * hedged


class TestRoutingExploration:
    """
    Routing must not starve the graph.

    Pure argmax means the robot that starts marginally ahead on a topic takes
    every question on it and its alternatives are never observed — so the
    accumulated-supervision result would hold only for topics where a supervisor
    happened to intervene early. The exploration rules fire on ties and on
    ignorance, never against confident evidence.
    """

    def test_a_confidently_better_robot_still_wins(self):
        # Exploration must never knowingly give visitors a worse answer.
        edges = [observed(A, 1.0, 10, robot=R), observed(A, 0.0, 10, robot=OTHER)]
        from decision.kg_infer import route
        who, why = route(edges, LINKS, A, [R, OTHER])
        assert who == R and why == "argmax"

    def test_a_tie_goes_to_the_less_observed_robot(self):
        from decision.kg_infer import route
        # A genuine posterior tie needs matched CLAMPED values, not matched
        # targets: weight and confidence trade off, so one strong correction and
        # six weak ones land in the same place.
        # R: 1 correction at 1.0 -> 0.5625.  OTHER: 6 at 0.6 -> 0.5620.
        edges = [observed(A, 1.0, 1, robot=R), observed(A, 0.6, 6, robot=OTHER)]
        who, _why = route(edges, LINKS, A, [R, OTHER])
        assert who == R          # the less-observed of the two

    def test_a_tie_switches_away_from_the_more_observed_leader(self):
        from decision.kg_infer import route
        # The branch that actually has to move the pick: the MORE observed robot
        # is fractionally ahead, so argmax would keep feeding it and the other
        # would never be seen.
        edges = [observed(A, 0.55, 5, robot=R), observed(A, 0.6, 2, robot=OTHER)]
        who, why = route(edges, LINKS, A, [R, OTHER])
        assert who == OTHER and "explore" in why

    def test_a_barely_observed_runner_up_gets_a_turn(self):
        from decision.kg_infer import route
        # R is established and only slightly ahead; OTHER has never been tried.
        edges = [observed(A, 0.6, 6, robot=R)]
        who, why = route(edges, LINKS, A, [R, OTHER])
        assert who == OTHER and "explore" in why

    def test_exploration_yields_to_a_clearly_better_robot(self):
        from decision.kg_infer import route
        # R is demonstrably strong here. Spending turns on an unknown robot
        # would mean knowingly giving visitors worse answers.
        edges = [observed(A, 1.0, 10, robot=R)]
        who, why = route(edges, LINKS, A, [R, OTHER])
        assert who == R and why == "argmax"

    def test_exploration_can_be_switched_off(self):
        from decision.kg_infer import route
        edges = [observed(A, 0.8, 6, robot=R)]
        who, why = route(edges, LINKS, A, [R, OTHER], explore=False)
        assert who == R and why == "argmax"

    def test_an_empty_fleet_is_not_an_error(self):
        from decision.kg_infer import route
        assert route([], LINKS, A, [])[0] is None

    def test_routing_is_deterministic(self):
        from decision.kg_infer import route
        edges = [observed(A, 1.0, 4, robot=R)]
        picks = {route(edges, LINKS, A, [R, OTHER])[0] for _ in range(20)}
        assert len(picks) == 1


class TestColdStartExploration:
    """
    Exploration must actually fire, especially when the graph knows nothing.

    Both of these were broken and neither was caught by the earlier tests,
    because those all constructed a graph that already had evidence in it.
    Measured on a stub campaign, exploration fired 7 times in 128 decisions.

      1. Unreachable thresholds. The leader's clamped weight against an
         unobserved rival grows 0.0625 / 0.1417 / 0.2031 at one, two and three
         observations. With EXPLORE_MIN_OBS at 3 and EXPLORE_MAX_GAP at 0.20,
         the leader passed the gap cap before it reached the observation
         minimum, so the two conditions never held simultaneously.

      2. Cold ties. With every robot unobserved, every posterior is 0.5 and
         rank_robots breaks the tie lexicographically — so the alphabetically
         first robot took every question on every topic and the rest were never
         tried. That is the exact starvation exploration exists to prevent.
    """

    ROBOTS = ["r_alpha", "r_beta", "r_gamma"]

    def test_the_explore_windows_overlap(self):
        from decision.kg_infer import route
        # Two observations on the leader, none on the rivals: inside both the
        # gap cap and the observation minimum. Previously impossible.
        edges = [observed(A, 1.0, 2, robot="r_alpha")]
        who, why = route(edges, LINKS, A, self.ROBOTS)
        assert who != "r_alpha" and "explore" in why

    def test_a_cold_tie_spreads_by_total_observations(self):
        from decision.kg_infer import route
        # Nobody observed on THIS topic, but r_alpha is well known elsewhere.
        # The per-topic counts are all zero and carry no information; the global
        # count says which robot the graph knows least about.
        edges = [observed("topic:elsewhere", 1.0, 5, robot="r_alpha")]
        who, why = route(edges, LINKS, A, self.ROBOTS)
        assert who != "r_alpha" and "cold tie" in why

    def test_a_fully_cold_graph_is_deterministic(self):
        from decision.kg_infer import route
        # With zero information there is nothing to prefer, so this stays
        # deterministic rather than random — a rollout must be reproducible.
        picks = {route([], LINKS, A, self.ROBOTS)[0] for _ in range(10)}
        assert len(picks) == 1

    def test_a_confident_leader_still_ends_exploration(self):
        from decision.kg_infer import route
        edges = [observed(A, 1.0, 10, robot="r_alpha")]
        who, why = route(edges, LINKS, A, self.ROBOTS)
        assert who == "r_alpha" and why == "argmax"

    def test_exploration_engages_within_a_few_turns(self):
        # The property that matters end to end: repeatedly routing a topic must
        # not feed one robot forever.
        from decision.kg_infer import route
        from decision.kg import Evidence, RobotTopicEdge
        edges, picked = {}, []
        for _ in range(8):
            who, _why = route(list(edges.values()), LINKS, A, self.ROBOTS)
            picked.append(who)
            key = (who, A)
            e = edges.get(key) or RobotTopicEdge(robot_id=who, topic_id=A)
            edges[key] = e.update(1.0, Evidence.SUPERVISOR)
        assert len(set(picked)) > 1, f"one robot took every turn: {picked}"


class TestStemming:
    """
    The resolver's morphology, and the trap in it.

    A first attempt stripped -ation directly, which sent 'navigation' to 'navig'
    while 'navigate' only reached 'navigat' — further apart than before
    stemming. Letting -ion and -e apply in separate passes lands both on
    'navigat'. Resolution went 84% -> 86% -> 91% across those two attempts.
    """

    @pytest.mark.parametrize("a,b", [
        ("navigate", "navigation"),
        ("coordinate", "coordination"),
        ("conversation", "conversational"),
        ("map", "mapping"),
        ("model", "models"),
    ])
    def test_inflections_of_one_word_share_a_stem(self, a, b):
        from decision.kg_policy import _stem
        assert _stem(a) == _stem(b)

    @pytest.mark.parametrize("a,b", [
        ("emotion", "motion"),
        ("speech", "speaker"),
        ("trust", "truth"),
        ("signal", "single"),
        ("hardware", "harder"),
    ])
    def test_unrelated_words_do_not_collide(self, a, b):
        # Over-stemming invents matches, which is worse than missing one: it
        # writes an observation against a topic nobody asked about.
        from decision.kg_policy import _stem
        assert _stem(a) != _stem(b)

    def test_a_known_gap_stays_documented(self):
        """
        'localise' reaches 'localis' but 'localisation' stops at 'localisat',
        because -ation is deliberately absent from the suffix list and -at is
        not a suffix. Adding it back would re-break navigate/navigation, which
        cost more.

        No corpus question hits this: the topic label already contains
        'localisation' and so does the question that targets it. Recorded so the
        next person to touch _SUFFIXES knows it is a known trade rather than an
        oversight.
        """
        from decision.kg_policy import _stem
        assert _stem("localise") != _stem("localisation")

    def test_stemming_is_bounded(self):
        from decision.kg_policy import _stem
        for word in ("ss", "sss", "ies", "es", "s", "e", "aaa"):
            assert len(_stem(word)) >= 1
