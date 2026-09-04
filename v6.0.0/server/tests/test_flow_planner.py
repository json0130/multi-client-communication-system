"""
tests/test_flow_planner.py
==========================
The flow graph and the ladder that plans against it.

Two things under test. The graph lifts constraints out of prose — "PROJECT and
QA are deliberately absent from COMPRESSIBLE", "CLOSING survives a
DROP_REMAINING" — into structure that can be checked. The planner turns "the
visitor has 15 minutes" into an ordered set of cuts.

The ordering is the part that matters. Cutting by what recovers the most time
would drop a whole project first; cutting by what a visitor notices tightens Q&A
first and drops a project last.
"""

from __future__ import annotations

import pytest

from decision.flow import DEFAULT_STEP_SEC, FlowGraph, StepRef
from decision.models import PlanOpKind, StepRole
from decision.planner import QA_FLOOR_SEC, block_importance, plan_for_budget
from demo.demo_script import build_script

GUIDE = "pepper_01"
A, B, C = "chatbox_jetson_001", "navel_001", "silbot_01"


@pytest.fixture
def graph():
    return FlowGraph.from_script(build_script(GUIDE, [A, B, C]))


@pytest.fixture
def importance():
    return {A: 0.8, B: 0.5, C: 0.4}


# ── The graph ─────────────────────────────────────────────────────────────────

class TestFlowGraph:

    def test_blocks_are_derived_from_the_tags_build_script_already_sets(self, graph):
        assert [b.robot_id for b in graph.blocks] == [A, B, C]

    def test_opening_and_closing_are_separated_from_blocks(self, graph):
        assert {s.step_id for s in graph.closing} == {"wrap_up", "open_floor"}
        assert all(s.block_robot_id is None for s in graph.opening + graph.closing)

    def test_research_content_is_never_compressible(self, graph):
        # The constraint that previously lived only in a comment.
        for b in graph.blocks:
            for s in b.steps:
                if s.role in (StepRole.PROJECT, StepRole.QA):
                    assert not s.compressible

    def test_an_untagged_script_yields_no_blocks_rather_than_wrong_ones(self):
        from demo.demo_orchestrator import DemoStep
        g = FlowGraph.from_script([
            DemoStep(step_id="a", robot_id="r", text="x"),
            DemoStep(step_id="b", robot_id="r", text="y"),
        ])
        # Better to plan nothing than to plan against a guessed structure.
        assert g.blocks == ()
        assert len(g.opening) == 2

    def test_qa_is_excluded_from_scripted_time(self, graph):
        b = graph.block(A)
        # Q&A is allocated, not predicted, so it must not appear in the part of
        # the estimate that comes from measured content.
        assert b.scripted_seconds({}) == DEFAULT_STEP_SEC * len(
            [s for s in b.steps if s.role != StepRole.QA])

    def test_the_estimate_separates_scripted_from_allocated(self, graph):
        e = graph.estimate(qa_budget=90)
        assert e["qa_sec"] == 90 * 3
        assert e["total_sec"] == e["fixed_sec"] + e["scripted_sec"] + e["qa_sec"]

    def test_a_bigger_qa_budget_only_moves_the_qa_term(self, graph):
        a, b = graph.estimate(qa_budget=60), graph.estimate(qa_budget=120)
        assert a["scripted_sec"] == b["scripted_sec"]
        assert b["qa_sec"] > a["qa_sec"]

    def test_measured_coverage_reports_how_much_is_guessed(self, graph):
        assert graph.measured_coverage({}) == 0.0
        some = {s.step_id: 5.0 for s in graph.blocks[0].steps}
        assert 0.0 < graph.measured_coverage(some) < 1.0

    def test_measured_durations_are_used_over_the_default(self, graph):
        step = graph.blocks[0].steps[0]
        slow = graph.estimate({step.step_id: DEFAULT_STEP_SEC * 4}, qa_budget=0)
        base = graph.estimate({}, qa_budget=0)
        assert slow["total_sec"] > base["total_sec"]


# ── Importance ────────────────────────────────────────────────────────────────

class TestImportance:

    def test_defaults_stand_alone_when_no_interest_is_stated(self, graph):
        # The common case: most visitors say nothing, and everything must not
        # collapse to the same score.
        got = block_importance(graph, defaults={A: 0.9, B: 0.2})
        assert got[A] > got[C] > got[B]

    def test_an_unranked_block_is_neutral(self, graph):
        assert block_importance(graph)[A] == 0.5

    def test_a_stated_interest_raises_the_block_that_covers_it(self, graph):
        from decision.kg import Evidence, RobotTopicEdge
        e = RobotTopicEdge(robot_id=B, topic_id="topic:emotion-recognition")
        for _ in range(12):
            e = e.update(1.0, Evidence.SUPERVISOR)
        got = block_importance(graph, defaults={A: 0.5, B: 0.5, C: 0.5},
                               visitor_topics=["topic:emotion-recognition"],
                               kg_edges=[e], kg_links=[])
        assert got[B] > got[A]

    def test_a_hand_set_default_still_carries_weight(self, graph):
        # A project the lab considers unmissable stays hard to cut even for a
        # visitor who did not ask for it.
        from decision.kg import Evidence, RobotTopicEdge
        e = RobotTopicEdge(robot_id=B, topic_id="topic:x")
        for _ in range(12):
            e = e.update(1.0, Evidence.SUPERVISOR)
        got = block_importance(graph, defaults={A: 1.0, B: 0.0, C: 0.0},
                               visitor_topics=["topic:x"], kg_edges=[e], kg_links=[])
        assert got[A] > 0.3

    def test_the_learned_layer_is_absent_not_stubbed(self, graph):
        # Layer 3 contributes only when passed. A placeholder silently scoring
        # zero would be indistinguishable from a working one with no data.
        base = block_importance(graph, defaults={A: 0.8})
        with_learned = block_importance(graph, defaults={A: 0.8}, learned={A: 0.0})
        assert with_learned[A] < base[A]


# ── The ladder ────────────────────────────────────────────────────────────────

class TestLadder:

    def test_a_generous_budget_changes_nothing(self, graph, importance):
        r = plan_for_budget(graph, 900, importance=importance)
        assert r["ops"] == [] and r["fits_already"]

    def test_mild_pressure_only_tightens_qa(self, graph, importance):
        r = plan_for_budget(graph, 420, importance=importance)
        assert {o.kind for o in r["ops"]} == {PlanOpKind.SET_QA_BUDGET}
        assert r["feasible"]

    def test_qa_is_tightened_before_anything_is_compressed(self, graph, importance):
        r = plan_for_budget(graph, 300, importance=importance)
        kinds = [o.kind for o in r["ops"]]
        assert kinds.index(PlanOpKind.SET_QA_BUDGET) < kinds.index(PlanOpKind.COMPRESS)

    def test_qa_is_never_squeezed_below_the_floor(self, graph, importance):
        r = plan_for_budget(graph, 120, importance=importance)
        budgets = [o.seconds for o in r["ops"] if o.kind is PlanOpKind.SET_QA_BUDGET]
        assert budgets and min(budgets) >= QA_FLOOR_SEC

    def test_nothing_is_compressed_before_qa_reaches_the_floor(self, graph, importance):
        r = plan_for_budget(graph, 300, importance=importance)
        budgets = [o.seconds for o in r["ops"] if o.kind is PlanOpKind.SET_QA_BUDGET]
        assert min(budgets) == QA_FLOOR_SEC

    def test_the_least_important_block_is_compressed_first(self, graph, importance):
        r = plan_for_budget(graph, 300, importance=importance)
        order = [o.robot_id for o in r["ops"] if o.kind is PlanOpKind.COMPRESS]
        assert order[0] == C          # 0.4, the lowest

    def test_the_least_important_block_is_skipped_first(self, graph, importance):
        r = plan_for_budget(graph, 120, importance=importance)
        skipped = [o.robot_id for o in r["ops"] if o.kind is PlanOpKind.SKIP]
        assert skipped and skipped[0] == C

    def test_the_most_important_block_survives(self, graph, importance):
        r = plan_for_budget(graph, 120, importance=importance)
        skipped = {o.robot_id for o in r["ops"] if o.kind is PlanOpKind.SKIP}
        assert A not in skipped

    def test_one_project_always_remains(self, graph, importance):
        # A tour with no projects is not a short tour, it is a different event.
        r = plan_for_budget(graph, 30, importance=importance)
        skipped = {o.robot_id for o in r["ops"] if o.kind is PlanOpKind.SKIP}
        assert len(skipped) < len(graph.blocks)

    def test_an_impossible_budget_is_reported_not_papered_over(self, graph, importance):
        r = plan_for_budget(graph, 30, importance=importance)
        assert r["feasible"] is False
        assert "cannot be met" in r["trace"][-1]

    def test_importance_changes_which_block_is_cut(self, graph):
        low_a = plan_for_budget(graph, 120, importance={A: 0.1, B: 0.9, C: 0.9})
        low_c = plan_for_budget(graph, 120, importance={A: 0.9, B: 0.9, C: 0.1})
        first_a = [o.robot_id for o in low_a["ops"] if o.kind is PlanOpKind.SKIP][0]
        first_c = [o.robot_id for o in low_c["ops"] if o.kind is PlanOpKind.SKIP][0]
        assert first_a == A and first_c == C

    def test_the_plan_reports_how_much_of_it_is_guessed(self, graph, importance):
        r = plan_for_budget(graph, 300, importance=importance)
        # With no measured durations the total is arithmetic, not a prediction.
        assert r["measured_coverage"] == 0.0

    def test_the_trace_explains_every_cut(self, graph, importance):
        r = plan_for_budget(graph, 120, importance=importance)
        assert len(r["trace"]) == len([t for t in r["trace"] if t])
        assert any("skipped" in t for t in r["trace"])


class TestDropRemainingRung:
    """
    Rung 4 — reached only when even the last project, fully compressed, does
    not fit. The docstring always promised four rungs; the implementation only
    had three until this was added, so a genuinely impossible-but-not-quite
    budget (too tight for one project, loose enough for none) silently reported
    infeasible instead of proposing the one plan that actually fits.
    """

    def test_a_budget_that_only_fits_with_no_projects_drops_the_last_one(
        self, graph, importance):
        # 120s cannot hold even one compressed block + floor Q&A (129s, per the
        # existing ladder tests), but 60s of opening+closing alone fits easily.
        r = plan_for_budget(graph, 120, importance=importance)
        assert PlanOpKind.DROP_REMAINING in [o.kind for o in r["ops"]]
        assert r["feasible"] is True
        assert r["estimate"]["total_sec"] <= 120

    def test_drop_remaining_only_fires_after_the_ladder_is_exhausted(
        self, graph, importance):
        # A budget the ladder can already satisfy must never reach for this.
        r = plan_for_budget(graph, 300, importance=importance)
        assert PlanOpKind.DROP_REMAINING not in [o.kind for o in r["ops"]]

    def test_a_budget_too_small_even_for_the_closing_stays_infeasible(
        self, graph, importance):
        # Opening + closing alone take ~60s; nothing can be done about that.
        r = plan_for_budget(graph, 30, importance=importance)
        assert r["feasible"] is False
        assert "cannot be met" in r["trace"][-1]

    def test_drop_remaining_still_reports_what_it_could_not_fix(
        self, graph, importance):
        r = plan_for_budget(graph, 30, importance=importance)
        assert PlanOpKind.DROP_REMAINING in [o.kind for o in r["ops"]]
        assert r["feasible"] is False

    def test_the_estimate_after_dropping_is_fixed_time_only(self, graph, importance):
        r = plan_for_budget(graph, 120, importance=importance)
        assert r["estimate"]["scripted_sec"] == 0.0
        assert r["estimate"]["qa_sec"] == 0.0
        assert r["estimate"]["total_sec"] == r["estimate"]["fixed_sec"]
