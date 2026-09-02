"""
tests/test_decision_policy.py
=============================
Behaviour parity, plus the new PLAN_REVISE rules.

The parity half is the important one. HeuristicPolicy is a move, not a rewrite:
the Q&A precedence chain that used to be spread across websocket_gateway and
http_gateway must produce exactly the same outcomes it did before, or a refactor
has quietly changed what visitors experience. The cases below are driven off the
real phrase lists rather than a sample of them, so adding a phrase to
QA_ADVANCE_PHRASES cannot pass without the rule still holding for it.

Assertions are on the mechanism as well as the action. A right answer from the
wrong rule is still a bug — the correction rate per mechanism is the number this
whole layer exists to produce.
"""

from __future__ import annotations

import pytest

from decision import (
    ActionKind,
    DecisionPoint,
    HeuristicPolicy,
    Mechanism,
    Observation,
    PlanOpKind,
    QA_ADVANCE_PHRASES,
    QA_CLOSING_PHRASES,
)

GUIDE = "pepper_01"
ROBOT_A = "chatbox_jetson_001"
ROBOT_B = "navel_001"

PEERS = (
    {"client_id": GUIDE, "robot_name": "Pepper", "robot_role": "guide",
     "access_level": "global"},
    {"client_id": ROBOT_A, "robot_name": "ChatBox", "robot_role": "RAG research",
     "access_level": "local"},
    {"client_id": ROBOT_B, "robot_name": "Navel", "robot_role": "emotion research",
     "access_level": "local"},
)


def visitor_turn(text: str, **kw) -> Observation:
    """An Observation for a visitor speaking during a Q&A window."""
    base = dict(
        step_id="qa_invite_chatbox_jetson_001",
        step_idx=6,
        total_steps=20,
        steps_remaining=14,
        demo_state="qa_window",
        last_speaker_id="visitor",
        user_utterance=text,
        connected_peers=PEERS,
        guide_robot_id=GUIDE,
        presenting_robot_id=ROBOT_A,
        decider_robot_id=ROBOT_A,
    )
    base.update(kw)
    return Observation(**base)


def robot_turn(text: str, **kw) -> Observation:
    """An Observation for a robot having just replied during a Q&A window."""
    base = dict(
        step_id="qa_invite_chatbox_jetson_001",
        step_idx=6,
        total_steps=20,
        demo_state="qa_window",
        last_speaker_id=ROBOT_A,
        last_robot_utterance=text,
        connected_peers=PEERS,
        guide_robot_id=GUIDE,
        presenting_robot_id=ROBOT_A,
        decider_robot_id=ROBOT_A,
    )
    base.update(kw)
    return Observation(**base)


def decide(obs, point=DecisionPoint.QA_ADVANCE, **kw):
    return HeuristicPolicy(**kw).decide(point, obs)


# ── Parity: the visitor-turn chain ────────────────────────────────────────────

class TestAdvancePhrasePrecedence:
    """Rule 1: an advance phrase closes the window, before anything else runs."""

    @pytest.mark.parametrize("phrase", QA_ADVANCE_PHRASES)
    def test_every_advance_phrase_advances(self, phrase):
        r = decide(visitor_turn(phrase))
        assert r.action.kind is ActionKind.ADVANCE
        assert r.mechanism == Mechanism.ADVANCE_PHRASE

    @pytest.mark.parametrize("phrase", QA_ADVANCE_PHRASES)
    def test_advance_phrase_matches_case_insensitively(self, phrase):
        r = decide(visitor_turn(f"Okay {phrase.upper()} please"))
        assert r.action.kind is ActionKind.ADVANCE

    def test_advance_phrase_beats_the_question_heuristic(self):
        # "can we continue?" is both an advance phrase and question-shaped.
        # The original checked phrases first; so must this.
        r = decide(visitor_turn("can we continue?"))
        assert r.mechanism == Mechanism.ADVANCE_PHRASE

    def test_advance_phrase_never_reaches_the_classifier(self):
        def boom(_):
            raise AssertionError("classifier must not run after a phrase match")
        r = decide(visitor_turn("move on"), intent_classifier=boom)
        assert r.action.kind is ActionKind.ADVANCE


class TestQuestionHeuristic:
    """Rule 2: a question keeps the window open without paying for an LLM call."""

    @pytest.mark.parametrize("text", [
        "What is RAG?",
        "how does the emotion model work",
        "why did you choose that approach",
        "tell me about your dataset",
        "explain the architecture",
        "do you use a transformer",
        "is there a paper for this",
    ])
    def test_questions_stay(self, text):
        r = decide(visitor_turn(text))
        assert r.action.kind is ActionKind.STAY
        assert r.mechanism == Mechanism.QUESTION_HEURISTIC

    def test_question_never_reaches_the_classifier(self):
        def boom(_):
            raise AssertionError("classifier must not run for an obvious question")
        r = decide(visitor_turn("what is RAG?"), intent_classifier=boom)
        assert r.action.kind is ActionKind.STAY


class TestIntentClassifier:
    """Rule 3: everything else goes to the LLM."""

    def test_done_advances(self):
        r = decide(visitor_turn("yeah I think so"), intent_classifier=lambda _: "done")
        assert r.action.kind is ActionKind.ADVANCE
        assert r.mechanism == Mechanism.LLM_CLASSIFIER

    def test_continue_stays(self):
        r = decide(visitor_turn("hmm interesting"), intent_classifier=lambda _: "continue")
        assert r.action.kind is ActionKind.STAY
        assert r.mechanism == Mechanism.LLM_CLASSIFIER

    def test_absent_classifier_defaults_to_continue(self):
        # classify_qa_intent's own contract: the safe default never skips a
        # real question, so a missing classifier must not advance either.
        r = decide(visitor_turn("hmm interesting"))
        assert r.action.kind is ActionKind.STAY

    def test_classifier_failure_defaults_to_continue(self):
        def broken(_):
            raise RuntimeError("ollama down")
        r = decide(visitor_turn("hmm interesting"), intent_classifier=broken)
        assert r.action.kind is ActionKind.STAY

    def test_unexpected_classifier_output_is_not_done(self):
        r = decide(visitor_turn("hmm"), intent_classifier=lambda _: "maybe")
        assert r.action.kind is ActionKind.STAY


# ── Parity: the robot-turn chain ──────────────────────────────────────────────

class TestRobotTurn:

    @pytest.mark.parametrize("phrase", QA_CLOSING_PHRASES)
    def test_every_closing_phrase_advances(self, phrase):
        r = decide(robot_turn(f"Sure. {phrase}"))
        assert r.action.kind is ActionKind.ADVANCE
        assert r.mechanism == Mechanism.CLOSING_PHRASE

    def test_closing_phrase_beats_the_moderator(self):
        def boom(_):
            raise AssertionError("moderator must not run after a phrase match")
        r = decide(robot_turn("hope that helps"), wrap_up_judge=boom)
        assert r.action.kind is ActionKind.ADVANCE

    def test_moderator_yes_interjects_as_the_guide(self):
        r = decide(robot_turn("We use a vision transformer."), wrap_up_judge=lambda _: True)
        assert r.action.kind is ActionKind.GUIDE_INTERJECT
        assert r.action.robot_id == GUIDE
        assert r.mechanism == Mechanism.LLM_MODERATOR

    def test_moderator_no_stays(self):
        r = decide(robot_turn("We use a vision transformer."), wrap_up_judge=lambda _: False)
        assert r.action.kind is ActionKind.STAY

    def test_moderator_failure_stays(self):
        def broken(_):
            raise RuntimeError("ollama down")
        r = decide(robot_turn("We use a transformer."), wrap_up_judge=broken)
        assert r.action.kind is ActionKind.STAY


# ── Routing ───────────────────────────────────────────────────────────────────

class TestRouting:

    def test_baseline_routes_to_whoever_heard_the_question(self):
        r = decide(visitor_turn("what is RAG?"), point=DecisionPoint.QA_ROUTE)
        assert r.action.kind is ActionKind.ROUTE_TO
        assert r.action.robot_id == ROBOT_A
        assert r.mechanism == Mechanism.RECEIVER


# ── PLAN_REVISE ───────────────────────────────────────────────────────────────

class TestPlanRevision:

    def test_no_trigger_and_no_budget_does_nothing(self):
        r = decide(visitor_turn("what is RAG?"), point=DecisionPoint.PLAN_REVISE)
        assert r.action.kind is ActionKind.STAY
        assert r.mechanism == Mechanism.NO_REVISION

    def test_the_clock_alone_does_nothing_without_a_budget(self):
        # No budget means no projected overrun, so inference from the clock
        # cannot fire. This is the opt-in: a run started without
        # time_budget_sec never has its script rewritten on its own.
        r = decide(visitor_turn("hmm, alright"), point=DecisionPoint.PLAN_REVISE)
        assert r.action.kind is ActionKind.STAY
        assert r.mechanism == Mechanism.NO_REVISION

    def test_a_visitor_saying_it_acts_even_without_a_budget(self):
        # A person stating they are short on time is a direct request, not an
        # inference, so it does not need a budget to be believed.
        r = decide(
            visitor_turn("we are running out of time"),
            point=DecisionPoint.PLAN_REVISE,
        )
        assert r.action.kind is ActionKind.REVISE
        assert r.mechanism == Mechanism.TIME_PRESSURE

    def test_stated_time_pressure_budgets_qa_then_compresses(self):
        obs = visitor_turn(
            "we are running out of time",
            engagement_by_robot={ROBOT_A: {"turns": 3, "questions": 2}},
        )
        r = decide(obs, point=DecisionPoint.PLAN_REVISE)
        assert r.action.kind is ActionKind.REVISE
        # The compression ladder: tighten Q&A before trimming anything. ROBOT_A
        # has already presented, so only the untouched project is affected.
        assert [o.kind for o in r.action.ops] == [
            PlanOpKind.SET_QA_BUDGET, PlanOpKind.COMPRESS]
        assert {o.robot_id for o in r.action.ops} == {ROBOT_B}
        assert r.action.ops[0].seconds > 0

    def test_severe_overrun_drops_to_the_wrap_up(self):
        obs = visitor_turn(
            "hmm",
            elapsed_sec=900.0,
            time_budget_sec=600.0,
            projected_overrun_sec=400.0,   # > 50% of the budget
        )
        r = decide(obs, point=DecisionPoint.PLAN_REVISE)
        assert [o.kind for o in r.action.ops] == [PlanOpKind.DROP_REMAINING]
        assert r.mechanism == Mechanism.TIME_PRESSURE

    def test_mild_overrun_trims_rather_than_drops(self):
        obs = visitor_turn(
            "hmm",
            elapsed_sec=650.0,
            time_budget_sec=600.0,
            projected_overrun_sec=60.0,
            engagement_by_robot={ROBOT_A: {"turns": 1, "questions": 1}},
        )
        r = decide(obs, point=DecisionPoint.PLAN_REVISE)
        kinds = [o.kind for o in r.action.ops]
        assert PlanOpKind.DROP_REMAINING not in kinds
        # Q&A is budgeted first — the cut a visitor is least likely to notice.
        assert kinds[0] is PlanOpKind.SET_QA_BUDGET

    def test_skip_request_targets_the_named_robot(self):
        r = decide(
            visitor_turn("let's skip the Navel part"),
            point=DecisionPoint.PLAN_REVISE,
        )
        assert r.mechanism == Mechanism.SKIP_REQUEST
        assert r.action.ops[0].kind is PlanOpKind.SKIP
        assert r.action.ops[0].robot_id == ROBOT_B

    def test_skip_request_without_a_name_targets_the_presenter(self):
        r = decide(visitor_turn("we can skip this one"), point=DecisionPoint.PLAN_REVISE)
        assert r.action.ops[0].robot_id == ROBOT_A

    def test_interest_request_extends_that_robots_qa(self):
        r = decide(
            visitor_turn("can we hear more about ChatBox"),
            point=DecisionPoint.PLAN_REVISE,
        )
        assert r.mechanism == Mechanism.INTEREST_REQUEST
        assert r.action.ops[0].kind is PlanOpKind.EXTEND_QA
        assert r.action.ops[0].robot_id == ROBOT_A

    def test_interest_request_naming_nobody_is_ignored(self):
        # No robot named and no fallback — guessing which project they meant
        # would rewrite the tour on a coin flip.
        r = decide(
            visitor_turn("I'd love to hear more about robotics generally"),
            point=DecisionPoint.PLAN_REVISE,
        )
        assert r.action.kind is ActionKind.STAY

    def test_explicit_skip_outranks_the_clock(self):
        obs = visitor_turn(
            "let's skip the Navel part",
            elapsed_sec=900.0,
            time_budget_sec=600.0,
            projected_overrun_sec=400.0,
        )
        r = decide(obs, point=DecisionPoint.PLAN_REVISE)
        assert r.mechanism == Mechanism.SKIP_REQUEST

    def test_guide_is_never_a_compression_target(self):
        obs = visitor_turn(
            "we are in a hurry",
            engagement_by_robot={ROBOT_A: {"turns": 1, "questions": 0},
                                 ROBOT_B: {"turns": 1, "questions": 0}},
        )
        r = decide(obs, point=DecisionPoint.PLAN_REVISE)
        # Both projects have presented and the guide is excluded, so there is
        # nothing left to trim.
        assert r.action.kind is ActionKind.STAY


# ── Reserved decision point ───────────────────────────────────────────────────

class TestReservedPoints:

    def test_delegate_initiative_is_declared_but_not_implemented(self):
        with pytest.raises(NotImplementedError, match="prompt_builder"):
            decide(visitor_turn("hi"), point=DecisionPoint.DELEGATE_INITIATIVE)
