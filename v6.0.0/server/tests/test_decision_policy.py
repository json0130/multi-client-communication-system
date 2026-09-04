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
    SKIP_PHRASES,
    TIME_PRESSURE_PHRASES,
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


class TestBareAffirmation:
    """
    Regression for a real misfire: a live run had the LLM classifier read a
    bare "yes" — replying to the guide's compound question "any other
    questions, or shall we continue?" — as "yes, I have a question", staying
    in the loop when the visitor meant "yes, let's continue".

    A short affirmative reply to that framing overwhelmingly means "continue":
    someone with an actual follow-up almost always asks it directly rather than
    replying with a bare "yes". Handled deterministically, before paying for an
    LLM call, and with an EXACT match — not the substring match QA_ADVANCE_PHRASES
    uses — specifically so it cannot fire on a real question that happens to
    contain the word "yes".
    """

    @pytest.mark.parametrize("text", [
        "yes", "Yes", "YES", "yeah", "yep", "yup", "sure", "correct",
        "exactly", "right", "that's right", "ok", "okay", "alright",
        "definitely", "absolutely", "yes.", "yes!", "  yes  ",
    ])
    def test_bare_affirmations_advance(self, text):
        r = decide(visitor_turn(text))
        assert r.action.kind is ActionKind.ADVANCE
        assert r.mechanism == Mechanism.BARE_AFFIRMATION

    def test_bare_affirmation_never_reaches_the_classifier(self):
        def boom(_):
            raise AssertionError("classifier must not run for a bare affirmation")
        r = decide(visitor_turn("yes"), intent_classifier=boom)
        assert r.action.kind is ActionKind.ADVANCE

    @pytest.mark.parametrize("text", [
        "yes, what about the sensors?",
        "yes I have another question",
        "yeah but how does it handle errors",
        "well yes and no",
    ])
    def test_a_real_follow_up_containing_yes_is_not_caught(self, text):
        # The precision guard: substring-matching "yes" would wrongly fire on
        # a genuine question that happens to contain the word. These must fall
        # through to the question heuristic or the classifier, not advance.
        r = decide(visitor_turn(text))
        assert r.mechanism != Mechanism.BARE_AFFIRMATION

    def test_bare_negatives_are_not_treated_as_affirmations(self):
        # Deliberately NOT auto-advanced: a bare "no" answering "shall we
        # continue?" would mean STAY, so guessing advance risks cutting off a
        # visitor who wanted to keep going — worse than one extra LLM call.
        r = decide(visitor_turn("no"), intent_classifier=lambda _: "continue")
        assert r.mechanism != Mechanism.BARE_AFFIRMATION

    def test_advance_phrase_list_still_takes_precedence(self):
        # "move on" also happens to be a case where BOTH could apply in
        # principle; the substring list runs first and must still win.
        r = decide(visitor_turn("move on"))
        assert r.mechanism == Mechanism.ADVANCE_PHRASE


class TestTimePressureAdvances:
    """
    Regression for a real misfire: a visitor said "i am running out of time
    so keep the demo short pls" during a Q&A window. PLAN_REVISE correctly
    read it as time pressure and shortened the tour, but QA_ADVANCE is a
    separate decision — it fell through to the LLM classifier, which read
    the same utterance as 'continue' and left the window open, so the
    visitor had to repeat themselves to a different robot before anything
    moved on. Stating time pressure is exactly the kind of explicit signal
    an advance phrase already outranks the classifier with; it gets the
    same treatment here.
    """

    @pytest.mark.parametrize("phrase", TIME_PRESSURE_PHRASES)
    def test_every_time_pressure_phrase_advances(self, phrase):
        r = decide(visitor_turn(phrase))
        assert r.action.kind is ActionKind.ADVANCE
        assert r.mechanism == Mechanism.TIME_PRESSURE

    def test_time_pressure_never_reaches_the_classifier(self):
        def boom(_):
            raise AssertionError("classifier must not run when time pressure is stated")
        r = decide(visitor_turn("we're running out of time"), intent_classifier=boom)
        assert r.action.kind is ActionKind.ADVANCE

    def test_advance_phrase_list_still_takes_precedence_over_time_pressure(self):
        r = decide(visitor_turn("move on, we're running out of time"))
        assert r.mechanism == Mechanism.ADVANCE_PHRASE


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

    @pytest.mark.parametrize("phrase", SKIP_PHRASES)
    def test_every_skip_phrase_is_recognized(self, phrase):
        # SKIP_PHRASES only had the declarative "we can skip" — a real visitor
        # asking "can we skip chatbox" as a question matched none of them, so
        # the request was silently dropped and that robot's block ran anyway.
        r = decide(visitor_turn(f"{phrase} chatbox"), point=DecisionPoint.PLAN_REVISE)
        assert r.mechanism == Mechanism.SKIP_REQUEST
        assert r.action.ops[0].kind is PlanOpKind.SKIP
        assert r.action.ops[0].robot_id == ROBOT_A

    def test_can_we_skip_targets_the_named_robot(self):
        r = decide(
            visitor_turn("can we skip chatbox"),
            point=DecisionPoint.PLAN_REVISE,
        )
        assert r.mechanism == Mechanism.SKIP_REQUEST
        assert r.action.ops[0].kind is PlanOpKind.SKIP
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
