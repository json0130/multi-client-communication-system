"""
decision/policy.py
==================
Who decides, and by what rule.

`HeuristicPolicy` is the current system, moved rather than rewritten. Today the
Q&A window is governed by five mechanisms scattered across websocket_gateway and
robot_instance — two phrase lists, a prefix heuristic, an LLM intent classifier
and an LLM moderator prompt. They already form a precedence chain; it just is not
written down anywhere. Writing it down here does three things:

  1. Present behaviour becomes a named, citable baseline instead of control flow.
  2. `mechanism` on every logged decision says which rule actually fired, so the
     phrase lists' hit rate is measurable rather than assumed.
  3. A learned policy becomes a drop-in — same Observation, same Action.

The precedence chain is preserved exactly. `tests/test_decision_policy.py`
asserts it against the real phrase lists; changing an outcome here is a
behaviour change to a live demo, not a refactor.

PLAN_REVISE has no prior implementation to preserve — a visitor saying "we are
running out of time" currently closes one window and nothing more. The rules
below are deliberately thin. They exist to make the decision point real and
loggable, and to be the weak baseline the learned policy has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from decision.models import Action, DecisionPoint, Observation, PlanOp, PlanOpKind
from decision.observation import looks_like_question


# ── Mechanism names ───────────────────────────────────────────────────────────
# Stable identifiers. The audit views GROUP BY these, so they are contract.

class Mechanism:
    ADVANCE_PHRASE = "advance_phrase"       # user said something in _QA_ADVANCE_PHRASES
    CLOSING_PHRASE = "closing_phrase"       # robot said something in _QA_CLOSING_PHRASES
    QUESTION_HEURISTIC = "question_heuristic"
    LLM_CLASSIFIER = "llm_classifier"       # RobotInstance.classify_qa_intent
    LLM_MODERATOR = "llm_moderator"         # the guide's wrap-up judgement
    RECEIVER = "receiver"                   # whoever heard the audio answers
    TIME_PRESSURE = "time_pressure"
    SKIP_REQUEST = "skip_request"
    INTEREST_REQUEST = "interest_request"
    NO_REVISION = "no_revision"


# ── Phrase lists ──────────────────────────────────────────────────────────────
# Moved verbatim from gateway/websocket_gateway.py. Do not edit without updating
# tests/test_decision_policy.py — these strings are the baseline being measured.

QA_ADVANCE_PHRASES = [
    "move on",
    "next project",
    "next robot",
    "next step",
    "proceed",
    "can we continue",
    "shall we continue",
    "ready to continue",
    "ready to move",
    "let's go",
    "let's continue",
    "let us continue",
    "no more questions",
    "no questions",
    "that's all",
    "no thank you",
    "continue the demo",
    "continue the demonstration",
    # Natural dismissals that the LLM classifier tends to misread
    "carry on",
    "move along",
    "all good",
    "we're good",
    "i'm good",
    "im good",
    "that's fine",
    "that's okay",
    "it's okay",
    "no worries",
    "never mind",
    "forget it",
    "done here",
    "we're done",
    "all done",
]

QA_CLOSING_PHRASES = [
    # Research robot wrapping up their explanation
    "let me know if you have any other questions",
    "any other questions",
    "feel free to ask",
    "hope that answers",
    "hope that helps",
    "is there anything else",
    "anything else i can help",
    "don't hesitate to ask",
    "please don't hesitate",
    "happy to answer more",
    "if you'd like to know more",
    # Pepper acknowledging a "move on" request from the visitor
    "let us move on",
    "let's move on",
    "moving on to",
    "moving on now",
    "let's proceed",
    "shall proceed",
    "proceed to the next",
    "sure thing",
    "great, moving",
]

# ── PLAN_REVISE triggers ──────────────────────────────────────────────────────
# New. No prior behaviour to preserve — see the module docstring.

TIME_PRESSURE_PHRASES = [
    "running out of time",
    "run out of time",
    "short on time",
    "pressed for time",
    "running late",
    "not much time",
    "little time",
    "keep it brief",
    "keep it short",
    "cut it short",
    "speed this up",
    "speed it up",
    "in a hurry",
    "we need to leave",
    "have to leave",
    "only have a few minutes",
]

SKIP_PHRASES = [
    "skip this",
    "skip that",
    "skip ahead",
    "skip the",
    "we can skip",
    "let's skip",
    "not interested in this",
    "not interested in that",
]

INTEREST_PHRASES = [
    "hear more about",
    "tell us more about",
    "tell me more about",
    "more about the",
    "spend more time on",
    "go deeper on",
    "go deeper into",
    "really interested in",
    "most interested in",
]

# Above this fraction of the budget in projected overrun, trimming individual
# project blocks will not recover the time — go straight to the wrap-up.
DROP_REMAINING_OVERRUN_RATIO = 0.5

# ── The compression ladder ────────────────────────────────────────────────────
# Faced with 15 minutes for a 25-minute tour, WHAT gets cut first matters more
# than how much. Ordered by what a visitor actually notices:
#
#   1. tighten Q&A      largest share of tour time, least noticed when shortened.
#                       Nobody remembers a question round that ended a minute
#                       early; everybody remembers a robot that never spoke.
#   2. compress blocks  drop the social scaffolding (intro, handoff, greeting,
#                       prompt) and keep the research talk.
#   3. skip a block     only when the first two cannot recover enough.
#
# Reaching for rung 3 first is the intuitive move and the wrong one.
QA_BUDGET_TIGHT_SEC = 60.0
"""What a Q&A window is cut to under time pressure. Deliberately a constant
rather than a measured figure: data.demo_duration_repo.suggested_qa_budget reads
observed windows, but it needs ~10 of them before it says anything, and until
then a round number honestly labelled beats a measurement of three windows."""


def _matches(text: str, phrases: list[str]) -> Optional[str]:
    """Return the first phrase present in `text`, or None."""
    t = (text or "").lower()
    for p in phrases:
        if p in t:
            return p
    return None


# ── Policy interface ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyResult:
    """
    The action, plus the name of the rule that produced it.

    `mechanism` is not the policy's name — it is the specific rule that fired.
    A single HeuristicPolicy emits several. That granularity is the whole point:
    it is how "the phrase list caught it" is separated from "the LLM caught it".
    """

    action: Action
    mechanism: str


class Policy(Protocol):
    """Anything that can answer a decision point."""

    name: str

    def decide(self, point: DecisionPoint, obs: Observation) -> PolicyResult: ...


class HeuristicPolicy:
    """
    The current system, made explicit.

    `intent_classifier` and `wrap_up_judge` are injected rather than imported so
    this package stays free of robot/ and gateway/ — the same reason core/rbac
    takes a writer callable. The simulator will pass stubs for both.
    """

    name = "heuristic_v1"

    def __init__(
        self,
        intent_classifier: Optional[Callable[[str], str]] = None,
        wrap_up_judge: Optional[Callable[[Observation], bool]] = None,
    ):
        self._classify = intent_classifier
        self._wrap_up = wrap_up_judge

    # ── Entry point ───────────────────────────────────────────────────────────

    def decide(self, point: DecisionPoint, obs: Observation) -> PolicyResult:
        if point is DecisionPoint.QA_ADVANCE:
            return self._decide_advance(obs)
        if point is DecisionPoint.QA_ROUTE:
            return self._decide_route(obs)
        if point is DecisionPoint.PLAN_REVISE:
            return self._decide_revise(obs)
        raise NotImplementedError(
            f"{point.value} has no heuristic baseline. DELEGATE_INITIATIVE is "
            "reserved for the initiative-arbitration phase; the current system "
            "hard-codes human-lead in robot/prompt_builder.build_delegation_prompt."
        )

    # ── QA_ADVANCE ────────────────────────────────────────────────────────────

    def _decide_advance(self, obs: Observation) -> PolicyResult:
        """
        Should the Q&A window close?

        Visitor turn — the original precedence, unchanged:
            advance phrase → advance
            looks like a question → stay
            otherwise → LLM intent classifier

        Robot turn — the two mechanisms that ran after a robot replied:
            closing phrase → advance
            otherwise → the guide's LLM moderator judgement
        """
        if obs.last_speaker_id == "visitor":
            if _matches(obs.user_utterance, QA_ADVANCE_PHRASES):
                return PolicyResult(Action.advance(), Mechanism.ADVANCE_PHRASE)

            if looks_like_question(obs.user_utterance):
                return PolicyResult(Action.stay(), Mechanism.QUESTION_HEURISTIC)

            # Safe default is 'continue' — matches classify_qa_intent's contract,
            # and applies when no classifier is wired in at all.
            intent = "continue"
            if self._classify is not None:
                try:
                    intent = self._classify(obs.user_utterance)
                except Exception as e:
                    print(f"[decision.policy] intent classifier failed: {e}")
            action = Action.advance() if intent == "done" else Action.stay()
            return PolicyResult(action, Mechanism.LLM_CLASSIFIER)

        # Robot turn
        if _matches(obs.last_robot_utterance, QA_CLOSING_PHRASES):
            return PolicyResult(Action.advance(), Mechanism.CLOSING_PHRASE)

        if self._wrap_up is not None:
            try:
                if self._wrap_up(obs):
                    return PolicyResult(
                        Action.guide_interject(obs.guide_robot_id),
                        Mechanism.LLM_MODERATOR,
                    )
            except Exception as e:
                print(f"[decision.policy] wrap-up judge failed: {e}")
        return PolicyResult(Action.stay(), Mechanism.LLM_MODERATOR)

    # ── QA_ROUTE ──────────────────────────────────────────────────────────────

    def _decide_route(self, obs: Observation) -> PolicyResult:
        """
        Who answers?

        The baseline has no routing at all: whichever robot's microphone picked
        the visitor up replies. Recording it as an explicit decision is the only
        change — the outcome is identical, and it gives the learned policy
        something to be compared against.
        """
        return PolicyResult(Action.route_to(obs.decider_robot_id), Mechanism.RECEIVER)

    # ── PLAN_REVISE ───────────────────────────────────────────────────────────

    def _decide_revise(self, obs: Observation) -> PolicyResult:
        """
        Should the remaining script change?

        Precedence: an explicit skip beats a stated interest, which beats an
        inferred time problem — a direct instruction from a visitor outranks
        anything derived from the clock.
        """
        utterance = obs.user_utterance or ""

        if _matches(utterance, SKIP_PHRASES):
            target = self._named_robot(utterance, obs) or obs.presenting_robot_id
            if target:
                return PolicyResult(
                    Action.revise([PlanOp(PlanOpKind.SKIP, robot_id=target)]),
                    Mechanism.SKIP_REQUEST,
                )

        if _matches(utterance, INTEREST_PHRASES):
            target = self._named_robot(utterance, obs)
            if target:
                return PolicyResult(
                    Action.revise([PlanOp(PlanOpKind.EXTEND_QA, robot_id=target)]),
                    Mechanism.INTEREST_REQUEST,
                )

        stated_time_pressure = _matches(utterance, TIME_PRESSURE_PHRASES) is not None
        overrun = obs.projected_overrun_sec

        if stated_time_pressure or (overrun is not None and overrun > 0):
            budget = obs.time_budget_sec
            severe = (
                overrun is not None
                and budget
                and overrun > budget * DROP_REMAINING_OVERRUN_RATIO
            )
            if severe:
                # Past saving by trimming. Closing steps survive, so the tour
                # still ends rather than stopping mid-sentence.
                return PolicyResult(
                    Action.revise([PlanOp(PlanOpKind.DROP_REMAINING)]),
                    Mechanism.TIME_PRESSURE,
                )

            remaining = self._remaining_projects(obs)
            if remaining:
                # Rungs 1 and 2 together, in that order: budget every upcoming
                # Q&A window first, then trim the scaffolding around the talks.
                # Applied to EVERY remaining block rather than one, so no project
                # is silently cut for being late in the running order.
                ops = [PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=r,
                              seconds=QA_BUDGET_TIGHT_SEC) for r in remaining]
                ops += [PlanOp(PlanOpKind.COMPRESS, robot_id=r) for r in remaining]
                return PolicyResult(Action.revise(ops), Mechanism.TIME_PRESSURE)

        return PolicyResult(Action.stay(), Mechanism.NO_REVISION)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _named_robot(self, utterance: str, obs: Observation) -> Optional[str]:
        """
        Which robot did the visitor name?

        Matches on robot_name and client_id only. Matching on role text was
        tempting and is wrong: roles are free-form sentences ("You are a helpful
        robot"), so common words in them would match almost any utterance.
        """
        t = (utterance or "").lower()
        for peer in obs.connected_peers:
            name = (peer.get("robot_name") or "").strip().lower()
            cid = (peer.get("client_id") or "").strip().lower()
            if name and len(name) > 2 and name in t:
                return peer.get("client_id")
            if cid and len(cid) > 2 and cid in t:
                return peer.get("client_id")
        return None

    def _remaining_projects(self, obs: Observation) -> list[str]:
        """
        Project robots whose block has not started yet.

        Derived from engagement rather than the step list: a robot that has
        already drawn visitor turns has presented, so compressing it would do
        nothing. The guide is never a project.
        """
        seen = set(obs.engagement_by_robot.keys())
        out = []
        for peer in obs.connected_peers:
            cid = peer.get("client_id")
            if not cid or cid == obs.guide_robot_id or cid in seen:
                continue
            out.append(cid)
        return out
