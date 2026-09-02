"""
tests/test_decision_observation.py
==================================
The state space.

build_observation is the only place a policy's inputs are assembled, so these
tests are really about a contract: the learned policy and the heuristic baseline
must see identical state, or the comparison in the paper is between two different
problems.

Two properties get the most attention here:

  connected_peers stays structured. It is where the relational knowledge graph
  attaches next; flattening it into a prompt string now would have to be undone.

  decider_access_level is populated, because it is the join key to rbac_audit_log
  — without it "did this policy widen context exposure?" stops being a query.

Nothing here touches a database, a robot or an LLM.
"""

from __future__ import annotations

import time

import pytest

from core.rbac import AccessLevel, RobotIdentity
from decision import DemoRunTracker, build_observation, looks_like_question
from decision.observation import _projected_overrun

GUIDE = "pepper_01"
ROBOT_A = "chatbox_jetson_001"
ROBOT_B = "navel_001"
SCENARIO = "lab_demo"


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeInstance:
    """Enough RobotInstance to build an Observation from."""

    def __init__(self, client_id, name, role, level=AccessLevel.LOCAL, session="sess-1"):
        self.client_id = client_id
        self.robot_name = name
        self.access_level = level
        self._role = role
        self._session = session

    @property
    def identity(self) -> RobotIdentity:
        return RobotIdentity(
            robot_id=self.client_id,
            scenario_id=SCENARIO,
            session_id=self._session,
            access_level=self.access_level,
            role=self._role,
        )


class FakeRegistry:
    def __init__(self, instances):
        self._instances = list(instances)

    def get_all(self, exclude_id=None):
        return [i for i in self._instances if i.client_id != exclude_id]


class ExplodingRegistry:
    def get_all(self, exclude_id=None):
        raise RuntimeError("registry is having a bad day")


def status(**kw) -> dict:
    """A DemoOrchestrator.get_status() payload, mid-tour by default."""
    base = {
        "state": "qa_window",
        "step_idx": 7,
        "total": 20,
        "step_id": f"qa_invite_{ROBOT_A}",
        "robot_id": GUIDE,
        "elapsed_sec": 300.0,
        "time_budget_sec": None,
        "steps": [
            {"step_id": "greeting", "robot_id": GUIDE},
            {"step_id": "lab_intro", "robot_id": GUIDE},
            {"step_id": f"intro_project_a", "robot_id": GUIDE},
            {"step_id": f"{ROBOT_A}_greeting", "robot_id": ROBOT_A},
            {"step_id": f"{ROBOT_A}_prompt", "robot_id": GUIDE},
            {"step_id": f"{ROBOT_A}_project", "robot_id": ROBOT_A},
            {"step_id": f"transition_to_{ROBOT_B}", "robot_id": GUIDE},
            {"step_id": f"qa_invite_{ROBOT_A}", "robot_id": GUIDE},
            {"step_id": f"{ROBOT_B}_project", "robot_id": ROBOT_B},
        ],
    }
    base.update(kw)
    return base


@pytest.fixture
def registry():
    return FakeRegistry([
        FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
        FakeInstance(ROBOT_A, "ChatBox", "RAG research"),
        FakeInstance(ROBOT_B, "Navel", "Emotion research"),
    ])


@pytest.fixture
def tracker():
    return DemoRunTracker()


# ── Peers stay structured ─────────────────────────────────────────────────────

class TestPeers:

    def test_peers_are_dicts_not_a_prompt_string(self, registry, tracker):
        obs = build_observation(status(), registry, tracker)
        assert len(obs.connected_peers) == 3
        for p in obs.connected_peers:
            assert set(p) == {"client_id", "robot_name", "robot_role", "access_level"}

    def test_peer_role_comes_from_the_rbac_identity(self, registry, tracker):
        # RobotInstance keeps _robot_role private and refreshes it from the DB
        # before each chat, so identity.role is the only current reading.
        obs = build_observation(status(), registry, tracker)
        roles = {p["client_id"]: p["robot_role"] for p in obs.connected_peers}
        assert roles[ROBOT_A] == "RAG research"

    def test_peer_access_levels_are_rendered(self, registry, tracker):
        obs = build_observation(status(), registry, tracker)
        levels = {p["client_id"]: p["access_level"] for p in obs.connected_peers}
        assert levels[GUIDE] == "global"
        assert levels[ROBOT_A] == "local"

    def test_a_broken_registry_does_not_stop_the_decision(self, tracker):
        # A decision still has to be made and logged when peer lookup fails —
        # degrading to no peers beats refusing to decide mid-demo.
        obs = build_observation(status(), ExplodingRegistry(), tracker)
        assert obs.connected_peers == ()
        assert obs.step_id == f"qa_invite_{ROBOT_A}"


# ── RBAC identifiers ──────────────────────────────────────────────────────────

class TestRbacContext:

    def test_decider_identity_is_recorded(self, registry, tracker):
        decider = FakeInstance(ROBOT_A, "ChatBox", "RAG research", session="sess-abc")
        obs = build_observation(status(), registry, tracker, decider=decider)
        assert obs.decider_robot_id == ROBOT_A
        assert obs.decider_access_level == "local"
        assert obs.scenario_id == SCENARIO
        # The join key to rbac_audit_log.
        assert obs.session_id == "sess-abc"

    def test_an_unparseable_access_level_is_recorded_verbatim(self, registry, tracker):
        # A decision row is not an access check. A malformed level must be
        # visible in the audit trail, not the cause of a crash mid-demo.
        decider = FakeInstance(ROBOT_A, "ChatBox", "role", level="omniscient")
        obs = build_observation(status(), registry, tracker, decider=decider)
        assert obs.decider_access_level == "omniscient"

    def test_no_decider_leaves_identity_empty(self, registry, tracker):
        obs = build_observation(status(), registry, tracker)
        assert obs.decider_robot_id is None
        assert obs.session_id is None


# ── Guide and presenter ───────────────────────────────────────────────────────

class TestGuideAndPresenter:

    def test_guide_is_the_robot_on_step_zero(self, registry, tracker):
        obs = build_observation(status(), registry, tracker)
        assert obs.guide_robot_id == GUIDE

    def test_presenter_is_the_last_non_guide_robot(self, registry, tracker):
        # The current step is a Q&A run by the guide, but the visitors are
        # asking about ChatBox. Reporting the guide as presenter would send
        # every "skip this one" to the wrong block.
        obs = build_observation(status(), registry, tracker)
        assert obs.presenting_robot_id == ROBOT_A

    def test_no_presenter_during_the_opening(self, registry, tracker):
        obs = build_observation(status(step_idx=1), registry, tracker)
        assert obs.presenting_robot_id is None

    def test_empty_script_is_survivable(self, registry, tracker):
        obs = build_observation({"steps": [], "total": 0}, registry, tracker)
        assert obs.guide_robot_id is None
        assert obs.presenting_robot_id is None


# ── Time budget ───────────────────────────────────────────────────────────────

class TestTimeBudget:

    def test_no_budget_means_no_projected_overrun(self, registry, tracker):
        obs = build_observation(status(), registry, tracker)
        assert obs.time_budget_sec is None
        assert obs.projected_overrun_sec is None

    def test_overrun_extends_the_observed_pace(self, registry, tracker):
        # 300s over 6 completed steps = 50s/step; 14 steps remain → 700s more,
        # 1000s total against a 600s budget.
        obs = build_observation(
            status(step_idx=6, total=20, elapsed_sec=300.0, time_budget_sec=600.0),
            registry, tracker,
        )
        assert obs.projected_overrun_sec == pytest.approx(400.0)

    def test_a_run_ahead_of_schedule_reports_a_negative_overrun(self, registry, tracker):
        obs = build_observation(
            status(step_idx=10, total=20, elapsed_sec=100.0, time_budget_sec=600.0),
            registry, tracker,
        )
        assert obs.projected_overrun_sec < 0

    def test_step_zero_has_no_pace_to_extrapolate_from(self):
        # Dividing by zero completed steps is the obvious bug here; the guard
        # also stops PLAN_REVISE firing on no evidence at all.
        assert _projected_overrun(0.0, 600.0, 0, 20) is None


# ── Run tracker ───────────────────────────────────────────────────────────────

class TestRunTracker:

    def test_visitor_turns_count_toward_the_addressed_robot(self, tracker):
        tracker.open_window()
        tracker.note_visitor_turn(ROBOT_A, "what is RAG?")
        tracker.note_visitor_turn(ROBOT_A, "interesting")
        tracker.note_visitor_turn(ROBOT_B, "how does that work?")
        snap = tracker.snapshot()
        assert snap["engagement_by_robot"][ROBOT_A] == {"turns": 2, "questions": 1}
        assert snap["engagement_by_robot"][ROBOT_B] == {"turns": 1, "questions": 1}
        assert snap["turns_in_window"] == 3

    def test_closing_a_window_resets_its_counters_but_not_engagement(self, tracker):
        tracker.open_window()
        tracker.note_visitor_turn(ROBOT_A, "what is RAG?")
        tracker.close_window()
        snap = tracker.snapshot()
        assert snap["turns_in_window"] == 0
        assert snap["seconds_in_window"] == 0.0
        # Interest accrues across the whole run — it is what "they clearly want
        # more of project B" is derived from.
        assert snap["engagement_by_robot"][ROBOT_A]["turns"] == 1

    def test_starting_a_run_clears_everything(self, tracker):
        tracker.open_window()
        tracker.note_visitor_turn(ROBOT_A, "what is RAG?")
        tracker.start_run()
        assert tracker.snapshot()["engagement_by_robot"] == {}

    def test_window_seconds_advance(self, tracker):
        tracker.open_window()
        time.sleep(0.02)
        assert tracker.seconds_in_window() > 0

    def test_robot_turns_set_the_last_speaker(self, tracker):
        tracker.note_robot_turn(ROBOT_A, "RAG combines retrieval with generation.")
        snap = tracker.snapshot()
        assert snap["last_speaker_id"] == ROBOT_A
        assert "retrieval" in snap["last_robot_utterance"]

    def test_visitor_turns_mark_the_speaker_as_visitor(self, tracker):
        # HeuristicPolicy branches on this: 'visitor' selects the advance chain,
        # anything else selects the robot-response chain.
        tracker.note_visitor_turn(ROBOT_A, "hello")
        assert tracker.snapshot()["last_speaker_id"] == "visitor"

    def test_explicit_utterance_overrides_the_tracked_one(self, registry, tracker):
        tracker.note_visitor_turn(ROBOT_A, "older message")
        obs = build_observation(
            status(), registry, tracker, user_utterance="the message being judged"
        )
        assert obs.user_utterance == "the message being judged"


# ── Serialization ─────────────────────────────────────────────────────────────

class TestSerialization:

    def test_as_dict_is_json_safe(self, registry, tracker):
        import json
        tracker.open_window()
        tracker.note_visitor_turn(ROBOT_A, "what is RAG?")
        obs = build_observation(
            status(time_budget_sec=600.0), registry, tracker, user_utterance="hi"
        )
        # The observation column is JSONB; anything unserializable here becomes
        # a dropped batch at write time, far from the cause.
        round_tripped = json.loads(json.dumps(obs.as_dict()))
        assert round_tripped["connected_peers"][0]["client_id"] == GUIDE
        assert round_tripped["engagement_by_robot"][ROBOT_A]["turns"] == 1


# ── Shared question heuristic ─────────────────────────────────────────────────

class TestQuestionHeuristic:

    @pytest.mark.parametrize("text,expected", [
        ("what is RAG?", True),
        ("How does it work", True),
        ("  tell me more", True),
        ("is there a paper", True),
        ("that is fascinating", False),
        ("okay", False),
        ("", False),
    ])
    def test_matches_the_original_prefilter(self, text, expected):
        assert looks_like_question(text) is expected
