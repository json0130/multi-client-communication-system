"""
tests/test_qa_auto_close.py
============================
Closing a Q&A window when the robot has audibly finished.

A live run had a robot say "If you change your mind, don't hesitate to ask.
Have a great day!" and the tour then sat there until someone said "move on"
out loud. The closing-phrase mechanism to handle that had been written long
before and never wired to a call site; this is it wired, plus the pause that
makes it safe.

THE PAUSE IS THE POINT
Closing the instant a robot invites more questions would cut off the visitor
who was drawing breath to take it up. So an ADVANCE from a robot turn
SCHEDULES the close, and anything the visitor says cancels it. Tests use a
tiny delay rather than the real five seconds — the property under test is
"cancellable, and fires only if nothing happened", not the constant.
"""

from __future__ import annotations

import threading
import time

import pytest

from core.rbac import AccessLevel, RobotIdentity
from decision import DecisionRecorder, MemoryDecisionSink
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
from demo.demo_script import build_script
from gateway.websocket_gateway import WebSocketGateway

GUIDE = "pepper_01"
A, B = "chatbox_01", "navel_01"
SCENARIO = "lab_demo"

SIGN_OFF = "Let me know if you have any other questions."
A_REAL_ANSWER = "We use a vector database to retrieve the relevant passage."


class FakeInstance:
    def __init__(self, client_id, name, role, level=AccessLevel.LOCAL):
        self.client_id, self.robot_name, self.access_level = client_id, name, level
        self._role = role

    @property
    def identity(self):
        return RobotIdentity(robot_id=self.client_id, scenario_id=SCENARIO,
                             session_id=f"sess-{self.client_id}",
                             access_level=self.access_level, role=self._role)

    def classify_qa_intent(self, message):
        return "continue"


class FakeRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}

    def get(self, client_id):
        return self._by_id.get(client_id)

    def get_all(self, exclude_id=None):
        return [i for i in self._by_id.values() if i.client_id != exclude_id]


@pytest.fixture
def wired():
    registry = FakeRegistry([
        FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
        FakeInstance(A, "ChatBox", "RAG research"),
        FakeInstance(B, "Navel", "Emotion research"),
    ])
    recorder = DecisionRecorder(MemoryDecisionSink())
    gw = WebSocketGateway(registry, recorder=recorder)
    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)
    orch._state = DemoState.QA_WINDOW
    orch._idx = next(i for i, s in enumerate(orch._script)
                     if s.block_robot_id == A and s.role == StepRole.QA)
    gw.on_qa_window_open()
    gw.send_to_robot = lambda cid, data: None

    closed = []
    real_qa_end = orch.qa_end

    def spy(source="auto", reason=""):
        closed.append(source)
        real_qa_end(source=source, reason=reason)
    orch.qa_end = spy
    return gw, orch, registry, closed


def _settle(seconds=0.15):
    time.sleep(seconds)


class TestSignOffSchedulesAClose:

    def test_a_sign_off_closes_the_window_after_the_pause(self, wired, monkeypatch):
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(A, SIGN_OFF)
        assert closed == [], "must not close instantly — the visitor gets the pause"
        _settle()
        assert closed == ["policy"]

    def test_an_ordinary_answer_schedules_nothing(self, wired, monkeypatch):
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        _settle()
        assert closed == []

    def test_the_guide_IS_heard_when_it_says_a_closing_phrase(self, wired, monkeypatch):
        """The guard covers the LLM moderator, not the phrase match.

        It used to cover both, so the guide could say "Let's continue with
        our tour" and nothing closed — the robot most likely to say a
        wrap-up line was the one robot that could not trigger one. A string
        match cannot recurse, so there is nothing to guard against.
        """
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(GUIDE, "You're welcome! Let's continue with our tour.")
        _settle()
        assert closed == ["policy"]

    def test_the_guide_is_still_guarded_from_the_moderator(self, wired, monkeypatch):
        # Ordinary guide chatter with no closing phrase in it must not be
        # handed to the wrap-up judge — that is the loop the guard exists for.
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(GUIDE, "Let me think about how best to explain that.")
        _settle()
        assert closed == []

    def test_nothing_is_scheduled_outside_a_qa_window(self, wired, monkeypatch):
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        orch._state = DemoState.RUNNING
        gw.check_qa_auto_close(A, SIGN_OFF)
        _settle()
        assert closed == []


class TestTheVisitorAlwaysWins:
    """Anything the visitor says means the window should not close for
    silence — that is what the pause was for."""

    def test_a_visitor_turn_cancels_a_pending_close(self, wired, monkeypatch):
        gw, orch, registry, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.3)
        gw.check_qa_auto_close(A, SIGN_OFF)
        gw.check_qa_advance_from_user(registry.get(A), "actually, one more thing")
        _settle(0.4)
        assert closed == [], "the visitor spoke — the pause is over, not elapsed"

    def test_an_explicit_move_on_still_closes_once(self, wired, monkeypatch):
        # The visitor cancels the timer AND asks to move on. Exactly one
        # close, from the phrase — not a second one from the stale timer.
        gw, orch, registry, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(A, SIGN_OFF)
        gw.check_qa_advance_from_user(registry.get(A), "move on")
        _settle(0.2)
        assert closed == ["policy"]

    def test_closing_the_window_cancels_a_pending_timer(self, wired, monkeypatch):
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.2)
        gw.check_qa_auto_close(A, SIGN_OFF)
        gw.on_qa_window_close()
        _settle(0.3)
        assert closed == []

    def test_a_new_window_cancels_the_previous_ones_timer(self, wired, monkeypatch):
        # A timer left over from the last window would close the new one
        # almost as soon as it opened.
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.2)
        gw.check_qa_auto_close(A, SIGN_OFF)
        gw.on_qa_window_open()
        _settle(0.3)
        assert closed == []

    def test_cancelling_twice_is_harmless(self, wired):
        gw, _, _, _ = wired
        gw.cancel_qa_auto_close()
        gw.cancel_qa_auto_close()

    def test_a_second_sign_off_replaces_the_first_timer(self, wired, monkeypatch):
        # Two sign-offs must not queue two closes.
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(A, SIGN_OFF)
        gw.check_qa_auto_close(A, SIGN_OFF)
        _settle(0.2)
        assert closed == ["policy"]


class TestTheTimerDoesNotOutliveItsWindow:
    def test_a_fired_timer_checks_the_window_is_still_open(self, wired, monkeypatch):
        # The timer fires into a window someone else already closed. It must
        # notice rather than closing whatever window is open by then.
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.1)
        gw._schedule_qa_auto_close(0.1)
        orch._state = DemoState.RUNNING       # closed by some other path
        _settle(0.2)
        assert closed == []

    def test_the_timer_is_a_daemon(self, wired, monkeypatch):
        # A non-daemon timer would keep the process alive for the full delay
        # on shutdown.
        gw, _, _, _ = wired
        gw._schedule_qa_auto_close(30.0)
        try:
            assert gw._auto_close_timer.daemon is True
        finally:
            gw.cancel_qa_auto_close()


class TestTheGuardUsesTheRealGuide:
    """
    Regression: the guide guard read status["robot_id"], which is the CURRENT
    STEP's robot, not the host. During a project block that IS the presenting
    robot — so the guard fired on exactly the robot most likely to sign off,
    and auto-close silently never ran. A live run had Silbot say "If you have
    more questions, feel free to ask" and the tour sat there waiting.
    """

    def test_the_presenting_robot_can_close_its_own_window(self, wired, monkeypatch):
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        # Park on A's own project step, so status["robot_id"] == A.
        orch._idx = next(i for i, s in enumerate(orch._script)
                         if s.block_robot_id == A and s.role == StepRole.PROJECT)
        assert orch.get_status()["robot_id"] == A, "fixture must reproduce the trap"
        gw.check_qa_auto_close(A, SIGN_OFF)
        _settle()
        assert closed == ["policy"]

    def test_a_presenting_robots_sign_off_still_closes(self, wired, monkeypatch):
        gw, orch, _, closed = wired
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.check_qa_auto_close(A, SIGN_OFF)
        _settle()
        assert closed == ["policy"]

    def test_the_interval_is_three_seconds(self):
        from gateway.websocket_gateway import QA_AUTO_CLOSE_SEC
        assert QA_AUTO_CLOSE_SEC == 3.0
