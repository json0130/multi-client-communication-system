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


# Captured before any test patches the module attribute, so the pure-function
# tests below measure the real estimator rather than the autouse stub.
from gateway.websocket_gateway import _speaking_seconds as REAL_SPEAKING_SECONDS


@pytest.fixture(autouse=True)
def _instant_speech(monkeypatch):
    """Neutralise the speaking-time allowance for every test here.

    The scheduled delay is speaking time + QA_AUTO_CLOSE_SEC, because the
    server cannot hear the end of a sentence. These tests are about the
    cancellation and guard logic, so they patch the constant to milliseconds
    — leaving the ~4s speaking estimate in would just make them slow and
    flaky. TestSpeakingTimeIsWaitedOut covers the allowance itself.
    """
    monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                        lambda _t: 0.0)


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


class TestSpeakingTimeIsWaitedOut:
    """
    The countdown starts when GENERATION finishes, not when the robot stops
    talking — there is no end-of-TTS signal for streamed sentences. So a
    robot that signs off by ASKING something ("Is there anything else you'd
    like to know?") had the tour move on before the question finished
    playing. Reported exactly that way.

    The fix waits out an estimate of the speech first, so the pause means
    what it says: silence AFTER the robot finishes.
    """

    LONG = ("Great! Is there anything else you'd like to know about how I "
            "work in real-life situations?")

    def test_a_longer_sign_off_waits_longer(self):
        assert (REAL_SPEAKING_SECONDS(self.LONG)
                > REAL_SPEAKING_SECONDS("Thanks!"))

    def test_the_estimate_is_capped(self):
        # A wrong estimate must not park the tour indefinitely.
        assert REAL_SPEAKING_SECONDS("word " * 5000) <= 20.0

    def test_empty_text_costs_nothing(self):
        assert REAL_SPEAKING_SECONDS("") == 0.0

    def test_the_allowance_is_added_to_the_pause(self, wired, monkeypatch):
        # With speaking time restored, a sign-off that takes seconds to say
        # must not close within the bare pause.
        gw, orch, _, closed = wired
        monkeypatch.undo()   # drop the autouse zeroing for this one
        monkeypatch.setattr("gateway.websocket_gateway.QA_AUTO_CLOSE_SEC", 0.05)
        gw.note_speech(A, {"event": "chat_sentence", "text": self.LONG})
        gw.check_qa_auto_close(A, self.LONG)
        _settle(0.3)
        assert closed == [], "closed while the robot was still speaking"


class TestTheSpeechClock:
    """
    The server never hears the end of a sentence — TTS happens on the robot,
    and only ACK-bearing demo steps report back. So "wait for the robot to
    finish" has to be estimated from what was sent, or it silently means
    "wait from the moment generation finished", with the whole utterance
    still ahead of it. That is what put the guide's transition one second
    into a nine-second answer.
    """

    def test_nothing_in_flight_costs_nothing(self, wired):
        gw, *_ = wired
        assert gw.seconds_until_quiet() == 0.0

    def test_sentences_of_one_answer_add_up(self, wired, monkeypatch):
        # Four sentences of an answer are spoken one after another, not at
        # once. Taking the longest would say the answer lasts as long as its
        # longest sentence.
        gw, *_ = wired
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: 1.0)
        for _ in range(4):
            gw.note_speech(A, {"event": "chat_sentence", "text": "a sentence"})
        assert 3.5 < gw.seconds_until_quiet() <= 4.0

    def test_an_acked_step_is_not_counted(self, wired, monkeypatch):
        # The robot reports back when it has finished speaking one of those,
        # and the run loop already waits for that. Counting it here would
        # charge for the same speech twice.
        gw, *_ = wired
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: 1.0)
        gw.note_speech(A, {"event": "demo_step", "text": "spoken", "require_ack": True})
        assert gw.seconds_until_quiet() == 0.0

    def test_an_unacked_step_is_counted(self, wired, monkeypatch):
        # "_qa_more_questions" and the guide's wrap-up are spoken and never
        # acknowledged; nothing else knows they are in flight.
        gw, *_ = wired
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: 1.0)
        gw.note_speech(A, {"event": "demo_step", "text": "spoken", "require_ack": False})
        assert gw.seconds_until_quiet() > 0.0

    def test_an_interrupt_clears_what_was_queued(self, wired, monkeypatch):
        gw, *_ = wired
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: 5.0)
        gw.note_speech(A, {"event": "chat_sentence", "text": "a long answer"})
        gw.note_speech(A, {"event": "tts_stop"})
        assert gw.seconds_until_quiet() == 0.0

    def test_an_emotion_tag_alone_is_not_speech(self, wired):
        gw, *_ = wired
        gw.note_speech(A, {"event": "chat_sentence", "text": "[DEFAULT]"})
        assert gw.seconds_until_quiet() == 0.0

    def test_the_wait_is_capped(self, wired, monkeypatch):
        # A muted or disconnected robot must not be able to park the tour.
        gw, *_ = wired
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: 1000.0)
        gw.note_speech(A, {"event": "chat_sentence", "text": "forever"})
        assert gw.seconds_until_quiet() == gw.MAX_QUIET_WAIT_SEC

    def test_it_is_fed_by_sending(self, wired, monkeypatch):
        # The estimate must not depend on any caller remembering to report —
        # every outbound utterance goes through send_to_robot.
        # The real method, not the fixture's capturing stub — the point is
        # that no caller has to remember to report speech.
        from gateway.websocket_gateway import WebSocketGateway
        gw, *_ = wired
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: 2.0)
        WebSocketGateway.send_to_robot(gw, A, {"event": "chat_sentence",
                                               "text": "an answer"})
        assert gw.seconds_until_quiet() > 0.0


class TestPickingUpAfterAQuestion:
    """
    When a Q&A window closes, the robot's next scripted sentence used to
    begin with no acknowledgement that the detour had ended — which reads as
    the robot having forgotten the exchange rather than returning from it.
    """

    def _orch(self):
        from demo.demo_orchestrator import DemoOrchestrator, DemoStep

        class Stub:
            def send_to_robot(self, *a, **k): pass
        o = DemoOrchestrator(Stub())
        o.load_script([DemoStep(step_id="s", robot_id=A, text="Explain.",
                                block_robot_id=A)])
        return o, o._script[0]

    def test_no_hint_before_any_question(self):
        o, step = self._orch()
        assert o._resuming_hint(step) == ""

    def test_the_same_block_picks_up(self):
        o, step = self._orch()
        o._resume_after_qa, o._qa_block = True, A
        assert "carrying on" in o._resuming_hint(step)

    def test_it_is_consumed_once(self):
        # Only the first step back carries it; the rest of the block runs
        # normally.
        o, step = self._orch()
        o._resume_after_qa, o._qa_block = True, A
        assert o._resuming_hint(step)
        assert o._resuming_hint(step) == ""

    def test_a_different_block_does_not_pick_up(self):
        # The guide's transition already covers a change of project.
        o, step = self._orch()
        o._resume_after_qa, o._qa_block = True, "someone_else"
        assert o._resuming_hint(step) == ""

    def test_a_step_belonging_to_no_block_does_not_pick_up(self):
        # The opening and the wrap-up belong to the tour, not to a project.
        # Nothing of theirs was ever interrupted by a project's Q&A.
        from demo.demo_orchestrator import DemoStep
        o, _ = self._orch()
        o._resume_after_qa, o._qa_block = True, A
        assert o._resuming_hint(
            DemoStep(step_id="wrap_up", robot_id=GUIDE, text="Thanks all.")) == ""


class TestOnlyAnInterruptionIsPickedUpFrom:
    """
    The lead-in belongs to the ad-hoc window a VISITOR opens by speaking up
    mid-presentation — the only case where a presentation really was cut off
    with something left to say.

    A block's own scripted Q&A step sits at the END of that block, so nothing
    is left to pick up and the next thing said is the guide moving on to
    another robot entirely. Arming it there had Pepper open every single
    transition with "Let me carry on where I left off. Coming back to our
    next project..." — announcing a return to something already finished.
    """

    def _orch(self):
        class Stub:
            def send_to_robot(self, *a, **k): pass
        o = DemoOrchestrator(Stub())
        o.load_script(build_script(GUIDE, [A, B]))
        return o

    def _qa_step_of(self, o, robot_id):
        return next(s for s in o._script
                    if s.block_robot_id == robot_id and s.role == StepRole.QA)

    def _run_scripted_window(self, o, robot_id):
        # A real one, opened and closed. _open_qa_window CLEARS the close
        # event on entry, so the window cannot be pre-closed — it has to be
        # given a budget instead, and a hundredth of a second of it is enough
        # to reach the exit path this test is about.
        from dataclasses import replace
        o._state = DemoState.QA_WINDOW
        o._open_qa_window(replace(self._qa_step_of(o, robot_id), qa_timeout=0.01))

    def test_a_scripted_qa_window_arms_nothing(self):
        o = self._orch()
        self._run_scripted_window(o, A)
        assert o._resume_after_qa is False

    def test_the_transition_after_a_scripted_window_has_no_lead_in(self):
        # The reported line, end to end: the guide's transition to the next
        # robot must not open by carrying on from anything.
        o = self._orch()
        self._run_scripted_window(o, A)
        transition = next(s for s in o._script
                          if s.role == StepRole.TRANSITION and s.block_robot_id == A)
        assert o._resuming_hint(transition) == ""

    def test_an_interruption_arms_the_lead_in_for_that_block(self):
        o = self._orch()
        o._idx = next(i for i, s in enumerate(o._script)
                      if s.block_robot_id == A and s.role == StepRole.PROJECT)
        o._state = DemoState.QA_WINDOW      # what qa_interrupt() sets
        o._qa_end_event.set()
        assert o._wait_if_interrupted_qa() is True
        assert o._resume_after_qa is True
        assert o._qa_block == A
        assert "carrying on" in o._resuming_hint(o._script[o._idx])

    def test_an_interruption_does_not_arm_another_robots_block(self):
        o = self._orch()
        o._idx = next(i for i, s in enumerate(o._script)
                      if s.block_robot_id == A and s.role == StepRole.PROJECT)
        o._state = DemoState.QA_WINDOW
        o._qa_end_event.set()
        o._wait_if_interrupted_qa()
        b_step = next(s for s in o._script if s.block_robot_id == B)
        assert o._resuming_hint(b_step) == ""


class TestTheVerbatimResumeSaysSoItself:
    """
    The remainder of an interrupted sentence is repeated verbatim, never
    regenerated, so it cannot be ASKED to introduce itself the way a
    generated step can. Without a lead-in of its own the robot answered the
    visitor's question and then, with no seam at all, resumed on the second
    half of a sentence.
    """

    def _orch_mid_project(self):
        class Stub:
            def send_to_robot(self, *a, **k): pass
        o = DemoOrchestrator(Stub())
        o.load_script(build_script(GUIDE, [A, B]))
        o._state = DemoState.QA_WINDOW
        o._idx = next(i for i, s in enumerate(o._script)
                      if s.block_robot_id == A and s.role == StepRole.PROJECT)
        return o

    def test_the_remainder_is_spoken_after_a_lead_in(self):
        o = self._orch_mid_project()
        cut = o._script[o._idx]
        o.note_interrupted_step(cut.robot_id, cut.step_id, "It matters because of X.")
        resume = o._script[o._idx + 1]
        assert resume.text.startswith(o.RESUME_LEAD_IN)
        assert resume.text.endswith("It matters because of X.")
        assert resume.generate is False

    def test_the_lead_in_lands_after_an_emotion_tag(self):
        # The client reads the tag off the front of the text; a lead-in
        # pushed in ahead of it would be spoken and the tag lost.
        o = self._orch_mid_project()
        cut = o._script[o._idx]
        o.note_interrupted_step(cut.robot_id, cut.step_id, "[HAPPY] And that is the impact.")
        assert o._script[o._idx + 1].text == f"[HAPPY] {o.RESUME_LEAD_IN}And that is the impact."

    def test_it_is_not_said_twice(self):
        o = self._orch_mid_project()
        assert (o._with_resume_lead_in(o.RESUME_LEAD_IN + "the rest of it")
                == o.RESUME_LEAD_IN + "the rest of it")

    def test_resume_suffixes_do_not_stack(self):
        # A resume that is itself interrupted stays "<step>_resume", never
        # "<step>_resume_resume_resume".
        o = self._orch_mid_project()
        cut = o._script[o._idx]
        o.note_interrupted_step(cut.robot_id, cut.step_id, "first remainder")
        o._idx += 1
        resume = o._script[o._idx]
        o.note_interrupted_step(resume.robot_id, resume.step_id, "second remainder")
        assert o._script[o._idx + 1].step_id == f"{cut.step_id}_resume"

    def test_a_verbatim_resume_consumes_the_generated_lead_in(self):
        # Otherwise the flag survives the resume — which already introduced
        # itself — and lands on a LATER generated step, which opens with
        # "let me carry on where I left off" minutes after the interruption.
        o = self._orch_mid_project()
        cut = o._script[o._idx]
        o.note_interrupted_step(cut.robot_id, cut.step_id, "the rest of it")
        o._resume_after_qa, o._qa_block = True, A
        o._send_step(o._script[o._idx + 1])
        assert o._resume_after_qa is False


class TestTheGuideWrapsUpAfterTheAnswer:
    """
    The guide's "Shall we move on to the next part of the demo?" is a
    judgement made when GENERATION finishes — and the robot that just
    answered is still working through its sentences at that point.

    A live run had Pepper start that line while Navel was two sentences into
    explaining the pipeline, and in the same second Navel was told to ask "Do
    you have any other questions, or shall we continue the demonstration?".
    Two voices, one second, contradicting each other about what was being
    asked. Both moves are reasonable alone; what was missing is that they are
    alternatives, and that one of them has to wait.
    """

    WRAP_UP = "Wonderful! Shall we move on to the next part of the demo?"

    def _interjecting(self, gw, monkeypatch):
        from decision.models import Action
        from decision.policy import PolicyResult

        def fake_decide(point, decider, user_utterance=""):
            gw._scratch.wrap_up_text = self.WRAP_UP
            return PolicyResult(Action.guide_interject(GUIDE), "llm_moderator")

        monkeypatch.setattr(gw, "_decide", fake_decide)

    def _capture(self, gw):
        sent = []
        gw.send_to_robot = lambda cid, data: sent.append((cid, data))
        return sent

    def _slow_speech(self, monkeypatch, seconds=0.2):
        """Give the pending wrap-up a window to be called off inside.

        The autouse fixture zeroes speaking time, which makes the timer fire
        almost instantly — fine for "does it fire at all", useless for "can
        it be stopped". The real delay is seconds long; this is the same
        shape, small enough to keep the test quick.
        """
        monkeypatch.setattr("gateway.websocket_gateway._speaking_seconds",
                            lambda _t: seconds)

    def test_it_reports_what_it_set_in_motion(self, wired, monkeypatch):
        gw, _orch, _r, _closed = wired
        self._capture(gw)
        self._interjecting(gw, monkeypatch)
        assert gw.check_qa_auto_close(A, A_REAL_ANSWER) == "wrap_up"

    def test_a_sign_off_reports_the_close_it_scheduled(self, wired):
        gw, _orch, _r, _closed = wired
        assert gw.check_qa_auto_close(A, SIGN_OFF) == "auto_close"

    def test_an_ordinary_answer_reports_nothing(self, wired):
        gw, _orch, _r, _closed = wired
        assert gw.check_qa_auto_close(A, A_REAL_ANSWER) is None

    def test_nothing_is_reported_outside_a_qa_window(self, wired):
        gw, orch, _r, _closed = wired
        orch._state = DemoState.RUNNING
        assert gw.check_qa_auto_close(A, SIGN_OFF) is None

    def test_the_wrap_up_waits_out_the_answer(self, wired, monkeypatch):
        # The reported symptom: the guide talking over the robot it is
        # reacting to.
        gw, _orch, _r, _closed = wired
        sent = self._capture(gw)
        monkeypatch.undo()          # restore the real speaking-time estimate
        self._interjecting(gw, monkeypatch)
        gw.note_speech(A, {"event": "chat_sentence", "text": "word " * 60})
        gw.check_qa_auto_close(A, "word " * 60)
        _settle(0.3)
        assert sent == [], "the guide spoke while the answer was still being said"

    def test_the_wrap_up_is_said_once_the_answer_is_done(self, wired, monkeypatch):
        gw, _orch, _r, _closed = wired
        sent = self._capture(gw)
        self._interjecting(gw, monkeypatch)   # speaking time zeroed by the autouse fixture
        monkeypatch.setattr("gateway.websocket_gateway.QA_WRAP_UP_SILENCE_SEC", 0.0)
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        _settle()
        assert [(cid, d["step_id"]) for cid, d in sent] == [(GUIDE, "_qa_wrap_up")]
        assert sent[0][1]["text"] == self.WRAP_UP

    def test_a_visitor_speaking_calls_the_wrap_up_off(self, wired, monkeypatch):
        # Whatever the guide was about to wrap up is stale the moment the
        # visitor asks something else. One timer slot, so the existing
        # cancel covers this too.
        gw, _orch, _r, _closed = wired
        sent = self._capture(gw)
        self._slow_speech(monkeypatch)
        self._interjecting(gw, monkeypatch)
        gw.note_speech(A, {"event": "chat_sentence", "text": A_REAL_ANSWER})
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        gw.cancel_qa_auto_close("visitor spoke")
        _settle(0.4)
        assert sent == []

    def test_the_wrap_up_does_not_outlive_its_window(self, wired, monkeypatch):
        gw, orch, _r, _closed = wired
        sent = self._capture(gw)
        self._slow_speech(monkeypatch)
        self._interjecting(gw, monkeypatch)
        gw.note_speech(A, {"event": "chat_sentence", "text": A_REAL_ANSWER})
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        orch._state = DemoState.RUNNING   # closed by someone else meanwhile
        _settle(0.4)
        assert sent == []


class TestTheWrapUpIsNotAnInterruption:
    """A live run had Pepper say "Wonderful! Shall we move on to the next
    part?" the moment every answer ended — three times in one window."""

    def _setup(self, gw, monkeypatch, silence):
        from decision.models import Action
        from decision.policy import PolicyResult

        def fake_decide(point, decider, user_utterance=""):
            gw._scratch.wrap_up_text = "Shall we move on?"
            return PolicyResult(Action.guide_interject(GUIDE), "llm_moderator")

        monkeypatch.setattr(gw, "_decide", fake_decide)
        monkeypatch.setattr("gateway.websocket_gateway.QA_WRAP_UP_SILENCE_SEC", silence)
        sent = []
        gw.send_to_robot = lambda cid, data: sent.append(data)
        return lambda: [d for d in sent if d.get("step_id") == "_qa_wrap_up"]

    def test_it_waits_for_silence(self, wired, monkeypatch):
        gw, _o, _r, _c = wired
        offers = self._setup(gw, monkeypatch, silence=0.3)
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        _settle(0.1)
        assert offers() == []
        _settle(0.4)
        assert len(offers()) == 1

    def test_a_visitor_speaking_first_cancels_it(self, wired, monkeypatch):
        gw, _o, _r, _c = wired
        offers = self._setup(gw, monkeypatch, silence=0.2)
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        gw.cancel_qa_auto_close("visitor spoke")
        _settle(0.4)
        assert offers() == []

    def test_it_is_offered_once_per_window(self, wired, monkeypatch):
        gw, _o, _r, _c = wired
        offers = self._setup(gw, monkeypatch, silence=0.0)
        for _ in range(3):
            gw.check_qa_auto_close(A, A_REAL_ANSWER)
            _settle(0.1)
        assert len(offers()) == 1

    def test_a_cancelled_offer_does_not_use_up_the_window(self, wired, monkeypatch):
        gw, _o, _r, _c = wired
        offers = self._setup(gw, monkeypatch, silence=0.2)
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        gw.cancel_qa_auto_close("visitor spoke")
        _settle(0.3)
        monkeypatch.setattr("gateway.websocket_gateway.QA_WRAP_UP_SILENCE_SEC", 0.0)
        gw.check_qa_auto_close(A, A_REAL_ANSWER)
        _settle(0.2)
        assert len(offers()) == 1


class _NoisyGateway:
    """A gateway whose robots are still speaking for `quiet_in` seconds."""

    def __init__(self, quiet_in=0.0):
        self.quiet_in = quiet_in
        self.sent = []

    def seconds_until_quiet(self):
        return self.quiet_in

    def send_to_robot(self, client_id, data):
        self.sent.append((client_id, data))


class TestTheTourWaitsForSilence:
    """
    A scripted step sent over the top of a live answer is the most confusing
    thing the tour does. A live run: the visitor asked Silbot a question, the
    answer went out at 15:24:56 and takes about nine seconds to say, and at
    15:24:57 the guide said "Thank you, Silbot, for that fascinating
    discussion. Let us move on to the next project!".

    The run loop had no way to know. A streamed answer is dispatched sentence
    by sentence and never acknowledged, so as far as the loop was concerned
    nothing was happening.
    """

    def _orch(self, ws):
        o = DemoOrchestrator(ws)
        o.load_script(build_script(GUIDE, [A, B]))
        o._state = DemoState.RUNNING   # a tour actually under way
        return o

    def test_a_gateway_that_cannot_answer_costs_nothing(self):
        # The orchestrator predates the estimate and must keep working
        # against a gateway with no such hook.
        class Old:
            def send_to_robot(self, *a, **k): pass
        assert self._orch(Old())._seconds_until_quiet() == 0.0

    def test_silence_is_not_waited_for(self):
        o = self._orch(_NoisyGateway(0.0))
        started = time.time()
        o._wait_for_quiet()
        assert time.time() - started < 0.1

    def test_speech_is_waited_out(self):
        ws = _NoisyGateway(0.3)
        o = self._orch(ws)

        def stop_soon():
            time.sleep(0.25)
            ws.quiet_in = 0.0
        threading.Thread(target=stop_soon, daemon=True).start()

        started = time.time()
        o._wait_for_quiet()
        assert time.time() - started >= 0.2, "did not wait for the robot to finish"

    def test_a_stopped_demo_does_not_keep_waiting(self):
        o = self._orch(_NoisyGateway(30.0))
        o._state = DemoState.IDLE
        started = time.time()
        o._wait_for_quiet()
        assert time.time() - started < 0.6

    def test_the_wait_is_capped(self, monkeypatch):
        # A muted or disconnected robot must not park the demonstration.
        o = self._orch(_NoisyGateway(30.0))
        monkeypatch.setattr(type(o), "MAX_QUIET_WAIT_SEC", 0.2)
        started = time.time()
        o._wait_for_quiet()
        assert time.time() - started < 0.6


class TestTheQABudgetCountsQuietSeconds:
    """
    A shortened window (see _shrink_if_already_engaged) is five seconds long,
    and a live run spent all five of them with Silbot mid-answer: the window
    opened, the visitor's question was answered into it, and the budget
    expired before the answer had finished playing. The visitor never got a
    moment of silence to ask a follow-up in — which is the only thing the
    budget is there to provide.
    """

    def _orch(self, ws):
        o = DemoOrchestrator(ws)
        o.load_script(build_script(GUIDE, [A, B]))
        return o

    def test_an_expired_budget_closes_when_all_is_quiet(self):
        o = self._orch(_NoisyGateway(0.0))
        assert o._wait_for_qa_close(0.05) is False

    def test_closing_the_window_wins_over_the_budget(self):
        o = self._orch(_NoisyGateway(0.0))
        o._qa_end_event.set()
        assert o._wait_for_qa_close(5.0) is True

    def test_the_budget_does_not_expire_mid_answer(self):
        ws = _NoisyGateway(0.4)
        o = self._orch(ws)

        def stop_soon():
            time.sleep(0.4)
            ws.quiet_in = 0.0
        threading.Thread(target=stop_soon, daemon=True).start()

        started = time.time()
        assert o._wait_for_qa_close(0.05) is False
        assert time.time() - started >= 0.4, \
            "the window closed while the answer was still being said"

    def test_an_answer_that_never_ends_cannot_hold_the_window_open(self, monkeypatch):
        # The estimate is a word count, not a microphone: a robot that goes
        # away mid-answer leaves it standing.
        o = self._orch(_NoisyGateway(5.0))
        monkeypatch.setattr(type(o), "MAX_QUIET_WAIT_SEC", 0.2)
        started = time.time()
        assert o._wait_for_qa_close(0.05) is False
        assert time.time() - started < 1.0

    def test_a_manual_only_window_still_waits_for_a_close(self):
        # timeout None means "operator closes it", and no amount of silence
        # should change that.
        o = self._orch(_NoisyGateway(0.0))
        o._qa_end_event.set()
        assert o._wait_for_qa_close(None) is True
