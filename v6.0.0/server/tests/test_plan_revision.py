"""
tests/test_plan_revision.py
===========================
DemoOrchestrator.revise_script and the invariant it exists to protect.

THE INVARIANT: only steps ahead of the play head are ever touched, and after a
revision `_idx` still points at the same step object it pointed at before.

That is not a style preference. The run loop is blocked on an ACK for the step at
`_idx` while a revision arrives on a Flask request thread. A revision that shifted
the index would strand that ACK and then silently skip or repeat a step — in
front of visitors, with no error anywhere. Every test below is ultimately about
that one property.

No network, no robots: the gateway is a stub that records what it was sent.
"""

from __future__ import annotations

import threading

import pytest

from decision import DecisionPoint, MemoryDecisionSink, DecisionRecorder, PlanOp, PlanOpKind
from demo.demo_orchestrator import DemoOrchestrator, DemoState, DemoStep, StepRole
from demo.demo_script import build_script

GUIDE = "pepper_01"
A = "chatbox_jetson_001"
B = "navel_001"
C = "silbot_01"


class StubGateway:
    """Records outbound traffic; generates nothing."""

    def __init__(self):
        self.sent = []

    def send_to_robot(self, client_id, data):
        self.sent.append((client_id, data))

    def generate_demo_step(self, robot_id, instruction):
        return instruction


@pytest.fixture
def gateway():
    return StubGateway()


@pytest.fixture
def sink():
    return MemoryDecisionSink()


@pytest.fixture
def orch(gateway, sink):
    """
    An orchestrator parked mid-tour without a runner thread.

    The state is set directly rather than by calling start(), because start()
    spawns a daemon that would begin dialling stub robots. revise_script only
    reads _state, _idx and _script, so this is the honest minimum.
    """
    o = DemoOrchestrator(
        gateway,
        recorder=DecisionRecorder(sink),
        session_context=lambda: {"scenario_id": "lab_demo", "session_id": "sess-1"},
    )
    o.load_script(build_script(GUIDE, [A, B, C]))
    o._state = DemoState.QA_WINDOW
    # Park on A's Q&A step — the realistic point for a visitor to ask for a change.
    o._idx = next(
        i for i, s in enumerate(o._script)
        if s.block_robot_id == A and s.role == StepRole.QA
    )
    return o


def blocks(o) -> list:
    """
    Project blocks with steps still ahead of the play head, in order.

    Parked on A's Q&A this reads [A, B, C], not [B, C] — A's farewell
    ("thanks ChatBox, moving on") comes after its Q&A and has not been said yet.
    """
    out = []
    for s in o._script[o._idx + 1:]:
        if s.block_robot_id and s.block_robot_id not in out:
            out.append(s.block_robot_id)
    return out


def step_ids(o) -> list:
    return [s.step_id for s in o._script]


# ── The invariant ─────────────────────────────────────────────────────────────

class TestPlayHeadInvariant:

    @pytest.mark.parametrize("op", [
        PlanOp(PlanOpKind.SKIP, robot_id=B),
        PlanOp(PlanOpKind.COMPRESS, robot_id=B),
        PlanOp(PlanOpKind.REORDER, robot_id=C, position=0),
        PlanOp(PlanOpKind.EXTEND_QA, robot_id=B),
        PlanOp(PlanOpKind.DROP_REMAINING),
    ])
    def test_current_step_object_survives_every_op(self, orch, op):
        current = orch._script[orch._idx]
        orch.revise_script([op])
        assert orch._script[orch._idx] is current

    @pytest.mark.parametrize("op", [
        PlanOp(PlanOpKind.SKIP, robot_id=B),
        PlanOp(PlanOpKind.DROP_REMAINING),
        PlanOp(PlanOpKind.REORDER, robot_id=C, position=0),
    ])
    def test_completed_steps_are_never_touched(self, orch, op):
        before = step_ids(orch)[: orch._idx + 1]
        orch.revise_script([op])
        assert step_ids(orch)[: orch._idx + 1] == before

    def test_skipping_the_currently_presenting_robot_leaves_its_current_step(self, orch):
        # A visitor says "skip this one" during A's own Q&A. A's remaining steps
        # go; the step being executed does not, because the run loop is inside it.
        current = orch._script[orch._idx]
        assert current.block_robot_id == A
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=A)])
        assert orch._script[orch._idx] is current
        assert A not in blocks(orch)

    def test_revision_during_waiting_ack_does_not_strand_the_ack(self, orch):
        orch._state = DemoState.WAITING_ACK
        current = orch._script[orch._idx]
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)])
        # receive_ack matches on the step at _idx; if the index moved, the ACK
        # the run loop is waiting for would be ignored forever.
        assert orch._script[orch._idx] is current
        orch.receive_ack(current.step_id)
        assert orch._ack_event.is_set()


# ── Individual operations ─────────────────────────────────────────────────────

class TestSkip:

    def test_removes_the_whole_block(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)])
        assert blocks(orch) == [A, C]
        assert not any(s.block_robot_id == B for s in orch._script[orch._idx + 1:])

    def test_takes_the_blocks_own_farewell_with_it(self, orch):
        # transition_to_<next> belongs to the block it signs off, so skipping B
        # must not leave "thanks Navel, moving on" behind.
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)])
        assert f"transition_to_{C}" not in step_ids(orch)

    def test_skipping_an_already_finished_block_is_ignored_not_an_error(self, orch):
        # By the time someone says "skip that one", it may already be over.
        result = orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=GUIDE)])
        assert result["applied"] == []
        assert result["ignored"][0]["kind"] == "skip"


class TestCompress:

    def test_keeps_the_research_and_the_qa(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.COMPRESS, robot_id=B)])
        kept = [s.role for s in orch._script if s.block_robot_id == B]
        assert StepRole.PROJECT in kept
        assert StepRole.QA in kept

    def test_drops_only_the_social_scaffolding(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.COMPRESS, robot_id=B)])
        kept = {s.role for s in orch._script if s.block_robot_id == B}
        assert not (kept & StepRole.COMPRESSIBLE)

    def test_leaves_other_blocks_alone(self, orch):
        before = [s.step_id for s in orch._script if s.block_robot_id == C]
        orch.revise_script([PlanOp(PlanOpKind.COMPRESS, robot_id=B)])
        after = [s.step_id for s in orch._script if s.block_robot_id == C]
        assert before == after

    def test_compressing_twice_is_a_no_op_the_second_time(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.COMPRESS, robot_id=B)])
        result = orch.revise_script([PlanOp(PlanOpKind.COMPRESS, robot_id=B)])
        assert result["applied"] == []


class TestDropRemaining:

    def test_keeps_the_closing_steps(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.DROP_REMAINING)])
        # A tour cut short for time still gets a proper goodbye.
        assert "wrap_up" in step_ids(orch)
        assert "open_floor" in step_ids(orch)

    def test_removes_every_remaining_project(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.DROP_REMAINING)])
        assert blocks(orch) == []

    def test_is_ignored_when_only_closing_steps_remain(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.DROP_REMAINING)])
        result = orch.revise_script([PlanOp(PlanOpKind.DROP_REMAINING)])
        assert result["applied"] == []


class TestReorder:

    def test_moves_a_block_to_the_front(self, orch):
        assert blocks(orch) == [A, B, C]
        orch.revise_script([PlanOp(PlanOpKind.REORDER, robot_id=C, position=0)])
        # A stays pinned at the front — its farewell is still pending.
        assert blocks(orch) == [A, C, B]

    def test_the_presenting_block_cannot_be_moved(self, orch):
        result = orch.revise_script([PlanOp(PlanOpKind.REORDER, robot_id=A, position=1)])
        assert result["applied"] == []
        assert "currently presenting" in result["ignored"][0]["why"]

    def test_moves_the_whole_block_not_just_one_step(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.REORDER, robot_id=C, position=0)])
        tail = orch._script[orch._idx + 1:]
        c_positions = [i for i, s in enumerate(tail) if s.block_robot_id == C]
        # Contiguous: the block moved intact rather than being interleaved.
        assert c_positions == list(range(c_positions[0], c_positions[0] + len(c_positions)))

    def test_closing_steps_stay_at_the_end(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.REORDER, robot_id=C, position=0)])
        assert step_ids(orch)[-1] == "open_floor"

    def test_out_of_range_position_clamps_rather_than_failing(self, orch):
        # A visitor request should not fail because there were fewer projects
        # left than they assumed.
        result = orch.revise_script([PlanOp(PlanOpKind.REORDER, robot_id=C, position=99)])
        assert result["applied"]
        assert blocks(orch) == [A, B, C]


class TestExtendQa:

    def test_widens_an_upcoming_window(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.EXTEND_QA, robot_id=B)])
        qa = next(s for s in orch._script if s.block_robot_id == B and s.role == StepRole.QA)
        assert qa.qa_window and qa.qa_timeout == 0.0

    def test_inserts_a_fresh_window_for_a_finished_block(self, orch):
        # The usual case: they ask for more after hearing something. It has to
        # open next, not after the whole tour.
        orch.revise_script([PlanOp(PlanOpKind.EXTEND_QA, robot_id=A)])
        nxt = orch._script[orch._idx + 1]
        assert nxt.block_robot_id == A
        assert nxt.role == StepRole.QA
        assert nxt.qa_window

    def test_the_inserted_step_is_spoken_by_the_guide(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.EXTEND_QA, robot_id=A)])
        assert orch._script[orch._idx + 1].robot_id == GUIDE


# ── Multiple ops, and rejection ───────────────────────────────────────────────

class TestBatches:

    def test_ops_apply_in_order(self, orch):
        orch.revise_script([
            PlanOp(PlanOpKind.SKIP, robot_id=B),
            PlanOp(PlanOpKind.COMPRESS, robot_id=C),
        ])
        assert blocks(orch) == [A, C]
        kept = {s.role for s in orch._script if s.block_robot_id == C}
        assert not (kept & StepRole.COMPRESSIBLE)

    def test_a_partially_valid_batch_applies_what_it_can(self, orch):
        # Half a revision applying is correct when a visitor asks for something
        # that is partly already true.
        result = orch.revise_script([
            PlanOp(PlanOpKind.SKIP, robot_id=B),
            PlanOp(PlanOpKind.SKIP, robot_id="nonexistent_robot"),
        ])
        assert len(result["applied"]) == 1
        assert len(result["ignored"]) == 1

    def test_op_without_a_robot_id_is_ignored(self, orch):
        result = orch.revise_script([PlanOp(PlanOpKind.SKIP)])
        assert result["applied"] == []

    @pytest.mark.parametrize("state", [DemoState.IDLE, DemoState.COMPLETED, DemoState.ERROR])
    def test_refused_when_no_demo_is_running(self, orch, state):
        orch._state = state
        before = step_ids(orch)
        result = orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)])
        assert result["applied"] == []
        assert step_ids(orch) == before


# ── Correction recording ──────────────────────────────────────────────────────

class TestCorrectionRecording:

    def test_an_operator_revision_is_recorded(self, orch, sink):
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)], source="operator",
                           reason="visitors are short on time")
        assert len(sink.corrections) == 1
        c = sink.corrections[0]
        assert c.decision_point == DecisionPoint.PLAN_REVISE.value
        assert c.reason == "visitors are short on time"
        assert c.session_id == "sess-1"
        assert c.scenario_id == "lab_demo"

    def test_the_systems_own_revision_is_not_a_correction(self, orch, sink):
        # It is a decision, already logged where it was made. Counting it here
        # too would make the correction rate meaningless.
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)], source="policy")
        assert sink.corrections == []

    def test_a_revision_that_changed_nothing_is_not_recorded(self, orch, sink):
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id="nobody")], source="operator")
        assert sink.corrections == []

    def test_operator_move_on_is_a_correction(self, orch, sink):
        orch.manual_next(source="operator", reason="dragging")
        assert len(sink.corrections) == 1
        assert sink.corrections[0].corrected_to_kind == "advance"

    def test_operator_interrupt_is_the_opposite_correction(self, orch, sink):
        orch._state = DemoState.RUNNING
        orch.qa_interrupt(source="operator", reason="someone had a question")
        assert sink.corrections[0].corrected_to_kind == "stay"

    def test_automatic_advances_are_not_corrections(self, orch, sink):
        orch.qa_end()                      # default source="auto"
        orch.qa_end(source="policy")
        assert sink.corrections == []

    def test_an_unannotated_click_is_still_recorded(self, orch, sink):
        # The timestamp and step are the label; the reason is a bonus. Requiring
        # one would discard most of the training signal.
        orch.manual_next(source="operator")
        assert len(sink.corrections) == 1
        assert sink.corrections[0].reason == ""

    def test_a_failing_recorder_never_breaks_the_demo(self, gateway):
        class BrokenRecorder:
            def clear(self): pass
            def flush(self): pass
            def live_decision_id(self, step_id): raise RuntimeError("db gone")
            def record_correction(self, e): raise RuntimeError("db gone")

        o = DemoOrchestrator(gateway, recorder=BrokenRecorder())
        o.load_script(build_script(GUIDE, [A]))
        o._state = DemoState.QA_WINDOW
        o.manual_next(source="operator")     # must not raise
        assert o._ack_event.is_set()


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_revisions_leave_a_consistent_script(self, orch):
        # Flask serves each control request on its own thread while the run loop
        # advances steps; revise_script must hold the lock for the whole splice.
        current = orch._script[orch._idx]
        errors = []

        def worker(robot_id):
            try:
                orch.revise_script([PlanOp(PlanOpKind.COMPRESS, robot_id=robot_id)])
            except Exception as e:      # pragma: no cover - the point of the test
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(r,)) for r in (B, C, B, C)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert orch._script[orch._idx] is current
        assert len(step_ids(orch)) == len(set(step_ids(orch)))


class TestQaBudget:
    """
    Q&A as an allocation, not a prediction.

    Scripted step length is a property of the content and averages usefully;
    Q&A length is a property of the operator and the group and does not. So the
    planner does not ask how long a window will run — it says how long it gets.

    That also makes it the right first cut. Q&A is the largest share of tour
    time and the least noticed when shortened: nobody remembers a question round
    that ended early, everybody remembers a robot that never spoke.
    """

    def test_a_budget_is_written_onto_the_upcoming_window(self, orch):
        orch.revise_script([PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B, seconds=90)])
        qa = next(s for s in orch._script
                  if s.block_robot_id == B and s.role == StepRole.QA)
        assert qa.qa_timeout == 90.0 and qa.qa_window

    def test_zero_restores_manual_advance(self, orch):
        # The right default for an unhurried tour: a budget is a response to
        # time pressure, not a standing policy.
        orch.revise_script([PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B, seconds=90)])
        orch.revise_script([PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B, seconds=0)])
        qa = next(s for s in orch._script
                  if s.block_robot_id == B and s.role == StepRole.QA)
        assert qa.qa_timeout == 0.0

    def test_a_budget_without_seconds_is_ignored(self, orch):
        result = orch.revise_script([PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B)])
        assert result["applied"] == []

    def test_a_block_with_no_upcoming_window_is_ignored(self, orch):
        result = orch.revise_script([
            PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id="nobody", seconds=60)])
        assert result["applied"] == []

    def test_budgeting_does_not_disturb_the_play_head(self, orch):
        current = orch._script[orch._idx]
        orch.revise_script([PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B, seconds=60)])
        assert orch._script[orch._idx] is current

    def test_budgeting_removes_no_steps(self, orch):
        before = len(orch._script)
        orch.revise_script([PlanOp(PlanOpKind.SET_QA_BUDGET, robot_id=B, seconds=60)])
        assert len(orch._script) == before


class TestCompressionLadder:
    """Time pressure tightens Q&A BEFORE it trims anything, and drops last."""

    def _obs(self, **kw):
        from tests.test_decision_policy import visitor_turn
        return visitor_turn("we are running out of time", **kw)

    def test_qa_is_budgeted_before_blocks_are_compressed(self):
        from decision import DecisionPoint, HeuristicPolicy
        r = HeuristicPolicy().decide(DecisionPoint.PLAN_REVISE, self._obs())
        kinds = [o.kind for o in r.action.ops]
        assert PlanOpKind.SET_QA_BUDGET in kinds
        assert kinds.index(PlanOpKind.SET_QA_BUDGET) < kinds.index(PlanOpKind.COMPRESS)

    def test_nothing_is_dropped_for_mild_pressure(self):
        from decision import DecisionPoint, HeuristicPolicy
        r = HeuristicPolicy().decide(DecisionPoint.PLAN_REVISE, self._obs())
        assert PlanOpKind.DROP_REMAINING not in [o.kind for o in r.action.ops]

    def test_severe_overrun_still_drops(self):
        from decision import DecisionPoint, HeuristicPolicy
        obs = self._obs(elapsed_sec=900.0, time_budget_sec=600.0,
                        projected_overrun_sec=400.0)
        r = HeuristicPolicy().decide(DecisionPoint.PLAN_REVISE, obs)
        assert [o.kind for o in r.action.ops] == [PlanOpKind.DROP_REMAINING]
