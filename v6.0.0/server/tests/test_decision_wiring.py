"""
tests/test_decision_wiring.py
=============================
The seam between the gateway, the orchestrator and the decision layer.

The unit tests prove the policy decides correctly and the orchestrator splices
safely. This proves the two are actually connected — that a visitor turn produces
a logged decision, that an operator's button press lands as a correction *of that
decision*, and that a dead logging database cannot take a live demo down with it.

Everything below the LLM and the database is real: real HeuristicPolicy, real
DecisionRecorder, real orchestrator. Only the robots, the LLM and the sink's
writer are faked.
"""

from __future__ import annotations

import warnings

import pytest

from core.rbac import AccessLevel, RobotIdentity
from decision import (
    ActionKind,
    DecisionPoint,
    DecisionRecorder,
    Mechanism,
    MemoryDecisionSink,
)
from demo.demo_orchestrator import DemoOrchestrator, DemoState, StepRole
from demo.demo_script import build_script
from gateway.websocket_gateway import WebSocketGateway

pytestmark = pytest.mark.integration

GUIDE = "pepper_01"
A = "chatbox_jetson_001"
B = "navel_001"
SCENARIO = "lab_demo"


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeInstance:
    def __init__(self, client_id, name, role, level=AccessLevel.LOCAL, intent="continue"):
        self.client_id = client_id
        self.robot_name = name
        self.access_level = level
        self._role = role
        self._intent = intent
        self.classified = []

    @property
    def identity(self):
        return RobotIdentity(
            robot_id=self.client_id, scenario_id=SCENARIO,
            session_id=f"sess-{self.client_id}",
            access_level=self.access_level, role=self._role,
        )

    def classify_qa_intent(self, message):
        self.classified.append(message)
        return self._intent


class FakeRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}

    def get(self, client_id):
        return self._by_id.get(client_id)

    def get_all(self, exclude_id=None):
        return [i for i in self._by_id.values() if i.client_id != exclude_id]


@pytest.fixture
def sink():
    return MemoryDecisionSink()


@pytest.fixture
def wired(sink):
    """A gateway and orchestrator wired exactly as app.create_app() wires them."""
    registry = FakeRegistry([
        FakeInstance(GUIDE, "Pepper", "Lab guide", AccessLevel.GLOBAL),
        FakeInstance(A, "ChatBox", "RAG research"),
        FakeInstance(B, "Navel", "Emotion research"),
    ])
    recorder = DecisionRecorder(sink)
    gw = WebSocketGateway(registry, recorder=recorder)

    orch = DemoOrchestrator(gw, recorder=recorder, session_context=gw.session_context)
    orch.load_script(build_script(GUIDE, [A, B]))
    gw.set_demo_orchestrator(orch)

    # Park mid-tour in an open Q&A window, without starting the runner thread.
    orch._state = DemoState.QA_WINDOW
    orch._idx = next(
        i for i, s in enumerate(orch._script)
        if s.block_robot_id == A and s.role == StepRole.QA
    )
    gw.on_qa_window_open()
    # Stub the transport — nothing is dialling a real robot here.
    gw.send_to_robot = lambda cid, data: gw._sent.append((cid, data))
    gw._sent = []
    return gw, orch, registry


# ── Decisions reach the log ───────────────────────────────────────────────────

class TestDecisionsAreRecorded:

    def test_a_visitor_turn_logs_a_decision(self, wired, sink):
        gw, _, registry = wired
        gw.check_qa_advance_from_user(registry.get(A), "what is RAG?")

        assert len(sink.decisions) == 1
        d = sink.decisions[0]
        assert d.decision_point == DecisionPoint.QA_ADVANCE.value
        assert d.mechanism == Mechanism.QUESTION_HEURISTIC
        assert d.action_kind == "stay"

    def test_the_logged_observation_carries_rbac_identifiers(self, wired, sink):
        gw, _, registry = wired
        gw.check_qa_advance_from_user(registry.get(A), "what is RAG?")

        d = sink.decisions[0]
        # The join to rbac_audit_log. Without these the exposure question the
        # schema exists to answer becomes unanswerable.
        assert d.session_id == f"sess-{A}"
        assert d.scenario_id == SCENARIO
        assert d.decider_access_level == "local"

    def test_the_logged_observation_carries_the_team(self, wired, sink):
        gw, _, registry = wired
        gw.check_qa_advance_from_user(registry.get(A), "what is RAG?")

        peers = sink.decisions[0].observation["connected_peers"]
        assert {p["client_id"] for p in peers} == {GUIDE, A, B}
        assert all("robot_role" in p for p in peers)

    def test_mechanism_is_never_null(self, wired, sink):
        # Every comparison in the analysis groups by mechanism, so a null makes
        # the row useless. The schema enforces it; this checks we never try.
        gw, _, registry = wired
        for msg in ["what is RAG?", "move on", "hmm alright"]:
            gw.check_qa_advance_from_user(registry.get(A), msg)
        assert all(d.mechanism for d in sink.decisions)


# ── Parity at the seam ────────────────────────────────────────────────────────

class TestBehaviourParity:

    def test_advance_phrase_closes_the_window(self, wired, sink):
        gw, orch, registry = wired
        result = gw.check_qa_advance_from_user(registry.get(A), "okay let's move on")

        assert result.action.kind is ActionKind.ADVANCE
        assert result.mechanism == Mechanism.ADVANCE_PHRASE
        assert orch._qa_end_event.is_set()

    def test_a_question_never_reaches_the_classifier(self, wired):
        # The cheap path stays cheap: an obvious question must not cost an LLM
        # call, which is why the heuristic exists at all.
        gw, _, registry = wired
        instance = registry.get(A)
        gw.check_qa_advance_from_user(instance, "how does retrieval work?")
        assert instance.classified == []

    def test_the_classifier_is_bound_to_the_robot_that_was_addressed(self, wired):
        gw, _, registry = wired
        addressed = registry.get(B)
        gw.check_qa_advance_from_user(addressed, "hmm alright then")
        # Not the guide, and not whichever robot spoke last.
        assert addressed.classified == ["hmm alright then"]
        assert registry.get(A).classified == []

    def test_classifier_done_closes_the_window(self, sink):
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG", intent="done"),
        ])
        recorder = DecisionRecorder(sink)
        gw = WebSocketGateway(registry, recorder=recorder)
        orch = DemoOrchestrator(gw, recorder=recorder)
        orch.load_script(build_script(GUIDE, [A]))
        gw.set_demo_orchestrator(orch)
        orch._state = DemoState.QA_WINDOW

        result = gw.check_qa_advance_from_user(registry.get(A), "no I think that's it")
        assert result.action.kind is ActionKind.ADVANCE
        assert result.mechanism == Mechanism.LLM_CLASSIFIER
        assert orch._qa_end_event.is_set()

    def test_no_decision_is_made_outside_a_qa_window(self, wired, sink):
        gw, orch, registry = wired
        orch._state = DemoState.RUNNING
        assert gw.check_qa_advance_from_user(registry.get(A), "move on") is None
        assert sink.decisions == []


# ── Corrections attach to decisions ───────────────────────────────────────────

class TestCorrections:

    def test_operator_move_on_attaches_to_the_live_decision(self, wired, sink):
        gw, orch, registry = wired
        gw.check_qa_advance_from_user(registry.get(A), "what is RAG?")
        decision_id = sink.decisions[0].decision_id

        orch.manual_next(source="operator", reason="dragging on")

        assert len(sink.corrections) == 1
        c = sink.corrections[0]
        # This link is what makes "how often is this mechanism overridden?" a
        # join rather than a guess.
        assert c.decision_id == decision_id
        assert c.corrected_to_kind == "advance"
        assert c.reason == "dragging on"

    def test_the_correction_carries_the_session_from_the_guide(self, wired, sink):
        gw, orch, registry = wired
        gw.check_qa_advance_from_user(registry.get(A), "what is RAG?")
        orch.manual_next(source="operator")

        # session_context() reads the guide, who is present for the whole run,
        # rather than a project robot that may disconnect mid-tour.
        assert sink.corrections[0].session_id == f"sess-{GUIDE}"

    def test_an_orphan_correction_is_still_recorded(self, wired, sink):
        # No decision was logged for this step, but the operator still told us
        # the window should have closed. That is the label that matters most.
        gw, orch, _ = wired
        orch.manual_next(source="operator", reason="nobody was talking")

        assert len(sink.corrections) == 1
        assert sink.corrections[0].decision_id is None

    def test_a_policy_advance_is_not_double_counted(self, wired, sink):
        # The policy closing a window is a decision, already logged. Recording
        # it as its own correction would make the correction rate meaningless.
        gw, _, registry = wired
        gw.check_qa_advance_from_user(registry.get(A), "move on")

        assert len(sink.decisions) == 1
        assert sink.corrections == []


# ── Plan revision at the seam ─────────────────────────────────────────────────

class TestPlanRevisionWiring:

    def test_a_visitor_asking_to_skip_shortens_the_script(self, wired, sink):
        gw, orch, registry = wired
        before = len(orch._script)

        changed = gw.check_plan_revision(registry.get(A), "can we skip the Navel part")

        assert changed
        assert len(orch._script) < before
        assert not any(s.block_robot_id == B for s in orch._script)

    def test_an_ordinary_question_changes_nothing(self, wired):
        gw, orch, registry = wired
        before = [s.step_id for s in orch._script]
        gw.check_plan_revision(registry.get(A), "what is retrieval augmented generation?")
        assert [s.step_id for s in orch._script] == before

    def test_a_revision_is_logged_as_a_decision(self, wired, sink):
        gw, _, registry = wired
        gw.check_plan_revision(registry.get(A), "we are running out of time")

        revisions = [d for d in sink.decisions
                     if d.decision_point == DecisionPoint.PLAN_REVISE.value]
        assert revisions
        assert revisions[-1].action_payload["ops"]


# ── Failure modes ─────────────────────────────────────────────────────────────

class TestDegradation:

    def test_a_dead_sink_does_not_stop_the_demo(self, wired):
        """
        Non-negotiable before any live demo: with the database gone, the tour
        still runs and the only symptom is a warning.
        """
        gw, orch, registry = wired

        class DeadSink:
            def record(self, e): raise RuntimeError("supabase unreachable")
            def record_correction(self, e): raise RuntimeError("supabase unreachable")
            def flush(self): raise RuntimeError("supabase unreachable")

        gw._recorder = DecisionRecorder(DeadSink())
        orch._recorder = gw._recorder

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = gw.check_qa_advance_from_user(registry.get(A), "move on")
            orch.manual_next(source="operator")

        # The decision was still made and still acted on.
        assert result.action.kind is ActionKind.ADVANCE
        assert orch._qa_end_event.is_set()

    def test_no_recorder_at_all_is_fine(self, wired):
        gw, orch, registry = wired
        gw._recorder = DecisionRecorder()       # NullDecisionSink
        orch._recorder = gw._recorder
        result = gw.check_qa_advance_from_user(registry.get(A), "what is RAG?")
        assert result.action.kind is ActionKind.STAY

    def test_a_disconnected_guide_does_not_break_session_context(self, wired):
        gw, _, registry = wired
        registry._by_id.pop(GUIDE)
        assert gw.session_context() == {}


class TestCorrectionAttachment:
    """
    A correction must attach to a decision of the SAME decision point.

    One visitor turn produces three decisions on one step — QA_ADVANCE,
    PLAN_REVISE, QA_ROUTE. Keying live decisions on step_id alone made every
    correction attach to whichever was logged last, so an operator clicking
    "Move On" was recorded as overriding the routing choice and the correction
    rate per mechanism measured nothing. Caught by tools/demo_sim.py.
    """

    def _turn(self, gw, orch, registry):
        from decision import DecisionPoint as DP
        instance = registry.get(A)
        gw.check_qa_advance_from_user(instance, "what is RAG?")
        gw.check_plan_revision(instance, "what is RAG?")
        gw._decide(DP.QA_ROUTE, instance, "what is RAG?")

    def test_move_on_attaches_to_the_advance_decision(self, wired, sink):
        gw, orch, registry = wired
        self._turn(gw, orch, registry)
        orch.manual_next(source="operator", reason="dragging")

        c = sink.corrections[0]
        parent = next(d for d in sink.decisions if d.decision_id == c.decision_id)
        assert parent.decision_point == DecisionPoint.QA_ADVANCE.value
        assert parent.decision_point == c.decision_point

    def test_interrupt_attaches_to_the_advance_decision(self, wired, sink):
        gw, orch, registry = wired
        self._turn(gw, orch, registry)
        orch._state = DemoState.RUNNING
        orch.qa_interrupt(source="operator")

        c = sink.corrections[0]
        parent = next(d for d in sink.decisions if d.decision_id == c.decision_id)
        assert parent.decision_point == DecisionPoint.QA_ADVANCE.value

    def test_a_revision_attaches_to_the_plan_revise_decision(self, wired, sink):
        gw, orch, registry = wired
        self._turn(gw, orch, registry)
        from decision import PlanOp, PlanOpKind
        orch.revise_script([PlanOp(PlanOpKind.SKIP, robot_id=B)],
                           source="operator", reason="short on time")

        c = sink.corrections[-1]
        parent = next(d for d in sink.decisions if d.decision_id == c.decision_id)
        assert parent.decision_point == DecisionPoint.PLAN_REVISE.value

    def test_every_correction_matches_its_parents_point(self, wired, sink):
        gw, orch, registry = wired
        self._turn(gw, orch, registry)
        orch.manual_next(source="operator")
        orch.qa_end(source="operator")

        by_id = {d.decision_id: d for d in sink.decisions}
        for c in sink.corrections:
            if c.decision_id:
                assert by_id[c.decision_id].decision_point == c.decision_point


class TestSinkWriteOrdering:
    """
    A correction must never reach the database before the decision it names.

    demo_correction_log.decision_id is a foreign key, so an out-of-order write is
    rejected with 23503 and the correction — the training signal — is lost. An
    earlier sink used one queue per event type and drained decisions first, which
    looks ordered but is not: the queues fill concurrently, so a correction
    enqueued just after a drain pass overtakes its parent. Seen in a live run.
    """

    def _sink(self, writes):
        from decision import BatchingDecisionSink
        return BatchingDecisionSink(
            decision_writer=lambda b: writes.extend(("decision", e.decision_id) for e in b),
            correction_writer=lambda b: writes.extend(("correction", e.decision_id) for e in b),
            batch_size=5, flush_interval_sec=0.05,
        )

    def test_interleaved_events_are_written_in_record_order(self):
        from decision import Action, DecisionPoint, Observation, build_correction, build_decision
        writes: list = []
        sink = self._sink(writes)

        expected = []
        for i in range(12):
            d = build_decision(DecisionPoint.QA_ADVANCE, Action.stay(),
                               "m", Observation(step_id=f"s{i}"))
            sink.record(d)
            expected.append(("decision", d.decision_id))
            if i % 2 == 0:
                c = build_correction(DecisionPoint.QA_ADVANCE, Action.advance(),
                                     "operator", decision_id=d.decision_id,
                                     step_id=f"s{i}")
                sink.record_correction(c)
                expected.append(("correction", d.decision_id))
        sink.shutdown()

        assert writes == expected

    def test_a_correction_never_precedes_its_decision(self):
        from decision import Action, DecisionPoint, Observation, build_correction, build_decision
        writes: list = []
        sink = self._sink(writes)

        for i in range(30):
            d = build_decision(DecisionPoint.QA_ADVANCE, Action.stay(),
                               "m", Observation(step_id=f"s{i}"))
            sink.record(d)
            sink.record_correction(build_correction(
                DecisionPoint.QA_ADVANCE, Action.advance(), "operator",
                decision_id=d.decision_id, step_id=f"s{i}"))
        sink.shutdown()

        seen: set = set()
        for kind, did in writes:
            if kind == "decision":
                seen.add(did)
            else:
                assert did in seen, "correction written before its decision"

    def test_a_failing_writer_does_not_stall_the_rest(self):
        from decision import Action, BatchingDecisionSink, DecisionPoint, Observation, build_decision
        ok: list = []
        sink = BatchingDecisionSink(
            decision_writer=lambda b: ok.extend(b),
            correction_writer=lambda b: (_ for _ in ()).throw(RuntimeError("nope")),
            batch_size=2, flush_interval_sec=0.05,
        )
        from decision import build_correction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sink.record_correction(build_correction(
                DecisionPoint.QA_ADVANCE, Action.advance(), "operator", step_id="s0"))
            for i in range(4):
                sink.record(build_decision(DecisionPoint.QA_ADVANCE, Action.stay(),
                                           "m", Observation(step_id=f"s{i}")))
            sink.shutdown()
        assert len(ok) == 4


class TestKGRouting:
    """
    QA_ROUTE backed by the competence graph, and the guarantee that it can never
    strand a question.
    """

    def _router(self, edges=(), links=(), topics=None):
        from decision.kg_policy import KGRouter
        return KGRouter(edges, links,
                        topics if topics is not None else [
                            {"id": "topic:rag", "label": "retrieval augmented generation"},
                            {"id": "topic:emotion", "label": "emotion recognition"},
                        ])

    def _wire(self, sink, router):
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG"),
            FakeInstance(B, "Navel", "emotion"),
        ])
        recorder = DecisionRecorder(sink)
        gw = WebSocketGateway(registry, recorder=recorder,
                              kg_router_factory=lambda: router)
        orch = DemoOrchestrator(gw, recorder=recorder)
        orch.load_script(build_script(GUIDE, [A, B]))
        gw.set_demo_orchestrator(orch)
        orch._state = DemoState.QA_WINDOW
        gw.send_to_robot = lambda cid, d: None
        return gw, registry

    def test_the_graph_can_override_the_receiver(self, sink):
        from decision.kg import Evidence, RobotTopicEdge
        e = RobotTopicEdge(robot_id=B, topic_id="topic:rag")
        for _ in range(10):
            e = e.update(1.0, Evidence.SUPERVISOR)
        gw, registry = self._wire(sink, self._router(edges=[e]))

        # A asked the question; the graph says B handles this subject.
        result = gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                            "how does retrieval augmented generation work")
        assert result.action.robot_id == B
        assert result.mechanism in ("kg_argmax", "kg_explore")

    def test_an_unresolvable_question_falls_back_to_the_receiver(self, sink):
        gw, registry = self._wire(sink, self._router())
        result = gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                            "what is the weather like today")
        assert result.action.robot_id == A
        assert result.mechanism == Mechanism.RECEIVER

    def test_an_unseeded_graph_falls_back_to_the_receiver(self, sink):
        gw, registry = self._wire(sink, None)
        result = gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                            "how does retrieval augmented generation work")
        assert result.mechanism == Mechanism.RECEIVER

    def test_a_broken_router_falls_back_rather_than_raising(self, sink):
        class Exploding:
            def decide(self, *a, **kw): raise RuntimeError("graph is on fire")
        gw, registry = self._wire(sink, Exploding())
        result = gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                            "retrieval augmented generation")
        assert result.action.robot_id == A
        assert result.mechanism == Mechanism.RECEIVER

    def test_the_routing_decision_is_still_logged(self, sink):
        from decision.kg import Evidence, RobotTopicEdge
        e = RobotTopicEdge(robot_id=B, topic_id="topic:emotion")
        for _ in range(10):
            e = e.update(1.0, Evidence.SUPERVISOR)
        gw, registry = self._wire(sink, self._router(edges=[e]))
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(A), "emotion recognition")

        routes = [d for d in sink.decisions
                  if d.decision_point == DecisionPoint.QA_ROUTE.value]
        assert routes and routes[-1].mechanism.startswith("kg_")

    def test_an_ambiguous_question_does_not_route(self, sink):
        # Two topics matching equally means the question named neither.
        router = self._router(topics=[
            {"id": "topic:a", "label": "emotion recognition"},
            {"id": "topic:b", "label": "emotion detection"},
        ])
        assert router.resolve_topic("tell me about emotion") is None


class TestOutcomeEmission:
    """
    The window-close signal — the only way the graph hears about routing that
    went RIGHT. Without it every edge is built from failures and human_share is
    trivially 100%.
    """

    def _wire(self, sink, router, observed):
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG"),
            FakeInstance(B, "Navel", "emotion"),
        ])
        gw = WebSocketGateway(registry, recorder=DecisionRecorder(sink),
                              kg_router_factory=lambda: router,
                              kg_observer=observed.extend)
        orch = DemoOrchestrator(gw, recorder=DecisionRecorder(sink))
        orch.load_script(build_script(GUIDE, [A, B]))
        gw.set_demo_orchestrator(orch)
        orch._state = DemoState.QA_WINDOW
        gw.send_to_robot = lambda cid, d: None
        return gw, registry

    def _router(self):
        from decision.kg_policy import KGRouter
        return KGRouter([], [], [
            {"id": "topic:rag", "label": "retrieval augmented generation"},
        ])

    def test_a_silent_window_credits_nothing(self, sink):
        observed = []
        gw, _ = self._wire(sink, self._router(), observed)
        gw.on_qa_window_open()
        gw.on_qa_window_close()
        assert observed == []

    def test_a_routed_question_is_credited_at_close(self, sink):
        observed = []
        gw, registry = self._wire(sink, self._router(), observed)
        gw.on_qa_window_open()
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                   "how does retrieval augmented generation work")
        gw.on_qa_window_close()
        assert len(observed) == 1
        assert observed[0].topic_id == "topic:rag"

    def test_a_corrected_window_credits_nothing(self, sink):
        observed = []
        gw, registry = self._wire(sink, self._router(), observed)
        gw.on_qa_window_open()
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(A),
                   "retrieval augmented generation")
        gw.note_routing_correction()
        gw.on_qa_window_close()
        assert observed == []

    def test_an_unresolvable_question_credits_nothing(self, sink):
        observed = []
        gw, registry = self._wire(sink, self._router(), observed)
        gw.on_qa_window_open()
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(A), "what is the weather")
        gw.on_qa_window_close()
        assert observed == []

    def test_state_does_not_leak_between_windows(self, sink):
        observed = []
        gw, registry = self._wire(sink, self._router(), observed)
        gw.on_qa_window_open()
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(A), "retrieval augmented generation")
        gw.on_qa_window_close()
        observed.clear()
        gw.on_qa_window_open()          # a second, silent window
        gw.on_qa_window_close()
        assert observed == []

    def test_a_failing_observer_does_not_break_the_demo(self, sink):
        def explode(_obs):
            raise RuntimeError("graph is unreachable")
        registry = FakeRegistry([
            FakeInstance(GUIDE, "Pepper", "guide", AccessLevel.GLOBAL),
            FakeInstance(A, "ChatBox", "RAG"),
        ])
        gw = WebSocketGateway(registry, recorder=DecisionRecorder(sink),
                              kg_router_factory=self._router, kg_observer=explode)
        orch = DemoOrchestrator(gw, recorder=DecisionRecorder(sink))
        orch.load_script(build_script(GUIDE, [A]))
        gw.set_demo_orchestrator(orch)
        orch._state = DemoState.QA_WINDOW
        gw.send_to_robot = lambda cid, d: None
        gw.on_qa_window_open()
        gw._decide(DecisionPoint.QA_ROUTE, registry.get(A), "retrieval augmented generation")
        gw.on_qa_window_close()          # must not raise
