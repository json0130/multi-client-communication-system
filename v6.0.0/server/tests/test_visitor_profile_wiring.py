"""
tests/test_visitor_profile_wiring.py
=====================================
Style framing actually reaching generation, and the fallback-safety bug found
while wiring it: comparing the "did generation succeed" check against the
wrong baseline would have made a genuine generation FAILURE speak the raw
framing directive text aloud instead of silently falling back.
"""

from __future__ import annotations

import pytest

from decision.visitor_profile import VisitorProfile
from demo.demo_orchestrator import DemoOrchestrator, DemoStep, StepRole

GUIDE = "pepper_01"
A = "chatbox_jetson_001"


class RecordingGateway:
    """Records exactly what instruction generate_demo_step was called with,
    and lets a test control whether generation "succeeds"."""

    def __init__(self, succeed=True):
        self.sent = []
        self.received_instructions = []
        self._succeed = succeed

    def send_to_robot(self, client_id, data):
        self.sent.append((client_id, data))

    def generate_demo_step(self, robot_id, instruction):
        self.received_instructions.append(instruction)
        if self._succeed:
            return f"[DEFAULT] a generated line about {instruction[:10]}"
        return instruction   # the real fallback behaviour: echoes its input


def make_orch(gateway, profile=None):
    orch = DemoOrchestrator(gateway, transition_delay=0.0)
    orch.load_script([
        DemoStep(step_id="s1", robot_id=A, text="Explain your research project.",
                generate=True, require_ack=False, block_robot_id=A, role=StepRole.PROJECT),
    ])
    orch._visitor_profile = profile
    return orch


class TestFramingReachesGeneration:

    def test_no_profile_sends_the_plain_instruction(self):
        gw = RecordingGateway()
        orch = make_orch(gw)
        orch._send_step(orch._script[0])
        assert gw.received_instructions == ["Explain your research project."]

    def test_a_style_profile_appends_its_framing(self):
        gw = RecordingGateway()
        profile = VisitorProfile(style="technical")
        orch = make_orch(gw, profile)
        orch._send_step(orch._script[0])
        assert gw.received_instructions[0].startswith("Explain your research project.")
        assert "technical audience" in gw.received_instructions[0]

    def test_general_style_appends_nothing(self):
        gw = RecordingGateway()
        profile = VisitorProfile(style="general")
        orch = make_orch(gw, profile)
        orch._send_step(orch._script[0])
        assert gw.received_instructions[0] == "Explain your research project."

    def test_framing_applies_regardless_of_which_robot_or_step(self):
        # Style is a visitor property, not a robot property — every generated
        # step gets it uniformly.
        gw = RecordingGateway()
        profile = VisitorProfile(style="business")
        orch = DemoOrchestrator(gw, transition_delay=0.0)
        orch.load_script([
            DemoStep(step_id="s1", robot_id=GUIDE, text="Welcome the visitors.",
                    generate=True, require_ack=False, role=StepRole.OPENING),
            DemoStep(step_id="s2", robot_id=A, text="Explain your research.",
                    generate=True, require_ack=False, block_robot_id=A, role=StepRole.PROJECT),
        ])
        orch._visitor_profile = profile
        orch._send_step(orch._script[0])
        orch._send_step(orch._script[1])
        assert all("business audience" in i for i in gw.received_instructions)


class TestFallbackSafety:
    """
    The bug: comparing generated output against step.text (the bare
    instruction) rather than against what generation was actually GIVEN
    (instruction + framing) meant every framed call looked like a "success"
    even when generation had genuinely failed and just echoed its input back —
    and the visible fallback branch, which speaks the plain instruction, would
    never fire. The robot would instead speak the framing directive itself
    aloud: "...Explain your research. Frame this for a technical audience..."
    """

    def test_a_genuine_failure_falls_back_to_the_plain_instruction(self):
        gw = RecordingGateway(succeed=False)   # echoes its input, like the real fallback
        profile = VisitorProfile(style="technical")
        orch = make_orch(gw, profile)
        orch._send_step(orch._script[0])
        spoken = orch.get_status()["text"]
        assert spoken == "Explain your research project."
        assert "technical audience" not in spoken

    def test_a_genuine_success_is_recognised_as_such(self):
        gw = RecordingGateway(succeed=True)
        profile = VisitorProfile(style="technical")
        orch = make_orch(gw, profile)
        orch._send_step(orch._script[0])
        spoken = orch.get_status()["text"]
        assert spoken.startswith("[DEFAULT] a generated line")

    def test_the_llm_was_still_given_the_framing_even_on_failure(self):
        # The framing must reach generation regardless of outcome — only the
        # SPOKEN fallback text must never include it.
        gw = RecordingGateway(succeed=False)
        profile = VisitorProfile(style="technical")
        orch = make_orch(gw, profile)
        orch._send_step(orch._script[0])
        assert "technical audience" in gw.received_instructions[0]


# ── Style fit reaching the instruction ────────────────────────────────────────

class TestStyleFitReachesGeneration:
    """
    A learned audience fit is only worth anything if it changes what the
    robot is actually briefed with. These run the real _send_step against a
    recording gateway and read the instruction that came out.
    """

    def _orch(self, style_fit=None):
        from decision.visitor_profile import VisitorProfile
        gw = RecordingGateway()
        o = DemoOrchestrator(gw, style_fit=style_fit)
        o.load_script([DemoStep(step_id="s", robot_id="silbot_01",
                                text="Explain your project.", generate=True)])
        o._visitor_profile = VisitorProfile(style="technical")
        return o, gw

    def test_with_no_lookup_the_instruction_is_the_plain_directive(self):
        from decision.visitor_profile import STYLE_FRAMING
        o, gw = self._orch(style_fit=None)
        o._send_step(o._script[0])
        assert gw.received_instructions[-1] == "Explain your project." + STYLE_FRAMING["technical"]

    def test_a_poorly_rated_robot_gets_the_corrective_note(self):
        from decision.style_fit import STYLE_REINFORCEMENT, StyleFit
        poor = StyleFit(robot_id="silbot_01", style="technical")
        for _ in range(3):
            poor = poor.record(0.0)
        o, gw = self._orch(style_fit=lambda r, s: poor)
        o._send_step(o._script[0])
        assert STYLE_REINFORCEMENT["technical"] in gw.received_instructions[-1]

    def test_a_well_rated_robot_does_not(self):
        from decision.style_fit import STYLE_REINFORCEMENT, StyleFit
        good = StyleFit(robot_id="silbot_01", style="technical")
        for _ in range(5):
            good = good.record(1.0)
        o, gw = self._orch(style_fit=lambda r, s: good)
        o._send_step(o._script[0])
        assert STYLE_REINFORCEMENT["technical"] not in gw.received_instructions[-1]

    def test_a_failing_lookup_falls_back_to_the_plain_directive(self):
        # Best-effort throughout: a style-fit lookup that raises must never
        # stop a robot speaking.
        from decision.visitor_profile import STYLE_FRAMING

        def boom(robot_id, style):
            raise RuntimeError("supabase down")
        o, gw = self._orch(style_fit=boom)
        o._send_step(o._script[0])
        assert gw.received_instructions[-1] == "Explain your project." + STYLE_FRAMING["technical"]

    def test_no_profile_means_no_framing_at_all(self):
        o, gw = self._orch(style_fit=None)
        o._visitor_profile = None
        o._send_step(o._script[0])
        assert gw.received_instructions[-1] == "Explain your project."


class TestQAAnswersAreStyledToo:
    """
    The visitor profile used to reach scripted steps only. A technical visitor
    got precise language from the project talk and generic language the moment
    they asked a follow-up — and Q&A is where a visitor spends most of their
    attention, so half the styling was missing from the half that matters most.
    """

    def _orch(self, style="technical"):
        from decision.visitor_profile import VisitorProfile
        gw = RecordingGateway()
        o = DemoOrchestrator(gw)
        o.load_script([DemoStep(step_id="s", robot_id="silbot_01", text="x")])
        o._visitor_profile = VisitorProfile(style=style)
        return o, gw

    def test_the_run_exposes_its_framing(self):
        from decision.visitor_profile import STYLE_FRAMING
        o, _ = self._orch("technical")
        assert o.framing_for_robot("silbot_01") == STYLE_FRAMING["technical"]

    def test_no_profile_means_no_framing(self):
        o, _ = self._orch()
        o._visitor_profile = None
        assert o.framing_for_robot("silbot_01") == ""

    def test_general_style_frames_nothing(self):
        o, _ = self._orch("general")
        assert o.framing_for_robot("silbot_01") == ""

    def test_the_gateway_lookup_is_best_effort(self):
        # A framing lookup must never be able to stop a robot answering.
        from gateway.websocket_gateway import WebSocketGateway

        class Boom:
            def framing_for_robot(self, robot_id):
                raise RuntimeError("nope")

        gw = WebSocketGateway.__new__(WebSocketGateway)
        gw._demo_orchestrator = Boom()
        assert WebSocketGateway.style_framing_for(gw, "silbot_01") == ""

    def test_no_orchestrator_means_no_framing(self):
        from gateway.websocket_gateway import WebSocketGateway
        gw = WebSocketGateway.__new__(WebSocketGateway)
        gw._demo_orchestrator = None
        assert WebSocketGateway.style_framing_for(gw, "silbot_01") == ""
