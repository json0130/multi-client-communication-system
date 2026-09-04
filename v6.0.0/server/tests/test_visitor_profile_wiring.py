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
