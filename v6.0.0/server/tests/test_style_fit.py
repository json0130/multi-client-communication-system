"""
tests/test_style_fit.py
========================
decision/style_fit.py — how well a robot pitches to an audience, and what a
learned value actually changes.

The properties worth pinning are the boundaries, because this is the third
learned quantity in the system and the whole point of adding it separately
was that it must not blur into the other two:

  * it must never affect routing — a robot that explains navigation badly is
    still the robot that knows navigation
  * it must move only on a person's judgement, never on a clean Q&A window,
    which says nothing about how an answer was pitched
  * a deployment with no ratings must generate exactly what it generated
    before the feature existed
"""

from __future__ import annotations

import pytest

from decision.style_fit import (REINFORCE_BELOW, STYLE_REINFORCEMENT,
                                StyleFit, framing_for)
from decision.visitor_profile import STYLE_FRAMING

R = "silbot_01"


def rated(style: str, target: float, n: int = 1, robot: str = R) -> StyleFit:
    fit = StyleFit(robot_id=robot, style=style)
    for _ in range(n):
        fit = fit.record(target)
    return fit


class TestNothingKnownChangesNothing:
    """The property that makes this safe to ship: with no ratings the
    generation path produces byte-identical instructions to the ones it
    produced before style fit existed."""

    def test_no_fit_gives_the_plain_directive(self):
        for style, base in STYLE_FRAMING.items():
            assert framing_for(style, None) == base

    def test_an_unrated_fit_gives_the_plain_directive(self):
        fit = StyleFit(robot_id=R, style="technical")
        assert framing_for("technical", fit) == STYLE_FRAMING["technical"]

    def test_general_style_stays_empty_even_when_rated_badly(self):
        # No directive was given, so there is nothing to correct toward.
        assert framing_for("general", rated("general", 0.0, n=8)) == ""

    def test_an_unknown_style_is_empty_not_an_error(self):
        assert framing_for("archaeological", None) == ""


class TestReinforcementNeedsRealEvidence:

    def test_one_poor_rating_does_not_change_the_briefing(self):
        # The clamp is doing the work: a single judgement lands near neutral,
        # so one bad night cannot permanently change how a robot is briefed.
        fit = rated("technical", 0.0, n=1)
        assert fit.clamped >= REINFORCE_BELOW
        assert framing_for("technical", fit) == STYLE_FRAMING["technical"]

    def test_two_poor_ratings_do(self):
        # Two independent judgements is the documented bar — see
        # REINFORCE_BELOW, which is set from the clamp curve rather than by eye.
        fit = rated("technical", 0.0, n=2)
        assert fit.clamped < REINFORCE_BELOW
        out = framing_for("technical", fit)
        assert out.startswith(STYLE_FRAMING["technical"])
        assert STYLE_REINFORCEMENT["technical"] in out

    def test_good_ratings_never_reinforce(self):
        fit = rated("business", 1.0, n=8)
        assert framing_for("business", fit) == STYLE_FRAMING["business"]

    def test_reinforcement_is_reversible(self):
        # A robot that improves stops getting the corrective note.
        fit = rated("business", 0.0, n=4)
        assert fit.needs_reinforcement
        for _ in range(8):
            fit = fit.record(1.0)
        assert not fit.needs_reinforcement

    def test_every_directive_style_has_a_correction(self):
        # A style that framed but could not be corrected would silently do
        # nothing when a robot was rated badly on it.
        for style, base in STYLE_FRAMING.items():
            if base:
                assert STYLE_REINFORCEMENT.get(style), f"{style} has no correction"


class TestArithmeticMatchesTheGraph:
    """Reused from decision/kg.py rather than reimplemented, so there is one
    update rule in the codebase."""

    def test_a_rating_moves_partway_never_all_the_way(self):
        fit = rated("technical", 1.0, n=1)
        assert 0.5 < fit.weight < 1.0

    def test_repeated_agreement_converges(self):
        fit = rated("technical", 1.0, n=20)
        assert fit.weight > 0.9

    def test_disagreement_averages_rather_than_thrashing(self):
        fit = StyleFit(robot_id=R, style="technical")
        for target in (1.0, 0.0) * 6:
            fit = fit.record(target)
        assert 0.3 < fit.weight < 0.7

    def test_confidence_grows_with_evidence(self):
        assert (rated("technical", 1.0, n=1).confidence
                < rated("technical", 1.0, n=5).confidence)

    def test_an_unrated_fit_clamps_to_exactly_neutral(self):
        assert StyleFit(robot_id=R, style="technical").clamped == 0.5

    def test_a_row_round_trips(self):
        fit = rated("business", 0.0, n=3)
        restored = StyleFit.from_row(fit.as_row())
        assert restored.robot_id == fit.robot_id
        assert restored.style == fit.style
        assert restored.n_supervisor == fit.n_supervisor
        assert restored.weight == pytest.approx(fit.weight, abs=1e-5)


class TestItIsPerRobot:
    def test_two_robots_are_tracked_separately(self):
        good = rated("technical", 1.0, n=5, robot="chatbox_01")
        poor = rated("technical", 0.0, n=5, robot="navel_01")
        assert not good.needs_reinforcement
        assert poor.needs_reinforcement

    def test_two_styles_are_tracked_separately(self):
        # The same robot can be fine with one audience and poor with another;
        # that is the whole reason this is keyed by style.
        assert rated("technical", 1.0, n=5).needs_reinforcement is False
        assert rated("business", 0.0, n=5).needs_reinforcement is True


class TestItNeverTouchesRouting:
    def test_style_fit_is_not_reachable_from_the_router(self):
        # Structural: a robot's audience fit must not be able to change who
        # answers. If kg_infer ever imports it, that separation is gone.
        import inspect
        from decision import kg_infer, kg_policy
        for module in (kg_infer, kg_policy):
            assert "style_fit" not in inspect.getsource(module)


class TestTheRatingPathIsReachable:
    """
    Rating was endpoint-only, which meant it was never going to happen during
    a live tour — so the table stayed empty and the framing never adapted.
    These pin the pieces the dashboard panel needs.
    """

    def test_the_run_reports_its_visitor_style(self):
        # The panel highlights the style in force so an operator does not
        # have to remember which audience is in front of them.
        from decision.visitor_profile import VisitorProfile
        from demo.demo_orchestrator import DemoOrchestrator, DemoStep

        class Stub:
            def send_to_robot(self, *a, **k): pass
        o = DemoOrchestrator(Stub())
        o.load_script([DemoStep(step_id="s", robot_id="r", text="x")])
        assert o.get_status()["visitor_style"] is None
        o._visitor_profile = VisitorProfile(style="business")
        assert o.get_status()["visitor_style"] == "business"

    def test_the_panel_threshold_matches_the_code(self):
        # The panel shows "corrective note on" below REINFORCE_BELOW. If the
        # constant moves and the panel does not, it reports the wrong state.
        import re
        src = open("robot-dashboard/src/components/KGTab.jsx").read()
        shown = re.search(r"clamped < ([0-9.]+)", src)
        assert shown, "the panel no longer shows the reinforcement threshold"
        assert float(shown.group(1)) == REINFORCE_BELOW

    def test_every_rateable_style_has_a_directive(self):
        # A style the panel offers but that frames nothing would collect
        # ratings that change no behaviour.
        import re
        src = open("robot-dashboard/src/components/KGTab.jsx").read()
        listed = re.search(r"const STYLES = \[([^\]]+)\]", src)
        assert listed
        for st in re.findall(r"'([a-z]+)'", listed.group(1)):
            assert STYLE_FRAMING.get(st), f"{st} is rateable but frames nothing"
