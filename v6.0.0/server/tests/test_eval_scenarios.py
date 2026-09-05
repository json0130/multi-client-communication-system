"""
tests/test_eval_scenarios.py
=============================
Runs tools/eval_scenarios.py's pre-registered scenarios as tests.

The scenarios exist to score conditions against each other later, but they
double as the sharpest regression net in the suite: each one asserts an
end-to-end OUTCOME a visitor would experience, derived by hand from the
documented rules, rather than any single function's return value. A change
that quietly reorders the compression ladder, breaks importance weighting,
stops routing acting on QA_ROUTE, or makes an absent robot answerable will
fail here even if every unit test still passes.

Kept as one test per scenario rather than a loop so a failure names which
scenario broke, and per-assertion detail so it names which property.
"""

from __future__ import annotations

import pytest

from tools.eval_scenarios import SCENARIOS, score


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_matches_its_pre_registered_expectation(scenario):
    results = score(scenario, scenario.run())
    failures = [f"{name}: {detail}" for name, passed, detail in results if not passed]
    assert not failures, (
        f"{scenario.id} diverged from its pre-registered expectation.\n"
        f"  Rationale for the expectation: {scenario.rationale}\n"
        f"  Failed: " + "; ".join(failures) + "\n"
        f"  Either the reasoning in the rationale is wrong or the "
        f"implementation changed — decide which before editing the expectation."
    )


def test_every_scenario_records_its_derivation():
    # An expectation with no written reasoning is one nobody can check, and
    # the first thing a future reader would do is re-derive it from the
    # implementation — which is exactly what pre-registration prevents.
    for sc in SCENARIOS:
        assert sc.rationale.strip(), f"{sc.id} has no rationale"
        assert len(sc.rationale) > 120, f"{sc.id}'s rationale is too thin to check"


def test_scenarios_cover_both_planning_and_routing():
    from tools.eval_scenarios import ExpectedPlan, ExpectedRoute
    kinds = {type(s.expected) for s in SCENARIOS}
    assert ExpectedPlan in kinds and ExpectedRoute in kinds


def test_s2_would_fail_if_importance_were_ignored(monkeypatch):
    """
    S2 is the one scenario that exists to prove importance reaches the
    planner, so it has to be able to FAIL when it does not. Its earlier form
    could not distinguish that cleanly: the two non-interest blocks were tied
    at the default, and the alphabet — not importance — decided which one was
    cut. This pins the discrimination directly by making the mutation and
    checking the scenario notices.
    """
    import tools.eval_scenarios as ev

    s2 = next(s for s in SCENARIOS if s.id.startswith("S2"))
    assert not [n for n, p, _ in ev.score(s2, s2.run()) if not p], \
        "S2 must pass before the mutation, or this test proves nothing"

    # The bug: importance is computed but every block comes out equal, so the
    # planner's (importance, robot_id) sort falls through to the alphabet.
    monkeypatch.setattr(
        ev, "block_importance",
        lambda graph, **kw: {b.robot_id: 0.5 for b in graph.blocks},
    )
    failures = [n for n, p, _ in ev.score(s2, s2.run()) if not p]
    assert "surviving_blocks" in failures
    assert "interest changed the cut" in failures


def test_hand_derived_defaults_are_all_distinct():
    """With any two blocks tied, the planner's sort reaches its robot_id
    tie-break and the alphabet decides — which would make an expectation
    about WHICH block was cut evidence about ordering rather than about
    importance. Distinct defaults keep the tie-break unreachable."""
    from tools.eval_scenarios import DEFAULTS
    assert len(set(DEFAULTS.values())) == len(DEFAULTS)


# ── Generated scenarios ───────────────────────────────────────────────────────

def _generated():
    from tools.eval_scenarios import generated_cases
    return generated_cases()


@pytest.mark.parametrize("case", _generated(), ids=[c["id"] for c in _generated()])
def test_generated_case_matches_the_rule_derived_expectation(case):
    """
    Inputs generated automatically; expected outputs derived by
    tools/eval_oracle.py from the documented rules, never read off what the
    planner produced. A suite whose expectations came from the system would
    pass on the day the ladder was reordered wrongly, because the
    expectations would reorder with it.
    """
    from tools.eval_scenarios import run_generated
    failures = [f"{n}: {d}" for n, p, d in run_generated(case) if not p]
    assert not failures, (
        f"{case['id']} disagrees with the rule-derived expectation: "
        + "; ".join(failures)
    )


def test_the_generated_grid_is_actually_broad():
    cases = _generated()
    plans = [c for c in cases if c["kind"] == "plan"]
    routes = [c for c in cases if c["kind"] == "route"]
    assert len(cases) >= 40, f"only {len(cases)} generated cases"
    # Every rung has to be reachable somewhere in the grid, or whole branches
    # of the ladder are untested no matter how many cases there are.
    outcomes = [c["expected"] for c in plans]
    assert any(not o.compressed_blocks and o.surviving_blocks for o in outcomes), \
        "no case fits without compressing — rung 0 untested"
    assert any(o.compressed_blocks and len(o.surviving_blocks) == len(c["projects"])
               for o, c in zip(outcomes, plans)), \
        "no case compresses without skipping — rung 2 untested"
    assert any(0 < len(o.surviving_blocks) < len(c["projects"])
               for o, c in zip(outcomes, plans)), "no case skips — rung 3 untested"
    assert any(not o.feasible for o in outcomes), "no infeasible case — rung 4 untested"
    assert {r.deferred_to is not None for r in (c["expected"] for c in routes)} == {True, False}, \
        "routing grid does not cover both deferred and answered"
