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
