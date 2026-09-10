"""
tests/test_grounding.py
========================
decision/grounding.py and the prompt rule it feeds.

The property that matters is narrow: a robot must not state a specific it
does not have. A live run had Silbot answer "Extended Kalman Filters for
state estimation and particle filters for tracking multiple people" and
ChatBox answer "a retrieval model like BM25" — neither string appears
anywhere in this system. The model filled a gap, confidently, in front of
visitors.

So the tests here are about what reaches the prompt: that unverified drafts
are visibly marked, that another robot's results never appear as this
robot's own, and that having NO facts still changes the instruction rather
than silently leaving the model free.
"""

from __future__ import annotations

from decision.grounding import UNVERIFIED, format_facts
from robot.prompt_builder import build_delegation_prompt

ROBOT = "silbot_01"


def row(**kw):
    base = {"topic_id": "topic:social-robot-navigation", "robot_id": ROBOT,
            "kind": "method", "fact": "Plans socially-aware paths.",
            "verified": True, "id": 1}
    base.update(kw)
    return base


def prompt(facts=()):
    system, _user = build_delegation_prompt(
        "Silbot", "navigation researcher", ["[DEFAULT]"],
        "which technique do you use", [], (), grounded_facts=facts)
    return system


class TestFormatting:
    def test_a_verified_fact_carries_no_marker(self):
        assert UNVERIFIED not in format_facts([row()])[0]

    def test_an_unverified_fact_is_marked(self):
        out = format_facts([row(verified=False)])[0]
        assert UNVERIFIED in out

    def test_a_topic_general_fact_is_not_claimed_as_this_robots_result(self):
        # Background about the field must not be narrated as "our finding".
        out = format_facts([row(robot_id=None)])[0]
        assert "background" in out.lower()

    def test_the_kind_is_stated(self):
        assert format_facts([row(kind="limitation")])[0].startswith("limitation:")

    def test_an_empty_fact_is_dropped_not_rendered_blank(self):
        # A blank bullet reads as a fact the robot could not recall.
        assert format_facts([row(fact="  "), row(fact="real")]) == \
            ["method: real"]

    def test_no_rows_gives_no_lines(self):
        assert format_facts([]) == [] and format_facts(None) == []


class TestThePromptRule:
    def test_the_anti_invention_rule_is_always_present(self):
        # Even with nothing to ground on — that is the case it exists for.
        assert "NEVER INVENT SPECIFICS" in prompt()
        assert "NEVER INVENT SPECIFICS" in prompt(["method: something"])

    def test_it_names_what_must_not_be_invented(self):
        p = prompt()
        for word in ("model", "algorithm", "dataset", "number", "paper"):
            assert word in p.lower(), f"the rule does not mention {word}"

    def test_declining_is_framed_as_the_good_outcome(self):
        # Without this the model reads the rule as "be vague", not "say so".
        assert "would need to check" in prompt()

    def test_the_rule_puts_the_answer_before_the_caveat(self):
        """Leading with the disclaimer is its own failure.

        The rule worked but produced "I can explain the approach, but I would
        have to check the exact model" as an OPENING line, twice in a row, to
        a visitor who had asked a simple question. Being honest about a gap is
        right; spending the first sentence on it is not — the visitor came for
        the answer, and the facts above usually contain most of one.
        """
        p = prompt()
        assert "LEAD WITH WHAT YOU DO KNOW" in p
        assert "not open with a disclaimer" in p.lower()

    def test_facts_reach_the_prompt(self):
        assert "Plans socially-aware paths." in prompt(format_facts([row()]))

    def test_the_unverified_caveat_appears_only_with_facts(self):
        assert "UNVERIFIED" in prompt(format_facts([row(verified=False)]))
        assert "WHAT IS ACTUALLY TRUE" not in prompt()

    def test_the_prompt_still_builds_with_no_facts_argument(self):
        # Callers that predate grounding must keep working.
        system, _ = build_delegation_prompt(
            "Silbot", "r", ["[DEFAULT]"], "hi", [], ())
        assert "NEVER INVENT SPECIFICS" in system


class TestRepositoryScoping:
    def test_another_robots_facts_are_never_returned(self, monkeypatch):
        # A robot must not narrate a colleague's results as its own work.
        from data import demo_facts_repo

        rows = [row(robot_id="silbot_01", id=1),
                row(robot_id="chatbox_01", id=2, fact="ChatBox's own result"),
                row(robot_id=None, id=3, fact="Shared background")]

        class FakeTable:
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def execute(self): return type("R", (), {"data": rows})()
        monkeypatch.setattr(demo_facts_repo, "get_client",
                            lambda: type("C", (), {"table": lambda s, n: FakeTable()})())
        got = demo_facts_repo.facts_for("topic:social-robot-navigation", "silbot_01")
        assert {r["id"] for r in got} == {1, 3}

    def test_limitations_are_ordered_first(self):
        # The most useful thing a research demo can say, and the least
        # likely to be invented.
        from data.demo_facts_repo import KIND_ORDER
        assert KIND_ORDER[0] == "limitation"
