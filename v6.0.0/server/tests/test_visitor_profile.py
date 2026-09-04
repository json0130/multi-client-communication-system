"""
tests/test_visitor_profile.py
==============================
The explicit trigger-type ordering, and the pre-demo profile it resolves
against.

resolve_emphasis makes explicit a sequence that used to be split across
decision/policy.py and app.py with no single place stating it: a fresh
utterance beats the standing profile, which beats having nothing at all.
"""

from __future__ import annotations

import pytest

from decision.planner import resolve_emphasis
from decision.visitor_profile import DEFAULT_STYLE, STYLE_FRAMING, VisitorProfile


class TestResolveEmphasis:

    def test_a_fresh_utterance_wins_over_the_profile(self):
        topics, source = resolve_emphasis(["topic:emotion"], ["topic:navigation"])
        assert topics == ("topic:emotion",) and source == "utterance"

    def test_the_profile_is_used_when_nothing_was_just_said(self):
        topics, source = resolve_emphasis(None, ["topic:navigation"])
        assert topics == ("topic:navigation",) and source == "profile"

    def test_neither_present_returns_empty_and_says_so(self):
        topics, source = resolve_emphasis(None, None)
        assert topics == () and source == "none"

    def test_an_empty_utterance_list_falls_through_to_the_profile(self):
        # [] and None must behave the same — both mean "nothing was resolved".
        topics, source = resolve_emphasis([], ["topic:navigation"])
        assert topics == ("topic:navigation",) and source == "profile"

    def test_a_refinement_replaces_rather_than_merges(self):
        # A visitor who stated an interest at the start and then asks about
        # something else mid-tour is refining, not adding — merging would keep
        # inflating a topic they have since moved past.
        topics, _source = resolve_emphasis(["topic:emotion"], ["topic:navigation"])
        assert "topic:navigation" not in topics


class TestVisitorProfile:

    def test_default_style_has_no_framing(self):
        p = VisitorProfile()
        assert p.style == DEFAULT_STYLE
        assert p.framing == ""

    @pytest.mark.parametrize("style", ["technical", "business", "interactive"])
    def test_every_non_default_style_has_a_framing_directive(self, style):
        p = VisitorProfile(style=style)
        assert len(p.framing) > 10

    def test_an_unknown_style_degrades_to_no_framing_not_an_error(self):
        p = VisitorProfile(style="made_up_style")
        assert p.framing == ""

    def test_frozen_profile_cannot_be_mutated(self):
        p = VisitorProfile(interest_text="x")
        with pytest.raises(Exception):
            p.interest_text = "y"

    def test_as_dict_round_trips_the_fields(self):
        p = VisitorProfile(interest_text="emotion recognition", style="business",
                          topics=("topic:emotion-recognition",))
        d = p.as_dict()
        assert d == {"interest_text": "emotion recognition", "style": "business",
                    "topics": ["topic:emotion-recognition"]}

    def test_every_framing_string_is_declared_for_a_real_style_key(self):
        # Guards against a typo in STYLE_FRAMING that silently never fires.
        assert set(STYLE_FRAMING) >= {"technical", "business", "interactive", "general"}
