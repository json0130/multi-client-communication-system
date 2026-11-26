"""
tests/test_llm_parse.py — the reply's tag, as parse_response reads it off the front of a reply.
"""
from modules.llm.llm_provider import parse_response


def test_one_tag():
    r = parse_response("[WAVE] Hello there!")
    assert (r.emotion_tag, r.clean_text) == ("WAVE", "Hello there!")


def test_no_tag():
    r = parse_response("Hello there!")
    assert (r.emotion_tag, r.clean_text) == ("", "Hello there!")


def test_a_real_tag_after_default_wins_and_both_are_stripped():
    r = parse_response("[DEFAULT] [GREETING] I can help with that!")
    assert (r.emotion_tag, r.clean_text) == ("GREETING", "I can help with that!")


def test_default_alone_stays_default():
    assert parse_response("[DEFAULT] Sure.").emotion_tag == "DEFAULT"
