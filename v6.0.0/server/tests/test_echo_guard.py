from gateway.websocket_gateway import _looks_like_echo

SAID = ["Hi everyone, I'm ChatBox, excited to share about our project!",
        "A robot might forget what you told it last time, like a companion robot that can't recall a past request."]


def test_mic_hearing_the_speaker_is_echo():
    heard = ("Hi, everyone. My chat box. It's kind of to share about that project. Oh, I see. "
             "A robot might forget what you told it last time. Like a companion robot that can't recall")
    assert _looks_like_echo(heard, SAID)


def test_a_visitor_question_is_not():
    assert not _looks_like_echo("How does the robot remember what I told it yesterday", SAID[:1])
    assert not _looks_like_echo("yes go ahead", SAID)
    assert not _looks_like_echo("anything at all here", [])
