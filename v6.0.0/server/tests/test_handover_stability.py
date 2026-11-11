from conversation.turn import HANDOVER_OFFER
from gateway.websocket_gateway import WebSocketGateway, _same_utterance, _words


def test_two_mics_one_utterance():
    a = set(_words("What about the gestures like co-speech motion just"))
    b = set(_words("Oh, I see. What about the gestures, like Coast Beach motion gestures?"))
    assert _same_utterance(a, b, 2.0)
    assert _same_utterance(set(_words("Yes.")), set(_words("yeah")), 1.0)
    assert not _same_utterance(set(_words("Yes.")), set(_words("yeah")), 5.0)
    assert not _same_utterance(a, set(_words("why do we need that system")), 1.0)


def test_only_the_first_mic_is_a_turn():
    gw = WebSocketGateway.__new__(WebSocketGateway)
    import threading
    gw._speech_lock, gw._heard_recent = threading.Lock(), []
    q = "What about the gestures like co-speech motion"
    assert not gw.heard_elsewhere("navel_001", q)
    assert gw.heard_elsewhere("chatbox_001", q)
    assert not gw.heard_elsewhere("navel_001", q)        # the same robot asking again is a new turn


def test_models_own_offer_is_never_spoken():
    for t in ("Shall I pass this to Navel?", "Would you like me to ask Navel?", "Should we hand this over?"):
        assert HANDOVER_OFFER.search(t), t
    assert not HANDOVER_OFFER.search("Navel handles gestures.")


def test_a_made_up_tag_is_not_spoken_or_read_as_a_gesture():
    from conversation.turn import strip_bad_tags
    assert strip_bad_tags("[I AM] [GREETING] Hello!") == "[GREETING] Hello!"
    assert strip_bad_tags("[DEFAULT] Hi") == "[DEFAULT] Hi"
    assert strip_bad_tags("[I AM] [NOT A TAG] Hi") == "Hi"
    assert strip_bad_tags("[navel_001] answer") == "[navel_001] answer"
