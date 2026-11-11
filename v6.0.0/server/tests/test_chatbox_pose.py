from types import SimpleNamespace

from gateway.websocket_gateway import WebSocketGateway


def _gw(name="ChatBox"):
    gw = WebSocketGateway.__new__(WebSocketGateway)
    gw._registry = SimpleNamespace(get=lambda cid: SimpleNamespace(robot_name=name))
    gw._last_tag = {}
    return gw


def _step(gw, text, cid="c1"):
    return gw._vary_chatbox_pose(cid, {"event": "demo_step", "text": text})["text"]


def test_a_second_default_in_a_row_becomes_point():
    gw = _gw()
    assert [_step(gw, t) for t in ("[GREET] Hi", "[DEFAULT] a", "[DEFAULT] b", "[DEFAULT] c", "[DEFAULT] d")] == [
        "[GREET] Hi", "[DEFAULT] a", "[POINT] b", "[DEFAULT] c", "[POINT] d"]


def test_sentences_count_too_and_the_pose_reset_does_not():
    gw = _gw()
    s = lambda tag: gw._vary_chatbox_pose("c1", {"event": "chat_sentence", "text": "words", "emotion_tag": tag})["emotion_tag"]
    assert [s("DEFAULT"), s(""), s("DEFAULT")] == ["DEFAULT", "", "POINT"]
    assert _step(gw, "[DEFAULT]") == "[DEFAULT]"                       # tag only: left alone


def test_other_robots_are_untouched():
    gw = _gw("Navel")
    assert [_step(gw, "[DEFAULT] a"), _step(gw, "[DEFAULT] b")] == ["[DEFAULT] a", "[DEFAULT] b"]
