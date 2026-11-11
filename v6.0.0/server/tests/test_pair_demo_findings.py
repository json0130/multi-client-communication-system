"""
tests/test_pair_demo_findings.py
================================
Three things a live ChatBox + Navel run got wrong:

3. Navel answered over ChatBox, which was still saying its own sentences and the hand-over line.
4. ChatBox asked "Would you like me to pass your question to Navel?" and handed over without waiting.
6. "How does it work?" got a general answer with the FAISS index and the model sitting in the facts.
"""

from __future__ import annotations

import time

import pytest

from conversation.turn import HANDOVER_OFFER
from gateway.delegation_handler import DelegationHandler
from robot.prompt_builder import build_delegation_prompt
from tests.test_delegation_handler import SOURCE, TARGET, _FakeGateway, _FakeRegistry, _FakeSource, _FakeTarget


# ── 3. The robot being handed to waits for the one handing over ───────────────

class _TalkingGateway(_FakeGateway):
    """Reports the source as still speaking for `talk` seconds after the hand-over line is sent."""

    def __init__(self, talk):
        super().__init__(None)
        self.talk, self.line_sent = talk, None

    def send_to_robot(self, client_id, data):
        super().send_to_robot(client_id, data)
        if client_id == SOURCE and data.get("event") == "chat_sentence":
            self.line_sent = time.time()

    def seconds_until_quiet(self):
        return max(0.0, (self.line_sent or 0) + self.talk - time.time()) if self.line_sent else 0.0


def test_the_target_starts_only_after_the_source_has_finished_speaking():
    gw = _TalkingGateway(talk=0.6)
    registry = _FakeRegistry([_FakeSource(SOURCE), _FakeTarget(TARGET, "Navel")])
    answered = []
    orig = gw.send_to_robot
    gw.send_to_robot = lambda cid, d: (answered.append(time.time()) if cid == TARGET else None, orig(cid, d))[1]

    DelegationHandler(registry, gw).execute_sync(SOURCE, TARGET, "explain co-speech motion")

    assert answered and answered[0] - gw.line_sent >= 0.55


# ── 4. A hand-over is not also a question ─────────────────────────────────────

@pytest.mark.parametrize("sentence", [
    "Would you like me to pass your question to Navel?",
    "Do you want me to ask Navel?",
    "Should I pass this question to Navel?",
])
def test_an_offer_to_hand_over_is_never_spoken(sentence):
    assert HANDOVER_OFFER.search(sentence)


@pytest.mark.parametrize("sentence", [
    "Navel, can you tell us more about that?",
    "Would you like to know more about retrieval?",
    "Shall we move on?",
    "My system uses a FAISS index.",
])
def test_ordinary_sentences_are_still_spoken(sentence):
    assert not HANDOVER_OFFER.search(sentence)


# ── 6. How it works gets the mechanism, by name ───────────────────────────────

FACTS = ["method: Dense vector retrieval over a FAISS IndexFlatL2 index", "model: Qwen2.5 at 7B parameters"]


def _prompt(message, facts=FACTS):
    return build_delegation_prompt("ChatBox", "role", ["[DEFAULT]"], message, [], [], grounded_facts=facts)[0]


@pytest.mark.parametrize("message", ["how does it work?", "Can you explain how your system works technically?",
                                     "what is under the hood", "how does it work in more detail?"])
def test_a_how_it_works_question_asks_for_the_named_mechanism(message):
    assert "THE VISITOR ASKED HOW IT WORKS" in _prompt(message)


def test_other_questions_and_ungrounded_answers_are_left_alone():
    assert "THE VISITOR ASKED HOW IT WORKS" not in _prompt("what is your name?")
    assert "THE VISITOR ASKED HOW IT WORKS" not in _prompt("how does it work?", facts=[])


# ── Q&A has no time limit on this branch ──────────────────────────────────────

import threading

from decision import DemoRunTracker
from demo.demo_orchestrator import DemoOrchestrator, DemoStep
from gateway.websocket_gateway import WebSocketGateway
from tests.test_qa_auto_close import A, FakeInstance, FakeRegistry, SIGN_OFF


@pytest.mark.no_qa_limits
def test_a_qa_window_stays_open_past_its_budget_until_someone_closes_it():
    class _WS:
        tracker = DemoRunTracker()
        def on_qa_window_open(self): pass
        def on_qa_window_close(self): pass

    orch = DemoOrchestrator(_WS(), duration_sink=lambda *a: None)
    step = DemoStep(step_id="qa_invite_a", robot_id="pepper_01", text="Questions?", qa_window=True, qa_timeout=0.2)
    done = threading.Event()
    threading.Thread(target=lambda: (orch._open_qa_window(step), done.set()), daemon=True).start()
    assert not done.wait(0.8)                  # past 4x the budget, still open
    orch.qa_end(source="visitor")
    assert done.wait(2)


@pytest.mark.no_qa_limits
def test_a_robot_signing_off_does_not_close_the_window():
    import tests.test_qa_auto_close as m
    closed = []
    orch = DemoOrchestrator(None)
    orch._state = m.DemoState.QA_WINDOW
    orch.qa_end = lambda **kw: closed.append(kw)
    gw = WebSocketGateway(FakeRegistry([FakeInstance(A, "ChatBox", "x")]), recorder=None)
    gw._demo_orchestrator = orch
    orch.get_status = lambda: {"state": "qa_window", "robot_id": A, "steps": []}
    gw.check_qa_auto_close(A, SIGN_OFF)
    time.sleep(0.2)
    assert closed == []


# ── Handing over: ask first, wait for the answer, then Navel answers ──────────

from tests.test_chatbox_navel_pair import CHATBOX, NAVEL, _instant, _pair  # noqa: F401  (_instant is autouse there)
from tests.test_stations_and_names import _ask

NAVEL_Q = "can i ask more about the emotion recognition?"


def _watch(c):
    """What each robot was sent and what Navel was asked, on top of the pair harness."""
    sent, asked = [], []
    gw = c.orch._ws
    gw.send_to_robot = lambda cid, data: sent.append((cid, data))
    nav = c.robots[NAVEL]
    orig = nav.process_chat_stream
    nav.process_chat_stream = lambda m, f, **kw: (asked.append(m), orig(m, f, **kw))[1]
    return sent, asked


def _spoken(sent, cid):
    return [d.get("text", "") for c_, d in sent if c_ == cid and d.get("event") in ("chat_sentence", "demo_step")]


@pytest.mark.ask_before_handover
def test_chatbox_asks_before_passing_a_question_to_navel_and_waits():
    c = _pair()
    sent, asked = _watch(c)
    _ask(c, NAVEL_Q, to=CHATBOX)
    assert not asked and not c.robots[NAVEL].calls                  # Navel has not been asked anything yet
    offer = next(t for t in _spoken(sent, CHATBOX) if "Would you like me to ask Navel?" in t)
    assert "specialises" in offer and "Navel" in offer
    assert not c.robots[CHATBOX].calls                              # and ChatBox did not answer it either


@pytest.mark.ask_before_handover
def test_once_the_visitor_says_yes_navel_answers_the_question_that_was_asked():
    c = _pair()
    sent, asked = _watch(c)
    _ask(c, NAVEL_Q, to=CHATBOX)
    _ask(c, "yes please", to=CHATBOX)
    assert asked == [NAVEL_Q]                                       # Navel answers the original question, not "yes please"
    assert any("Navel" in t for t in _spoken(sent, CHATBOX))        # after ChatBox has said the hand-over line


@pytest.mark.ask_before_handover
def test_if_the_visitor_says_no_chatbox_answers_it_itself():
    c = _pair()
    sent, asked = _watch(c)
    _ask(c, NAVEL_Q, to=CHATBOX)
    _ask(c, "no thanks", to=CHATBOX)
    assert not asked and c.robots[CHATBOX].calls


@pytest.mark.ask_before_handover
def test_a_different_question_lets_the_offer_lapse():
    c = _pair()
    sent, asked = _watch(c)
    _ask(c, NAVEL_Q, to=CHATBOX)
    _ask(c, "can i ask more about retrieval augmented generation?", to=CHATBOX)
    assert not asked and c.robots[CHATBOX].calls
    assert c.orch._ws.take_pending_handover() is None


@pytest.mark.ask_before_handover
def test_navel_asks_any_other_questions_even_though_chatbox_already_did():
    c = _pair()
    sent, asked = _watch(c)
    _ask(c, "can i ask more about retrieval augmented generation?", to=CHATBOX)     # ChatBox answers, asks
    _ask(c, NAVEL_Q, to=CHATBOX)
    _ask(c, "yes", to=CHATBOX)
    prompts = lambda cid: [d for c_, d in sent if c_ == cid and d.get("step_id") == "_qa_more_questions"]
    assert len(prompts(CHATBOX)) == 1 and len(prompts(NAVEL)) == 1


def test_a_chatbox_project_talk_is_problem_with_examples_then_how_it_works_then_more_use_cases():
    from demo.demo_script import _project_point_steps
    problem, approach, impact = _project_point_steps("chatbox_01")
    assert "example" in problem.text and "companion robot" in problem.text
    assert "technical details" in approach.text and "retrieval method" in approach.text
    assert "use cases" in impact.text and "why it matters" in impact.text
    assert "technical details" not in _project_point_steps("navel_01")[1].text        # other robots keep the default wording
    assert [s.step_id for s in _project_point_steps("chatbox_01")] == [
        "chatbox_01_project_problem", "chatbox_01_project_approach", "chatbox_01_project_impact"]


# ── ...and the same the other way: Navel hands a ChatBox question to ChatBox ──

RAG_Q = "can i ask more about retrieval augmented generation?"


def _watch_chatbox(c):
    sent = []
    c.orch._ws.send_to_robot = lambda cid, data: sent.append((cid, data))
    box = c.robots[CHATBOX]
    asked = []
    orig = box.process_chat_stream
    box.process_chat_stream = lambda m, f, **kw: (asked.append(m), orig(m, f, **kw))[1]
    return sent, asked


@pytest.mark.ask_before_handover
def test_navel_asks_before_passing_a_chatbox_question_to_chatbox():
    c = _pair()
    sent, asked = _watch_chatbox(c)
    _ask(c, RAG_Q, to=NAVEL)
    assert not asked and not c.robots[NAVEL].calls
    offer = next(t for t in _spoken(sent, NAVEL) if "Would you like me to ask ChatBox?" in t)
    assert "specialises" in offer
    _ask(c, "yes", to=NAVEL)
    assert asked == [RAG_Q]


@pytest.mark.ask_before_handover
def test_the_robot_that_asked_hands_over_even_if_another_robots_microphone_heard_the_yes():
    c = _pair()
    sent, asked = _watch_chatbox(c)
    _ask(c, RAG_Q, to=NAVEL)                         # Navel asks...
    sent.clear()
    _ask(c, "yes please", to=CHATBOX)                # ...and ChatBox's microphone hears the answer
    assert asked == [RAG_Q]
    assert any("ChatBox" in t for t in _spoken(sent, NAVEL))        # the hand-over line is Navel's, not ChatBox's


@pytest.mark.ask_before_handover
def test_a_no_heard_by_the_other_robot_returns_the_question_to_the_robot_that_asked():
    c = _pair()
    sent, asked = _watch_chatbox(c)
    _ask(c, RAG_Q, to=NAVEL)
    _ask(c, "no", to=CHATBOX)
    assert not asked and c.robots[NAVEL].calls


# ── A hand-over the model chose by itself is asked about too ──────────────────

@pytest.mark.ask_before_handover
def test_a_hand_over_the_model_decides_on_is_asked_about_before_it_happens():
    from tests.test_stations_and_names import _Result
    c = _pair()
    sent, asked = _watch(c)
    box = c.robots[CHATBOX]

    def delegating(message, on_sentence, **kw):
        on_sentence("[DEFAULT] Navel knows more about that.", "DEFAULT")
        r = _Result("[DEFAULT] Navel knows more about that.")
        r.is_delegation, r.delegation_target = True, NAVEL
        r.response = '[DEFAULT] Navel knows more about that.\n```json\n{"target_robot_id": "%s", "task": "explain"}\n```' % NAVEL
        return r
    box.process_chat_stream = delegating
    _ask(c, "something the graph has no opinion on", to=CHATBOX)
    assert not asked                                                    # nothing was delegated yet
    assert any("Would you like me to ask Navel?" in t for t in _spoken(sent, CHATBOX))
    _ask(c, "yes", to=CHATBOX)
    assert asked == ["something the graph has no opinion on"]


# ── Follow-ups belong to the robot that just spoke ────────────────────────────

@pytest.mark.ask_before_handover
def test_yes_please_heard_by_chatbox_after_navel_asked_a_question_goes_to_navel():
    c = _pair()
    sent, asked = _watch(c)
    _ask(c, NAVEL_Q, to=CHATBOX)
    _ask(c, "yes please", to=CHATBOX)                    # the visitor agrees to the hand-over; Navel answers
    chatbox_calls = len(c.robots[CHATBOX].calls)
    _ask(c, "yes pls", to=CHATBOX)                       # ...then agrees to Navel's "would you like to know more?"
    assert asked == [NAVEL_Q, "yes pls"]                 # Navel gets it, with its own conversation behind it
    assert len(c.robots[CHATBOX].calls) == chatbox_calls  # and ChatBox improvised nothing


def test_what_counts_as_a_follow_up():
    from conversation.turn import is_continuation
    for yes in ("yes pls", "tell me more", "go on", "why?", "can you explain that", "sure"):
        assert is_continuation(yes), yes
    for no in ("what is a companion robot", "how does retrieval work in detail please do tell", "no thanks", ""):
        assert not is_continuation(no), no


# ── a tag the model puts mid-answer is not spoken ─────────────────────────────

def test_a_tag_inside_a_later_sentence_is_not_read_out():
    c = _pair()
    sent, _ = _watch(c)
    nav = c.robots[NAVEL]

    def answering(message, on_sentence, **kw):
        on_sentence("[LISTEN] Exactly!", "LISTEN")
        on_sentence("[DEFAULT] Would you like to know more about how I use gestures?", "")
        r = __import__("tests.test_stations_and_names", fromlist=["_Result"])._Result("Exactly!")
        return r
    nav.process_chat_stream = answering
    _ask(c, "navel, tell us about your project", to=NAVEL)
    spoken = [d for cid, d in sent if cid == NAVEL and d.get("event") == "chat_sentence"]
    assert spoken and all("[" not in d["text"] for d in spoken)
    assert spoken[1]["emotion_tag"] == "DEFAULT"


# ── the ChatBox Socket.IO client speaks chat_response, so chat_sentence is sent as that ───────────────

def test_the_chatbox_client_gets_streamed_sentences_as_ordered_chat_responses():
    import socketio
    from tests.test_socketio_robots import server as _server  # noqa: F401
    from tests.test_inbound_robots import _wait
    # built by hand: the fixture is a generator fixture in another module
    import threading
    from flask import Flask
    from werkzeug.serving import make_server
    from gateway.websocket_gateway import WebSocketGateway
    from tests.test_inbound_robots import _Registry
    reg = _Registry()
    gw = WebSocketGateway(reg, recorder=None)
    api = Flask(__name__)
    srv = make_server("127.0.0.1", 0, api, threaded=True)
    api.wsgi_app = gw.socketio_app(api.wsgi_app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sio, got, init = socketio.Client(), [], []
    sio.on("client_init_response", init.append)
    sio.on("chat_response", got.append)
    sio.connect(f"http://127.0.0.1:{srv.server_port}", transports=["websocket", "polling"])
    sio.emit("client_init", {"robot_name": "ChatBox", "client_id": "chatbox_jetson_001"})
    assert _wait(lambda: init)
    for i, text in enumerate(["Navel specialises in this.", "Would you like me to ask Navel?", "Navel, over to you."]):
        gw.send_to_robot("chatbox_jetson_001", {"event": "chat_sentence", "text": text,
                                                "emotion_tag": "DEFAULT" if i == 0 else ""})
    assert _wait(lambda: len(got) == 3)
    assert [g["clean_text"] for g in got] == ["Navel specialises in this.", "Would you like me to ask Navel?", "Navel, over to you."]
    assert got[0]["response"].startswith("[DEFAULT] ") and got[0]["emotion_tag"] == "DEFAULT"
    sio.disconnect(); srv.shutdown()
