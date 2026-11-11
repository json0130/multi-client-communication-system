"""
conversation/turn.py
====================
One visitor turn, from "someone said something" to "a robot has answered".

There used to be three copies of this: the WebSocket `chat` handler, the
dashboard's HTTP /chat route, and the WebSocket `speech` handler. The first two
were kept in step by hand and had drifted (only one short-circuited
acknowledgements, only the other invited more questions). The third was
older still: it waited for the whole reply before sending anything and skipped
routing, grounding and style framing entirely. Every voice robot takes that
path, so spoken questions were the slowest and the least decided.

Now all three call handle_turn(). Speech is just STT followed by the same call.

The gateway is passed in rather than imported: it still owns the Q&A window
state (timers, who was handed off to, the more-questions claim) that this
pipeline reads and writes.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from decision import ActionKind, guide_and_presenter, is_acknowledgement
from conversation.method_b import is_navel
from decision.policy import QA_CLOSING_PHRASES, _matches
from gateway import websocket_gateway as _wsg
from gateway.websocket_gateway import _speaking_seconds, _spoken_text
from modules.llm.llm_provider import strip_bad_tags

# One id per turn, increasing across every robot and both transports. Sent on
# tts_stop and chat_sentence so a client can drop sentences from a turn older
# than the last barge-in it was told about.
_turn_ids = itertools.count(1)

MORE_QUESTIONS_TEXT = (
    "[DEFAULT] Do you have any other questions, or shall we continue the demonstration?"
)

# "Would you like me to pass your question to Navel?" The prompt says to hand over without asking, but
# the model still asks, then the hand-over JSON follows in the same reply, so the visitor was asked a
# question the robot never waited for. The offer is never spoken.
HANDOVER_OFFER = re.compile(
    r"\b(would you like|do you want|shall|should) (me|us|i|we)( to)? (pass|hand|forward|delegate|ask|check with|bring in|get)\b"
    r"|\bpass (your|the|this) question\b.*\?", re.I)


_YES = ("yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "go ahead", "do it", "of course",
        "definitely", "why not", "alright", "sounds good", "absolutely", "by all means", "ask")
_NO = ("no", "nope", "nah", "no thanks", "no thank you", "not now", "you answer", "you do it", "don't", "dont")


def handover_verdict(message: str) -> Optional[str]:
    """"yes" or "no" when a visitor's reply is an answer to "would you like me to ask Navel?", else None.

    Short replies only, and by their first words: "yes please", "no, you tell me", "sure, go ahead". A
    full sentence is a new question, and the offer lapses.
    """
    words = re.sub(r"[^a-z' ]", " ", (message or "").lower()).split()
    if not words or len(words) > 6:
        return None
    text = " ".join(words)
    for no in _NO:                       # "no" first: "no thanks" must not read as the "thanks" of a yes
        if text == no or text.startswith(no + " "):
            return "no"
    for yes in _YES:
        if text == yes or text.startswith(yes + " "):
            return "yes"
    return None


_CONTINUE = ("tell me more", "more", "go on", "continue", "keep going", "and then", "why", "how so",
             "what do you mean", "can you explain", "could you explain", "explain", "elaborate", "go ahead")


def is_continuation(message: str) -> bool:
    """A short reply that only makes sense as a follow-up to what was just said: "yes pls" to "Would you like
    to know more?", "tell me more", "why?". Whoever heard it, it is for the robot that just spoke."""
    words = re.sub(r"[^a-z' ]", " ", (message or "").lower()).split()
    if not words or len(words) > 7:
        return False
    text = " ".join(words)
    return handover_verdict(message) == "yes" or any(text == c or text.startswith(c + " ") for c in _CONTINUE)


class TurnCancelled(Exception):
    """The visitor started another turn while this one was answering."""


@dataclass
class Turn:
    """What happened on one turn, for callers that need to report it."""
    outcome: str                      # advance | acknowledged | deferred | answered | cancelled
    target_id: str = ""               # who answered (may differ from who heard it)
    handoff: str = ""                 # what the receiver said before handing over
    text: str = ""                    # fixed line spoken, for advance/acknowledged
    result: object = None             # ChatResult when answered
    delegation_result: Optional[dict] = None
    timings: dict = field(default_factory=dict)
    turn_id: int = 0


def handle_turn(gw, instance, client_id: str, message: str, *,
                sync_delegation: bool = False, stt_sec: Optional[float] = None,
                cancelled: Optional[threading.Event] = None) -> Turn:
    """Run one visitor utterance through the Q&A pipeline.

    `sync_delegation` runs a delegation on this thread so the caller gets the
    delegated answer back (the HTTP route returns it to the dashboard); the
    WebSocket path launches it in the background instead.

    `stt_sec` is how long transcription took, when the turn was spoken, so the
    timing line covers the whole turn.

    `cancelled`, once set, stops a streaming answer at its next sentence.
    """
    t0 = time.perf_counter()
    tid = next(_turn_ids)
    marks: dict = {} if stt_sec is None else {"stt": stt_sec}
    try:
        turn = _run(gw, instance, client_id, message, sync_delegation, t0, marks,
                    tid, cancelled or threading.Event())
    except TurnCancelled:
        turn = Turn("cancelled", target_id=client_id)
    turn.turn_id = tid
    marks["total"] = time.perf_counter() - t0 + (stt_sec or 0.0)
    turn.timings = {k: round(v, 3) for k, v in marks.items()}
    print(f"[Turn] {client_id}->{turn.target_id or client_id} {turn.outcome} "
          + " ".join(f"{k}={v:.2f}s" for k, v in turn.timings.items()))
    return turn


def _run(gw, instance, client_id, message, sync_delegation, t0, marks,
         tid, cancelled) -> Turn:
    orch = gw._demo_orchestrator

    # Barge-in: user talking = robot listens. A raw phrase check, not a
    # decision — it has to happen before any LLM call.
    if not any(p in message.lower() for p in gw._QA_ADVANCE_PHRASES):
        gw.send_to_robot(client_id, {"event": "tts_stop", "turn_id": tid})
        if orch and orch.get_status()["state"] in ("running", "waiting_ack"):
            orch.qa_interrupt(source="auto")

    # A hand-over the visitor was just asked about ("Navel specialises in this. Would you like me to ask
    # Navel?"). Their reply answers it, so it is read before anything else can mistake a bare "no" for
    # "no more questions" or a "sure" for a nod. Anything longer is a new question and the offer lapses.
    pending = gw.take_pending_handover() if _wsg.ASK_BEFORE_HANDOVER else None
    verdict = handover_verdict(message) if pending else None
    if verdict:
        message = pending["question"]       # what is answered now is the question that was asked
        decision, advancing = None, False
    else:
        decision = gw.check_qa_advance_from_user(instance, message)
        advancing = decision is not None and decision.action.kind is ActionKind.ADVANCE

    # PLAN_REVISE unconditionally, before any early return: "we're running
    # out of time" can close this window AND shorten the tour, and the two
    # decisions are independent.
    gw.check_plan_revision(instance, message)

    if advancing:
        # Every advance mechanism ends the turn with a fixed line from the
        # GUIDE. The window is closing; there is nothing to answer, and a
        # generated reply from whichever robot held the mic has asked the
        # visitor to run the tour. See ADVANCE_ACK.
        ack = gw.advance_ack_text(decision.mechanism)
        if ack:
            guide_id = guide_and_presenter(orch.get_status())[0] if orch else None
            gw.send_to_robot(guide_id or client_id, {
                "event": "demo_step",
                "step_id": "_qa_advance_ack",
                "text": ack,
                "require_ack": False,
            })
        marks["decide"] = time.perf_counter() - t0
        return Turn("advance", text=ack or "")

    # A nod, not a question — a fixed line, not the LLM. Told not to narrate
    # the tour, a 7B model still answered "okay thank you" by announcing the
    # next project.
    if is_acknowledgement(message):
        gw.send_to_robot(client_id, {
            "event": "chat_sentence", "text": "Of course.", "emotion_tag": "DEFAULT",
            "turn_id": tid,
        })
        marks["decide"] = time.perf_counter() - t0
        return Turn("acknowledged", target_id=client_id, text="Of course.")

    # Who answers. May name a different robot than the one that heard it;
    # the receiver then says a one-line hand-off first.
    # Whoever asked the visitor is the one who hands over (or keeps the question), whichever robot's
    # microphone heard the reply.
    asker_id = (pending or {}).get("from_id") or client_id
    asker = gw._registry.get(asker_id) or instance
    speaker_id = asker_id if verdict else client_id
    if verdict == "yes":
        target_id = pending["target_id"]
        target_instance = gw._registry.get(target_id)
        handoff = None
        if target_instance is None:
            target_instance, target_id = asker, asker_id
        else:
            gw.note_floor(target_id)
            handoff = gw._handoff_line(asker_id, getattr(target_instance, "robot_name", None) or target_id, message)
    elif verdict == "no":
        target_instance, target_id, handoff = asker, asker_id, None         # the robot that asked answers it itself
    else:
        target_instance, target_id, handoff = gw.route_question(
            instance, message, ask_first=_wsg.ASK_BEFORE_HANDOVER)
        previous = gw.last_answerer()
        if (target_instance is instance and not handoff and previous and previous != client_id
                and gw._registry.get(previous) is not None and is_continuation(message)):
            # "yes pls" heard by ChatBox's microphone, to a question NAVEL had just asked. ChatBox has none
            # of that conversation and answered with a hand-over line; the robot that asked has it all.
            target_instance, target_id = gw._registry.get(previous), previous
        if _wsg.ASK_BEFORE_HANDOVER and getattr(getattr(gw, "_scratch", None), "asks_first", False):
            offer = gw.handover_offer(client_id, target_id, message)
            gw.set_pending_handover(client_id, target_id, message)
            gw.send_to_robot(client_id, {
                "event": "chat_sentence", "text": offer, "emotion_tag": "DEFAULT", "turn_id": tid,
            })
            marks["decide"] = time.perf_counter() - t0
            return Turn("asked_handover", target_id=client_id, text=offer)
    if handoff:
        gw.send_to_robot(speaker_id, {
            "event": "chat_sentence", "text": handoff, "emotion_tag": "DEFAULT",
            "turn_id": tid,
        })
    marks["decide"] = time.perf_counter() - t0

    # Deferred to that robot's station: nobody answers now.
    if target_instance is None:
        return Turn("deferred", target_id=target_id, handoff=handoff or "")

    # Let the hand-off be HEARD before the answer starts — two robots, two
    # speakers, nothing else serialising them. Only the first sentence waits.
    wait = {"left": _speaking_seconds(handoff) if handoff else 0.0}

    def _on_sentence(clean_text, emotion_tag):
        if "first_sentence" not in marks:
            marks["first_sentence"] = time.perf_counter() - t0
        if wait["left"] > 0:
            time.sleep(wait["left"])
            wait["left"] = 0.0
        if cancelled.is_set():
            raise TurnCancelled()     # unwinds out of the LLM stream
        if "```" in clean_text:          # never speak a delegation JSON block
            return
        clean_text = strip_bad_tags(clean_text)
        tag = re.match(r"\s*\[([A-Z_]+)\]\s*", clean_text)
        if tag:                          # a tag the model put mid-answer would be read out as "default"
            emotion_tag = emotion_tag or tag.group(1)
            clean_text = clean_text[tag.end():]
        if HANDOVER_OFFER.search(clean_text):
            return
        if not _spoken_text(clean_text):  # a bare emotion tag is not speech
            return
        send_sentence({
            "event": "chat_sentence", "text": clean_text, "emotion_tag": emotion_tag,
            "turn_id": tid,
        })

    # Navel's sentences are tagged by Method B (conversation/method_b.py) while the reply is still streaming.
    mb = getattr(gw, "method_b", None)
    stream = (mb.stream(target_id, message, lambda p: gw.send_to_robot(target_id, p))
              if mb is not None and is_navel(target_instance) else None)
    send_sentence = stream.put if stream else (lambda p: gw.send_to_robot(target_id, p))
    try:
        result = target_instance.process_chat_stream(
            message, _on_sentence, **gw.answer_kwargs(target_id, message),
        )
    finally:
        if stream:
            if cancelled.is_set():
                stream.cancel()
            stream.close()          # everything queued has gone out before anything else is said
    marks["generate"] = time.perf_counter() - t0
    gw.note_answer(target_id)

    # Did the robot just sign off? Then the window closes on its own after a
    # pause. `closing` says a closing move is already in flight, which
    # replaces the more-questions prompt below.
    closing = gw.check_qa_auto_close(target_id, result.clean_text or "")

    # Delegation before the prompt: DelegationHandler asks "any other
    # questions?" itself once the delegated reply lands, from the right robot.
    delegated, delegation_result = False, None
    if (result.is_delegation and result.delegation_target and _wsg.ASK_BEFORE_HANDOVER
            and result.delegation_target != target_id and gw._registry.get(result.delegation_target) is not None):
        # The model decided on its own to hand over (the router had no opinion). That is a hand-over too,
        # so the visitor is asked first, exactly as above; their "yes" sends the question to that robot.
        offer = gw.handover_offer(target_id, result.delegation_target, message)
        gw.set_pending_handover(target_id, result.delegation_target, message)
        gw.send_to_robot(target_id, {
            "event": "chat_sentence", "text": offer, "emotion_tag": "DEFAULT", "turn_id": tid,
        })
        delegated = True
    elif result.is_delegation and result.delegation_target:
        from gateway.delegation_handler import DelegationHandler
        handler = DelegationHandler(gw._registry, gw)
        if sync_delegation:
            del_target, task = handler._extract(result.response)
            if del_target and task:
                delegation_result = handler.execute_sync(target_id, del_target, task)
            delegated = bool(delegation_result)
        else:
            delegated = handler.handle(target_id, result.response)

    # "Any other questions?" once per window, from whoever just answered — not
    # when the answer already invited more, and not when a closing move is in
    # flight. `advancing` is False here by construction: every advance
    # returned above, before qa_end()'s state change could race this check.
    already_invited = _matches(result.clean_text or "", QA_CLOSING_PHRASES)
    if (orch and not delegated and not already_invited and not closing
            and gw.claim_more_questions_prompt(target_id)
            and orch.get_status()["state"] == "qa_window"):
        gw.send_to_robot(target_id, {
            "event": "demo_step",
            "step_id": "_qa_more_questions",
            "text": MORE_QUESTIONS_TEXT,
            "require_ack": False,
        })

    return Turn("answered", target_id=target_id, handoff=handoff or "",
                result=result, delegation_result=delegation_result)
