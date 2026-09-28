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
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from decision import ActionKind, guide_and_presenter, is_acknowledgement
from decision.policy import QA_CLOSING_PHRASES, _matches
from gateway.websocket_gateway import _speaking_seconds, _spoken_text

# One id per turn, increasing across every robot and both transports. Sent on
# tts_stop and chat_sentence so a client can drop sentences from a turn older
# than the last barge-in it was told about.
_turn_ids = itertools.count(1)

MORE_QUESTIONS_TEXT = (
    "[DEFAULT] Do you have any other questions, or shall we continue the demonstration?"
)


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
    target_instance, target_id, handoff = gw.route_question(instance, message)
    if handoff:
        gw.send_to_robot(client_id, {
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
        if not _spoken_text(clean_text):  # a bare emotion tag is not speech
            return
        gw.send_to_robot(target_id, {
            "event": "chat_sentence", "text": clean_text, "emotion_tag": emotion_tag,
            "turn_id": tid,
        })

    result = target_instance.process_chat_stream(
        message, _on_sentence, **gw.answer_kwargs(target_id, message),
    )
    marks["generate"] = time.perf_counter() - t0

    # Did the robot just sign off? Then the window closes on its own after a
    # pause. `closing` says a closing move is already in flight, which
    # replaces the more-questions prompt below.
    closing = gw.check_qa_auto_close(target_id, result.clean_text or "")

    # Delegation before the prompt: DelegationHandler asks "any other
    # questions?" itself once the delegated reply lands, from the right robot.
    delegated, delegation_result = False, None
    if result.is_delegation and result.delegation_target:
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
            and gw.claim_more_questions_prompt()
            and orch.get_status()["state"] == "qa_window"):
        gw.send_to_robot(target_id, {
            "event": "demo_step",
            "step_id": "_qa_more_questions",
            "text": MORE_QUESTIONS_TEXT,
            "require_ack": False,
        })

    return Turn("answered", target_id=target_id, handoff=handoff or "",
                result=result, delegation_result=delegation_result)
