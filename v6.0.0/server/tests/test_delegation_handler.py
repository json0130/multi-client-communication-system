"""
tests/test_delegation_handler.py
=================================
gateway/delegation_handler.py::DelegationHandler.execute_sync — the LLM-driven
handoff path (Pepper's own LLM decides to delegate via a JSON block), distinct
from decision-layer QA_ROUTE (see gateway/websocket_gateway.py::route_question
and tests/test_qa_more_questions_race.py).

Two regressions from a real run, both fixed in the same change:

  1. The spoken handoff line echoed the LLM-generated `task` text verbatim —
     "Navel, Please explain more about your research project." — stilted
     because `task` was never meant to be read aloud. Now generic and fixed,
     regardless of what `task` says; Navel still receives the real `task`
     text for what it actually answers.

  2. The "any other questions?" follow-up used to be the CALLER's job, fired
     right after Pepper's own reply stream finished — which, for the
     synchronous caller (http_gateway.py), was BEFORE delegation had even
     run, and for the async caller (websocket_gateway.py's handle()/
     _execute() background thread), never reliably happened at all. A real
     run had the prompt land on Pepper while Navel's answer was still being
     generated. Moving it into execute_sync() means it fires once, at the
     one moment that's actually correct, regardless of which caller invoked
     it.
"""

from __future__ import annotations

from gateway.delegation_handler import DelegationHandler

SOURCE = "pepper_01"
TARGET = "navel_01"


class _ChatResult:
    def __init__(self, text):
        self.response = text
        self.emotion_tag = "DEFAULT"
        self.clean_text = text


class _FakeSource:
    """No _get_rag_context — context serialization degrades safely without it,
    which is exactly the behaviour under test elsewhere; not the concern here."""
    def __init__(self, client_id):
        self.client_id = client_id


class _FakeTarget:
    def __init__(self, client_id, name):
        self.client_id = client_id
        self.robot_name = name
        self.task_received = None

    def process_chat(self, task, is_delegated=False, delegated_context=None, task_id=None):
        self.task_received = task
        return _ChatResult(f"Answer to: {task}")


class _FakeGrantStore:
    def issue(self, grant):
        pass

    def revoke_task(self, task_id):
        return 0


class _FakeRegistry:
    def __init__(self, instances):
        self._by_id = {i.client_id: i for i in instances}
        self.grants = _FakeGrantStore()

    def get(self, client_id):
        return self._by_id.get(client_id)


class _FakeOrchestrator:
    def __init__(self, state="qa_window"):
        self._state = state

    def get_status(self):
        return {"state": self._state}


class _FakeGateway:
    def __init__(self, demo_orchestrator=None):
        self.sent = []
        self._demo_orchestrator = demo_orchestrator

    def send_to_robot(self, client_id, data):
        self.sent.append((client_id, data))


def _make(orch_state="qa_window"):
    gw = _FakeGateway(_FakeOrchestrator(orch_state) if orch_state is not None else None)
    target = _FakeTarget(TARGET, "Navel")
    registry = _FakeRegistry([_FakeSource(SOURCE), target])
    return DelegationHandler(registry, gw), gw, target


class TestVerbalHandoff:
    def test_handoff_names_the_target_but_does_not_echo_the_raw_task(self):
        handler, gw, _ = _make()
        handler.execute_sync(SOURCE, TARGET, "Please explain more about your research project.")

        handoff = next(d["text"] for cid, d in gw.sent
                       if cid == SOURCE and d.get("event") == "chat_sentence")
        assert "Navel" in handoff
        assert "Please explain more about your research project." not in handoff

    def test_the_target_still_receives_the_real_task_text(self):
        handler, gw, target = _make()
        handler.execute_sync(SOURCE, TARGET, "Please explain more about your research project.")
        assert target.task_received == "Please explain more about your research project."


class TestFollowUpPrompt:
    def test_prompts_the_target_when_still_in_a_qa_window(self):
        handler, gw, _ = _make(orch_state="qa_window")
        handler.execute_sync(SOURCE, TARGET, "explain more")

        step_ids = [d.get("step_id") for cid, d in gw.sent
                   if cid == TARGET and d.get("event") == "demo_step"]
        assert "_qa_more_questions" in step_ids

    def test_does_not_prompt_when_no_longer_in_a_qa_window(self):
        handler, gw, _ = _make(orch_state="running")
        handler.execute_sync(SOURCE, TARGET, "explain more")

        step_ids = [d.get("step_id") for _, d in gw.sent if d.get("event") == "demo_step"]
        assert "_qa_more_questions" not in step_ids

    def test_does_not_prompt_when_no_orchestrator_is_wired(self):
        handler, gw, _ = _make(orch_state=None)
        handler.execute_sync(SOURCE, TARGET, "explain more")

        step_ids = [d.get("step_id") for _, d in gw.sent if d.get("event") == "demo_step"]
        assert "_qa_more_questions" not in step_ids

    def test_the_prompt_is_sent_after_the_answer_not_before(self):
        handler, gw, _ = _make()
        handler.execute_sync(SOURCE, TARGET, "explain more")

        events = [d.get("event") for cid, d in gw.sent if cid == TARGET]
        assert events.index("chat_response") < events.index("demo_step")

    def test_the_prompt_goes_to_the_target_not_the_source(self):
        handler, gw, _ = _make()
        handler.execute_sync(SOURCE, TARGET, "explain more")

        prompts = [cid for cid, d in gw.sent if d.get("step_id") == "_qa_more_questions"]
        assert prompts == [TARGET]
