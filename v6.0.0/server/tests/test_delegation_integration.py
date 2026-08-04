"""
tests/test_delegation_integration.py
====================================
End-to-end Contextual Inheritance across the hand-off.

The three properties the paper's Context Serialization requires:

  1. A Worker cannot retrieve the Manager's records directly.
  2. During a delegated task it *can* see the granted snippet in its composite
     system prompt.
  3. After expiry it loses that access again — and the snippet was never written
     into its own memory.

Everything below the LLM and the database is real: real RBACFilter, real
GrantStore, real policy, real prompt_builder. Only the LLM call, the Supabase
writes and the Ollama embedding call are faked.
"""

from __future__ import annotations
from datetime import timedelta

import pytest

from core.rbac import (
    AccessLevel,
    GrantStore,
    MemoryRecord,
    RBACFilter,
    RobotIdentity,
    Visibility,
    new_grant,
)
from tests.conftest import MANAGER_ID, SCENARIO, WORKER_ID

pytestmark = pytest.mark.integration


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeLLMResponse:
    def __init__(self, text):
        self.text = text
        self.emotion_tag = "DEFAULT"
        self.clean_text = text.replace("[DEFAULT] ", "")


class CapturingLLM:
    """Captures the composite system prompt instead of calling a model."""

    def __init__(self):
        self.system_prompts: list[str] = []

    def is_available(self): return True

    def generate_with_history(self, system, history, user_message):
        self.system_prompts.append(system)
        return FakeLLMResponse("[DEFAULT] On it.")

    @property
    def last_prompt(self) -> str:
        return self.system_prompts[-1] if self.system_prompts else ""


class FakeRag:
    """Stands in for the FAISS layer, returning fixed candidates through RBAC."""

    def __init__(self, records, rbac, grants_provider):
        self._records = records
        self._rbac = rbac
        self._grants_provider = grants_provider
        self.added: list[str] = []

    def is_available(self): return True

    def search(self, query, requester, rbac, grants=(), top_k=5, now=None):
        return rbac.filter_records(requester, self._records, grants, now, store="faiss")[:top_k]

    def add(self, message, subject_user_id=None, visibility=None):
        self.added.append(message)


@pytest.fixture
def shared():
    """One RBAC filter and one grant store, shared as the registry shares them."""
    return RBACFilter(), GrantStore()


MANAGER_SNIPPET = MemoryRecord(
    record_id="faiss:1#0",
    content="the visitor is a wheelchair user and asked about step-free routes",
    source_robot_id=MANAGER_ID,
    scenario_id=SCENARIO,
    session_id="sess-m",
    visibility=Visibility.GLOBAL,
    subject_user_id=42,
)

WORKER_OWN = MemoryRecord(
    record_id="faiss:2#0",
    content="I explained the navigation stack",
    source_robot_id=WORKER_ID,
    scenario_id=SCENARIO,
    session_id="sess-w",
    visibility=Visibility.LOCAL,
    subject_user_id=42,
)


@pytest.fixture
def worker_instance(shared, monkeypatch):
    """A Worker RobotInstance with the DB and LLM stubbed out."""
    import data.robot_repo as robot_repo
    import data.chat_log_repo as chat_log_repo
    from robot.robot_instance import RobotInstance

    rbac, grants = shared

    monkeypatch.setattr(
        robot_repo, "get_robot",
        lambda cid: robot_repo.RobotRecord(
            client_id=WORKER_ID, robot_name="Silbot", robot_role="Guide",
            is_active=True, allowed_tags=["[DEFAULT]"], modules=["gpt", "rag"],
            access_level="local", scenario_id=SCENARIO,
        ),
    )
    # The non-delegated path builds a peer list from the DB. Stub it so the
    # suite never touches the network.
    monkeypatch.setattr(robot_repo, "get_all_active_robots", lambda **k: [])
    inserted: list[dict] = []
    monkeypatch.setattr(
        chat_log_repo, "insert",
        lambda *a, **k: inserted.append({"args": a, "kwargs": k}),
    )

    inst = RobotInstance(
        client_id=WORKER_ID, robot_name="Silbot", user_id=2,
        enabled_modules={"gpt", "rag"}, rbac=rbac, grants=grants,
        access_level=AccessLevel.LOCAL, scenario_id=SCENARIO,
        session_id="sess-w", default_visibility=Visibility.LOCAL,
    )
    inst.llm = CapturingLLM()
    inst.rag = FakeRag([MANAGER_SNIPPET, WORKER_OWN], rbac, lambda: grants)
    inst._inserted = inserted           # for assertions
    return inst


# ── 1. Standing isolation ─────────────────────────────────────────────────────

def test_worker_cannot_retrieve_the_managers_records_directly(worker_instance):
    """Without a grant, the Manager's snippet is simply not reachable."""
    cleared = worker_instance._get_rag_context("step-free routes")
    texts = [c.text for c in cleared]

    assert MANAGER_SNIPPET.content not in texts
    assert texts == [WORKER_OWN.content]


def test_worker_normal_chat_prompt_excludes_the_managers_records(worker_instance):
    worker_instance.process_chat("what routes are there?")
    prompt = worker_instance.llm.last_prompt

    assert MANAGER_SNIPPET.content not in prompt
    assert WORKER_OWN.content in prompt


# ── 2. During the delegated task ──────────────────────────────────────────────

def test_worker_sees_the_granted_snippet_in_its_composite_prompt(
    worker_instance, shared
):
    _, grants = shared
    grants.issue(new_grant(
        snippet_ids=[MANAGER_SNIPPET.record_id],
        granted_to=WORKER_ID, granted_by=MANAGER_ID, task_id="task-1",
        session_id="sess-m", ttl_sec=180,
    ))

    worker_instance.process_chat(
        "take the visitor to lab 2", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )
    prompt = worker_instance.llm.last_prompt

    assert MANAGER_SNIPPET.content in prompt
    assert "CONTEXT SHARED BY YOUR TEAMMATE" in prompt


def test_the_grant_does_not_widen_standing_access(worker_instance, shared):
    """
    The whole point: even while the grant is live, an ordinary retrieval still
    cannot reach the Manager's records. Only the serialized snippet is visible.
    """
    _, grants = shared
    grants.issue(new_grant(
        snippet_ids=[MANAGER_SNIPPET.record_id],
        granted_to=WORKER_ID, granted_by=MANAGER_ID, task_id="task-1",
        session_id="sess-m", ttl_sec=180,
    ))

    cleared = worker_instance._get_rag_context("step-free routes")
    assert [c.text for c in cleared] == [WORKER_OWN.content]


def test_payload_without_a_grant_confers_nothing(worker_instance):
    """Handing a Worker context is not enough — the grant carries the authority."""
    worker_instance.process_chat(
        "take the visitor to lab 2", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )
    assert MANAGER_SNIPPET.content not in worker_instance.llm.last_prompt


def test_a_grant_for_a_different_snippet_does_not_admit_this_one(
    worker_instance, shared
):
    _, grants = shared
    grants.issue(new_grant(
        snippet_ids=["faiss:1#999"],
        granted_to=WORKER_ID, granted_by=MANAGER_ID, task_id="task-1",
        session_id="sess-m", ttl_sec=180,
    ))
    worker_instance.process_chat(
        "task", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )
    assert MANAGER_SNIPPET.content not in worker_instance.llm.last_prompt


# ── 3. After expiry / revocation ──────────────────────────────────────────────

def test_worker_loses_the_snippet_after_expiry(worker_instance, shared):
    _, grants = shared
    grants.issue(new_grant(
        snippet_ids=[MANAGER_SNIPPET.record_id],
        granted_to=WORKER_ID, granted_by=MANAGER_ID, task_id="task-1",
        session_id="sess-m", ttl_sec=-1,            # already expired
    ))

    worker_instance.process_chat(
        "take the visitor to lab 2", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )
    assert MANAGER_SNIPPET.content not in worker_instance.llm.last_prompt


def test_worker_loses_the_snippet_after_revocation(worker_instance, shared):
    _, grants = shared
    grants.issue(new_grant(
        snippet_ids=[MANAGER_SNIPPET.record_id],
        granted_to=WORKER_ID, granted_by=MANAGER_ID, task_id="task-1",
        session_id="sess-m", ttl_sec=180,
    ))
    worker_instance.process_chat(
        "task", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )
    assert MANAGER_SNIPPET.content in worker_instance.llm.last_prompt

    grants.revoke_task("task-1")

    worker_instance.process_chat(
        "task again", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )
    assert MANAGER_SNIPPET.content not in worker_instance.llm.last_prompt


def test_granted_snippet_is_never_written_into_the_workers_own_memory(
    worker_instance, shared
):
    """
    Prompt-time only. After a delegated task the Worker's stores contain its own
    message and response, and nothing that was granted to it.
    """
    _, grants = shared
    grants.issue(new_grant(
        snippet_ids=[MANAGER_SNIPPET.record_id],
        granted_to=WORKER_ID, granted_by=MANAGER_ID, task_id="task-1",
        session_id="sess-m", ttl_sec=180,
    ))

    worker_instance.process_chat(
        "take the visitor to lab 2", is_delegated=True,
        delegated_context=[MANAGER_SNIPPET], task_id="task-1",
    )

    assert MANAGER_SNIPPET.content in worker_instance.llm.last_prompt
    assert MANAGER_SNIPPET.content not in worker_instance.rag.added
    assert worker_instance.rag.added == ["take the visitor to lab 2"]

    logged = " ".join(
        str(v) for row in worker_instance._inserted
        for v in list(row["args"]) + list(row["kwargs"].values())
    )
    assert MANAGER_SNIPPET.content not in logged


def test_records_are_persisted_with_provenance_and_visibility(worker_instance):
    worker_instance.process_chat("hello")
    assert worker_instance._inserted, "nothing was logged"
    kwargs = worker_instance._inserted[-1]["kwargs"]
    assert kwargs["source_robot_id"] == WORKER_ID
    assert kwargs["scenario_id"] == SCENARIO
    assert kwargs["visibility"] == Visibility.LOCAL.value


# ── The temporal buffer ───────────────────────────────────────────────────────

def test_temporal_buffer_never_crosses_robot_instances(shared, worker_instance):
    """
    _history is per-instance. Asserted rather than filtered: if this ever becomes
    shared state, this test fails and the buffer needs its own RBAC filter.
    """
    from robot.robot_instance import RobotInstance

    rbac, grants = shared
    other = RobotInstance(
        client_id=MANAGER_ID, robot_name="Pepper", user_id=1,
        enabled_modules=set(), rbac=rbac, grants=grants,
        access_level=AccessLevel.GLOBAL, scenario_id=SCENARIO,
    )
    worker_instance._history.append({"role": "user", "content": "private to the worker"})

    assert other._history == []
    assert other._history is not worker_instance._history
