"""
tests/test_rag_rebuild.py
=========================
RagModule rebuilds a missing local index from chat_logs through the data/
layer, keyed by the connection's session_id, so one session's visitors never
seed another's index.
"""

from __future__ import annotations

import pytest

from core.rbac import Visibility
from data import chat_log_repo


class _Embeddings:
    def create(self, model, input):
        return type("R", (), {"data": [type("D", (), {"embedding": [float(len(t)), 0.0]})()
                                       for t in input]})()


@pytest.fixture
def mod(rag_config, monkeypatch):
    from modules.rag.rag_module import RagModule
    m = RagModule(user_id=5, client_id="navel_001", scenario_id="lab_demo",
                  session_id="navel_001:100", default_visibility=Visibility.LOCAL)
    m._embed_client = type("C", (), {"embeddings": _Embeddings()})()
    monkeypatch.setattr(m, "_save", lambda: None)
    return m


def test_the_rebuild_reads_only_this_session(mod, monkeypatch):
    asked = []
    monkeypatch.setattr(chat_log_repo, "get_session_rows", lambda sid: asked.append(sid) or [
        {"message": "hello", "session_id": sid, "visibility": "local"},
        {"message": "   "},
    ])
    mod._build_from_supabase()
    assert asked == ["navel_001:100"]
    assert [r["text"] for r in mod._records] == ["hello"]
    assert mod._records[0]["source_robot_id"] == "navel_001"    # filled from identity


def test_no_history_is_an_empty_index(mod, monkeypatch):
    monkeypatch.setattr(chat_log_repo, "get_session_rows", lambda sid: [])
    mod._build_from_supabase()
    assert mod._records == [] and mod._index is None


class TestRepo:
    def _client(self, monkeypatch, fail=False):
        calls = []

        class Q:
            def __getattr__(self, name):
                def f(*a, **k):
                    calls.append((name, a))
                    if fail and name == "execute":
                        raise RuntimeError("down")
                    return type("R", (), {"data": [{"message": "hi"}]})() \
                        if name == "execute" else self
                return f

        monkeypatch.setattr(chat_log_repo, "get_client",
                            lambda: type("C", (), {"table": lambda s, n: Q()})())
        return calls

    def test_it_is_keyed_by_session(self, monkeypatch):
        calls = self._client(monkeypatch)
        assert chat_log_repo.get_session_rows("s:1") == [{"message": "hi"}]
        assert ("eq", ("session_id", "s:1")) in calls
        assert not any(c[0] == "eq" and c[1][0] == "user_id" for c in calls)

    def test_no_session_reads_nothing(self, monkeypatch):
        calls = self._client(monkeypatch)
        assert chat_log_repo.get_session_rows(None) == [] and calls == []

    def test_a_failure_is_empty(self, monkeypatch):
        self._client(monkeypatch, fail=True)
        assert chat_log_repo.get_session_rows("s:1") == []
