"""
tests/conftest.py
=================
Shared fixtures.

Nothing here touches the network, Supabase, Ollama or the filesystem outside
tmp_path. The server's only Supabase seam is the module-global `_client` in
data/connection.py, and RagModule reads cfg.rag.* in __init__ and calls Ollama
for embeddings — both are faked below.

This is the first automated test suite in the repository; the pre-existing
tools/check_*.py scripts are manual checkpoint scripts that require live
infrastructure.
"""

from __future__ import annotations
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

# Server modules are imported as top-level packages (app.py runs from this dir).
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from core.rbac import (  # noqa: E402
    AccessLevel,
    MemoryAuditSink,
    MemoryRecord,
    RBACFilter,
    RobotIdentity,
    Visibility,
    new_grant,
)

SCENARIO = "lab_demo"
MANAGER_ID = "pepper_01"
WORKER_ID = "silbot_01"
OTHER_WORKER_ID = "navel_001"


# ── Time ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def now() -> datetime:
    return datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc)


# ── Identities ────────────────────────────────────────────────────────────────

@pytest.fixture
def manager() -> RobotIdentity:
    """A Manager: global access within the lab_demo scenario."""
    return RobotIdentity(
        robot_id=MANAGER_ID, scenario_id=SCENARIO, session_id="sess-m",
        access_level=AccessLevel.GLOBAL, role="Guide",
    )


@pytest.fixture
def worker() -> RobotIdentity:
    """A Worker: local isolation."""
    return RobotIdentity(
        robot_id=WORKER_ID, scenario_id=SCENARIO, session_id="sess-w",
        access_level=AccessLevel.LOCAL, role="Navigation researcher",
    )


# ── Records ───────────────────────────────────────────────────────────────────

def make_record(
    record_id: str = "faiss:1#0",
    content: str = "the visitor mentioned they are allergic to peanuts",
    source: str | None = WORKER_ID,
    visibility: object = Visibility.LOCAL,
    scenario: str | None = SCENARIO,
    subject_user_id: int | None = 42,
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        content=content,
        source_robot_id=source,
        scenario_id=scenario,
        session_id="sess-x",
        visibility=visibility,
        subject_user_id=subject_user_id,
    )


@pytest.fixture
def record_factory():
    return make_record


# ── RBAC plumbing ─────────────────────────────────────────────────────────────

@pytest.fixture
def audit() -> MemoryAuditSink:
    return MemoryAuditSink()


@pytest.fixture
def rbac(audit) -> RBACFilter:
    return RBACFilter(audit_sink=audit)


@pytest.fixture
def grant_factory(now):
    """Build a grant relative to the frozen `now`."""
    def _make(snippet_ids, granted_to=WORKER_ID, granted_by=MANAGER_ID,
              task_id="task-1", ttl_sec=180):
        return new_grant(
            snippet_ids=snippet_ids, granted_to=granted_to, granted_by=granted_by,
            task_id=task_id, session_id="sess-m", ttl_sec=ttl_sec, now=now,
        )
    return _make


# ── Fakes for the two external seams ──────────────────────────────────────────

class FakeSupabaseTable:
    """Records calls and replays canned rows. Mirrors the supabase-py chain API."""

    def __init__(self, rows=None, sink=None, name=""):
        self._rows = list(rows or [])
        self._sink = sink if sink is not None else []
        self._name = name

    def select(self, *_a, **_k): return self
    def eq(self, *_a, **_k): return self
    def neq(self, *_a, **_k): return self
    def in_(self, *_a, **_k): return self
    def order(self, *_a, **_k): return self
    def limit(self, *_a, **_k): return self
    def single(self, *_a, **_k): return self

    def insert(self, payload):
        rows = payload if isinstance(payload, list) else [payload]
        self._sink.extend(rows)
        self._rows = [{"id": 1, **rows[0]}] if rows else []
        return self

    def update(self, payload):
        self._sink.append({"_update": payload})
        return self

    def upsert(self, payload, **_k):
        self._sink.append(payload)
        return self

    def delete(self): return self

    def execute(self):
        class R:
            pass
        r = R()
        r.data = self._rows
        return r


class FakeSupabaseClient:
    def __init__(self, tables=None):
        self.tables = tables or {}
        self.writes: dict[str, list] = {}

    def table(self, name):
        self.writes.setdefault(name, [])
        return FakeSupabaseTable(
            rows=self.tables.get(name, []), sink=self.writes[name], name=name
        )


@pytest.fixture
def fake_supabase(monkeypatch):
    """
    Replace data/connection.get_client — the single Supabase seam.
    Patches the symbol in every module that imported it by name.
    """
    client = FakeSupabaseClient()

    def _get_client():
        return client

    import data.connection
    monkeypatch.setattr(data.connection, "get_client", _get_client)
    for mod_name in ("data.chat_log_repo", "data.robot_repo",
                     "data.project_repo", "data.rbac_audit_repo"):
        try:
            mod = __import__(mod_name, fromlist=["get_client"])
            monkeypatch.setattr(mod, "get_client", _get_client, raising=False)
        except ImportError:
            pass
    return client


@pytest.fixture(autouse=True)
def no_network(monkeypatch, request):
    """
    Fail any test that reaches for a real Supabase client.

    The suite must be hermetic — it runs without a .env, without a database and
    without Ollama. This caught _get_active_peers() quietly calling Supabase from
    the non-delegated chat path. Opt out with @pytest.mark.allow_db if a test
    ever genuinely needs the real client.
    """
    if request.node.get_closest_marker("allow_db"):
        return

    def _boom():
        raise AssertionError(
            "A test called data.connection.get_client(). Tests must not touch "
            "the database — stub the repo function, or use the fake_supabase "
            "fixture."
        )

    import data.connection
    monkeypatch.setattr(data.connection, "get_client", _boom)
    for mod_name in ("data.chat_log_repo", "data.robot_repo",
                     "data.project_repo", "data.rbac_audit_repo"):
        try:
            mod = __import__(mod_name, fromlist=["get_client"])
            monkeypatch.setattr(mod, "get_client", _boom, raising=False)
        except ImportError:
            pass


@pytest.fixture
def rag_config(monkeypatch, tmp_path):
    """
    Give RagModule a usable cfg. It reads cfg.rag.* in __init__, so it cannot be
    constructed while core.config.cfg is None (which is what happens without a
    .env file).
    """
    import core.config as core_config

    class _Rag:
        index_dir = str(tmp_path / "rag_indexes")
        embed_model = "nomic-embed-text"
        ollama_host = "127.0.0.1"
        ollama_port = 11434

    class _Cfg:
        rag = _Rag()

    monkeypatch.setattr(core_config, "cfg", _Cfg(), raising=False)
    import modules.rag.rag_module as rag_module
    monkeypatch.setattr(rag_module, "cfg", _Cfg(), raising=False)
    return _Cfg()
