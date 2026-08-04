"""
tests/test_rbac_grants.py
=========================
Grant lifecycle and the enforcement filter: issue -> resolve -> expire -> revoke.

Also covers the clearance stamp, which is what stops an unfiltered read path
reaching prompt assembly.
"""

from __future__ import annotations
from datetime import timedelta

import pytest

from core.rbac import (
    ClearanceError,
    ClearedRecord,
    Decision,
    GrantStore,
    MemoryRecord,
    RBACFilter,
    Reason,
    Visibility,
    assert_cleared,
    make_record_id,
    new_grant,
)
from tests.conftest import MANAGER_ID, OTHER_WORKER_ID, WORKER_ID, make_record


# ── GrantStore lifecycle ──────────────────────────────────────────────────────

class TestGrantStore:

    def test_issue_then_resolve(self, grant_factory, now):
        store = GrantStore()
        g = store.issue(grant_factory(["a", "b"]))
        active = store.active_for(WORKER_ID, now=now)
        assert [x.grant_id for x in active] == [g.grant_id]

    def test_expired_grants_are_never_returned(self, grant_factory, now):
        store = GrantStore()
        store.issue(grant_factory(["a"], ttl_sec=60))
        assert store.active_for(WORKER_ID, now=now) != []
        assert store.active_for(WORKER_ID, now=now + timedelta(seconds=61)) == []

    def test_revoke_on_task_completion(self, grant_factory, now):
        store = GrantStore()
        store.issue(grant_factory(["a"], task_id="t1"))
        assert store.revoke_task("t1") == 1
        assert store.active_for(WORKER_ID, now=now) == []

    def test_revoking_an_unknown_task_is_harmless(self):
        assert GrantStore().revoke_task("nope") == 0

    def test_grants_are_scoped_to_their_recipient(self, grant_factory, now):
        store = GrantStore()
        store.issue(grant_factory(["a"], granted_to=OTHER_WORKER_ID))
        assert store.active_for(WORKER_ID, now=now) == []
        assert store.active_for(OTHER_WORKER_ID, now=now) != []

    def test_narrowing_by_task(self, grant_factory, now):
        store = GrantStore()
        store.issue(grant_factory(["a"], task_id="t1"))
        store.issue(grant_factory(["b"], task_id="t2"))
        assert len(store.active_for(WORKER_ID, now=now)) == 2
        assert len(store.active_for(WORKER_ID, now=now, task_id="t1")) == 1

    def test_purge_expired(self, grant_factory, now):
        store = GrantStore()
        store.issue(grant_factory(["a"], ttl_sec=60, task_id="t1"))
        store.issue(grant_factory(["b"], ttl_sec=6000, task_id="t2"))
        assert store.purge_expired(now + timedelta(seconds=61)) == 1
        assert len(store) == 1

    def test_snippet_ids_are_frozen(self, grant_factory):
        g = grant_factory(["a", "b"])
        assert isinstance(g.snippet_ids, tuple)
        with pytest.raises((AttributeError, TypeError)):
            g.snippet_ids.append("c")   # type: ignore[attr-defined]

    def test_a_grant_never_holds_a_query_or_wildcard(self, grant_factory):
        """
        Structural guarantee: the only thing a grant can carry is a list of IDs.
        There is no field, and no constructor argument, for a pattern.
        """
        g = grant_factory(["a"])
        assert set(vars(g)) == {
            "grant_id", "snippet_ids", "granted_to", "granted_by",
            "session_id", "task_id", "expires_at",
        }

    def test_grant_with_no_expiry_is_inactive(self, now):
        """Fail closed if expires_at is somehow absent."""
        g = new_grant(["a"], WORKER_ID, MANAGER_ID, "t", now=now)
        broken = type(g)(**{**vars(g), "expires_at": None})
        assert not broken.is_active(now)


# ── The filter ────────────────────────────────────────────────────────────────

class TestFilter:

    def test_returns_only_accessible_records(self, worker, rbac, now):
        records = [
            make_record(record_id="r1", source=WORKER_ID),
            make_record(record_id="r2", source=MANAGER_ID, visibility=Visibility.GLOBAL),
            make_record(record_id="r3", source=WORKER_ID),
        ]
        cleared = rbac.filter_records(worker, records, (), now)
        assert [c.record_id for c in cleared] == ["r1", "r3"]

    def test_preserves_relevance_order(self, manager, rbac, now):
        records = [
            make_record(record_id=f"r{i}", source=WORKER_ID, visibility=Visibility.GLOBAL)
            for i in range(5)
        ]
        cleared = rbac.filter_records(manager, records, (), now)
        assert [c.record_id for c in cleared] == [f"r{i}" for i in range(5)]

    def test_every_decision_is_audited(self, worker, rbac, audit, now):
        records = [
            make_record(record_id="r1", source=WORKER_ID),
            make_record(record_id="r2", source=MANAGER_ID),
        ]
        rbac.filter_records(worker, records, (), now, store="faiss")
        assert len(audit.events) == 2
        assert audit.denials_by_reason() == {Reason.LOCAL_ISOLATION: 1}
        assert audit.denials_by_robot() == {WORKER_ID: 1}
        assert {e.store for e in audit.events} == {"faiss"}

    def test_a_failing_audit_sink_does_not_break_retrieval(self, worker, now):
        class Exploding:
            def record(self, event): raise RuntimeError("audit database is down")
            def flush(self): raise RuntimeError("still down")

        rbac = RBACFilter(audit_sink=Exploding())
        with pytest.warns(RuntimeWarning):
            cleared = rbac.filter_records(
                worker, [make_record(source=WORKER_ID)], (), now
            )
        assert len(cleared) == 1   # retrieval still works

    def test_empty_input_is_empty_output(self, worker, rbac, now):
        assert rbac.filter_records(worker, [], (), now) == []


# ── The clearance stamp ───────────────────────────────────────────────────────

class TestClearanceStamp:

    def test_cleared_records_cannot_be_forged(self):
        with pytest.raises(ClearanceError):
            ClearedRecord(
                record=MemoryRecord("x", "leak"),
                decision=Decision(True, "own_record"),
            )

    def test_forging_with_a_guessed_token_fails(self):
        with pytest.raises(ClearanceError):
            ClearedRecord(
                record=MemoryRecord("x", "leak"),
                decision=Decision(True, "own_record"),
                _token=object(),
            )

    def test_assert_cleared_rejects_raw_strings(self):
        with pytest.raises(ClearanceError, match="no RBAC clearance stamp"):
            assert_cleared(["a raw retrieved string"], "test")

    def test_assert_cleared_rejects_raw_memory_records(self):
        with pytest.raises(ClearanceError):
            assert_cleared([make_record()], "test")

    def test_assert_cleared_accepts_filtered_output(self, worker, rbac, now):
        cleared = rbac.filter_records(worker, [make_record(source=WORKER_ID)], (), now)
        assert assert_cleared(cleared, "test") == cleared

    def test_assert_cleared_allows_nothing(self):
        assert assert_cleared([], "test") == []
        assert assert_cleared(None, "test") == []


def test_record_ids_are_store_scoped():
    """Two stores must never collide on an ID, or a grant could cross stores."""
    assert make_record_id("faiss", "1#0") != make_record_id("chat_logs", "1#0")
