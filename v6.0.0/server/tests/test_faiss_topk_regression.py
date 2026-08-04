"""
tests/test_faiss_topk_regression.py
===================================
Regression: a Worker must still get k results when the neighbourhood is
dominated by records it cannot read.

This is the failure mode that motivated filtering on metadata rather than
post-filtering a fixed top-k. With naive post-filtering, a Worker asking for 5
results while the 20 nearest vectors all belong to a Manager gets zero — its
retrieval is silently emptied and the LLM loses all episodic context.

Uses a real faiss.IndexFlatL2 with deterministic 2-D embeddings so distances are
exactly controllable. Only the Ollama embedding call is faked.
"""

from __future__ import annotations

import pytest

from core.rbac import AccessLevel, RBACFilter, RobotIdentity, Visibility
from tests.conftest import MANAGER_ID, SCENARIO, WORKER_ID

pytestmark = pytest.mark.integration


# Vectors live on a line. "m<i>" sits at distance i+1 from the query, "w<i>" at
# 100+i — so every Manager record is strictly nearer than every Worker record.
def _fake_embed(text: str):
    text = text.strip()
    if text == "query":
        return [0.0, 0.0]
    kind, num = text[0], int(text[1:])
    return [float(num + 1) if kind == "m" else float(100 + num), 0.0]


@pytest.fixture
def index(rag_config, monkeypatch):
    """
    An index of 20 Manager records (near) and 10 Worker records (far), all in
    one FAISS index — the shared-pool layout a Manager's cross-client view needs.
    """
    from modules.rag.rag_module import RagModule

    mod = RagModule(
        user_id=1, client_id=MANAGER_ID, scenario_id=SCENARIO,
        session_id="sess", default_visibility=Visibility.GLOBAL,
    )
    monkeypatch.setattr(mod, "_embed", _fake_embed)
    monkeypatch.setattr(mod, "_save", lambda: None)

    for i in range(20):
        mod.add(f"m{i}")

    # Same index, different author.
    mod._client_id = WORKER_ID
    mod._default_visibility = Visibility.LOCAL
    for i in range(10):
        mod.add(f"w{i}")

    mod._available = True
    return mod


@pytest.fixture
def worker_identity():
    return RobotIdentity(WORKER_ID, SCENARIO, "sess-w", AccessLevel.LOCAL)


@pytest.fixture
def manager_identity():
    return RobotIdentity(MANAGER_ID, SCENARIO, "sess-m", AccessLevel.GLOBAL)


def test_worker_still_gets_k_results_behind_inaccessible_records(
    index, worker_identity, rbac
):
    """The regression. Naive post-filtering of top-5 would return 0 here."""
    results = index.search("query", requester=worker_identity, rbac=rbac, top_k=5)

    assert len(results) == 5, (
        f"Worker got {len(results)} of 5 requested — its results were emptied by "
        f"the 20 nearer Manager records."
    )
    assert all(r.record.source_robot_id == WORKER_ID for r in results)
    assert [r.text for r in results] == ["w0", "w1", "w2", "w3", "w4"]


def test_results_stay_distance_ordered_after_filtering(index, worker_identity, rbac):
    results = index.search("query", requester=worker_identity, rbac=rbac, top_k=8)
    assert [r.text for r in results] == [f"w{i}" for i in range(8)]


def test_escalation_stops_at_the_index_size(index, worker_identity, rbac):
    """Asking for more than exists returns everything accessible, not an error."""
    results = index.search("query", requester=worker_identity, rbac=rbac, top_k=50)
    assert len(results) == 10
    assert all(r.record.source_robot_id == WORKER_ID for r in results)


def test_manager_sees_the_near_global_records(index, manager_identity, rbac):
    """The Manager's own view is unchanged — it gets the nearest records."""
    results = index.search("query", requester=manager_identity, rbac=rbac, top_k=5)
    assert [r.text for r in results] == ["m0", "m1", "m2", "m3", "m4"]


def test_manager_cannot_see_worker_local_records(index, manager_identity, rbac):
    results = index.search("query", requester=manager_identity, rbac=rbac, top_k=30)
    assert all(r.record.source_robot_id == MANAGER_ID for r in results)
    assert len(results) == 20


def test_each_record_is_audited_at_most_once_per_search(
    index, worker_identity, rbac, audit
):
    """
    Escalation re-runs the FAISS search with a larger k, which returns a superset.
    Only the new tail is decided, so denial counts stay meaningful.
    """
    index.search("query", requester=worker_identity, rbac=rbac, top_k=5)
    seen = [e.record_id for e in audit.events]
    assert len(seen) == len(set(seen)), "a record was audited twice in one search"


def test_empty_index_returns_nothing(rag_config, monkeypatch, worker_identity, rbac):
    from modules.rag.rag_module import RagModule
    mod = RagModule(user_id=99, client_id=WORKER_ID, scenario_id=SCENARIO)
    monkeypatch.setattr(mod, "_embed", _fake_embed)
    assert mod.search("query", requester=worker_identity, rbac=rbac, top_k=5) == []


def test_results_carry_a_clearance_stamp(index, worker_identity, rbac):
    """search() must return stamped records, or prompt assembly will reject them."""
    from core.rbac import ClearedRecord, assert_cleared
    results = index.search("query", requester=worker_identity, rbac=rbac, top_k=3)
    assert all(isinstance(r, ClearedRecord) for r in results)
    assert assert_cleared(results, "test") == results


# ── Sidecar format ────────────────────────────────────────────────────────────

class TestSidecarCompatibility:

    def test_v1_bare_list_is_upgraded_and_attributed_to_the_owning_robot(
        self, rag_config, monkeypatch, tmp_path
    ):
        """
        Legacy sidecars carry no provenance. Backfilling with the index's owning
        robot reproduces pre-RBAC behaviour: the robot could only ever see its
        own index, and still can.
        """
        import json
        import faiss
        import numpy as np
        from modules.rag.rag_module import RagModule

        mod = RagModule(user_id=7, client_id=WORKER_ID, scenario_id=SCENARIO)
        idx = faiss.IndexFlatL2(2)
        idx.add(np.array([[1.0, 0.0], [2.0, 0.0]], dtype="float32"))
        faiss.write_index(idx, str(mod._faiss_path))
        mod._texts_path.write_text(json.dumps(["hello", "world"]))

        mod._load_local()

        assert [r["text"] for r in mod._records] == ["hello", "world"]
        assert all(r["source_robot_id"] == WORKER_ID for r in mod._records)
        assert all(r["visibility"] == Visibility.LOCAL.value for r in mod._records)
        assert all(r["scenario_id"] is None for r in mod._records)

    def test_v1_records_remain_readable_by_their_owner(
        self, rag_config, monkeypatch, tmp_path, rbac, worker_identity
    ):
        import json
        import faiss
        import numpy as np
        from modules.rag.rag_module import RagModule

        mod = RagModule(user_id=8, client_id=WORKER_ID, scenario_id=SCENARIO)
        idx = faiss.IndexFlatL2(2)
        idx.add(np.array([[1.0, 0.0]], dtype="float32"))
        faiss.write_index(idx, str(mod._faiss_path))
        mod._texts_path.write_text(json.dumps(["legacy memory"]))
        mod._load_local()
        mod._available = True
        monkeypatch.setattr(mod, "_embed", lambda t: [0.0, 0.0])

        results = mod.search("query", requester=worker_identity, rbac=rbac, top_k=5)
        assert [r.text for r in results] == ["legacy memory"]

    def test_v2_envelope_round_trips(self, rag_config, monkeypatch):
        import json
        from modules.rag.rag_module import RagModule

        mod = RagModule(user_id=9, client_id=WORKER_ID, scenario_id=SCENARIO)
        monkeypatch.setattr(mod, "_embed", _fake_embed)
        mod.add("m0")
        mod.add("m1")

        raw = json.loads(mod._texts_path.read_text())
        assert raw["version"] == 2
        assert [r["text"] for r in raw["records"]] == ["m0", "m1"]
        assert raw["records"][0]["source_robot_id"] == WORKER_ID

        reloaded = RagModule(user_id=9, client_id=WORKER_ID, scenario_id=SCENARIO)
        reloaded._load_local()
        assert [r["text"] for r in reloaded._records] == ["m0", "m1"]

    def test_unrecognised_sidecar_is_ignored_not_trusted(self, rag_config, monkeypatch):
        import json
        import faiss
        import numpy as np
        from modules.rag.rag_module import RagModule

        mod = RagModule(user_id=10, client_id=WORKER_ID)
        idx = faiss.IndexFlatL2(2)
        idx.add(np.array([[1.0, 0.0]], dtype="float32"))
        faiss.write_index(idx, str(mod._faiss_path))
        mod._texts_path.write_text(json.dumps("not a list or envelope"))

        mod._load_local()
        assert mod._records == []
