"""
modules/rag/rag_module.py
=========================
Per-robot Retrieval-Augmented Generation module.

What it does:
  - Embeds user messages using a local Ollama embedding model (nomic-embed-text)
  - Stores vectors in a local FAISS index (one file per user)
  - On each chat, searches for the top-K most similar past messages
  - Filters them through RBAC, then returns them for prompt injection

Storage layout (all local, no vectors in Supabase):
  ./rag_indexes/
      user_<id>.faiss          FAISS index
      user_<id>_texts.json     parallel record metadata sidecar

Sidecar format
--------------
v1 (legacy)  a bare JSON array of strings, positionally aligned to FAISS rows.
v2 (current) {"version": 2, "records": [{text, source_robot_id, scenario_id,
             session_id, visibility, subject_user_id, created_at}, ...]}

v1 files are read and upgraded in memory on load, with source_robot_id
backfilled to the owning robot's client_id — the index file belongs to one
robot's user, so that is the provenance that already exists. The file is
rewritten as v2 on the next add().

RBAC
----
search() filters on record metadata, never by post-filtering a fixed top-k.
It over-fetches and escalates k until it has k accessible results or the index
is exhausted, so a Worker still gets its k results when the neighbourhood is
dominated by records it cannot read.

Why over-fetch rather than per-scope indices: the index is IndexFlatL2 — exact
brute-force — so over-fetching costs no recall, unlike an IVF/HNSW index where
a widened search changes which candidates are visited. Corpora here are small.
Per-scope indices would instead force a Manager's cross-client query to merge N
indices and would multiply an already large rag_indexes/ directory.

Supabase is only used during first-boot to rebuild the index from chat history.

Requires: pip install faiss-cpu openai numpy
"""

from __future__ import annotations
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np
from openai import OpenAI

from modules.base import BaseModule
from core.config import cfg
from core.rbac import (
    ClearedRecord,
    DelegationGrant,
    MemoryRecord,
    RBACFilter,
    RobotIdentity,
    Visibility,
    make_record_id,
)

STORE_NAME = "faiss"

# How many candidates to pull per accessible result wanted, before escalating.
OVERFETCH_FACTOR = 4


class RagModule(BaseModule):

    def __init__(
        self,
        user_id: int,
        client_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        session_id: Optional[str] = None,
        default_visibility: object = Visibility.LOCAL,
    ):
        self._user_id = user_id
        # Provenance stamped onto records this module writes, and used to
        # backfill v1 sidecars on load.
        self._client_id = client_id
        self._scenario_id = scenario_id
        self._session_id = session_id
        self._default_visibility = Visibility(default_visibility) \
            if not isinstance(default_visibility, Visibility) else default_visibility

        self._index: Optional[faiss.Index] = None
        self._records: list[dict] = []
        self._lock = threading.RLock()
        self._available = False

        # Ollama embedding client (same base_url pattern as LLM)
        self._embed_client = OpenAI(
            base_url=f"http://{cfg.rag.ollama_host}:{cfg.rag.ollama_port}/v1",
            api_key="ollama",
        )
        self._embed_model = cfg.rag.embed_model

        # Local file paths
        index_dir = Path(cfg.rag.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._faiss_path = index_dir / f"user_{user_id}.faiss"
        self._texts_path = index_dir / f"user_{user_id}_texts.json"

    # ── BaseModule interface ───────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Load existing local FAISS index, or build one from Supabase chat history.
        Returns True even if the index is empty — an empty index is valid.
        """
        try:
            if self._faiss_path.exists() and self._texts_path.exists():
                self._load_local()
            else:
                self._build_from_supabase()

            self._available = True
            count = self._index.ntotal if self._index else 0
            print(f"[RagModule] user={self._user_id} ready — {count} vectors")
            return True

        except Exception as e:
            print(f"[RagModule] initialize error: {e}")
            # Still mark available with empty index — don't block the robot
            self._available = True
            return True

    def is_available(self) -> bool:
        return self._available

    def get_status(self) -> dict:
        count = self._index.ntotal if self._index else 0
        return {
            "module": "rag",
            "available": self._available,
            "user_id": self._user_id,
            "source_robot_id": self._client_id,
            "scenario_id": self._scenario_id,
            "vector_count": count,
            "embed_model": self._embed_model,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        requester: RobotIdentity,
        rbac: RBACFilter,
        grants: Sequence[DelegationGrant] = (),
        top_k: int = 5,
        now: Optional[datetime] = None,
    ) -> list[ClearedRecord]:
        """
        Return up to top_k accessible past messages most similar to the query.

        Filtering happens on record metadata inside the retrieval loop, not by
        trimming a fixed top-k afterwards: candidates are over-fetched and k is
        escalated until top_k accessible records are found or the index is
        exhausted. A Worker therefore still receives its k results even when the
        nearest neighbours all belong to a Manager.

        Returns an empty list if the index is empty or Ollama is unreachable.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            try:
                vec = self._embed(query)
                if vec is None:
                    return []
                q = np.array([vec], dtype="float32")

                ntotal = self._index.ntotal
                fetch = min(ntotal, max(top_k * OVERFETCH_FACTOR, top_k))
                cleared: list[ClearedRecord] = []
                processed = 0

                while True:
                    _, indices = self._index.search(q, fetch)
                    ordered = [
                        i for i in indices[0]
                        if 0 <= i < len(self._records)
                    ]
                    # A larger k returns a superset in the same order, so only
                    # the new tail needs deciding — this keeps the audit log
                    # free of duplicate decisions for the same record.
                    new_rows = ordered[processed:]
                    processed = len(ordered)

                    candidates = [self._to_memory_record(i) for i in new_rows]
                    cleared.extend(
                        rbac.filter_records(
                            requester, candidates, grants, now, store=STORE_NAME
                        )
                    )

                    if len(cleared) >= top_k or fetch >= ntotal:
                        break
                    fetch = min(ntotal, fetch * 2)

                return cleared[:top_k]

            except Exception as e:
                print(f"[RagModule] search error: {e}")
                return []

    def add(
        self,
        message: str,
        subject_user_id: Optional[int] = None,
        visibility: object = None,
    ):
        """
        Embed a new user message and add it to the local index, stamped with the
        provenance and visibility of the robot that generated it.
        Called after every successful chat exchange.
        Non-blocking on embedding failure — just skips silently.
        """
        message = message.strip()
        if not message:
            return

        with self._lock:
            try:
                vec = self._embed(message)
                if vec is None:
                    return
                arr = np.array([vec], dtype="float32")

                if self._index is None:
                    self._index = faiss.IndexFlatL2(len(vec))

                self._index.add(arr)
                self._records.append(
                    self._new_record(message, subject_user_id, visibility)
                )
                self._save()
            except Exception as e:
                print(f"[RagModule] add error: {e}")

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        """
        Resolve one record by ID. Used when a delegated snippet needs to be
        looked up again; the caller must still run it through the RBAC filter.
        """
        with self._lock:
            for i in range(len(self._records)):
                mr = self._to_memory_record(i)
                if mr.record_id == record_id:
                    return mr
        return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _new_record(
        self,
        text: str,
        subject_user_id: Optional[int] = None,
        visibility: object = None,
    ) -> dict:
        vis = visibility if visibility is not None else self._default_visibility
        vis = vis.value if isinstance(vis, Visibility) else str(vis)
        return {
            "text": text,
            "source_robot_id": self._client_id,
            "scenario_id": self._scenario_id,
            "session_id": self._session_id,
            "visibility": vis,
            "subject_user_id": subject_user_id if subject_user_id is not None else self._user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _to_memory_record(self, position: int) -> MemoryRecord:
        """Adapt a sidecar row into the store-agnostic type the policy operates on."""
        row = self._records[position]
        return MemoryRecord(
            record_id=make_record_id(STORE_NAME, f"{self._user_id}#{position}"),
            content=row.get("text", ""),
            source_robot_id=row.get("source_robot_id"),
            scenario_id=row.get("scenario_id"),
            session_id=row.get("session_id"),
            visibility=row.get("visibility"),
            subject_user_id=row.get("subject_user_id"),
        )

    def _embed(self, text: str) -> Optional[list[float]]:
        """Call Ollama embedding API. Returns None on failure."""
        try:
            resp = self._embed_client.embeddings.create(
                model=self._embed_model,
                input=text[:8000],
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"[RagModule] embed error: {e}")
            return None

    def _load_local(self):
        """Load FAISS index and the record sidecar from disk, upgrading v1 to v2."""
        self._index = faiss.read_index(str(self._faiss_path))
        raw = json.loads(self._texts_path.read_text())
        self._records = self._parse_sidecar(raw)
        print(f"[RagModule] Loaded local index — {self._index.ntotal} vectors")

    def _parse_sidecar(self, raw: object) -> list[dict]:
        """
        Accept both sidecar formats.

        v1 is a bare list of strings with no metadata at all. Those records are
        backfilled with this index's owning robot as source and visibility
        'local', which reproduces today's behaviour exactly: before RBAC each
        robot could only ever see its own index.
        """
        if isinstance(raw, dict) and "records" in raw:
            records = raw.get("records") or []
            return [r for r in records if isinstance(r, dict)]

        if isinstance(raw, list):
            return [
                {
                    "text": t,
                    "source_robot_id": self._client_id,
                    "scenario_id": None,
                    "session_id": None,
                    "visibility": Visibility.LOCAL.value,
                    "subject_user_id": self._user_id,
                    "created_at": None,
                }
                for t in raw
                if isinstance(t, str)
            ]

        print(f"[RagModule] Unrecognised sidecar format in {self._texts_path.name} — ignoring")
        return []

    def _build_from_supabase(self):
        """
        Rebuild the local index from Supabase chat_logs on first boot.
        Batches embedding calls to stay within token limits.
        """
        print(f"[RagModule] No local index found — rebuilding from Supabase...")
        try:
            from data.connection import get_client

            def _fetch(columns: str):
                return (
                    get_client()
                    .table("chat_logs")
                    .select(columns)
                    .eq("user_id", self._user_id)
                    .order("id")
                    .execute()
                )

            try:
                resp = _fetch(
                    "message, source_robot_id, scenario_id, session_id, "
                    "visibility, subject_user_id"
                )
            except Exception as e:
                # 002_rbac.sql has not been applied yet. Fall back to the
                # pre-RBAC column set so the server still boots; the records
                # below are then stamped with this robot's own identity, which
                # is the same provenance the migration's backfill would recover.
                print(f"[RagModule] RBAC columns unavailable ({e}) — "
                      f"rebuilding without provenance. Apply 002_rbac.sql.")
                resp = _fetch("message")

            rows = [r for r in (resp.data or []) if (r.get("message") or "").strip()]
            texts = [r["message"].strip() for r in rows]

            if not texts:
                print(f"[RagModule] No chat history for user {self._user_id} — empty index")
                return

            # Batch embed
            BATCH = 64
            all_vecs = []
            for i in range(0, len(texts), BATCH):
                chunk = texts[i:i + BATCH]
                try:
                    resp = self._embed_client.embeddings.create(
                        model=self._embed_model,
                        input=chunk,
                    )
                    all_vecs.extend([d.embedding for d in resp.data])
                except Exception as e:
                    print(f"[RagModule] Batch embed error (skipping batch): {e}")

            if not all_vecs:
                return

            arr = np.array(all_vecs, dtype="float32")
            self._index = faiss.IndexFlatL2(arr.shape[1])
            self._index.add(arr)
            # Carry through whatever provenance the interaction log has; fall back
            # to this robot's own identity and a fail-closed 'local' visibility.
            self._records = [
                {
                    "text": r["message"].strip(),
                    "source_robot_id": r.get("source_robot_id") or self._client_id,
                    "scenario_id": r.get("scenario_id") or self._scenario_id,
                    "session_id": r.get("session_id"),
                    "visibility": r.get("visibility") or Visibility.LOCAL.value,
                    "subject_user_id": r.get("subject_user_id") or self._user_id,
                    "created_at": None,
                }
                for r in rows[: len(all_vecs)]
            ]
            self._save()
            print(f"[RagModule] Built index — {len(all_vecs)} vectors")

        except Exception as e:
            print(f"[RagModule] Supabase rebuild error: {e}")

    def _save(self):
        """Persist index and the v2 record sidecar to disk."""
        try:
            if self._index:
                faiss.write_index(self._index, str(self._faiss_path))
                self._texts_path.write_text(
                    json.dumps({"version": 2, "records": self._records})
                )
        except Exception as e:
            print(f"[RagModule] save error: {e}")
