"""
modules/rag/rag_module.py
=========================
Per-robot Retrieval-Augmented Generation module.

What it does:
  - Embeds user messages using a local Ollama embedding model (nomic-embed-text)
  - Stores vectors in a local FAISS index (one file per user)
  - On each chat, searches for the top-K most similar past messages
  - Returns them as context strings to inject into the LLM prompt

Storage layout (all local, no vectors in Supabase):
  ./rag_indexes/
      user_<id>.faiss          FAISS index
      user_<id>_texts.json     parallel list of raw message strings

Supabase is only used during first-boot to rebuild the index from chat history.

Requires: pip install faiss-cpu openai numpy
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI

from modules.base import BaseModule
from core.config import cfg


class RagModule(BaseModule):

    def __init__(self, user_id: int):
        self._user_id = user_id
        self._index: Optional[faiss.Index] = None
        self._texts: list[str] = []
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
            "vector_count": count,
            "embed_model": self._embed_model,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Return the top_k most similar past user messages to the query.
        Returns an empty list if index is empty or Ollama is unreachable.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            try:
                vec = self._embed(query)
                if vec is None:
                    return []
                q = np.array([vec], dtype="float32")
                k = min(top_k, self._index.ntotal)
                _, indices = self._index.search(q, k)
                return [
                    self._texts[i]
                    for i in indices[0]
                    if 0 <= i < len(self._texts)
                ]
            except Exception as e:
                print(f"[RagModule] search error: {e}")
                return []

    def add(self, message: str):
        """
        Embed a new user message and add it to the local index.
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
                self._texts.append(message)
                self._save()
            except Exception as e:
                print(f"[RagModule] add error: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

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
        """Load FAISS index and text list from disk."""
        self._index = faiss.read_index(str(self._faiss_path))
        self._texts = json.loads(self._texts_path.read_text())
        print(f"[RagModule] Loaded local index — {self._index.ntotal} vectors")

    def _build_from_supabase(self):
        """
        Rebuild the local index from Supabase chat_logs on first boot.
        Batches embedding calls to stay within token limits.
        """
        print(f"[RagModule] No local index found — rebuilding from Supabase...")
        try:
            from data.connection import get_client
            resp = (
                get_client()
                .table("chat_logs")
                .select("message")
                .eq("user_id", self._user_id)
                .order("id")
                .execute()
            )
            rows = resp.data or []
            texts = [r["message"].strip() for r in rows if r.get("message", "").strip()]

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
            self._texts = texts[: len(all_vecs)]
            self._save()
            print(f"[RagModule] Built index — {len(all_vecs)} vectors")

        except Exception as e:
            print(f"[RagModule] Supabase rebuild error: {e}")

    def _save(self):
        """Persist index and texts to disk."""
        try:
            if self._index:
                faiss.write_index(self._index, str(self._faiss_path))
                self._texts_path.write_text(json.dumps(self._texts))
        except Exception as e:
            print(f"[RagModule] save error: {e}")