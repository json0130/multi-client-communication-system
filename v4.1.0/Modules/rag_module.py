"""
rag_module.py – Per-user RAG component (Supabase backend, local FAISS cache)

Local files (./rag_indexes):
    user_<user_id>.faiss        : FAISS vectors
    user_<user_id>_texts.json   : parallel list of raw message strings

Supabase table 'chat_logs' stores: id (PK), user_id, message, response, created_at
Embeddings are *not* stored in Supabase.

Usage:
    rag = RagModule(user_id, supabase_client)
    ctx_texts = rag.search(query, top_k=8)
    row_id = db.insert_chat_log(user_id, message, response)
    rag.add(message)   # after insert (row_id not needed locally)
"""

from __future__ import annotations
import json, threading
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from openai import OpenAI
from supabase import Client as SupabaseClient

EMBED_MODEL = "text-embedding-ada-002"
INDEX_DIR = Path("./rag_indexes")
INDEX_DIR.mkdir(exist_ok=True)


class RagModule:
    def __init__(self, user_id: int, supabase: SupabaseClient):
        self.user_id = user_id
        self.supabase = supabase

        self._index: Optional[faiss.Index] = None
        self._embedded_texts: List[str] = []   # parallel list of message strings
        self._lock = threading.RLock()

        self._client = OpenAI()  # needs OPENAI_API_KEY
        self.faiss_path = INDEX_DIR / f"user_{user_id}.faiss"
        self.texts_path = INDEX_DIR / f"user_{user_id}_texts.json"

        self._load_or_build()

    # -------------------- Public API --------------------

    def search(self, query: str, top_k: int = 8) -> List[str]:
        """Return top_k similar past *user messages* (strings)."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            q_vec = np.array([self._embed(query)], dtype="float32")
            k = min(top_k, self._index.ntotal)
            D, I = self._index.search(q_vec, k)
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self._embedded_texts):
                    results.append(self._embedded_texts[idx])
            return results

    def add(self, message: str):
        """
        Add a new user message to the local index.
        Call AFTER inserting chat log row in Supabase.
        (We don't need the row_id locally.)
        """
        with self._lock:
            vec = np.array([self._embed(_normalize(message))], dtype="float32")
            if self._index is None:
                self._index = faiss.IndexFlatL2(vec.shape[1])
            self._index.add(vec)
            self._embedded_texts.append(_normalize(message))
            self._save()

    # -------------------- Internal helpers --------------------

    def _embed(self, text: str) -> List[float]:
        text = text[:8000]  # safety truncation
        resp = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return resp.data[0].embedding

    def _load_or_build(self):
        """Try to load existing local cache; otherwise rebuild from Supabase."""
        if self.faiss_path.exists() and self.texts_path.exists():
            try:
                self._index = faiss.read_index(str(self.faiss_path))
                self._embedded_texts = json.loads(self.texts_path.read_text())
                print(f"[RAG] Loaded index for user {self.user_id} "
                      f"({self._index.ntotal} vectors).")
                return
            except Exception as e:
                print(f"[RAG] Failed to load existing index → rebuilding. ({e})")

        # Rebuild from Supabase chat logs (all past messages)
        print(f"[RAG] Building index for user {self.user_id} from Supabase…")
        data = (
            self.supabase.table("chat_logs")
            .select("message")
            .eq("user_id", self.user_id)
            .order("id")
            .execute()
            .data
        ) or []

        if not data:
            print(f"[RAG] No chat logs for user {self.user_id}. Starting empty.")
            return

        texts = [_normalize(row["message"]) for row in data if row.get("message")]
        if not texts:
            print(f"[RAG] No non-empty messages for user {self.user_id}.")
            return

        # Batch embed to stay under token limits
        BATCH = 96
        all_vecs = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i + BATCH]
            resp = self._client.embeddings.create(
                model=EMBED_MODEL,
                input=chunk
            )
            all_vecs.extend([d.embedding for d in resp.data])

        arr = np.array(all_vecs, dtype="float32")
        self._index = faiss.IndexFlatL2(arr.shape[1])
        self._index.add(arr)
        self._embedded_texts = texts
        self._save()
        print(f"[RAG] Built index for user {self.user_id} with {len(texts)} messages.")

    def _save(self):
        try:
            if self._index:
                faiss.write_index(self._index, str(self.faiss_path))
                self.texts_path.write_text(json.dumps(self._embedded_texts))
        except Exception as e:
            print(f"[RAG] Save error: {e}")


def _normalize(s: str) -> str:
    return (s or "").strip()
