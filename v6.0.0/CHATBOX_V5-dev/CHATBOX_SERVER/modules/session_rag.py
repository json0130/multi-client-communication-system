"""
App-layer RAG over the SQLite session transcripts (Phase 2).

Embeds each turn (child + reply) once — cached in the `embedding` column of the
SessionStore — and retrieves the most relevant past turns for a query, blended
with recency so the timeline matters. Used to (a) enrich the live prompt with
relevant past conversation and (b) power the viz "click a topic → history".

Vector search uses FAISS (IndexFlatIP on L2-normalized vectors) when available,
falling back to a NumPy dot product. The embedding function is INJECTED
(`embed_fn(text) -> list[float]`), so this module has no Ollama/graph/PAD imports
beyond numpy/faiss.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:  # noqa: BLE001
    _HAVE_FAISS = False

EmbedFn = Callable[[str], List[float]]


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SessionRAG:
    """Timeline-aware retrieval over transcript turns. Lazily embeds new turns."""

    def __init__(self, store, embed_fn: EmbedFn, *, recency_weight: float = 0.15):
        self.store = store
        self.embed_fn = embed_fn
        self.recency_weight = recency_weight

    def reindex(self) -> int:
        """Embed and store any turns missing an embedding. Returns how many added.
        Safe to call often; embedding failures are skipped (retried next time)."""
        added = 0
        for turn_id, text in self.store.turns_needing_embedding():
            try:
                v = self.embed_fn(text)
            except Exception:  # noqa: BLE001
                v = None
            if v:
                self.store.set_embedding(turn_id, v)
                added += 1
        return added

    def search(self, query: str, *, top_k: int = 4,
               person_id: Optional[str] = None) -> List[dict]:
        """Return up to top_k turns most relevant to `query`, recency-blended and
        returned in timeline order. Each item: {ts, person_id, emotion, child,
        reply, score}. Empty list if nothing embedded or the query can't embed."""
        self.reindex()
        rows = self.store.embedded_turns(person_id)
        if not rows:
            return []
        try:
            qv = self.embed_fn(query)
        except Exception:  # noqa: BLE001
            qv = None
        if not qv:
            return []

        mat = _normalize(np.asarray([r["vec"] for r in rows], dtype="float32"))
        q = _normalize(np.asarray([qv], dtype="float32"))[0]

        if _HAVE_FAISS:
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            k = min(max(top_k * 3, top_k), len(rows))
            sims, idxs = index.search(q[None, :].astype("float32"), k)
            cand = [(int(i), float(s)) for i, s in zip(idxs[0], sims[0]) if i >= 0]
        else:
            sims = mat @ q
            order = np.argsort(-sims)[: max(top_k * 3, top_k)]
            cand = [(int(i), float(sims[i])) for i in order]

        # Recency blend: newest turn gets +recency_weight, oldest +0.
        n = len(rows)
        blended = []
        for i, sim in cand:
            recency = i / (n - 1) if n > 1 else 1.0
            blended.append((i, sim + self.recency_weight * recency))
        blended.sort(key=lambda x: -x[1])
        chosen = blended[:top_k]

        # Present in timeline order (oldest → newest).
        chosen.sort(key=lambda x: rows[x[0]]["ts"])
        out = []
        for i, score in chosen:
            r = rows[i]
            out.append({"ts": r["ts"], "person_id": r["person_id"],
                        "emotion": r["emotion"], "child": r["child"],
                        "reply": r["reply"], "score": round(score, 3)})
        return out
