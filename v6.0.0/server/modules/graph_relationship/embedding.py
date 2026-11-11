"""
Embedding-based capability↔topic matcher (optional, opt-in).

Builds a `Matcher` (see topics.Matcher) that, for a topic label, selects the
robot's best-fitting capability item by embedding cosine similarity — so a topic
the child raised connects to a capability even when the words differ
('addition' → 'good at math', 'planets' → 'knows about space').

Why argmax + floor (not a fixed threshold): single-word embeddings cluster
around ~0.5, so a global threshold can't separate a true near-match from an
unrelated pair. Instead we pick the SINGLE best capability per topic and require
it to clear a floor, which cleanly rejects unrelated topics.

Design contract
---------------
* The embedding backend is INJECTED as `embed_fn(text) -> list[float]`, so this
  module has no hard dependency on any model server — graph_relationship/ stays
  copy-pasteable. `ollama_embed_fn` is a convenience provider using only stdlib
  urllib. numpy is used for the cosine (optional dependency, imported here only).
* No PAD, no kg_bridge.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, List, Optional

import numpy as np

from .topics import Matcher

EmbedFn = Callable[[str], List[float]]

# Leading filler stripped before embedding so the core concept dominates:
# 'good at math' -> 'math', 'knows about space' -> 'space'.
_FILLER = re.compile(
    r"^(knows about|knows|good at|great at|likes|loves|enjoys|can|is|about|the)\s+",
    re.IGNORECASE,
)


def capability_core(item: str) -> str:
    """Strip leading filler words to the core concept of a capability item."""
    s = item.strip().lower()
    while True:
        stripped = _FILLER.sub("", s)
        if stripped == s:
            return s
        s = stripped


def ollama_embed_fn(
    model: str = "nomic-embed-text",
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout: float = 30.0,
) -> EmbedFn:
    """Return an embed_fn hitting Ollama's /api/embeddings (stdlib only)."""
    url = f"http://{host}:{port}/api/embeddings"

    def embed(text: str) -> List[float]:
        req = urllib.request.Request(
            url,
            data=json.dumps({"model": model, "prompt": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp).get("embedding", [])

    return embed


def make_embedding_matcher(
    embed_fn: EmbedFn, *, floor: float = 0.5, strip_filler: bool = True,
) -> Matcher:
    """Build a Matcher that selects the best capability item for a topic by
    embedding cosine similarity (argmax over items, must clear `floor`).

    Embeddings are cached per string. Any embedding failure → returns None (the
    caller then simply doesn't link — never raises).
    """
    cache: dict = {}

    def _vec(text: str) -> Optional[np.ndarray]:
        if text not in cache:
            try:
                v = np.asarray(embed_fn(text), dtype=float)
                n = np.linalg.norm(v)
                cache[text] = (v / n) if n > 0 else None
            except Exception:
                cache[text] = None
        return cache[text]

    def matcher(items: List[str], topic_label: str) -> Optional[str]:
        if not items:
            return None
        tv = _vec(topic_label)
        if tv is None:
            return None
        best_item, best_sim = None, -1.0
        for item in items:
            key = capability_core(item) if strip_filler else item
            iv = _vec(key)
            if iv is None:
                continue
            sim = float(iv @ tv)
            if sim > best_sim:
                best_sim, best_item = sim, item
        return best_item if best_sim >= floor else None

    return matcher
