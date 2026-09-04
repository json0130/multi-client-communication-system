"""
data/connection.py
==================
Creates and holds the single Supabase client instance for the whole app.
Every repository imports `get_client()` — nothing else creates a supabase client.
"""

from __future__ import annotations
import logging
from typing import Optional

import httpx
from supabase import create_client, Client
from supabase.lib.client_options import SyncClientOptions
from core.config import cfg

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


class _RetryingTransport(httpx.HTTPTransport):
    """
    Retries once on a dropped connection.

    This process writes to Supabase sparsely — minutes can pass between demo
    events — so a pooled keep-alive connection going stale between writes is
    the common case, not an edge case. httpx's own keepalive_expiry narrows
    the window (it's already a short 5s default) but cannot close it: the
    server can close a connection for its own reasons at any moment, and the
    client has no way to know until it tries to reuse it. That surfaces as
    httpx.RemoteProtocolError ("Server disconnected without sending a
    response") — a transport-level failure, not a real one. A live demo run
    lost a knowledge-graph observation permanently to exactly this, silently,
    because the caller's best-effort error handling (correctly, for demo
    continuity) just swallowed it and moved on. This is the actual fix, not
    another log line asking a human to notice mid-demo.

    Safe to retry: postgrest sends request bodies as fully-buffered JSON
    bytes, not a one-shot stream, so resending the same Request object just
    re-reads already-materialized bytes — never replays arbitrary I/O — and
    every write this app makes through here is an insert/upsert of an
    already-computed row, safe to repeat if the first attempt never actually
    reached the server.
    """

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return super().handle_request(request)
        except httpx.RemoteProtocolError as e:
            logger.warning(f"[Supabase] Retrying after dropped connection: {e}")
            return super().handle_request(request)


def get_client() -> Client:
    """
    Return the shared Supabase client, creating it on first call.
    Thread-safe enough for our single-process server; if you move to
    multiprocessing, add a lock here.
    """
    global _client
    if _client is None:
        if cfg is None:
            raise RuntimeError(
                "Config not loaded — SUPABASE_URL and SUPABASE_KEY must be set."
            )
        httpx_client = httpx.Client(
            transport=_RetryingTransport(
                limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=15.0),
            ),
        )
        _client = create_client(
            cfg.db.url, cfg.db.key, options=SyncClientOptions(httpx_client=httpx_client)
        )
    return _client