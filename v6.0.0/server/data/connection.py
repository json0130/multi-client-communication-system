"""
data/connection.py
==================
Creates and holds the single Supabase client instance for the whole app.
Every repository imports `get_client()` — nothing else creates a supabase client.
"""

from __future__ import annotations
from typing import Optional

from supabase import create_client, Client
from core.config import cfg

_client: Optional[Client] = None


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
        _client = create_client(cfg.db.url, cfg.db.key)
    return _client