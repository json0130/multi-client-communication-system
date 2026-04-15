"""
data/chat_log_repo.py
=====================
All database operations that touch the 'chat_logs' table.

Supabase schema expected:
  chat_logs (
    id          serial PRIMARY KEY,
    user_id     integer REFERENCES users(user_id),
    message     text,
    response    text,
    created_at  timestamptz DEFAULT now()
  )
"""

from __future__ import annotations
from typing import Optional

from data.connection import get_client


def insert(user_id: int, message: str, response: str) -> Optional[int]:
    """
    Persist a user message + bot response pair.
    Returns the new row id, or None on failure.
    """
    try:
        resp = (
            get_client()
            .table("chat_logs")
            .insert({"user_id": user_id, "message": message, "response": response})
            .execute()
        )
        return resp.data[0]["id"]
    except Exception as e:
        print(f"[chat_log_repo] insert error: {e}")
        return None


def get_recent_messages(user_id: int, limit: int = 100) -> list[str]:
    """
    Return the most recent <limit> user message strings for a given user.
    Used by the RAG module and topic-inference endpoint.
    """
    try:
        resp = (
            get_client()
            .table("chat_logs")
            .select("message")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return [r["message"] for r in (resp.data or []) if r.get("message")]
    except Exception as e:
        print(f"[chat_log_repo] get_recent_messages error: {e}")
        return []