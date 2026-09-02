"""
App-layer SQLite store for conversation transcripts.

Session history (every turn) lives HERE, not in the knowledge graph — so the KG
stays focused on relationships/topics/interests and the viz is no longer cluttered
with per-session nodes. The graph keeps only lightweight links (topic notes,
Interaction rapport/trust/count); full transcripts are keyed by
(person, robot, session, timestamp) in this DB and surfaced via RAG (Phase 2).

Pure stdlib `sqlite3` — no graph_relationship / LLM / PAD imports. An `embedding`
column is reserved for the Phase-2 FAISS/RAG index (populated later).

Row shape returned to callers matches the extraction input contract:
  {"turn": int, "ts": iso8601, "emotion": str|None, "child": str|None, "reply": str|None}
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional

DEFAULT_DB = "sessions.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """SQLite-backed transcript store. One row per conversation turn."""

    def __init__(self, path: str = DEFAULT_DB) -> None:
        self.path = path
        # check_same_thread=False + a lock: the viz server is a ThreadingHTTPServer,
        # so requests touch this connection from different threads. The lock
        # serializes all access (execute + fetch) so it stays consistent.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init()

    def _init(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                person_id   TEXT NOT NULL,
                robot_id    TEXT NOT NULL,
                turn_idx    INTEGER NOT NULL,
                ts          TEXT NOT NULL,
                emotion     TEXT,
                child       TEXT,
                reply       TEXT,
                topics      TEXT,          -- JSON list of detected topic labels
                embedding   TEXT,          -- JSON list[float] for RAG (Phase 2), nullable
                extracted   INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_turns_person  ON turns(person_id);
            CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_turns_extract ON turns(person_id, extracted);
            """
        )
        self._conn.commit()

    # ── writes ────────────────────────────────────────────────────────────────

    def append_turn(
        self, *, session_id: str, person_id: str, robot_id: str,
        emotion: Optional[str] = None, child: Optional[str] = None,
        reply: Optional[str] = None, topics: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Append one turn; turn_idx auto-increments within the session. Returns row id."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(turn_idx), 0) FROM turns WHERE session_id = ?",
                (session_id,),
            )
            next_idx = int(cur.fetchone()[0]) + 1
            cur = self._conn.execute(
                "INSERT INTO turns(session_id, person_id, robot_id, turn_idx, ts, "
                "emotion, child, reply, topics, embedding) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (session_id, person_id, robot_id, next_idx, _now(), emotion, child, reply,
                 json.dumps(topics) if topics else None,
                 json.dumps(embedding) if embedding else None),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_turn_topics(self, row_id: int, topics: Optional[List[str]]) -> None:
        """Set the `topics` label on an already-inserted turn (by row id). Used when
        the live topic label is computed asynchronously, after the turn is recorded.
        Thread-safe (holds the store lock)."""
        with self._lock:
            self._conn.execute(
                "UPDATE turns SET topics = ? WHERE id = ?",
                (json.dumps(topics) if topics else None, int(row_id)),
            )
            self._conn.commit()

    def rename_person(self, old_person_id: str, new_person_id: str) -> int:
        """Re-key a person's transcript rows (e.g. 'guest_3' -> 'jay' once they tell
        us their name). Returns the number of turns re-pointed."""
        if old_person_id == new_person_id:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "UPDATE turns SET person_id = ? WHERE person_id = ?",
                (new_person_id, old_person_id),
            )
            self._conn.commit()
            return cur.rowcount

    def mark_extracted(self, person_id: str) -> int:
        """Mark all of a person's un-extracted turns as extracted. Returns count."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE turns SET extracted = 1 WHERE person_id = ? AND extracted = 0",
                (person_id,),
            )
            self._conn.commit()
            return cur.rowcount

    # ── reads ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _turn_row(r: sqlite3.Row) -> dict:
        return {"turn": r["turn_idx"], "ts": r["ts"], "emotion": r["emotion"],
                "child": r["child"], "reply": r["reply"]}

    def unextracted_turns(self, person_id: str) -> List[dict]:
        """A person's turns not yet fed to extraction, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM turns WHERE person_id = ? AND extracted = 0 "
                "ORDER BY id ASC", (person_id,),
            ).fetchall()
        return [self._turn_row(r) for r in rows]

    def recent_turns(self, person_id: str, limit: int = 5) -> List[dict]:
        """A person's most recent turns, returned oldest→newest (chronological).
        Used to feed recent conversation flow into the prompt across sessions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM turns WHERE person_id = ? ORDER BY id DESC LIMIT ?",
                (person_id, int(limit)),
            ).fetchall()
        return [self._turn_row(r) for r in reversed(rows)]

    def turns_for_topic(self, topic_label: str, person_id: Optional[str] = None,
                        limit: int = 20) -> List[dict]:
        """Turns whose detected `topics` include `topic_label` (timeline order).
        A cheap keyword fallback; Phase-2 RAG adds semantic retrieval."""
        q = "SELECT * FROM turns WHERE topics IS NOT NULL"
        params: list = []
        if person_id:
            q += " AND person_id = ?"
            params.append(person_id)
        q += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        out = []
        for r in rows:
            try:
                tops = json.loads(r["topics"]) or []
            except (TypeError, ValueError):
                tops = []
            if any(str(t).lower() == topic_label.lower() for t in tops):
                out.append(self._turn_row(r))
        return out[-limit:]

    def person_turn_count(self, person_id: str, robot_id: Optional[str] = None) -> int:
        with self._lock:
            if robot_id:
                r = self._conn.execute(
                    "SELECT COUNT(*) FROM turns WHERE person_id = ? AND robot_id = ?",
                    (person_id, robot_id)).fetchone()
            else:
                r = self._conn.execute(
                    "SELECT COUNT(*) FROM turns WHERE person_id = ?", (person_id,)).fetchone()
        return int(r[0])

    def session_count(self) -> int:
        """Distinct sessions (conversations) recorded — drives the consolidation cadence."""
        with self._lock:
            r = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM turns").fetchone()
        return int(r[0])

    def people(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT person_id FROM turns").fetchall()
        return [r[0] for r in rows]

    # ── embeddings (Phase-2 RAG) ──────────────────────────────────────────────

    def turns_needing_embedding(self) -> List[tuple]:
        """[(turn_id, text), ...] for turns that have text but no embedding yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, child, reply FROM turns "
                "WHERE embedding IS NULL AND (child IS NOT NULL OR reply IS NOT NULL)"
            ).fetchall()
        out = []
        for r in rows:
            text = f"{r['child'] or ''} {r['reply'] or ''}".strip()
            if text:
                out.append((r["id"], text))
        return out

    def set_embedding(self, turn_id: int, vec: List[float]) -> None:
        with self._lock:
            self._conn.execute("UPDATE turns SET embedding = ? WHERE id = ?",
                               (json.dumps(list(vec)), turn_id))
            self._conn.commit()

    def embedded_turns(self, person_id: Optional[str] = None) -> List[dict]:
        """Turns that have an embedding, with the vector, for RAG search."""
        q = ("SELECT id, person_id, robot_id, ts, emotion, child, reply, embedding "
             "FROM turns WHERE embedding IS NOT NULL")
        params: list = []
        if person_id:
            q += " AND person_id = ?"
            params.append(person_id)
        q += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        out = []
        for r in rows:
            try:
                vec = json.loads(r["embedding"])
            except (TypeError, ValueError):
                vec = None
            if vec:
                out.append({"id": r["id"], "person_id": r["person_id"],
                            "robot_id": r["robot_id"], "ts": r["ts"],
                            "emotion": r["emotion"], "child": r["child"],
                            "reply": r["reply"], "vec": vec})
        return out

    def close(self) -> None:
        self._conn.close()
