#!/usr/bin/env python3
"""rag.py - L1.5 external semantic recall store for chatd.

Implements the request-scoped RAG sidecar layer (L1.5) - a chatd-owned
SQLite store completely separate from mempalace. Conversation turns are
embedded via the active backend and retrieved by cosine similarity for
the current user message.

L1.5 is always injected into the system prompt alongside KG recall.
It does not replace or modify any mempalace layer.

Set SUBCONSCIOUS_RAG_ENABLED=false to disable the layer entirely without
touching any other RAG settings.
"""
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from subconscious import backends
from subconscious.runtime.config import (
    RAG_DB_PATH,
    RAG_EMBED_MODEL,
    RAG_ENABLED,
    RAG_MAX_CHARS_PER_CHUNK,
    RAG_MIN_SCORE,
    RAG_TOP_K,
)

log = logging.getLogger("chatd.rag")


def _db_path() -> Path:
    return Path(os.path.expanduser(RAG_DB_PATH))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def init() -> None:
    """Initialise the L1.5 external RAG store."""
    if not RAG_ENABLED:
        log.info("RAG: disabled via SUBCONSCIOUS_RAG_ENABLED")
        return
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk TEXT NOT NULL,
                embedding TEXT NOT NULL,
                added_at TEXT NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_added_at ON rag_chunks(added_at)"
        )

        # Migration: user_id added after initial deployments. SQLite has no
        # ADD COLUMN IF NOT EXISTS, so probe first. Existing rows stay NULL
        # and are treated as legacy by retrieve().
        if not _has_column(conn, "rag_chunks", "user_id"):
            conn.execute("ALTER TABLE rag_chunks ADD COLUMN user_id TEXT")
            log.info("RAG: added user_id column (existing rows remain NULL)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_user ON rag_chunks(user_id)"
        )

        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        nulls = conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE user_id IS NULL"
        ).fetchone()[0]
        log.info(
            "RAG: store ready at %s (total: %d, no user_id: %d)",
            _db_path(), total, nulls,
        )
        if nulls > 0:
            log.warning(
                "RAG: %d legacy chunk(s) without user_id; they remain "
                "visible to every query until re-indexed",
                nulls,
            )
    finally:
        conn.close()


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _chunk_turn(user: str, assistant: str) -> str:
    user = _truncate(user, RAG_MAX_CHARS_PER_CHUNK // 2)
    assistant = _truncate(assistant, RAG_MAX_CHARS_PER_CHUNK // 2)
    return f"User: {user}\nAssistant: {assistant}".strip()


def _embed(text: str) -> List[float]:
    """Embed *text* using the backend selected for RAG_EMBED_MODEL."""
    backend = backends.get_backend(RAG_EMBED_MODEL)
    return backend.embed(text, RAG_EMBED_MODEL)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def _count_total() -> int:
    """Return total number of chunks in the RAG store."""
    try:
        conn = _connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        finally:
            conn.close()
    except Exception:
        return -1


def index_turn(
    source: str, user: str, assistant: str,
    user_id: Optional[str] = None,
) -> None:
    """Index one conversation turn into the L1.5 RAG store."""
    if not RAG_ENABLED:
        return
    chunk = _chunk_turn(user, assistant)
    if not chunk:
        return
    chunk_len = len(chunk)
    try:
        embedding = _embed(chunk)
    except Exception as e:
        log.warning("RAG: embedding failed during index_turn: %s", e)
        return

    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rag_chunks(source, chunk, embedding, added_at, user_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source,
                chunk,
                json.dumps(embedding),
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                user_id,
            ),
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        log.debug(
            "RAG: indexed chunk source=%s chunk_len=%d uid=%s db_total=%d",
            source, chunk_len, user_id or "-", total,
        )
    except Exception as e:
        log.warning("RAG: failed to index chunk for %s: %s", source, e)
    finally:
        conn.close()


def retrieve(
    query: str, top_k: int = RAG_TOP_K,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """Return top-k semantically similar past turns for *query*.

    user_id filters to that user's chunks plus legacy chunks with NULL
    user_id (pre-migration). Without user_id the query is not filtered,
    preserving pre-migration behaviour for callers that do not know
    who is asking.
    """
    if not RAG_ENABLED:
        return None
    query = (query or "").strip()
    if not query:
        return None

    try:
        qvec = _embed(query)
    except Exception as e:
        log.warning("RAG: embedding failed during retrieve: %s", e)
        return None

    conn = _connect()
    try:
        if user_id is None:
            rows = conn.execute(
                "SELECT source, chunk, embedding, added_at FROM rag_chunks "
                "ORDER BY id DESC LIMIT 200"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT source, chunk, embedding, added_at FROM rag_chunks "
                "WHERE user_id = ? OR user_id IS NULL "
                "ORDER BY id DESC LIMIT 200",
                (user_id,),
            ).fetchall()
    except Exception as e:
        log.warning("RAG: retrieval query failed: %s", e)
        return None
    finally:
        conn.close()

    scored = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
        except Exception:
            continue
        score = _cosine_similarity(qvec, emb)
        if score >= RAG_MIN_SCORE:
            scored.append((score, row["source"], row["chunk"], row["added_at"]))

    if not scored:
        log.debug("RAG: no chunks above min_score=%.3f (scanned %d)", RAG_MIN_SCORE, len(rows))
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    sources = [s for _, s, _, _ in top]
    total_chars = sum(len(c) for _, _, c, _ in top)
    log.debug(
        "RAG: retrieved %d/%d chunks, total_chars=%d, sources=%s, scores=%s",
        len(top),
        len(scored),
        total_chars,
        sources,
        [f"{sc:.3f}" for sc, _, _, _ in top],
    )

    blocks = []
    for score, source, chunk, added_at in top:
        blocks.append(
            f"[source={source} score={score:.3f} at={added_at}]\n{chunk}"
        )
    return "\n\n---\n\n".join(blocks)
