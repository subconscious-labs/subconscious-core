"""Event ledger on SQLite.

Append-only from the caller's perspective; internal status is mutable
(pending -> processing -> processed/linked/skipped). SQLite is stdlib,
so this adds no runtime dependency. Pattern mirrors memory/rag.py:
open per call, parameterised SQL, no ORM.

Keep this module a leaf: no imports from other project modules except
runtime.config. That way chat.api, worker.tick and tools.internal can
all depend on it without cycles.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from subconscious.runtime.config import EVENTS_DB_PATH

log = logging.getLogger("subconscious.events.store")

# Priority scale (smaller = more urgent):
#   0 critical   - immediate, ahead of interactive
#   1 interactive - webui, xmpp, email
#   2 normal      - alerts, monitor, other async
#   3 background  - ticks, housekeeping

PRIORITY_CRITICAL    = 0
PRIORITY_INTERACTIVE = 1
PRIORITY_NORMAL      = 2
PRIORITY_BACKGROUND  = 3


def _db_path() -> Path:
    return Path(os.path.expanduser(EVENTS_DB_PATH))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _new_id() -> str:
    return "ev-" + uuid.uuid4().hex[:12]


def init() -> None:
    """Create table and indexes if they do not exist."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id             TEXT PRIMARY KEY,
                ts             TEXT NOT NULL,
                source         TEXT NOT NULL,
                user_id        TEXT,
                event_type     TEXT NOT NULL,
                priority       INTEGER NOT NULL DEFAULT 2,
                status         TEXT NOT NULL DEFAULT 'pending',
                goal_id        TEXT,
                correlation_id TEXT,
                payload        TEXT NOT NULL,
                processed_at   TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_pending "
            "ON events(status, priority, ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_goal "
            "ON events(goal_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_correlation "
            "ON events(correlation_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_user "
            "ON events(user_id)"
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        log.info("events store ready at %s (total: %d)", _db_path(), total)
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = {k: row[k] for k in row.keys()}
    try:
        d["payload"] = json.loads(d["payload"])
    except Exception:
        pass
    return d


def append(
    source: str,
    event_type: str,
    payload: Dict[str, Any],
    user_id: Optional[str] = None,
    priority: int = PRIORITY_NORMAL,
    correlation_id: Optional[str] = None,
) -> str:
    """Insert one event. Returns its id."""
    event_id = _new_id()
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO events "
            "(id, ts, source, user_id, event_type, priority, status, "
            " correlation_id, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                event_id,
                _now_iso(),
                source,
                user_id,
                event_type,
                priority,
                correlation_id,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        conn.commit()
        return event_id
    finally:
        conn.close()


def mark(
    event_id: str,
    status: str,
    goal_id: Optional[str] = None,
) -> bool:
    """Update status (and optionally goal_id). Returns True if a row matched."""
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE events SET status=?, goal_id=COALESCE(?, goal_id), "
            "processed_at=? WHERE id=?",
            (status, goal_id, _now_iso(), event_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def link(event_id: str, goal_id: str) -> bool:
    """Attach an event to a goal and mark it linked."""
    return mark(event_id, status="linked", goal_id=goal_id)


def get(event_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM events WHERE id=?", (event_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def next_pending() -> Optional[Dict[str, Any]]:
    """Lowest-priority-number pending event, oldest first."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM events WHERE status='pending' "
            "ORDER BY priority ASC, ts ASC LIMIT 1"
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_events(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    goal_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []
    if user_id is not None:
        clauses.append("user_id=?")
        params.append(user_id)
    if status is not None:
        clauses.append("status=?")
        params.append(status)
    if goal_id is not None:
        clauses.append("goal_id=?")
        params.append(goal_id)
    if correlation_id is not None:
        clauses.append("correlation_id=?")
        params.append(correlation_id)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    limit = max(1, min(int(limit), 200))
    params.append(limit)

    conn = _connect()
    try:
        rows = conn.execute(
            f"SELECT * FROM events{where} ORDER BY ts DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()
