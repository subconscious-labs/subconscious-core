"""Tests for user_id propagation through sessions and RAG."""
import json
import sqlite3
import sys

import pytest


@pytest.fixture()
def rag(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBCONSCIOUS_RAG_DB_PATH", str(tmp_path / "rag.sqlite3"))
    monkeypatch.setenv("SUBCONSCIOUS_RAG_ENABLED", "true")
    for mod in list(sys.modules):
        if mod == "subconscious.runtime.config" or mod.startswith("subconscious.memory"):
            del sys.modules[mod]
    from subconscious.memory import rag as _rag
    return _rag


@pytest.fixture()
def sess(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBCONSCIOUS_SESSION_DIR", str(tmp_path / "sessions"))
    for mod in list(sys.modules):
        if mod == "subconscious.runtime.config" or mod.startswith("subconscious.chat"):
            del sys.modules[mod]
    from subconscious.chat import session as _sess
    return _sess


def test_rag_migration_adds_user_id_column(rag):
    rag.init()
    conn = sqlite3.connect(rag._db_path())
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(rag_chunks)").fetchall()]
        assert "user_id" in cols
    finally:
        conn.close()


def test_rag_retrieve_filters_by_user(rag, monkeypatch):
    rag.init()
    conn = sqlite3.connect(rag._db_path())
    try:
        for src, uid in [("a", "a@b.c"), ("b", "b@x.y"), ("legacy", None)]:
            conn.execute(
                "INSERT INTO rag_chunks(source, chunk, embedding, added_at, user_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (src, f"chunk {src}", json.dumps([1.0, 0.0, 0.0]),
                 "2026-01-01T00:00:00Z", uid),
            )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(rag, "_embed", lambda _: [1.0, 0.0, 0.0])
    monkeypatch.setattr(rag, "RAG_MIN_SCORE", -1.0)

    result = rag.retrieve("q", user_id="a@b.c")
    assert result is not None
    assert "chunk a" in result
    assert "chunk b" not in result
    assert "chunk legacy" in result  # NULL visible during migration

    unfiltered = rag.retrieve("q", user_id=None)
    assert "chunk b" in unfiltered


def test_session_stores_user_id(sess):
    s = sess.get_session("test-1", user_id="a@b.c")
    assert s.user_id == "a@b.c"
    sess.record_turn(s, "hi", "hello", user_id="a@b.c")

    path = s._jsonl_path()
    first = json.loads(path.read_text().splitlines()[0])
    assert first["user_id"] == "a@b.c"


def test_session_user_id_does_not_flip_silently(sess):
    s = sess.get_session("test-2", user_id="a@b.c")
    sess.get_session("test-2", user_id="x@y.z")
    assert s.user_id == "a@b.c"  # original preserved
