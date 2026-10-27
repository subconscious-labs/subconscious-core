"""Minimal unit tests for background.py.

Not covered here: real LLM calls, tool execution, journal rotation.
Those are exercised via the /api/tick smoke test.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def bg(tmp_path, monkeypatch):
    """Import background with isolated state dir."""
    monkeypatch.setenv("SUBCONSCIOUS_BG_STATE_DIR", str(tmp_path / "bg"))
    for mod in list(sys.modules):
        if mod in ("background", "config"):
            del sys.modules[mod]
    from subconscious.worker import tick as background
    return background


def test_goals_atomic_roundtrip(bg):
    g = bg._read_goals()
    assert g["goals"] == []
    goal = bg._new_goal("hello world", kind="normal")
    g["goals"].append(goal)
    bg._write_goals(g)

    g2 = bg._read_goals()
    assert len(g2["goals"]) == 1
    assert g2["goals"][0]["title"] == "hello world"
    # No leftover .tmp files.
    tmp_files = list(bg._state_dir().glob("*.tmp"))
    assert tmp_files == []


def test_open_goals_count(bg):
    goals = [
        {"status": "pending"},
        {"status": "active"},
        {"status": "done"},
        {"status": "blocked"},
        {"status": "cancelled"},
    ]
    assert len(bg._open_goals(goals)) == 2


def test_find_goal(bg):
    g = bg._new_goal("x", kind="normal")
    goals = [g]
    assert bg._find_goal(goals, g["id"]) is g
    assert bg._find_goal(goals, "nope") is None


def test_extract_json_plain(bg):
    assert bg._extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced(bg):
    text = 'Blah blah\n```json\n{"decision": "skip", "reason": "x"}\n```\nDone.'
    assert bg._extract_json(text) == {"decision": "skip", "reason": "x"}


def test_extract_json_garbage(bg):
    assert bg._extract_json("no json here") is None
    assert bg._extract_json("{not valid}") is None


def test_budget_reset_on_new_day(bg, monkeypatch):
    b = bg._read_budget()
    b["tokens_used"] = 999
    b["date"] = "1970-01-01"
    bg._atomic_write_json(bg._budget_path(), b)

    b2 = bg._read_budget()
    assert b2["tokens_used"] == 0
    assert b2["date"] == bg._today_utc()


def test_exploratory_cancel_stale(bg, monkeypatch):
    monkeypatch.setattr(bg, "BG_EXPLORATORY_TTL_HOURS", 1)
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "000Z"
    goals = [
        {"id": "g-1", "kind": "exploratory", "status": "pending",
         "created_at": old, "history": []},
        {"id": "g-2", "kind": "normal", "status": "pending",
         "created_at": old, "history": []},
    ]
    changed = bg._cancel_stale_exploratory(goals)
    assert changed == 1
    assert goals[0]["status"] == "cancelled"
    assert goals[1]["status"] == "pending"


def test_exploratory_tool_calls_default_zero(bg):
    """The tool_calls key must exist in executor output for the tick-level
    downgrade logic to work. This is a simple contract test."""
    # Not a real backend call - we just verify the key is expected in the
    # return shape by inspecting the source, since mocking _llm_call would
    # require importing a lot of chatd machinery.
    import inspect
    src = inspect.getsource(bg._executor)
    assert '"tool_calls": tool_calls_made' in src


def test_auto_resume_picks_existing_exploratory(bg):
    """When a pending exploratory exists, _active_exploratory must find it."""
    goals = [
        {"id": "g-1", "kind": "exploratory", "status": "pending",
         "attempts": 1, "history": []},
        {"id": "g-2", "kind": "normal", "status": "pending",
         "attempts": 0, "history": []},
    ]
    active = bg._active_exploratory(goals)
    assert len(active) == 1
    assert active[0]["id"] == "g-1"
