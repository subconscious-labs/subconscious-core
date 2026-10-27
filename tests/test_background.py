"""Minimal unit tests for worker.tick and worker.ledger.

Not covered here: real LLM calls, tool execution, journal rotation.
Those are exercised via the /api/tick smoke test.
"""
import inspect
import sys

import pytest


@pytest.fixture()
def bg(tmp_path, monkeypatch):
    """Import worker.tick with isolated state dir."""
    monkeypatch.setenv("SUBCONSCIOUS_BG_STATE_DIR", str(tmp_path / "bg"))
    for mod in list(sys.modules):
        if mod == "subconscious.runtime.config" or mod.startswith("subconscious.worker"):
            del sys.modules[mod]
    from subconscious.worker import tick
    return tick


@pytest.fixture()
def lgr(bg):
    """Import worker.ledger after bg set up the isolated state dir."""
    from subconscious.worker import ledger
    return ledger


# ── ledger tests ─────────────────────────────────────────────────────────────

def test_goals_atomic_roundtrip(lgr):
    g = lgr.read_goals()
    assert g["goals"] == []
    goal = lgr.new_goal("hello world", kind="normal")
    g["goals"].append(goal)
    lgr.write_goals(g)

    g2 = lgr.read_goals()
    assert len(g2["goals"]) == 1
    assert g2["goals"][0]["title"] == "hello world"
    tmp_files = list(lgr.state_dir().glob("*.tmp"))
    assert tmp_files == []


def test_open_goals_count(lgr):
    goals = [
        {"status": "pending"},
        {"status": "active"},
        {"status": "done"},
        {"status": "blocked"},
        {"status": "cancelled"},
    ]
    assert len(lgr.open_goals(goals)) == 2


def test_find_goal(lgr):
    g = lgr.new_goal("x", kind="normal")
    goals = [g]
    assert lgr.find_goal(goals, g["id"]) is g
    assert lgr.find_goal(goals, "nope") is None


def test_budget_reset_on_new_day(lgr):
    b = lgr.read_budget()
    b["tokens_used"] = 999
    b["date"] = "1970-01-01"
    lgr.atomic_write_json(lgr.budget_path(), b)

    b2 = lgr.read_budget()
    assert b2["tokens_used"] == 0
    assert b2["date"] == lgr.today_utc()


def test_exploratory_cancel_stale(lgr, monkeypatch):
    from datetime import datetime, timezone, timedelta
    monkeypatch.setattr(lgr, "BG_EXPLORATORY_TTL_HOURS", 1)
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "000Z"
    goals = [
        {"id": "g-1", "kind": "exploratory", "status": "pending",
         "created_at": old, "history": []},
        {"id": "g-2", "kind": "normal", "status": "pending",
         "created_at": old, "history": []},
    ]
    changed = lgr.cancel_stale_exploratory(goals)
    assert changed == 1
    assert goals[0]["status"] == "cancelled"
    assert goals[1]["status"] == "pending"


def test_auto_resume_picks_existing_exploratory(lgr):
    goals = [
        {"id": "g-1", "kind": "exploratory", "status": "pending",
         "attempts": 1, "history": []},
        {"id": "g-2", "kind": "normal", "status": "pending",
         "attempts": 0, "history": []},
    ]
    active = lgr.active_exploratory(goals)
    assert len(active) == 1
    assert active[0]["id"] == "g-1"


# ── tick tests ───────────────────────────────────────────────────────────────

def test_extract_json_plain(bg):
    assert bg._extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced(bg):
    text = 'Blah blah\n```json\n{"decision": "skip", "reason": "x"}\n```\nDone.'
    assert bg._extract_json(text) == {"decision": "skip", "reason": "x"}


def test_extract_json_garbage(bg):
    assert bg._extract_json("no json here") is None
    assert bg._extract_json("{not valid}") is None


def test_exploratory_tool_calls_default_zero(bg):
    """The tool_calls key must exist in executor output."""
    src = inspect.getsource(bg._executor)
    assert '"tool_calls": tool_calls_made' in src
