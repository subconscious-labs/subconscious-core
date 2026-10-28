"""Minimal unit tests for worker.tick and worker.ledger.

Not covered here: real LLM calls, tool execution, journal rotation.
Those are exercised via the /api/tick smoke test.
"""
import inspect
import sys

import pytest


@pytest.fixture()
def bg(tmp_path, monkeypatch):
    """Import worker.tick with isolated state dir.

    Clears subconscious.worker and subconscious.runtime.config so each
    test gets a fresh ledger bound to a fresh BG_STATE_DIR.

    subconscious.tools.internal is cleared too, because it holds
    `from subconscious.worker import ledger` at module level - the
    captured reference would otherwise stay bound to a previous test's
    directory. Only that one submodule is targeted, not the whole
    subconscious.tools package: chat.prompt and chat.stream hold
    module-level references to subconscious.tools.registry, and
    reloading registry would leave those references dangling (a
    monkeypatched list_tools in another test would land on the fresh
    module while callers kept reading the old one).

    Clearing sys.modules alone is not enough - the submodule attribute
    stays on the parent package, and `from subconscious.tools import
    internal` prefers the attribute over sys.modules. delattr forces a
    real re-import.
    """
    monkeypatch.setenv("SUBCONSCIOUS_BG_STATE_DIR", str(tmp_path / "bg"))
    for mod in list(sys.modules):
        if (
            mod == "subconscious.runtime.config"
            or mod.startswith("subconscious.worker")
            or mod == "subconscious.tools.internal"
        ):
            del sys.modules[mod]
    import subconscious.tools
    if hasattr(subconscious.tools, "internal"):
        delattr(subconscious.tools, "internal")
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


def test_create_goal_via_internal(lgr, tmp_path):
    from subconscious.tools import internal
    result = internal.call_internal(
        "subcortex_goals_create",
        {"title": "test goal"},
    )
    assert "created" in result
    assert result["created"]["title"] == "test goal"


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


def test_cancel_cascade(lgr, bg):
    from subconscious.tools import internal

    g1 = lgr.new_goal("parent", kind="normal")
    g2 = lgr.new_goal("child1", kind="normal", parent_goal_id=g1["id"])
    g3 = lgr.new_goal("child2", kind="normal", parent_goal_id=g1["id"])
    goals = {
        "version": lgr.GOALS_VERSION,
        "goals": [g1, g2, g3],
        "updated_at": lgr.now_iso(),
    }
    lgr.atomic_write_json(lgr.goals_path(), goals)

    result = internal.call_internal(
        "subcortex_goals_cancel",
        {"goal_id": g1["id"], "reason": "test", "cascade": True},
    )
    assert result["cancelled_count"] == 3

    reloaded = lgr.read_goals()
    assert reloaded["goals"] == []


def test_cancel_no_cascade(lgr, bg):
    from subconscious.tools import internal

    g1 = lgr.new_goal("parent", kind="normal")
    g2 = lgr.new_goal("child1", kind="normal", parent_goal_id=g1["id"])
    goals = {
        "version": lgr.GOALS_VERSION,
        "goals": [g1, g2],
        "updated_at": lgr.now_iso(),
    }
    lgr.atomic_write_json(lgr.goals_path(), goals)

    result = internal.call_internal(
        "subcortex_goals_cancel",
        {"goal_id": g1["id"], "reason": "test", "cascade": False},
    )
    assert result["cancelled_count"] == 1

    reloaded = lgr.read_goals()
    assert len(reloaded["goals"]) == 1
    assert reloaded["goals"][0]["id"] == g2["id"]
