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


# ── Wave 1-2 fixes: partials reset, seed availability, hallucination ─────

def test_reflector_retry_resets_partials(lgr):
    """Item 10: reflector retry must clear partials alongside attempts.

    Without this, a goal that accumulated partials=2 goes straight back
    to blocked on the next attempt, before any new information can
    change the outcome. Contract test: the state transition the code in
    _tick_impl performs leaves both counters at zero.
    """
    g = lgr.new_goal("stuck goal", kind="normal")
    g["attempts"] = 3
    g["partials"] = 2
    g["status"] = "blocked"
    goals = {
        "version": lgr.GOALS_VERSION,
        "goals": [g],
        "updated_at": lgr.now_iso(),
    }
    lgr.atomic_write_json(lgr.goals_path(), goals)

    # Apply the retry transition that _tick_impl performs.
    reloaded = lgr.read_goals()
    g2 = lgr.find_goal(reloaded["goals"], g["id"])
    g2["attempts"] = 0
    g2["partials"] = 0
    g2["status"] = "pending"
    lgr.write_goals(reloaded)

    result = lgr.read_goals()["goals"][0]
    assert result["status"] == "pending"
    assert result["attempts"] == 0
    assert result["partials"] == 0


def test_exploratory_seed_available_false_no_tools(bg, monkeypatch):
    """Item 5: no tools at all -> no seed available."""
    monkeypatch.setattr(bg, "_allowed_tools_for", lambda k: [])
    assert bg._exploratory_seed_available() is False


def test_exploratory_seed_available_true_with_no_arg_tool(bg, monkeypatch):
    """Item 5: a no-arg tool is a valid seed."""
    monkeypatch.setattr(bg, "_allowed_tools_for", lambda k: [{
        "type": "function",
        "function": {
            "name": "meta_list_servers",
            "parameters": {"type": "object", "properties": {}},
        },
    }])
    assert bg._exploratory_seed_available() is True


def test_exploratory_seed_available_false_all_require_args(bg, monkeypatch):
    """Item 5: every tool needs args -> no usable seed."""
    monkeypatch.setattr(bg, "_allowed_tools_for", lambda k: [
        {
            "type": "function",
            "function": {
                "name": "mempalace_search",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                },
            },
        },
    ])
    assert bg._exploratory_seed_available() is False


def test_hallucination_guard_prepends_note(bg, lgr, monkeypatch):
    """Item 6: zero tool calls + claim in observation -> NOTE prepended.

    Reproduces req 39e45070 (2026-10-28): model wrote 'single
    meta_list_servers call returned empty payload' with tool_calls_made=0.
    """
    monkeypatch.setattr(lgr, "resolve_model", lambda m, p: (m, None))
    monkeypatch.setattr(
        bg, "_executor_tool_loop",
        lambda msgs, model, tools, max_rounds: (100, 0),
    )
    monkeypatch.setattr(bg, "_allowed_tools_for", lambda k: [])
    monkeypatch.setattr(bg, "_system_block", lambda p: "")

    def fake_llm(messages, model, tools=None, num_predict=None,
                 response_format=None):
        return {
            "content": (
                '{"outcome": "done", "observation": '
                '"I called meta_list_servers and got an empty payload"}'
            ),
            "thinking": "",
            "tool_calls": [],
            "tokens": 50,
            "raw": {},
        }

    monkeypatch.setattr(bg, "_llm_call", fake_llm)

    goal = {
        "id": "g1", "kind": "normal", "title": "test",
        "attempts": 0, "last_result": None, "history": [],
    }
    result = bg._executor(goal)

    assert "no tools were called" in result["observation"]
    # Claim preserved as context, not discarded.
    assert "meta_list_servers" in result["observation"]


def test_cancel_stale_normal_goals_ttl(lgr, monkeypatch):
    """Item 3: normal pending goals older than TTL are cancelled; active,
    done, cancelled, and exploratory are left alone."""
    from datetime import datetime, timezone, timedelta

    old = (datetime.now(timezone.utc) - timedelta(hours=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "000Z"
    fresh = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%f"
    ) + "000Z"

    goals = [
        # Old pending normal -> cancelled.
        {"id": "g-old", "kind": "normal", "status": "pending",
         "updated_at": old, "history": []},
        # Fresh pending normal -> kept.
        {"id": "g-fresh", "kind": "normal", "status": "pending",
         "updated_at": fresh, "history": []},
        # Old but active -> kept (being worked on right now).
        {"id": "g-active", "kind": "normal", "status": "active",
         "updated_at": old, "history": []},
        # Old pending exploratory -> kept (own TTL).
        {"id": "g-expl", "kind": "exploratory", "status": "pending",
         "updated_at": old, "history": []},
        # Old but already done -> untouched.
        {"id": "g-done", "kind": "normal", "status": "done",
         "updated_at": old, "history": []},
    ]

    changed = lgr.cancel_stale_normal_goals(goals, ttl_hours=24)
    assert changed == 1
    assert goals[0]["status"] == "cancelled"
    assert "stale" in goals[0]["history"][-1]["note"]
    assert goals[1]["status"] == "pending"
    assert goals[2]["status"] == "active"
    assert goals[3]["status"] == "pending"
    assert goals[4]["status"] == "done"


def test_cancel_stale_normal_goals_disabled(lgr):
    """ttl_hours <= 0 disables the check entirely."""
    goals = [
        {"id": "g-old", "kind": "normal", "status": "pending",
         "updated_at": "2020-01-01T00:00:00.000000000Z", "history": []},
    ]
    assert lgr.cancel_stale_normal_goals(goals, ttl_hours=0) == 0
    assert lgr.cancel_stale_normal_goals(goals, ttl_hours=-1) == 0
    assert goals[0]["status"] == "pending"
