"""Tests for _apply_assess_verdict and subcortex_goals_resolve (D3a).

_apply_assess_verdict transitions the goal's status based on the
assessor's verdict. _record_assess_verdict (D2, tested elsewhere) sets
the assess metadata; these tests assume that has already run.

subcortex_goals_resolve is the operator-facing action for needs-review
goals. Only goals in that status can be resolved; anything else is
rejected so the operator notices a wrong id.

open_goals and write_goals invariants: needs-review is neither opened
by the planner nor archived.
"""
import sys

import pytest


@pytest.fixture()
def tick_mod(tmp_path, monkeypatch):
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

    from subconscious.worker import tick, ledger
    return tick, ledger


@pytest.fixture()
def internal_mod(tick_mod):
    tick, ledger = tick_mod
    from subconscious.tools import internal
    return tick, ledger, internal


# ── _apply_assess_verdict ───────────────────────────────────────────────────

def test_apply_verified_keeps_done(tick_mod):
    tick, _ = tick_mod
    g = {"id": "g1", "title": "t", "status": "done", "history": []}
    tick._apply_assess_verdict(g, {
        "verdict": "verified", "reason": "ok",
    })
    assert g["status"] == "done"
    assert "regressed_count" not in g


def test_apply_regressed_below_limit_returns_pending(tick_mod, monkeypatch):
    tick, _ = tick_mod
    monkeypatch.setattr(tick, "BG_MAX_REGRESSED_PER_GOAL", 2)
    g = {
        "id": "g1", "title": "t", "status": "done", "history": [],
        "attempts": 1, "partials": 2,
    }
    tick._apply_assess_verdict(g, {
        "verdict": "regressed", "reason": "vague",
    })
    assert g["status"] == "pending"
    assert g["attempts"] == 0
    assert g["partials"] == 0
    assert g["regressed_count"] == 1
    events = [h.get("event") for h in g["history"]]
    assert "regressed_pending" in events


def test_apply_regressed_at_limit_needs_review(tick_mod, monkeypatch):
    tick, _ = tick_mod
    monkeypatch.setattr(tick, "BG_MAX_REGRESSED_PER_GOAL", 2)
    g = {
        "id": "g1", "title": "t", "status": "done", "history": [],
        "regressed_count": 1,
    }
    tick._apply_assess_verdict(g, {
        "verdict": "regressed", "reason": "vague",
    })
    assert g["status"] == "needs-review"
    assert g["regressed_count"] == 2
    events = [h.get("event") for h in g["history"]]
    assert "needs-review" in events


def test_apply_unclear_needs_review(tick_mod):
    tick, _ = tick_mod
    g = {"id": "g1", "title": "t", "status": "done", "history": []}
    tick._apply_assess_verdict(g, {
        "verdict": "unclear", "reason": "cannot verify from here",
    })
    assert g["status"] == "needs-review"
    assert g["history"][-1]["event"] == "needs-review"


# ── subcortex_goals_resolve ────────────────────────────────────────────────

def _seed(internal, ledger, goal):
    goals = {
        "version": ledger.GOALS_VERSION,
        "goals": [goal],
        "updated_at": ledger.now_iso(),
    }
    ledger.atomic_write_json(ledger.goals_path(), goals)


def test_resolve_verified(internal_mod):
    tick, ledger, internal = internal_mod
    _seed(internal, ledger, {
        "id": "g1", "kind": "normal", "status": "needs-review",
        "title": "t", "attempts": 0, "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(), "history": [],
    })
    result = internal.call_internal(
        "subcortex_goals_resolve",
        {"goal_id": "g1", "action": "verified", "reason": "checked"},
    )
    assert result["resolved"] is True
    reloaded = ledger.read_goals()
    assert reloaded["goals"] == []  # done -> archived


def test_resolve_cancelled(internal_mod):
    tick, ledger, internal = internal_mod
    _seed(internal, ledger, {
        "id": "g1", "kind": "normal", "status": "needs-review",
        "title": "t", "attempts": 0, "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(), "history": [],
    })
    result = internal.call_internal(
        "subcortex_goals_resolve",
        {"goal_id": "g1", "action": "cancelled"},
    )
    assert result["resolved"] is True
    reloaded = ledger.read_goals()
    assert reloaded["goals"] == []


def test_resolve_retry_resets_counters(internal_mod):
    tick, ledger, internal = internal_mod
    _seed(internal, ledger, {
        "id": "g1", "kind": "normal", "status": "needs-review",
        "title": "t", "attempts": 2, "partials": 1, "regressed_count": 2,
        "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(), "history": [],
    })
    result = internal.call_internal(
        "subcortex_goals_resolve",
        {"goal_id": "g1", "action": "retry"},
    )
    assert result["resolved"] is True
    reloaded = ledger.read_goals()
    assert len(reloaded["goals"]) == 1
    g = reloaded["goals"][0]
    assert g["status"] == "pending"
    assert g["attempts"] == 0
    assert g["partials"] == 0
    assert g["regressed_count"] == 0


def test_resolve_wrong_status_rejected(internal_mod):
    tick, ledger, internal = internal_mod
    _seed(internal, ledger, {
        "id": "g1", "kind": "normal", "status": "pending",
        "title": "t", "attempts": 0, "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(), "history": [],
    })
    result = internal.call_internal(
        "subcortex_goals_resolve",
        {"goal_id": "g1", "action": "verified"},
    )
    assert "error" in result
    assert "needs-review" in result["error"]


def test_resolve_unknown_goal(internal_mod):
    tick, ledger, internal = internal_mod
    result = internal.call_internal(
        "subcortex_goals_resolve",
        {"goal_id": "g-unknown", "action": "verified"},
    )
    assert "error" in result


# ── ledger invariants ─────────────────────────────────────────────────────

def test_needs_review_not_in_open_goals(tick_mod):
    tick, ledger = tick_mod
    goals = [
        {"id": "g1", "status": "needs-review"},
        {"id": "g2", "status": "pending"},
        {"id": "g3", "status": "active"},
    ]
    open_ids = [g["id"] for g in ledger.open_goals(goals)]
    assert open_ids == ["g2", "g3"]


def test_needs_review_not_archived_on_write(tick_mod):
    tick, ledger = tick_mod
    g = {
        "id": "g1", "kind": "normal", "status": "needs-review",
        "title": "t", "attempts": 0,
        "created_at": ledger.now_iso(), "updated_at": ledger.now_iso(),
        "history": [],
    }
    goals = {
        "version": ledger.GOALS_VERSION,
        "goals": [g],
        "updated_at": ledger.now_iso(),
    }
    ledger.write_goals(goals)
    # needs-review stays in goals.json, not archive
    reloaded = ledger.read_goals()
    assert len(reloaded["goals"]) == 1
    assert reloaded["goals"][0]["status"] == "needs-review"


# ── read_recent_regressed + planner context ───────────────────────────────

def test_read_recent_regressed_from_archive(tick_mod):
    tick, ledger = tick_mod
    regressed = {
        "id": "g-old", "kind": "normal", "status": "done",
        "title": "Investigate flaky service X",
        "assess_verdict": "regressed",
        "last_result": "claimed done but was vague",
        "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(),
        "history": [],
    }
    ledger.append_archive(regressed)
    # Non-regressed entry must be ignored.
    ledger.append_archive({
        "id": "g-ok", "kind": "normal", "status": "done",
        "title": "Fine goal", "assess_verdict": "verified",
        "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(),
        "history": [],
    })

    results = ledger.read_recent_regressed(limit=3)
    assert len(results) == 1
    assert results[0]["id"] == "g-old"


def test_read_recent_regressed_from_active(tick_mod):
    tick, ledger = tick_mod
    active = {
        "id": "g-pending", "kind": "normal", "status": "pending",
        "title": "In-flight goal",
        "assess_verdict": "regressed",
        "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(),
        "history": [],
    }
    goals = {
        "version": ledger.GOALS_VERSION,
        "goals": [active],
        "updated_at": ledger.now_iso(),
    }
    ledger.atomic_write_json(ledger.goals_path(), goals)

    results = ledger.read_recent_regressed(limit=3)
    assert len(results) == 1
    assert results[0]["id"] == "g-pending"


def test_planner_includes_recent_regressed(tick_mod, monkeypatch):
    """Planner's user_text mentions regressed goal titles so the model
    does not re-propose them."""
    tick, ledger = tick_mod
    ledger.append_archive({
        "id": "g-old", "kind": "normal", "status": "done",
        "title": "Investigate flaky service X",
        "assess_verdict": "regressed",
        "last_result": "claimed done, was vague",
        "created_at": ledger.now_iso(),
        "updated_at": ledger.now_iso(),
        "history": [],
    })

    captured: dict = {}

    def fake_llm(messages, model, tools=None, num_predict=None,
                 response_format=None):
        captured["messages"] = messages
        return {
            "content": '{"decision": "skip", "reason": "test"}',
            "thinking": "",
            "tool_calls": [],
            "tokens": 0,
            "raw": {},
        }

    monkeypatch.setattr(tick, "_llm_call", fake_llm)
    monkeypatch.setattr(
        ledger, "resolve_model", lambda m, phase: (m, None)
    )

    tick._planner()

    user_msg = next(
        m["content"] for m in captured["messages"] if m["role"] == "user"
    )
    assert "Investigate flaky service X" in user_msg
    assert "do NOT re-propose" in user_msg
