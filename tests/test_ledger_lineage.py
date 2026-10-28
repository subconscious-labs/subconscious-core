"""Tests for goal lineage: parent_goal_id and descendant traversal."""
import sys

import pytest


@pytest.fixture()
def lgr(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBCONSCIOUS_BG_STATE_DIR", str(tmp_path / "bg"))
    for mod in list(sys.modules):
        if mod == "subconscious.runtime.config" \
                or mod.startswith("subconscious.worker"):
            del sys.modules[mod]
    from subconscious.worker import ledger
    return ledger


def test_new_goal_without_parent(lgr):
    g = lgr.new_goal("x", kind="normal")
    assert "parent_goal_id" not in g


def test_new_goal_with_parent(lgr):
    g = lgr.new_goal("x", kind="normal", parent_goal_id="g-abc")
    assert g["parent_goal_id"] == "g-abc"


def test_new_goal_parent_optional_empty_string(lgr):
    """Empty string is treated as no parent, not as a parent id."""
    g = lgr.new_goal("x", kind="normal", parent_goal_id="")
    assert "parent_goal_id" not in g


def test_find_descendants_linear(lgr):
    g1 = {"id": "g1"}
    g2 = {"id": "g2", "parent_goal_id": "g1"}
    g3 = {"id": "g3", "parent_goal_id": "g2"}
    result = lgr.find_descendants([g1, g2, g3], "g1")
    assert {g["id"] for g in result} == {"g2", "g3"}


def test_find_descendants_branching(lgr):
    g1 = {"id": "g1"}
    g2a = {"id": "g2a", "parent_goal_id": "g1"}
    g2b = {"id": "g2b", "parent_goal_id": "g1"}
    g3a = {"id": "g3a", "parent_goal_id": "g2a"}
    result = lgr.find_descendants([g1, g2a, g2b, g3a], "g1")
    assert {g["id"] for g in result} == {"g2a", "g2b", "g3a"}


def test_find_descendants_leaf(lgr):
    g1 = {"id": "g1"}
    assert lgr.find_descendants([g1], "g1") == []


def test_find_descendants_missing_root(lgr):
    assert lgr.find_descendants([{"id": "g1"}], "g-unknown") == []


def test_find_descendants_cycle_guard(lgr):
    """Corrupted ledger with a cycle must not hang or return the root."""
    g1 = {"id": "g1", "parent_goal_id": "g2"}
    g2 = {"id": "g2", "parent_goal_id": "g1"}
    result = lgr.find_descendants([g1, g2], "g1")
    ids = {g["id"] for g in result}
    assert "g1" not in ids
    assert "g2" in ids

