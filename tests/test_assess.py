"""Unit tests for the assessor phase (D2 scaffold).

In D2 the assessor records a verdict in goal.history and does not
change the goal's status. These tests cover the two building blocks
(_assessor and _record_assess_verdict) in isolation. Integration
through _tick_impl is covered by smoke, not here.

_llm_call is mocked per test. The mock returns the SAME normalised
shape that the real _llm_call returns:

    {"content": str, "thinking": str, "tool_calls": list,
     "tokens": int, "raw": dict}

Do not return raw backend responses ({"message": {...}}) here; that
shape belongs to the backend layer, and _llm_call has already
translated it by the time _assessor sees the result.

ledger.resolve_model is stubbed to bypass budget logic; budget is
tested separately in test_background.
"""
import sys

import pytest


@pytest.fixture()
def tick_mod(tmp_path, monkeypatch):
    """Import worker.tick fresh with a stubbed LLM primitive."""
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

    responses = []

    def fake_llm(messages, model, tools=None, num_predict=None,
                 response_format=None):
        if not responses:
            raise AssertionError("fake_llm exhausted")
        return responses.pop(0)

    monkeypatch.setattr(tick, "_llm_call", fake_llm)
    monkeypatch.setattr(
        ledger, "resolve_model", lambda m, phase: (m, None)
    )

    return tick, responses


def _normalised(content="", thinking="", tokens=0):
    """Shape returned by _llm_call, which is what _assessor consumes."""
    return {
        "content": content,
        "thinking": thinking,
        "tool_calls": [],
        "tokens": tokens,
        "raw": {},
    }


# ── _assessor ───────────────────────────────────────────────────────────────

def test_assessor_verified(tick_mod):
    tick, responses = tick_mod
    responses.append(_normalised(
        content=(
            '{"verdict": "verified", "reason": "checked", '
            '"evidence": "up=7 series"}'
        ),
        tokens=100,
    ))
    result = tick._assessor({
        "id": "g1", "title": "check", "last_result": "verified up=7",
    })
    assert result["verdict"] == "verified"
    assert result["tokens"] == 100
    assert "up=7" in result["evidence"]


def test_assessor_regressed(tick_mod):
    tick, responses = tick_mod
    responses.append(_normalised(
        content='{"verdict": "regressed", "reason": "vague claim"}',
        tokens=50,
    ))
    result = tick._assessor({"id": "g1", "title": "t"})
    assert result["verdict"] == "regressed"


def test_assessor_invalid_verdict_coerced_to_unclear(tick_mod):
    tick, responses = tick_mod
    responses.append(_normalised(
        content='{"verdict": "banana", "reason": "x"}',
        tokens=10,
    ))
    result = tick._assessor({"id": "g1", "title": "t"})
    assert result["verdict"] == "unclear"


def test_assessor_llm_error_returns_unclear(tick_mod, monkeypatch):
    tick, _ = tick_mod

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(tick, "_llm_call", boom)
    result = tick._assessor({"id": "g1", "title": "t"})
    assert result["verdict"] == "unclear"
    assert "network down" in result["reason"]


def test_assessor_empty_content_falls_back_to_thinking(tick_mod):
    """Some models emit the JSON in `thinking` when content is empty."""
    tick, responses = tick_mod
    responses.append(_normalised(
        content="",
        thinking='{"verdict": "verified", "reason": "ok"}',
        tokens=20,
    ))
    result = tick._assessor({"id": "g1", "title": "t"})
    assert result["verdict"] == "verified"


def test_assessor_unparseable_content_returns_unclear(tick_mod):
    """No JSON in content or thinking -> unclear, not an exception."""
    tick, responses = tick_mod
    responses.append(_normalised(
        content="this is not json",
        tokens=5,
    ))
    result = tick._assessor({"id": "g1", "title": "t"})
    assert result["verdict"] == "unclear"


# ── _record_assess_verdict ─────────────────────────────────────────────────

def test_record_sets_fields(tick_mod):
    tick, _ = tick_mod
    g = {
        "id": "g1", "title": "x", "status": "done",
        "history": [{"ts": "t0", "event": "created"}],
    }
    tick._record_assess_verdict(g, {
        "verdict": "verified",
        "reason": "specific facts cited",
        "evidence": "up=7",
    })
    assert g["assessed"] is True
    assert g["assess_verdict"] == "verified"
    assert g["assessed_at"]
    events = [h.get("event") for h in g["history"]]
    assert "assessed=verified" in events


def test_record_does_not_change_status(tick_mod):
    """D2 policy: verdict is recorded, status is NOT touched."""
    tick, _ = tick_mod
    g = {"id": "g1", "title": "x", "status": "done", "history": []}
    tick._record_assess_verdict(g, {
        "verdict": "regressed", "reason": "vague",
    })
    assert g["status"] == "done"


def test_record_truncates_long_reason(tick_mod):
    tick, _ = tick_mod
    g = {"id": "g1", "title": "x", "history": []}
    long_reason = "x" * 1000
    tick._record_assess_verdict(g, {
        "verdict": "unclear", "reason": long_reason,
    })
    history_note = g["history"][-1]["note"]
    assert len(history_note) == 500


def test_record_preserves_existing_history(tick_mod):
    tick, _ = tick_mod
    prior = {"ts": "t0", "event": "created", "by": "system"}
    g = {"id": "g1", "title": "x", "history": [prior]}
    tick._record_assess_verdict(g, {
        "verdict": "verified", "reason": "ok",
    })
    assert g["history"][0] == prior
    assert len(g["history"]) == 2




# ── system block integration (item 7) ────────────────────────────────────

def test_assessor_uses_system_block(tick_mod, monkeypatch):
    """Item 7: _assessor must consult _system_block("assessor") so
    BG_WAKEUP_MODE_ASSESSOR works symmetrically with other phases."""
    tick, _ = tick_mod
    captured = {"messages": None}

    def fake_llm(messages, model, tools=None, num_predict=None,
                 response_format=None):
        captured["messages"] = messages
        return _normalised(
            content='{"verdict": "verified", "reason": "ok"}',
        )

    monkeypatch.setattr(tick, "_llm_call", fake_llm)
    monkeypatch.setattr(
        tick, "_system_block", lambda phase: f"WAKEUP:{phase}"
    )

    tick._assessor({"id": "g1", "title": "t"})

    # Wake-up is first.
    assert captured["messages"][0] == {
        "role": "system", "content": "WAKEUP:assessor"
    }
    # Instructions still present as a second system message.
    assert any(
        m["role"] == "system" and "assessor" in m["content"].lower()
        for m in captured["messages"][1:]
    )
    # User text last.
    assert captured["messages"][-1]["role"] == "user"


def test_assessor_no_system_block_when_disabled(tick_mod, monkeypatch):
    """With BG_WAKEUP_MODE=none, _system_block returns "" and the
    assessor instructions become the first system message (old
    behaviour)."""
    tick, _ = tick_mod
    captured = {"messages": None}

    def fake_llm(messages, model, tools=None, num_predict=None,
                 response_format=None):
        captured["messages"] = messages
        return _normalised(content='{"verdict": "verified"}')

    monkeypatch.setattr(tick, "_llm_call", fake_llm)
    monkeypatch.setattr(tick, "_system_block", lambda phase: "")

    tick._assessor({"id": "g1", "title": "t"})

    assert captured["messages"][0]["role"] == "system"
    assert "WAKEUP" not in captured["messages"][0]["content"]
