"""Unit tests for the non-streaming tool loop.

The loop depends on three sources that are patched per test:
  - subconscious.backends.get_backend -> FakeBackend
  - subconscious.chat.tools_io.call_tool -> FakeBackend tool dispatch
  - subconscious.tools.registry.list_tools -> controlled tool schema list

No real backend is initialized, no MCP client starts, no network.
"""
import pytest

from subconscious.chat import tools_loop
from subconscious.chat.tools_loop import (
    TOOL_ROUND_NUM_PREDICT_BOOST,
    run_tool_loop,
)


# ── fake backend ─────────────────────────────────────────────────────────────

class FakeBackend:
    """Returns pre-scripted responses in order. Records every payload it
    receives. Raises on overrun so a test cannot pass by silently
    getting fewer/more LLM calls than expected."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_sync(self, payload):
        self.calls.append(payload)
        if not self.responses:
            raise AssertionError(
                f"FakeBackend exhausted after {len(self.calls)} calls"
            )
        return self.responses.pop(0)


def _content_resp(text):
    return {"message": {"role": "assistant", "content": text}}


def _tool_resp(*tool_calls):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": list(tool_calls),
        }
    }


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def no_tools(monkeypatch):
    monkeypatch.setattr(
        "subconscious.tools.registry.list_tools", lambda: []
    )


@pytest.fixture
def one_tool(monkeypatch):
    """One tool registered; ensures tools key is actually present in
    payloads so the last-round strip can be asserted."""
    monkeypatch.setattr(
        "subconscious.tools.registry.list_tools",
        lambda: [{
            "type": "function",
            "function": {
                "name": "t",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )


@pytest.fixture
def install_backend(monkeypatch):
    def _install(responses):
        backend = FakeBackend(responses)
        monkeypatch.setattr(
            "subconscious.backends.get_backend", lambda m: backend
        )
        return backend
    return _install


@pytest.fixture
def install_call_tool(monkeypatch):
    def _install(result_map=None):
        calls = []
        result_map = result_map or {}

        def fake_call_tool(name, args):
            calls.append({"name": name, "args": args})
            if name in result_map:
                return result_map[name]
            return {"ok": True, "tool": name}

        monkeypatch.setattr(
            "subconscious.chat.tools_io.call_tool", fake_call_tool
        )
        return calls
    return _install


# Convenience header used across tests where layer_sizes is irrelevant.
_LAYERS = {"L0+L1": 100, "L1.5_kg": 0, "L1.5_rag": 0}


# ── happy paths ────────────────────────────────────────────────────────────

def test_direct_final_answer(install_backend, no_tools, install_call_tool):
    install_call_tool()
    backend = install_backend([_content_resp("hello")])

    out = run_tool_loop(
        "deepseek/deepseek-flash",
        [{"role": "user", "content": "hi"}],
        {"num_predict": 512},
        _LAYERS,
        "req1",
    )

    assert out == "hello"
    assert len(backend.calls) == 1


def test_single_tool_round_then_final(install_backend, no_tools, install_call_tool):
    calls = install_call_tool()
    backend = install_backend([
        _tool_resp({"id": "c1", "name": "meta_search", "arguments": {"q": "x"}}),
        _content_resp("done"),
    ])
    messages = [{"role": "user", "content": "hi"}]

    out = run_tool_loop("m", messages, {"num_predict": 512}, _LAYERS, "req2")

    assert out == "done"
    assert len(backend.calls) == 2
    assert calls == [{"name": "meta_search", "args": {"q": "x"}}]
    # The caller's messages list was mutated by the loop:
    # user -> assistant(tool_calls) -> tool.
    assert any(m.get("role") == "tool" for m in messages)


def test_multiple_tools_in_one_round(install_backend, no_tools, install_call_tool):
    calls = install_call_tool()
    backend = install_backend([
        _tool_resp(
            {"id": "c1", "name": "meta_search", "arguments": {"q": "a"}},
            {"id": "c2", "name": "meta_call",   "arguments": {"r": "b"}},
        ),
        _content_resp("ok"),
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req3")

    assert out == "ok"
    assert len(calls) == 2
    assert calls[0]["name"] == "meta_search"
    assert calls[1]["name"] == "meta_call"
    assert len(backend.calls) == 2


def test_tool_error_does_not_abort(install_backend, no_tools, install_call_tool):
    """A tool returning {"error": ...} is logged and fed back; loop continues."""
    calls = install_call_tool(
        result_map={"bad_tool": {"error": "nope", "tool": "bad_tool"}}
    )
    backend = install_backend([
        _tool_resp({"id": "c1", "name": "bad_tool", "arguments": {}}),
        _content_resp("recovered"),
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req4")

    assert out == "recovered"
    assert calls == [{"name": "bad_tool", "args": {}}]


# ── repeat detection ──────────────────────────────────────────────────────

def test_repeat_detection_aborts(install_backend, no_tools, install_call_tool):
    """Three identical consecutive tool_calls abort the loop before
    MAX_TOOL_ROUNDS. Dispatch count is 3; the fourth round breaks
    before invoking call_tool."""
    calls = install_call_tool()
    same_tc = {"id": "c1", "name": "meta_search", "arguments": {"q": "x"}}
    backend = install_backend([
        _tool_resp(same_tc),   # round 0: prev=None -> rep=0
        _tool_resp(same_tc),   # round 1: same -> rep=1
        _tool_resp(same_tc),   # round 2: same -> rep=2
        _tool_resp(same_tc),   # round 3: same -> rep=3 -> break
        _content_resp("unreachable"),
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req5")

    # Loop broke on a tool_calls response, so final content is empty.
    assert out == ""
    assert len(calls) == 3
    assert len(backend.calls) == 4


def test_repeat_resets_on_different_signature(
    install_backend, no_tools, install_call_tool
):
    """A/A/B/B/A pattern: rep is reset when the signature changes, so
    the loop does not abort prematurely."""
    calls = install_call_tool()
    a = {"id": "c1", "name": "tool_a", "arguments": {}}
    b = {"id": "c2", "name": "tool_b", "arguments": {}}
    backend = install_backend([
        _tool_resp(a),
        _tool_resp(a),   # rep=1
        _tool_resp(b),   # reset rep=0
        _tool_resp(b),   # rep=1
        _content_resp("done"),
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req6")

    assert out == "done"
    assert len(backend.calls) == 5
    assert len(calls) == 4


# ── last-round behaviour ──────────────────────────────────────────────────

def test_last_round_strips_tools_and_instructs(
    install_backend, one_tool, install_call_tool, monkeypatch
):
    """When i == MAX_TOOL_ROUNDS-1: tools key is popped from the payload
    and a final-answer instruction is appended to the messages."""
    monkeypatch.setattr(tools_loop, "MAX_TOOL_ROUNDS", 2)
    install_call_tool()
    tc = {"id": "c1", "name": "t", "arguments": {}}
    backend = install_backend([
        _tool_resp(tc),        # round 0: normal, tools present
        _content_resp("final"),  # round 1: last, tools stripped
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req7")

    assert out == "final"
    assert len(backend.calls) == 2
    # Round 0: tools key present (list_tools returned one entry).
    assert "tools" in backend.calls[0]
    # Round 1 (last): tools key popped.
    assert "tools" not in backend.calls[1]
    # Round 1's messages include the final-answer instruction.
    round1_msgs = backend.calls[1]["messages"]
    assert any(
        "tool-call limit" in (m.get("content") or "")
        for m in round1_msgs
    )


def test_boosted_num_predict_on_later_rounds(
    install_backend, no_tools, install_call_tool
):
    """Round 1+ payloads carry the boosted num_predict."""
    install_call_tool()
    backend = install_backend([
        _tool_resp({"id": "c1", "name": "t", "arguments": {}}),
        _content_resp("ok"),
    ])
    run_tool_loop(
        "m", [{"role": "user", "content": "hi"}],
        {"num_predict": 512}, {}, "req8",
    )
    assert backend.calls[0]["options"]["num_predict"] == 512
    assert (
        backend.calls[1]["options"]["num_predict"]
        == 512 + TOOL_ROUND_NUM_PREDICT_BOOST
    )


# ── empty-content retry ───────────────────────────────────────────────────

def test_empty_content_retry_with_final_num_predict(
    install_backend, no_tools, install_call_tool
):
    """Empty content with final_num_predict set triggers a boosted
    retry. The retry result is returned."""
    install_call_tool()
    backend = install_backend([
        _content_resp(""),          # triggers retry
        _content_resp("recovered"),  # retry content
    ])

    out = run_tool_loop(
        "m", [{"role": "user", "content": "hi"}],
        {"num_predict": 512}, {}, "req9",
        final_num_predict=2048,
    )

    assert out == "recovered"
    assert len(backend.calls) == 2
    assert backend.calls[1]["options"]["num_predict"] == 2048


def test_empty_content_no_retry_without_final_num_predict(
    install_backend, no_tools, install_call_tool
):
    """Without final_num_predict, empty content is not retried."""
    install_call_tool()
    backend = install_backend([_content_resp("")])

    out = run_tool_loop(
        "m", [{"role": "user", "content": "hi"}], {}, {}, "req10",
    )

    assert out == ""
    assert len(backend.calls) == 1


# ── content sanitisation ──────────────────────────────────────────────────

def test_dsml_stripped_from_final(install_backend, no_tools, install_call_tool):
    install_call_tool()
    dsml = 'text <｜｜DSML｜｜ invoke name="x">\nmore'
    install_backend([_content_resp(dsml)])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req11")

    assert "DSML" not in out
    assert "text" in out
    assert "more" in out


def test_return_value_is_stripped_string(
    install_backend, no_tools, install_call_tool
):
    install_call_tool()
    install_backend([_content_resp("  spaces  \n")])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req12")

    assert isinstance(out, str)
    assert out == "spaces"


# ── argument normalisation ────────────────────────────────────────────────

def test_string_arguments_parsed(install_backend, no_tools, install_call_tool):
    """Backend that serialises arguments as a JSON string is parsed
    before dispatch."""
    calls = install_call_tool()
    install_backend([
        _tool_resp({"id": "c1", "name": "t", "arguments": '{"k": "v"}'}),
        _content_resp("ok"),
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req13")

    assert out == "ok"
    assert calls == [{"name": "t", "args": {"k": "v"}}]


def test_malformed_string_arguments_become_empty(
    install_backend, no_tools, install_call_tool
):
    """Non-JSON string arguments fall back to {} rather than raising."""
    calls = install_call_tool()
    install_backend([
        _tool_resp({"id": "c1", "name": "t", "arguments": "not json"}),
        _content_resp("ok"),
    ])

    out = run_tool_loop("m", [{"role": "user", "content": "hi"}], {}, {}, "req14")

    assert out == "ok"
    assert calls == [{"name": "t", "args": {}}]
