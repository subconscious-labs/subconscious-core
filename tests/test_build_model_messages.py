"""Unit tests for build_model_messages.

The function sanitises history before every backend call:
- strips `<think>...</think>` and legacy [→ tool] markers from prior
  assistant turns
- drops reasoning (`thinking`) from history - DeepSeek rejects
  reasoning_content on input in some API versions, and re-feeding it
  bloats context
- keeps flat tool_calls on assistant turns
- drops messages with unknown roles
- trims history to max_history_turns pairs without leaving orphaned
  `tool` messages at the head
- routes multimodal content through _extract_text_from_content so
  array-shaped content from Open WebUI does not crash here

No mocks, no fixtures. Pure function.
"""
import pytest

from subconscious.chat.prompt import build_model_messages
from subconscious.runtime.config import MAX_HISTORY_TURNS


def test_simple_turn_preserved():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = build_model_messages(msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2] == {"role": "assistant", "content": "hello"}


def test_missing_system_is_not_invented():
    msgs = [{"role": "user", "content": "hi"}]
    out = build_model_messages(msgs)
    assert len(out) == 1
    assert out[0]["role"] == "user"


def test_think_block_stripped_from_assistant():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>reasoning</think>answer"},
    ]
    out = build_model_messages(msgs)
    assert out[1]["content"] == "answer"


def test_legacy_tool_arrow_marker_stripped():
    """Older sessions may contain the pre-normalisation [→ name] marker."""
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "[→ meta_search] result"},
    ]
    out = build_model_messages(msgs)
    assert "[→" not in out[1]["content"]
    assert "result" in out[1]["content"]


def test_dsml_stripped_from_assistant():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": 'text <｜｜DSML｜｜ invoke name="foo">\nmore',
        },
    ]
    out = build_model_messages(msgs)
    assert "DSML" not in out[1]["content"]


def test_thinking_dropped_from_history():
    """Assistant `thinking` must not survive into the next backend call."""
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "answer",
            "thinking": "internal reasoning",
        },
    ]
    out = build_model_messages(msgs)
    assert "thinking" not in out[1]


def test_tool_calls_preserved():
    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "meta_search", "arguments": {"q": "x"}}
            ],
        },
        {"role": "tool", "content": "result"},
    ]
    out = build_model_messages(msgs)
    assert out[1]["tool_calls"] == [
        {"id": "c1", "name": "meta_search", "arguments": {"q": "x"}}
    ]
    assert out[2]["role"] == "tool"


def test_unknown_roles_dropped():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "system_diagnostic", "content": "should be dropped"},
        {"role": "assistant", "content": "a"},
    ]
    out = build_model_messages(msgs)
    roles = [m["role"] for m in out]
    assert "system_diagnostic" not in roles
    assert roles == ["user", "assistant"]


def test_multimodal_content_flattened():
    """Array-shaped content from Open WebUI must be converted to text,
    not crash."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
        {"role": "assistant", "content": "a"},
    ]
    out = build_model_messages(msgs)
    assert out[0]["content"] == "describe this"


def test_history_trim_to_limit():
    """Excess history is cut from the head, keeping the last N pairs."""
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    out = build_model_messages(msgs, max_history_turns=5)
    assert len(out) == 10
    assert out[0]["content"] == "u25"
    assert out[-1]["content"] == "a29"


def test_trim_never_leaves_orphan_tool():
    """When the cut lands on a tool message, skip forward so the first
    surviving message is not a tool without its preceding assistant
    tool_calls. OpenAI-compatible backends reject that payload."""
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "x", "arguments": {}}],
        },
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
    ]
    # max_history_turns=1 -> 2 slots, cutoff at index 4 -> tool
    out = build_model_messages(msgs, max_history_turns=1)
    assert out[0]["role"] != "tool"


def test_empty_messages():
    assert build_model_messages([]) == []


def test_default_max_history_uses_config():
    """The default trims to MAX_HISTORY_TURNS pairs from config."""
    msgs = []
    for i in range(MAX_HISTORY_TURNS + 5):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    out = build_model_messages(msgs)
    assert len(out) == MAX_HISTORY_TURNS * 2
