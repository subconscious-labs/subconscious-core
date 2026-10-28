"""Tests for late-connection MCP handling.

Scenario: subconscious starts before the backend is ready. The
reconnect loop must keep running in the background, and mcp_server_restart
must be able to register the tools once the backend comes up.
"""
import sys
import threading
from unittest.mock import MagicMock

import pytest

from subconscious.tools import mcp_client


@pytest.fixture()
def client(monkeypatch):
    # Very short initial wait so tests do not sleep.
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_INITIAL_BACKOFF_SEC", 0.05)
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_MAX_BACKOFF_SEC", 0.1)
    c = mcp_client.MCPClient(
        {"transport": "http", "url": "http://127.0.0.1:9999/mcp"},
        name="test-late",
    )
    yield c
    c._shutdown_event.set()
    c._disconnect_event.set()
    if c._thread:
        c._thread.join(timeout=2)


def test_start_returns_false_loop_continues(client, monkeypatch):
    """start() must return False without killing the loop."""
    attempts = {"n": 0}

    async def failing_run_session():
        attempts["n"] += 1
        raise ConnectionError("down")

    monkeypatch.setattr(client, "_run_session", failing_run_session)

    ready = client.start(wait_timeout=0.2)
    assert ready is False

    # Loop must be alive and trying.
    import time
    time.sleep(0.5)
    assert attempts["n"] >= 2, (
        f"expected 2+ attempts after start() returned False, "
        f"got {attempts['n']}"
    )
    assert client._thread.is_alive()
    assert not client._shutdown_event.is_set()


def test_start_returns_true_when_ready(client, monkeypatch):
    """When the initial connect succeeds, start() returns True."""
    async def ok_run_session():
        client._connection_ready.set()
        while not client._shutdown_event.is_set():
            import asyncio
            await asyncio.sleep(0.1)

    monkeypatch.setattr(client, "_run_session", ok_run_session)

    ready = client.start(wait_timeout=2.0)
    assert ready is True
    assert client._connection_ready.is_set()


def test_restart_returns_false_when_down(client, monkeypatch):
    async def failing_run_session():
        raise ConnectionError("down")

    monkeypatch.setattr(client, "_run_session", failing_run_session)

    ok = client.restart(wait_timeout=0.2)
    assert ok is False
    assert client._thread.is_alive() or not client._shutdown_event.is_set()


def test_refresh_server_replaces_tools():
    """refresh_server drops old tools for the named server and adds new."""
    for mod in list(sys.modules):
        if mod == "subconscious.tools.registry":
            del sys.modules[mod]
    from subconscious.tools import registry

    fake_client = MagicMock()
    fake_client.name = "lazy"

    fake_other = MagicMock()
    fake_other.name = "other"

    registry.TOOLS = [
        {"type": "function", "function": {"name": "meta_search"}},
        {"type": "function", "function": {"name": "other_tool"}},
    ]
    registry.REGISTRY = {
        "meta_search": fake_client,
        "other_tool": fake_other,
    }

    new_tools = [
        {"type": "function", "function": {"name": "meta_list_servers"}},
        {"type": "function", "function": {"name": "meta_call"}},
    ]
    new_reg = {"meta_list_servers": fake_client, "meta_call": fake_client}

    registry.refresh_server("lazy", new_tools, new_reg)

    tool_names = [t["function"]["name"] for t in registry.TOOLS]
    assert "meta_search" not in tool_names
    assert "other_tool" in tool_names  # other server untouched
    assert "meta_list_servers" in tool_names
    assert "meta_call" in tool_names

    assert "meta_search" not in registry.REGISTRY
    assert "other_tool" in registry.REGISTRY
    assert registry.REGISTRY["meta_call"] is fake_client


def test_mcp_server_restart_refreshes_tools(client, monkeypatch):
    """_mcp_server_restart refreshes tool schema after reconnect."""
    # Reset registry state.
    for mod in list(sys.modules):
        if mod in ("subconscious.tools.registry", "subconscious.tools.internal"):
            del sys.modules[mod]
    from subconscious.tools import registry, internal

    # Fake a client that is always in the _MCP_CLIENTS list.
    fake_result = MagicMock()
    content = MagicMock()
    content.text = '{"ok": true}'
    fake_result.content = [content]

    def make_tool(name):
        t = MagicMock()
        t.name = name
        t.description = "desc"
        t.inputSchema = {"type": "object", "properties": {}}
        return t

    client.restart = lambda force=False, wait_timeout=30.0: True
    client.list_tools = lambda: [make_tool("meta_search"), make_tool("meta_list_servers")]

    registry.reset([], {}, [client])

    # Patch _find_mcp_client in internal to return our client.
    monkeypatch.setattr(internal, "_find_mcp_client", lambda n: client if n == "test-late" else None)

    result = internal.call_internal(
        "mcp_server_restart", {"name": "test-late"},
    )

    assert result["status"] == "restarted"
    assert result["tools_count"] == 2
    assert set(result["tools"]) == {"meta_search", "meta_list_servers"}

    # Registry must now have these tools.
    tool_names = [t["function"]["name"] for t in registry.TOOLS]
    assert "meta_search" in tool_names
    assert "meta_list_servers" in tool_names
