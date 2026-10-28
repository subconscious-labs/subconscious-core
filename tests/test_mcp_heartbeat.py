"""Unit tests for MCP http idle handling.

Heartbeat and retry logic tested without a live MCP server. The
connection loop itself is covered by smoke (a real lazy-mcp process).

Two independent behaviours:

- _wait_for_reconnect polls until _session is set and _connection_ready
  is set, then clears _disconnect_event. Returns False on timeout.
- _call_tool_inner retries once after reconnect when the first call
  fails with the disconnect flag set. Does not retry when the flag is
  clear (real tool error).
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from subconscious.tools import mcp_client
from subconscious.tools.mcp_client import MCPClient


@pytest.fixture()
def client():
    """A minimal MCPClient with transport http, no real connection."""
    c = MCPClient(
        {"transport": "http", "url": "http://127.0.0.1:9999/mcp"},
        name="test-lazy",
    )
    # Test does not start the connection loop; we drive the state
    # manually.
    yield c
    # No real resources to release.


# ── _wait_for_reconnect ────────────────────────────────────────────────────

def test_wait_for_reconnect_immediate(client):
    client._session = MagicMock()
    client._connection_ready.set()
    client._disconnect_event.set()

    ok = client._wait_for_reconnect(timeout=1.0)

    assert ok is True
    assert client._disconnect_event.is_set() is False


def test_wait_for_reconnect_times_out(client):
    client._session = None
    client._connection_ready.clear()
    client._disconnect_event.set()

    start = time.time()
    ok = client._wait_for_reconnect(timeout=0.3)
    elapsed = time.time() - start

    assert ok is False
    assert 0.2 <= elapsed < 1.0
    assert client._disconnect_event.is_set() is True


def test_wait_for_reconnect_polls_until_ready(client):
    """Connection becomes ready mid-wait; method must return True."""
    client._session = None
    client._connection_ready.clear()
    client._disconnect_event.set()

    def become_ready():
        time.sleep(0.15)
        client._session = MagicMock()
        client._connection_ready.set()

    t = threading.Thread(target=become_ready, daemon=True)
    t.start()

    ok = client._wait_for_reconnect(timeout=1.0)
    assert ok is True
    assert client._disconnect_event.is_set() is False


# ── _call_tool_inner retry ─────────────────────────────────────────────────

def _fake_result(text="ok"):
    """Build a minimal result object shaped like MCP SDK's CallToolResult."""
    content = MagicMock()
    content.text = text
    result = MagicMock()
    result.content = [content]
    return result


def test_call_tool_retries_after_disconnect(client, monkeypatch):
    """First call raises with disconnect flag set, second succeeds."""
    client._session = MagicMock()
    client._connection_ready.set()

    calls = {"count": 0}

    def fake_run(coro, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            # Simulate _run behaviour: set disconnect, raise.
            client._disconnect_event.set()
            raise RuntimeError("Session terminated")
        return _fake_result("recovered")

    def fake_reconnect(timeout):
        # Connection comes back immediately; clear flag as real
        # _wait_for_reconnect would after finding a live session.
        client._disconnect_event.clear()
        return True

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(client, "_wait_for_reconnect", fake_reconnect)

    result = client._call_tool_inner("meta_search", {"query": "x"})

    assert result == "recovered"
    assert calls["count"] == 2


def test_call_tool_no_retry_when_flag_clear(client, monkeypatch):
    """Error without disconnect flag = real tool error, no retry."""
    client._session = MagicMock()
    client._connection_ready.set()

    calls = {"count": 0}

    def fake_run(coro, timeout):
        calls["count"] += 1
        raise RuntimeError("drawer not found")

    monkeypatch.setattr(client, "_run", fake_run)

    with pytest.raises(RuntimeError, match="drawer not found"):
        client._call_tool_inner("mempalace_get_drawer", {"drawer_id": "x"})

    assert calls["count"] == 1


def test_call_tool_propagates_second_failure(client, monkeypatch):
    """If retry also fails, propagate the second error."""
    client._session = MagicMock()
    client._connection_ready.set()

    calls = {"count": 0}

    def fake_run(coro, timeout):
        calls["count"] += 1
        client._disconnect_event.set()
        raise RuntimeError(f"fail #{calls['count']}")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(
        client, "_wait_for_reconnect", lambda timeout: True,
    )

    with pytest.raises(RuntimeError, match="fail #2"):
        client._call_tool_inner("x", {})

    assert calls["count"] == 2


def test_call_tool_propagates_when_reconnect_fails(client, monkeypatch):
    """If reconnect never becomes available, propagate first error."""
    client._session = MagicMock()
    client._connection_ready.set()

    calls = {"count": 0}

    def fake_run(coro, timeout):
        calls["count"] += 1
        client._disconnect_event.set()
        raise RuntimeError("Session terminated")

    monkeypatch.setattr(client, "_run", fake_run)
    monkeypatch.setattr(
        client, "_wait_for_reconnect", lambda timeout: False,
    )

    with pytest.raises(RuntimeError, match="Session terminated"):
        client._call_tool_inner("x", {})

    assert calls["count"] == 1
