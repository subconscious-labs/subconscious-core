"""Tests for the reconnect backoff and loop robustness.

The reconnect loop runs in a real thread with a real event loop; tests
exercise it by mocking _run_session to raise, then observing the loop
through the log output and the attempt counter. Timing-sensitive but
bounded by very short backoff intervals (100ms initial, 400ms cap).
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from subconscious.tools import mcp_client
from subconscious.tools.mcp_client import MCPClient


@pytest.fixture()
def client(monkeypatch):
    """MCPClient http, reconnect timings accelerated for tests."""
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_INITIAL_BACKOFF_SEC", 0.1)
    monkeypatch.setattr(mcp_client, "MCP_RECONNECT_MAX_BACKOFF_SEC", 0.4)
    c = MCPClient(
        {"transport": "http", "url": "http://127.0.0.1:9999/mcp"},
        name="test-reconnect",
    )
    yield c
    c._shutdown_event.set()
    c._disconnect_event.set()


def test_reconnect_loop_survives_many_failures(client, monkeypatch, caplog):
    """Loop must keep retrying after 5+ consecutive failures."""
    import logging
    caplog.set_level(logging.INFO, logger="subconscious.tools.mcp_client")

    attempts = {"n": 0}

    async def failing_run_session():
        attempts["n"] += 1
        raise ConnectionError("backend down")

    monkeypatch.setattr(client, "_run_session", failing_run_session)

    thread = threading.Thread(target=client._connection_loop, daemon=True)
    thread.start()

    # Wait long enough for at least 5 attempts.
    # Backoff 0.1 -> 0.2 -> 0.4 -> 0.4 -> 0.4, total ~1.5s to attempt 5.
    time.sleep(2.0)

    client._shutdown_event.set()
    thread.join(timeout=3)

    assert attempts["n"] >= 5, (
        f"expected 5+ attempts, got {attempts['n']} - "
        f"loop is stopping early"
    )

    # Every attempt should be logged with its number.
    attempt_logs = [
        r for r in caplog.records
        if "attempt #" in r.getMessage() and "failed" in r.getMessage()
    ]
    assert len(attempt_logs) >= 5


def test_reconnect_loop_exits_on_shutdown(client, monkeypatch):
    """Setting _shutdown_event must terminate the loop promptly."""
    async def failing_run_session():
        raise ConnectionError("down")

    monkeypatch.setattr(client, "_run_session", failing_run_session)

    thread = threading.Thread(target=client._connection_loop, daemon=True)
    thread.start()
    time.sleep(0.3)  # let one attempt happen

    client._shutdown_event.set()
    thread.join(timeout=3)

    assert not thread.is_alive(), (
        "reconnect loop did not exit after _shutdown_event"
    )


def test_reconnect_loop_uses_fresh_loop_per_attempt(client, monkeypatch):
    """Each attempt must set self._loop to a new event loop."""
    seen_loops = set()
    original_new_loop = mcp_client.asyncio.new_event_loop

    def tracking_new_loop():
        loop = original_new_loop()
        seen_loops.add(id(loop))
        return loop

    monkeypatch.setattr(mcp_client.asyncio, "new_event_loop", tracking_new_loop)

    async def failing_run_session():
        raise ConnectionError("down")

    monkeypatch.setattr(client, "_run_session", failing_run_session)

    thread = threading.Thread(target=client._connection_loop, daemon=True)
    thread.start()
    time.sleep(0.8)  # room for ~3 attempts
    client._shutdown_event.set()
    thread.join(timeout=3)

    assert len(seen_loops) >= 2, (
        f"expected 2+ distinct event loops, got {len(seen_loops)}"
    )


def test_backoff_caps_at_max(client, monkeypatch, caplog):
    """Attempt intervals should grow then cap."""
    import logging
    caplog.set_level(logging.INFO, logger="subconscious.tools.mcp_client")

    async def failing_run_session():
        raise ConnectionError("down")

    monkeypatch.setattr(client, "_run_session", failing_run_session)

    thread = threading.Thread(target=client._connection_loop, daemon=True)
    thread.start()
    time.sleep(1.5)
    client._shutdown_event.set()
    thread.join(timeout=3)

    waits = []
    for r in caplog.records:
        msg = r.getMessage()
        if "reconnect attempt #" in msg and " in " in msg:
            # "reconnect attempt #N in 0.1s (last outcome: X)"
            try:
                part = msg.split(" in ")[1].split("s ")[0]
                waits.append(float(part))
            except (IndexError, ValueError):
                continue

    assert waits, "no reconnect-attempt log lines captured"
    # First wait is initial, second is doubled, rest cap at max.
    assert waits[0] == pytest.approx(0.1, rel=0.01)
    assert waits[1] == pytest.approx(0.2, rel=0.01)
    if len(waits) >= 3:
        assert waits[2] == pytest.approx(0.4, rel=0.01)
    for w in waits:
        assert w <= 0.4 + 1e-6, f"backoff exceeded cap: {w}"
