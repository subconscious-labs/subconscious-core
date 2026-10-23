import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
import time

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, Optional, Callable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
try:
    # Newer SDK (mcp >= 1.10): headers via create_mcp_http_client
    from mcp.client.streamable_http import (
        streamable_http_client,
        create_mcp_http_client,
    )
    _HTTP_NEW_API = True
except ImportError:
    # Older SDK: streamablehttp_client accepts headers= directly
    from mcp.client.streamable_http import (
        streamablehttp_client as streamable_http_client,
    )
    create_mcp_http_client = None
    _HTTP_NEW_API = False

from config import MCP_ENV_PREFIXES

log = logging.getLogger("chatd.mcp_client")

from config import _env

MCP_LIST_TOOLS_TIMEOUT: float = float(_env("MCP_LIST_TOOLS_TIMEOUT", "30"))
MCP_CALL_TOOL_TIMEOUT:  float = float(_env("MCP_CALL_TOOL_TIMEOUT",  "60"))


# ---------------------------------------------------------------------------
# MCP server auto-discovery
#
# Env var conventions:
#   SUBCONSCIOUS_MCP_STDIO_<NAME>=<command>
#       Spawn <command> as a subprocess; talk stdin/stdout.
#
#   SUBCONSCIOUS_MCP_HTTP_<NAME>_URL=<url>
#   SUBCONSCIOUS_MCP_HTTP_<NAME>_TOKEN=<bearer>     (optional)
#       Connect to a remote MCP server over streamable HTTP.
#
# Returns {name: config}, where config is one of:
#   {"transport": "stdio", "cmd": [argv...]}
#   {"transport": "http",  "url": "...", "token": "..."}
# ---------------------------------------------------------------------------

def discover_mcp_servers() -> dict[str, dict]:
    servers: dict[str, dict] = {}

    for key, raw in os.environ.items():
        prefix = next((p for p in MCP_ENV_PREFIXES if key.startswith(p)), None)
        if prefix is None:
            continue
        val = (raw or "").strip()
        rest = key[len(prefix):]
        if rest.startswith("STDIO_"):
            name = rest[len("STDIO_"):].lower()
            if not name or not val:
                continue
            if name in servers:
                log.warning(
                    "[mcp] duplicate config for %s, ignoring stdio entry", name
                )
                continue
            servers[name] = {
                "transport": "stdio",
                "cmd": shlex.split(val),
            }

        elif rest.startswith("HTTP_"):
            body = rest[len("HTTP_"):]
            for suffix, field in (("_URL", "url"), ("_TOKEN", "token")):
                if not body.endswith(suffix):
                    continue
                name = body[:-len(suffix)].lower()
                if not name:
                    break
                entry = servers.setdefault(name, {"transport": "http"})
                if entry.get("transport") != "http":
                    log.warning(
                        "[mcp] %s already configured as stdio; ignoring http %s",
                        name, field,
                    )
                    break
                entry[field] = val
                break

    pruned: dict[str, dict] = {}
    for name, cfg in servers.items():
        if cfg.get("transport") == "http" and not cfg.get("url"):
            log.warning("[mcp] %s has no URL, skipping", name)
            continue
        if cfg.get("transport") == "stdio" and not cfg.get("cmd"):
            log.warning("[mcp] %s has empty stdio command, skipping", name)
            continue
        pruned[name] = cfg

    return pruned


def _kill_process(cmd: list[str]) -> None:
    """Best-effort: SIGKILL a hung stdio MCP server by matching its command.

    Triggered on call_tool/list_tools timeout. Kills the ENTIRE stdio
    process, which means all of that server's backends (in the case of a
    proxy like lazy-mcp) disappear until the next mcp_server_restart.
    This is by design: if the transport is stuck for 60s, the connection
    is unusable and continuing to use it produces only cryptic errors.

    Callers should be aware that a single slow operation through a proxy
    takes down every backend that proxy manages. Do not route long-running
    operations (grep -r, full-catalog dumps) through a shared MCP proxy;
    use the specific backend's tool directly.
    """
    if not cmd:
        return
    try:
        result = subprocess.run(
            ["pgrep", "-f", " ".join(cmd)],
            capture_output=True, text=True,
        )
        for pid_str in result.stdout.splitlines():
            try:
                pid = int(pid_str.strip())
                os.kill(pid, signal.SIGKILL)
                log.warning(
                    "[mcp] killed hung MCP process pid=%d cmd=%s", pid, cmd[0]
                )
            except (ValueError, ProcessLookupError, PermissionError) as e:
                log.debug("[mcp] kill pid=%s failed: %s", pid_str.strip(), e)
    except FileNotFoundError:
        log.debug("[mcp] pgrep not available, cannot kill hung process")


def _wrap_with_pidfile(cmd: list[str], pidfile: Path) -> list[str]:
    """Wrap a command so the child writes its PID to `pidfile` before exec.

    `exec "$@"` replaces the sh wrapper with the real command - same PID.
    The PID file therefore contains the real server process's PID, not the
    wrapper's.
    """
    return ["sh", "-c", f'echo $$ > "{pidfile}" && exec "$@"', "_"] + list(cmd)


def _is_zombie(pid: int) -> bool:
    """Return True if pid is a zombie (dead but not reaped) or gone."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            stat = f.read()
    except FileNotFoundError:
        return True
    except Exception:
        return False
    idx = stat.rfind(")")
    if idx < 0 or idx + 2 >= len(stat):
        return False
    return stat[idx + 2] == "Z"


class MCPClient:
    """Long-lived MCP client with auto-reconnect and progressive backoff."""

    def __init__(self, cfg: dict, name: str = "?", on_event: Optional[Callable] = None):
        self.name = name
        self.on_event = on_event
        self.transport: str = cfg.get("transport", "stdio")
        self.cmd: list[str] = cfg.get("cmd", []) or []
        self.url: str = cfg.get("url", "") or ""
        self.token: str = cfg.get("token", "") or ""

        self._session: Optional[ClientSession] = None
        self._stack: Optional[AsyncExitStack] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        self._pid: Optional[int] = None
        self._pidfile: Optional[Path] = None
        self._op_lock = threading.RLock()
        # Connection management
        self._shutdown_event = threading.Event()
        self._connection_ready = threading.Event()
        self._disconnect_event = threading.Event()

    async def _on_notification(self, message):
        """Handle server-initiated JSON-RPC notifications."""
        try:
            if not hasattr(message, "root"):
                return
            root = message.root
            method = getattr(root, "method", None)
            if method != "notifications/event":
                return

            params = getattr(root, "params", None) or {}
            if not isinstance(params, dict):
                return

            # Check allowlist: empty list means reject all.
            from config import MCP_EVENT_ALLOWED
            if not ("*" in MCP_EVENT_ALLOWED or self.name in MCP_EVENT_ALLOWED):
                log.warning("[mcp] %s: event rejected (not in MCP_EVENT_ALLOWED)", self.name)
                return

            log.info("[mcp] %s: event received: type=%s", self.name, params.get("type"))
            if self.on_event:
                self.on_event(self.name, params)
        except Exception as e:
            log.error("[mcp] %s: notification handler error: %s", self.name, e)

    async def _watchdog(self):
        """Ping the server to detect silent connection drops."""
        while not self._shutdown_event.is_set() and not self._disconnect_event.is_set():
            await asyncio.sleep(10)
            if self._shutdown_event.is_set() or not self._session:
                break
            try:
                # MCP Protocol supports ping
                if hasattr(self._session, "send_ping"):
                    await self._session.send_ping()
            except Exception as e:
                log.warning("[mcp] %s: ping failed: %s", self.name, e)
                self._disconnect_event.set()
                break

    async def _run_session(self):
        """Runs the entire MCP session lifecycle in one task."""
        self._disconnect_event.clear()
        try:
            # The AsyncExitStack MUST be inside the task that uses it.
            async with AsyncExitStack() as stack:
                self._stack = stack

                if self.transport == "stdio":
                    self._pidfile = Path(f"/tmp/mcp_{self.name}_{os.getpid()}.pid")
                    try:
                        self._pidfile.unlink()
                    except FileNotFoundError:
                        pass
                    wrapped = _wrap_with_pidfile(self.cmd, self._pidfile)
                    read, write = await stack.enter_async_context(
                        stdio_client(StdioServerParameters(
                            command=wrapped[0], args=wrapped[1:], env=os.environ.copy(),
                        ))
                    )
                elif self.transport == "http":
                    headers: dict[str, str] = {}
                    if self.token:
                        headers["Authorization"] = f"Bearer {self.token}"

                    if _HTTP_NEW_API:
                        http_client = create_mcp_http_client(headers=headers or None)
                        await stack.enter_async_context(http_client)
                        streams = await stack.enter_async_context(
                            streamable_http_client(self.url, http_client=http_client)
                        )
                        read, write = streams[0], streams[1]
                    else:
                        streams = await stack.enter_async_context(
                            streamable_http_client(self.url, headers=headers)
                        )
                        read, write = streams[0], streams[1]
                else:
                    raise ValueError(f"unknown transport: {self.transport}")

                self._session = await stack.enter_async_context(
                    ClientSession(read, write, message_handler=self._on_notification)
                )
                await self._session.initialize()

                log.info("[mcp] %s: connected (%s)", self.name, self.transport)

                if self.transport == "stdio" and self._pidfile:
                    try:
                        self._pid = int(self._pidfile.read_text().strip())
                    except Exception as e:
                        log.warning("[mcp] %s: pid capture failed: %s", self.name, e)
                        self._pid = None

                self._connection_ready.set()

                watchdog_task = asyncio.create_task(self._watchdog())
                try:
                    # Block this task until disconnect or shutdown is requested
                    while not self._disconnect_event.is_set() and not self._shutdown_event.is_set():
                        await asyncio.sleep(0.5)
                finally:
                    watchdog_task.cancel()
                    try:
                        await watchdog_task
                    except asyncio.CancelledError:
                        pass
        except asyncio.CancelledError:
            log.debug("[mcp] %s: session cancelled", self.name)
            raise
        except Exception as e:
            log.warning("[mcp] %s: session error: %s", self.name, e)
            raise
        finally:
            self._session = None
            self._stack = None
            log.info("[mcp] %s: disconnected", self.name)

    def _connection_loop(self):
        asyncio.set_event_loop(self._loop)
        backoff = 1.0

        while not self._shutdown_event.is_set():
            try:
                self._loop.run_until_complete(self._run_session())
            except asyncio.CancelledError:
                # Loop is shutting down or task was cancelled
                if not self._shutdown_event.is_set():
                    log.debug("[mcp] %s: task cancelled unexpectedly", self.name)
            except Exception as e:
                if not self._shutdown_event.is_set():
                    log.warning("[mcp] %s: connection error: %s", self.name, e)

            if self._shutdown_event.is_set():
                break

            log.info("[mcp] %s: reconnecting in %.1fs...", self.name, backoff)
            if self._shutdown_event.wait(backoff):
                break

            backoff = min(backoff * 2, 60.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._loop = asyncio.new_event_loop()
        self._shutdown_event.clear()
        self._connection_ready.clear()

        self._thread = threading.Thread(
            target=self._connection_loop, daemon=True,
            name=f"mcp-loop-{self.name}",
        )
        self._thread.start()

        # Wait up to 30s for the initial connection to succeed
        if not self._connection_ready.wait(timeout=30):
            log.error("[mcp] %s: initial connection timed out", self.name)
            self.stop()
            raise RuntimeError(f"MCP server {self.name} initial connection timed out")

    def stop(self) -> None:
        self._shutdown_event.set()
        self._disconnect_event.set()

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if self._loop:
            self._loop.close()
            self._loop = None

        self._session = None
        self._stack = None
        self._pid = None

    def signal(self, sig: int) -> bool:
        """Send a signal to the stdio child process."""
        if self.transport != "stdio" or not self._pid:
            return False
        try:
            os.kill(self._pid, sig)
            log.info("[mcp] %s: signal %d -> pid %d", self.name, sig, self._pid)
            return True
        except ProcessLookupError:
            log.warning("[mcp] %s: pid %d already gone", self.name, self._pid)
            self._pid = None
            return False
        except PermissionError as e:
            log.error("[mcp] %s: signal(%d) permission denied: %s", self.name, sig, e)
            return False

    def restart(self, force: bool = False) -> bool:
        """Stop and start again. force=True means SIGKILL immediately."""
        log.info("[mcp] %s: restarting (force=%s)", self.name, force)
        if force and self.transport == "stdio" and self._pid:
            log.warning("[mcp] %s: force-restart (SIGKILL pid %d)", self.name, self._pid)
            try:
                os.kill(self._pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.stop()
        try:
            self.start()
            return True
        except Exception as e:
            log.error("[mcp] %s: restart failed: %s", self.name, e)
            return False

    def reload(self, wait: float = 2.0) -> Dict[str, Any]:
        """Try SIGHUP; fall back to restart if the server dies or stops working."""
        if self.transport != "stdio":
            return {"status": "failed", "error": f"reload not supported for {self.transport}"}
        if not self._pid:
            return {"status": "failed", "error": "no pid captured"}

        old_pid = self._pid
        if not self.signal(signal.SIGHUP):
            return self._restart_after_reload_failure(old_pid, "process gone before signal")

        time.sleep(wait)

        if not self._pid or _is_zombie(self._pid):
            return self._restart_after_reload_failure(old_pid, "process died after SIGHUP")

        try:
            tools = self.list_tools()
            log.info("[mcp] %s: SIGHUP ok, still serving %d tools", self.name, len(tools))
            return {"status": "reloaded", "pid": self._pid}
        except Exception as e:
            return self._restart_after_reload_failure(old_pid, f"list_tools failed after SIGHUP: {e}")

    def _restart_after_reload_failure(self, old_pid: int, reason: str) -> Dict[str, Any]:
        log.warning("[mcp] %s: %s, falling back to restart", self.name, reason)
        force = not self._pid or _is_zombie(self._pid)
        ok = self.restart(force=force)
        if not ok:
            return {"status": "failed", "error": f"restart failed after: {reason}"}
        log.info("[mcp] %s: restarted after failed reload, new pid=%s", self.name, self._pid)
        return {
            "status": "restarted_after_sighup",
            "pid": self._pid,
            "reason": reason,
            "old_pid": old_pid,
        }

    def _run(self, coro, timeout: float):
        if self._loop is None or self._session is None:
            raise RuntimeError(f"MCPClient not connected: {self.name}")

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except Exception:
            # If the underlying connection dropped, ensure the loop knows
            # to reconnect on the next cycle.
            self._disconnect_event.set()
            raise

    def list_tools(self) -> list:
        if not self._op_lock.acquire(timeout=1.0):
            raise RuntimeError(f"MCP server {self.name} is busy (restart/reload in progress)")
        try:
            try:
                result = self._run(self._session.list_tools(), MCP_LIST_TOOLS_TIMEOUT)
                return result.tools
            except asyncio.TimeoutError:
                log.error("[mcp] list_tools timed out after %.0fs: %s", MCP_LIST_TOOLS_TIMEOUT, self.name)
                if self.transport == "stdio":
                    _kill_process(self.cmd)
                self._disconnect_event.set()
                raise RuntimeError(f"MCP list_tools timeout ({MCP_LIST_TOOLS_TIMEOUT:.0f}s): {self.name}")
        finally:
            self._op_lock.release()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._op_lock.acquire(timeout=1.0):
            raise RuntimeError(f"MCP server {self.name} is busy (restart/reload in progress)")
        try:
            return self._call_tool_inner(name, arguments)
        finally:
            self._op_lock.release()

    def _call_tool_inner(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            result = self._run(
                self._session.call_tool(name, arguments=arguments),
                MCP_CALL_TOOL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.error("[mcp] call_tool '%s' timed out after %.0fs: %s", name, MCP_CALL_TOOL_TIMEOUT, self.name)
            if self.transport == "stdio":
                _kill_process(self.cmd)
            self._disconnect_event.set()
            raise RuntimeError(f"MCP call_tool timeout ({MCP_CALL_TOOL_TIMEOUT:.0f}s): {name}")

        contents = getattr(result, "content", None) or []
        if not contents:
            return None

        first = contents[0]
        text = getattr(first, "text", None)
        if text is None and isinstance(first, dict):
            text = first.get("text")
        if text is None:
            return str(first)

        try:
            return json.loads(text)
        except Exception:
            return text
