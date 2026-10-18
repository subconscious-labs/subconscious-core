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
from typing import Any, Dict, Optional

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
    """Long-lived MCP client. Supports stdio and streamable HTTP transports.

    Call start() once after construction, stop() on shutdown.
    list_tools() and call_tool() reuse the persistent session.
    """

    def __init__(self, cfg: dict, name: str = "?"):
        self.name = name
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
        # RLock because restart() re-enters via stop_graceful().
        # Serializes signal/restart against concurrent call_tool/list_tools.
        self._op_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _loop_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _start_async(self) -> None:
        self._stack = AsyncExitStack()

        if self.transport == "stdio":
            self._pidfile = Path(f"/tmp/mcp_{self.name}_{os.getpid()}.pid")
            try:
                self._pidfile.unlink()
            except FileNotFoundError:
                pass
            wrapped = _wrap_with_pidfile(self.cmd, self._pidfile)
            read, write = await self._stack.enter_async_context(
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
                await self._stack.enter_async_context(http_client)
                streams = await self._stack.enter_async_context(
                    streamable_http_client(self.url, http_client=http_client)
                )
                # Newer SDK returns 2-tuple (TransportStreams);
                # older returns 3-tuple. Slicing works for both.
                read, write = streams[0], streams[1]
            else:
                streams = await self._stack.enter_async_context(
                    streamable_http_client(self.url, headers=headers)
                )
                read, write = streams[0], streams[1]
        else:
            raise ValueError(f"unknown transport: {self.transport}")

        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()

        if self.transport == "stdio" and self._pidfile:
            try:
                self._pid = int(self._pidfile.read_text().strip())
                log.debug("[mcp] %s: captured pid=%d", self.name, self._pid)
            except Exception as e:
                log.warning("[mcp] %s: pid capture failed: %s", self.name, e)
                self._pid = None


    def start(self) -> None:
        if self._loop is not None:
            log.warning("[mcp] %s: start() called but already running", self.name)
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop_forever, daemon=True,
            name=f"mcp-loop-{self.name}",
        )
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._start_async(), self._loop)
        try:
            fut.result(timeout=30)
            log.info("[mcp] started: %s (%s)", self.name, self.transport)
        except Exception as e:
            log.error("[mcp] failed to start %s: %s", self.name, e)
            self._shutdown_loop()
            raise

    def _shutdown_loop(self) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop.close()
        self._loop = None
        self._thread = None
        self._session = None
        self._stack = None

    def stop(self) -> None:
        if self._stack and self._loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._stack.aclose(), self._loop
                )
                fut.result(timeout=10)
            except Exception as e:
                log.debug("[mcp] stop aclose error: %s", e)
        self._shutdown_loop()
        log.info("[mcp] stopped: %s", self.name)

    def signal(self, sig: int) -> bool:
        """Send a signal to the stdio child process. Returns True on success."""
        if self.transport != "stdio":
            log.warning("[mcp] %s: signals not supported for %s transport",
                        self.name, self.transport)
            return False
        if not self._pid:
            log.warning("[mcp] %s: no pid captured, cannot signal", self.name)
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
            log.error("[mcp] %s: signal(%d) permission denied: %s",
                      self.name, sig, e)
            return False

    def _wait_exit(self, timeout: float) -> bool:
        """Poll until the child exits, becomes a zombie, or times out."""
        if not self._pid:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                os.kill(self._pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            if _is_zombie(self._pid):
                return True
            time.sleep(0.1)
        return False

    def stop_graceful(self, term_timeout: float = 5.0,
                      kill_timeout: float = 2.0) -> bool:
        """Stop the child: aclose session, wait, SIGTERM, SIGKILL if needed.

        Called from restart(). Safe to call on a live client.
        """
        with self._op_lock:
            # Close the async stack first - this closes stdin, often
            # enough to make the child exit gracefully on its own.
            if self._stack and self._loop:
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        self._stack.aclose(), self._loop
                    )
                    fut.result(timeout=3.0)
                except Exception as e:
                    log.debug("[mcp] %s: aclose during stop: %s", self.name, e)

            if self.transport == "stdio" and self._pid:
                if not self._wait_exit(2.0):
                    log.info("[mcp] %s: sending SIGTERM to pid %d",
                             self.name, self._pid)
                    try:
                        os.kill(self._pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    if not self._wait_exit(term_timeout):
                        log.warning(
                            "[mcp] %s: SIGTERM ignored after %.1fs, sending SIGKILL",
                            self.name, term_timeout,
                        )
                        try:
                            os.kill(self._pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        self._wait_exit(kill_timeout)

            self._shutdown_loop()
            self._pid = None
            if self._pidfile:
                try:
                    self._pidfile.unlink()
                except FileNotFoundError:
                    pass
            log.info("[mcp] %s: stopped gracefully", self.name)
            return True

    def restart(self, force: bool = False) -> bool:
        """Stop and start again. force=True means SIGKILL immediately."""
        with self._op_lock:
            if force and self.transport == "stdio" and self._pid:
                log.warning("[mcp] %s: force-restart (SIGKILL pid %d)",
                            self.name, self._pid)
                try:
                    os.kill(self._pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._wait_exit(2.0)
                if self._stack and self._loop:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._stack.aclose(), self._loop
                        ).result(timeout=2.0)
                    except Exception:
                        pass
                self._shutdown_loop()
                self._pid = None
                if self._pidfile:
                    try:
                        self._pidfile.unlink()
                    except FileNotFoundError:
                        pass
            else:
                self.stop_graceful()

            try:
                self.start()
                return True
            except Exception as e:
                log.error("[mcp] %s: restart failed: %s", self.name, e)
                return False

    def reload(self, wait: float = 2.0) -> Dict[str, Any]:
        """Try SIGHUP; fall back to restart if the server dies or stops working.

        Many servers (Go binaries like lazy-mcp, time-mcp) either ignore
        SIGHUP silently or die on it - SIGHUP's default action is termination
        unless a handler is installed. Even servers with a handler registered
        (visible in /proc/<pid>/status SigCgt) sometimes do nothing useful
        with it. Only the post-signal check is authoritative.

        Returns a dict describing what actually happened:
          {"status": "reloaded", "pid": <pid>}
          {"status": "restarted_after_sighup",
           "pid": <new_pid>,
           "reason": "process died" | "list_tools failed" | "process gone"}
          {"status": "failed", "error": "..."}
        """
        if self.transport != "stdio":
            return {"status": "failed",
                    "error": f"reload not supported for {self.transport}"}
        if not self._pid:
            return {"status": "failed", "error": "no pid captured"}

        with self._op_lock:
            old_pid = self._pid

            if not self.signal(signal.SIGHUP):
                reason = "process gone before signal"
                log.warning("[mcp] %s: %s, falling back to restart",
                            self.name, reason)
                return self._restart_after_reload_failure(old_pid, reason)

            time.sleep(wait)

            if not self._pid or _is_zombie(self._pid):
                reason = "process died after SIGHUP"
                log.warning("[mcp] %s: %s, falling back to restart",
                            self.name, reason)
                return self._restart_after_reload_failure(old_pid, reason)

            try:
                tools = self.list_tools()
                log.info("[mcp] %s: SIGHUP ok, still serving %d tools",
                         self.name, len(tools))
                return {"status": "reloaded", "pid": self._pid}
            except Exception as e:
                reason = f"list_tools failed after SIGHUP: {e}"
                log.warning("[mcp] %s: %s, falling back to restart",
                            self.name, reason)
                return self._restart_after_reload_failure(old_pid, reason)

    def _restart_after_reload_failure(self, old_pid: int,
                                      reason: str) -> Dict[str, Any]:
        """Restart the client after a failed SIGHUP. Caller holds _op_lock."""
        # If the process is dead, skip the graceful path - there's nothing
        # left to be graceful about. If it's still alive (SIGHUP was
        # ignored, nothing changed), we still need a real restart to pick
        # up a new config, so graceful stop is fine.
        force = not self._pid or _is_zombie(self._pid)
        ok = self.restart(force=force)
        if not ok:
            return {"status": "failed",
                    "error": f"restart failed after: {reason}"}
        log.info("[mcp] %s: restarted after failed reload (%s), new pid=%s",
                 self.name, reason, self._pid)
        return {
            "status": "restarted_after_sighup",
            "pid": self._pid,
            "reason": reason,
            "old_pid": old_pid,
        }

    def _run(self, coro, timeout: float):
        if self._loop is None or self._session is None:
            raise RuntimeError(f"MCPClient not started: {self.name}")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tools(self) -> list:
        if not self._op_lock.acquire(timeout=1.0):
            raise RuntimeError(
                f"MCP server {self.name} is busy (restart/reload in progress)"
            )
        try:
            try:
                result = self._run(self._session.list_tools(), MCP_LIST_TOOLS_TIMEOUT)
                return result.tools
            except asyncio.TimeoutError:
                log.error(
                    "[mcp] list_tools timed out after %.0fs: %s",
                    MCP_LIST_TOOLS_TIMEOUT, self.name,
                )
                if self.transport == "stdio":
                    _kill_process(self.cmd)
                raise RuntimeError(
                    f"MCP list_tools timeout ({MCP_LIST_TOOLS_TIMEOUT:.0f}s): {self.name}"
                )
        finally:
            self._op_lock.release()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._op_lock.acquire(timeout=1.0):
            raise RuntimeError(
                f"MCP server {self.name} is busy (restart/reload in progress)"
            )
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
            log.error(
                "[mcp] call_tool '%s' timed out after %.0fs: %s",
                name, MCP_CALL_TOOL_TIMEOUT, self.name,
            )
            if self.transport == "stdio":
                _kill_process(self.cmd)
            raise RuntimeError(
                f"MCP call_tool timeout ({MCP_CALL_TOOL_TIMEOUT:.0f}s): {name}"
            )

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
