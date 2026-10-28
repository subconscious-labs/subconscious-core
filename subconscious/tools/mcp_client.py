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

from subconscious.runtime.config import (
    MCP_ENV_PREFIXES,
    MCP_EVENT_ALLOWED,
    MCP_HTTP_HEARTBEAT_SEC,
    MCP_RECONNECT_WAIT_SEC,
    MCP_RECONNECT_MAX_BACKOFF_SEC,
    MCP_RECONNECT_INITIAL_BACKOFF_SEC,
    _env,
)

log = logging.getLogger("subconscious.tools.mcp_client")

# The MCP SDK's HTTP client is very noisy when connections drop,
# logging full tracebacks for network errors that we already handle
# via reconnects.
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("httpcore2").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.ERROR)

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
        # Incremented every time _run_session establishes a fresh
        # connection. _wait_for_reconnect compares the pre-disconnect
        # generation against the current one, so a retry after
        # disconnect does not slip through on the old (still-present)
        # session object.
        self._session_generation: int = 0

    async def _on_notification(self, message):
        """Handle server-initiated JSON-RPC notifications."""
        try:
            # SDK may pass either a JSONRPCMessage wrapper (with .root)
            # or the parsed notification object directly.
            root = getattr(message, "root", message)
            method = getattr(root, "method", None)

            log.debug("[mcp] %s: notification received: method=%s", self.name, method)

            if method == "notifications/message":
                params = getattr(root, "params", None)
                if not params:
                    return

                # Params could be a dict or a Pydantic model
                logger_name = getattr(params, "logger", None) or (params.get("logger") if isinstance(params, dict) else None)
                data = getattr(params, "data", None) or (params.get("data") if isinstance(params, dict) else None)

                if logger_name != "event":
                    return

                if not isinstance(data, dict):
                    # data might be a Pydantic model
                    data = getattr(data, "model_dump", lambda: data)()

                if not isinstance(data, dict):
                    return

                # Check allowlist
                if not ("*" in MCP_EVENT_ALLOWED or self.name in MCP_EVENT_ALLOWED):
                    log.warning("[mcp] %s: event rejected (not in MCP_EVENT_ALLOWED)", self.name)
                    return

                log.info("[mcp] %s: event received: type=%s", self.name, data.get("type"))
                if self.on_event:
                    self.on_event(self.name, data)

        except Exception as e:
            log.error("[mcp] %s: notification handler error: %s", self.name, e)

    async def _watchdog(self):
        """Ping the server to detect silent connection drops."""
        ping_failures = 0
        while not self._shutdown_event.is_set() and not self._disconnect_event.is_set():
            await asyncio.sleep(30)  # Увеличили интервал до 30 секунд
            if self._shutdown_event.is_set() or not self._session:
                break
            try:
                if hasattr(self._session, "send_ping"):
                    await self._session.send_ping()
                ping_failures = 0
            except Exception as e:
                err_str = str(e).lower()
                # Если сервер не реализовал ping, просто отключаем watchdog
                if "method not found" in err_str or "-32601" in err_str:
                    log.info("[mcp] %s: server does not support ping, disabling watchdog", self.name)
                    return

                # Для остальных ошибок делаем 3 попытки перед разрывом
                ping_failures += 1
                log.warning("[mcp] %s: ping failed (%d/3): %s", self.name, ping_failures, e)
                if ping_failures >= 3:
                    log.error("[mcp] %s: too many consecutive ping failures, disconnecting", self.name)
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

                # Fresh session established. Bump the generation
                # before signalling ready - a reader in _wait_for_reconnect
                # that grabbed the old generation must see the change.
                self._session_generation += 1
                self._disconnect_event.clear()
                self._connection_ready.set()

                # HTTP transports have native keep-alive and error handling
                # via the MCP SDK. Our JSON-RPC ping interferes with it.
                # Instead we POST a cheap list_tools call periodically:
                # lazy-mcp (and any streamable-http MCP server with an
                # idle reaper) treats a quiet client as dead after
                # session_ttl, and the next real POST gets 404. The
                # heartbeat keeps LastSeen fresh regardless of the
                # server's TTL.
                watchdog_task = None
                if self.transport == "stdio":
                    watchdog_task = asyncio.create_task(self._watchdog())

                try:
                    last_heartbeat = time.time()
                    while (
                        not self._disconnect_event.is_set()
                        and not self._shutdown_event.is_set()
                    ):
                        await asyncio.sleep(0.5)
                        if self.transport == "http" and MCP_HTTP_HEARTBEAT_SEC > 0:
                            now = time.time()
                            if now - last_heartbeat >= MCP_HTTP_HEARTBEAT_SEC:
                                try:
                                    await self._session.list_tools()
                                    last_heartbeat = now
                                    log.debug(
                                        "[mcp] %s: heartbeat ok", self.name,
                                    )
                                except Exception as e:
                                    log.warning(
                                        "[mcp] %s: heartbeat failed: %s",
                                        self.name, e,
                                    )
                                    self._disconnect_event.set()
                finally:
                    if watchdog_task:
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
            self._connection_ready.clear()
            log.info("[mcp] %s: disconnected", self.name)

    def _connection_loop(self):
        """Independent reconnect loop.

        Each attempt gets a fresh event loop and a fresh AsyncExitStack
        inside _run_session. The MCP SDK's streamable_http_client uses a
        TaskGroup internally; when the initial connect fails, the
        TaskGroup cancellation can leave the shared loop in a state
        where a subsequent run_until_complete on the same loop does not
        start (silent stall after two attempts, observed 2026-10-28). A
        fresh loop per attempt is cheap and removes the entire class of
        interference.

        Backoff grows from MCP_RECONNECT_INITIAL_BACKOFF_SEC by doubling
        up to MCP_RECONNECT_MAX_BACKOFF_SEC. Loop runs until
        _shutdown_event is set; every iteration logs the attempt number,
        wait, and outcome so a stuck retry is visible in journald.
        """
        backoff = MCP_RECONNECT_INITIAL_BACKOFF_SEC
        attempt = 0

        while not self._shutdown_event.is_set():
            attempt += 1

            # Fresh loop per attempt: see docstring.
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)

            error_class = None
            try:
                loop.run_until_complete(self._run_session())
            except asyncio.CancelledError:
                if not self._shutdown_event.is_set():
                    log.debug(
                        "[mcp] %s: attempt #%d cancelled",
                        self.name, attempt,
                    )
            except Exception as e:
                error_class = type(e).__name__
                if not self._shutdown_event.is_set():
                    log.warning(
                        "[mcp] %s: attempt #%d failed (%s): %s",
                        self.name, attempt, error_class, e,
                    )
            finally:
                try:
                    loop.close()
                except Exception as e:
                    log.debug(
                        "[mcp] %s: loop close failed after #%d: %s",
                        self.name, attempt, e,
                    )

            if self._shutdown_event.is_set():
                log.info(
                    "[mcp] %s: reconnect loop exiting (shutdown)",
                    self.name,
                )
                break

            log.info(
                "[mcp] %s: reconnect attempt #%d in %.1fs "
                "(last outcome: %s)",
                self.name, attempt + 1, backoff,
                error_class or "disconnect",
            )
            if self._shutdown_event.wait(backoff):
                log.info(
                    "[mcp] %s: reconnect loop exiting (shutdown in backoff)",
                    self.name,
                )
                break

            backoff = min(backoff * 2, MCP_RECONNECT_MAX_BACKOFF_SEC)

        log.info("[mcp] %s: reconnect loop terminated", self.name)

    def start(self, wait_timeout: float = 30.0) -> bool:
        """Start the connection loop.

        Returns True if the initial connection succeeded within
        wait_timeout, False otherwise. In both cases the connection loop
        runs in the background and keeps retrying with progressive
        backoff until stop() is called.

        Does NOT raise on initial timeout - the caller decides whether
        a not-yet-connected client is acceptable. Raising and stopping
        the loop (previous behaviour) meant that a backend slower than
        the timeout could never be used for the rest of the process's
        lifetime; the loop was killed and only a full restart of the
        whole agent brought it back.
        """
        if self._thread and self._thread.is_alive():
            return self._connection_ready.is_set()

        self._loop = asyncio.new_event_loop()
        self._shutdown_event.clear()
        self._connection_ready.clear()

        self._thread = threading.Thread(
            target=self._connection_loop, daemon=True,
            name=f"mcp-loop-{self.name}",
        )
        self._thread.start()

        ready = self._connection_ready.wait(timeout=wait_timeout)
        if not ready:
            log.warning(
                "[mcp] %s: not connected after %.0fs, "
                "reconnect loop continues in background",
                self.name, wait_timeout,
            )
        return ready

    def stop(self) -> None:
        self._shutdown_event.set()
        self._disconnect_event.set()

        # Nudge the loop so run_until_complete returns promptly instead
        # of waiting for a backoff. Safe to call from another thread;
        # the resulting RuntimeError out of run_until_complete is caught
        # in _connection_loop's generic except.
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception as e:
                log.debug(
                    "[mcp] %s: stop nudge failed: %s", self.name, e,
                )

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        if self._loop is not None:
            try:
                self._loop.close()
            except Exception as e:
                log.debug(
                    "[mcp] %s: final loop close failed: %s", self.name, e,
                )
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

    def restart(
        self, force: bool = False, wait_timeout: float = 30.0,
    ) -> bool:
        """Stop and start again. force=True means SIGKILL immediately.

        Returns True if the fresh connection is up within wait_timeout.
        Returns False if the connection is not ready yet - the
        reconnect loop continues in the background either way.
        """
        log.info("[mcp] %s: restarting (force=%s)", self.name, force)
        if force and self.transport == "stdio" and self._pid:
            log.warning("[mcp] %s: force-restart (SIGKILL pid %d)", self.name, self._pid)
            try:
                os.kill(self._pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        self.stop()
        return self.start(wait_timeout=wait_timeout)

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

    def _wait_for_reconnect(
        self, timeout: float = 10.0, generation_before: Optional[int] = None,
    ) -> bool:
        """Wait until the connection loop has established a NEW session.

        A retry after a disconnect cannot trust "self._session is not
        None and _connection_ready is set" alone - those were already
        true before the disconnect, because _run_session only tears them
        down on its way out, and there is a window where the old session
        object is still present after _disconnect_event is set.

        Instead, compare the current session generation with the value
        observed before the failed call. Returns True only when a session
        with a strictly newer generation exists.

        If generation_before is None (called without a prior observation),
        fall back to the simple readiness check - useful for tests and
        callers that do not have a pre-disconnect snapshot.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            session_ready = (
                self._session is not None
                and self._connection_ready.is_set()
            )
            if generation_before is None:
                # Caller did not snapshot the generation before the
                # disconnect (e.g. tests, or a fresh caller that has
                # nothing to compare against). Fall back to the plain
                # readiness check - same contract as before the
                # generation counter existed.
                ready = session_ready
            else:
                # Strictly newer session required. A caller that took
                # a snapshot before the failed call must not retry
                # against a stale object still present in _session.
                ready = session_ready and self._session_generation > generation_before
            if ready:
                self._disconnect_event.clear()
                return True
            time.sleep(0.2)
        return False

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
            self._disconnect_event.set()
            raise RuntimeError(
                f"MCP call_tool timeout ({MCP_CALL_TOOL_TIMEOUT:.0f}s): {name}"
            )
        except Exception as exc:
            # If the underlying connection dropped (session expired,
            # network, proxy restart), _run already set _disconnect_event.
            # Wait for reconnect and retry once. If the second attempt
            # also fails, propagate - by then it is not a stale session
            # but a real problem.
            if not self._disconnect_event.is_set():
                raise
            # Snapshot the generation BEFORE waiting - _run_session
            # will bump it on the next successful connect. Waiting on
            # "no strict increase" prevents retry against the old
            # session object that is still present after disconnect.
            gen_before = self._session_generation
            log.info(
                "[mcp] %s: call_tool %r failed (%s), waiting for reconnect "
                "(generation %d)",
                self.name, name, exc, gen_before,
            )
            if not self._wait_for_reconnect(
                timeout=MCP_RECONNECT_WAIT_SEC,
                generation_before=gen_before,
            ):
                log.warning(
                    "[mcp] %s: reconnect did not complete within %.0fs",
                    self.name, MCP_RECONNECT_WAIT_SEC,
                )
                raise
            log.info("[mcp] %s: retrying call_tool %r after reconnect",
                     self.name, name)
            result = self._run(
                self._session.call_tool(name, arguments=arguments),
                MCP_CALL_TOOL_TIMEOUT,
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
