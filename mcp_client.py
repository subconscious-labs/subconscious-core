#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
from contextlib import AsyncExitStack
from typing import Any, Optional

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

from config import MCP_ENV_PREFIX

log = logging.getLogger("chatd.mcp_client")

MCP_LIST_TOOLS_TIMEOUT: float = float(os.environ.get("CHATD_MCP_LIST_TOOLS_TIMEOUT", "30"))
MCP_CALL_TOOL_TIMEOUT:  float = float(os.environ.get("CHATD_MCP_CALL_TOOL_TIMEOUT",  "60"))


# ---------------------------------------------------------------------------
# MCP server auto-discovery
#
# Env var conventions:
#   CHATD_MCP_STDIO_<NAME>=<command>
#       Spawn <command> as a subprocess; talk stdin/stdout.
#
#   CHATD_MCP_HTTP_<NAME>_URL=<url>
#   CHATD_MCP_HTTP_<NAME>_TOKEN=<bearer>     (optional)
#       Connect to a remote MCP server over streamable HTTP.
#
# Returns {name: config}, where config is one of:
#   {"transport": "stdio", "cmd": [argv...]}
#   {"transport": "http",  "url": "...", "token": "..."}
# ---------------------------------------------------------------------------

def discover_mcp_servers() -> dict[str, dict]:
    servers: dict[str, dict] = {}

    for key, raw in os.environ.items():
        if not key.startswith(MCP_ENV_PREFIX):
            continue
        val = (raw or "").strip()
        rest = key[len(MCP_ENV_PREFIX):]

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
    """Best-effort: find and SIGKILL child processes matching cmd.
    Only meaningful for stdio transports.
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _loop_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _start_asyn(cself) -> None:
        self._stack = AsyncExitStack()

        if self.transport == "stdio":
            command = self.cmd[0]
            args = self.cmd[1:]
            read, write = await self._stack.enter_async_context(
                stdio_client(StdioServerParameters(
                    command=command, args=args, env=os.environ.copy(),
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
                # Newer SDK returns TransportStreams (2-tuple); older returns 3-tuple.
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

    def start(self) -> None:
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

    def _run(self, coro, timeout: float):
        if self._loop is None or self._session is None:
            raise RuntimeError(f"MCPClient not started: {self.name}")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_tools(self) -> list:
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

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
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
