"""Tool dispatch and MCP client lifecycle.

Owns the active MCP clients and the tool registry. Provides load_tools,
init_tools, call_tool, and shutdown.

Note on cycles: load_tools installs an event callback that calls
process_event. The import is deferred to the callback body - the tool
bus cannot import the event loop at module load time without creating
a cycle through chat.api. Same for call_tool's internal-tools branch
that touches session state.

The module-level singletons (TOOLS, TOOL_REGISTRY, _MCP_CLIENTS) are
duplicated in tools/registry.py after init_tools; the registry is what
tools_loop and prompt read. The duplicates here are kept so that
call_tool can distinguish "internal tool" from "unknown tool" without
an extra dict lookup, and so that the atexit cleanup has a stable
reference even if the registry is reset.
"""
from __future__ import annotations

import atexit
import json
import logging
from typing import Any, Dict, List

from subconscious.chat import session as sess
from subconscious.memory import wakeup as memory
from subconscious.runtime.config import (
    MEMPALACE_WRITE_TOOLS,
    TOOLS_ALLOWED,
    TOOLS_FILTER,
)
from subconscious.tools import internal as internal_tools
from subconscious.tools import registry as mcp_registry
from subconscious.tools.mcp_client import MCPClient, discover_mcp_servers

log = logging.getLogger("subconscious.chat.tools_io")

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}
_MCP_CLIENTS: List[MCPClient] = []


def load_tools():
    tools    = []
    registry = {}
    clients  = []

    def _mcp_event_cb(server_name: str, params: dict):
        """Normalise an inbound MCP notification into process_event() inputs.

        Payload shape is not standardised across servers. Three forms are
        handled:

        1. Structured dict:   {"payload": {"from": "...", "body": "..."}}
        2. Bare string:       {"payload": "user@domain: message"}  - xmpp-mcp
        3. Fields on data:    {"from": "...", "body": "..."}        - no envelope

        A bare string is preserved as `text` so session extraction can still
        attempt to recover the sender from a "user@domain: " prefix.
        """
        from subconscious.chat.event import process_event

        raw = params.get("payload")

        if isinstance(raw, dict) and raw:
            payload = raw
        elif isinstance(raw, str) and raw.strip():
            payload = {"text": raw.strip()}
        elif any(k in params for k in ("from", "body", "room", "text")):
            payload = {
                k: v for k, v in params.items()
                if k not in ("type", "source", "model", "options")
            }
        else:
            payload = {}

        process_event(
            event_type=params.get("type") or "event",
            event_source=params.get("source") or server_name,
            event_payload=payload,
            model=params.get("model"),
            options=params.get("options"),
            user_id=payload.get("user_id") if isinstance(payload, dict) else None,
        )

    for server_name, cfg in discover_mcp_servers().items():
        client = MCPClient(cfg, name=server_name, on_event=_mcp_event_cb)
        ready = client.start()
        # Always register the client: atexit needs it for cleanup,
        # mcp_server_restart needs it to find the client when the
        # backend comes up later.
        clients.append(client)

        if not ready:
            log.warning(
                "[tools] %s: connecting in background, tools unavailable "
                "until mcp_server_restart after the backend is up",
                server_name,
            )
            continue

        try:
            server_tools = client.list_tools()
        except Exception as e:
            log.error("[tools] %s: failed to list tools: %s", server_name, e)
            continue
        log.info("[tools] %s: %d tools loaded", server_name, len(server_tools))

        for t in server_tools:
            name         = getattr(t, "name", None)
            description  = getattr(t, "description", "") or ""
            input_schema = (
                getattr(t, "inputSchema", None)
                or getattr(t, "input_schema", None)
            )

            if not name or not input_schema:
                log.warning(
                    "[tools] %s: skipping tool without name/schema: %r",
                    server_name, t,
                )
                continue

            registry[name] = client

            if TOOLS_FILTER:
                if name not in TOOLS_ALLOWED:
                    log.debug("[tools] hiding from model context: %s", name)
                    continue

            tools.append({
                "type": "function",
                "function": {
                    "name":        name,
                    "description": description,
                    "parameters":  input_schema,
                },
            })
            log.debug("[tools] registered: %s", name)

    for schema in internal_tools.get_schemas():
        fn = schema.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        tools.append(schema)
        registry[name] = None      # marker: not an MCP client
        log.debug("[tools] registered internal: %s", name)

    return tools, registry, clients


def _stop_all_mcp_clients() -> None:
    for client in _MCP_CLIENTS:
        try:
            client.stop()
        except Exception as e:
            log.debug("[mcp] atexit stop error: %s", e)


def init_tools() -> None:
    global TOOLS, TOOL_REGISTRY, _MCP_CLIENTS
    log.info("[tools] initializing MCP tools...")
    TOOLS, TOOL_REGISTRY, _MCP_CLIENTS = load_tools()
    mcp_registry.reset(TOOLS, TOOL_REGISTRY, _MCP_CLIENTS)
    atexit.register(_stop_all_mcp_clients)
    log.info("[tools] total: %d tools in registry", len(TOOLS))


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Invoke a tool by name.

    Dispatch order:
      1. Internal tools (in-process) -> internal_tools.call_internal.
         Session invalidation on goal mutations because the cached
         wake-up may embed the goal list.
      2. MCP tools -> registry.get_client -> client.call_tool.
         Memory invalidation on mempalace write tools because the cached
         KG snippet is now stale.
      3. Unknown tool -> error dict with the requested name.
    """
    if internal_tools.is_internal(name):
        log.debug("[tool] calling internal %s with args: %s", name, arguments)
        result = internal_tools.call_internal(name, arguments)
        if name in ("subcortex_goals_create", "subcortex_goals_cancel"):
            sess.invalidate_all_prompt_caches()
        return result

    client = mcp_registry.get_client(name)
    if client is None:
        log.error("[tool] unknown tool requested: %s", name)
        return {"error": f"Unknown tool: {name}", "tool": name, "args": arguments}

    log.debug("[tool] calling %s with args: %s", name, arguments)
    try:
        result = client.call_tool(name, arguments)
    except Exception as exc:
        log.error("[tool] %s raised exception: %s", name, exc)
        return {"error": str(exc), "tool": name, "args": arguments}

    log.debug("[tool] %s result: %s", name, str(result)[:300])

    if isinstance(result, dict) and result.get("success") is False:
        err_msg = result.get("error") or "unknown MCP error"
        log.warning("[tool] %s returned MCP error: %s", name, err_msg)
        return {
            "error":       err_msg,
            "tool":        name,
            "args":        arguments,
            "mcp_result":  result,
        }

    if name in MEMPALACE_WRITE_TOOLS:
        memory.invalidate()
        sess.invalidate_all_prompt_caches()

    return result
