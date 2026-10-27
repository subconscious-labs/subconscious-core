#!/usr/bin/env python3
"""mcp_registry.py - shared MCP tool registry.

Both chatd.py (during startup) and memory.py (for wake-up) need access to
the same MCP clients. This module owns the registry and exposes a minimal
API for looking up tools.
"""
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("chatd.mcp_registry")

TOOLS: List[Dict] = []
REGISTRY: Dict[str, Any] = {}   # tool_name -> MCPClient


def reset(tools: List[Dict], registry: Dict[str, Any]) -> None:
    """Replace the registry. Called once at startup by chatd.init_tools().

    Internal tools (in-process, registered with client=None) stay in TOOLS
    so they reach the LLM schema, but are excluded from REGISTRY - the
    background tick calls into mcp_registry, and internal tools must not be
    reachable through that path.
    """
    global TOOLS, REGISTRY
    TOOLS = tools
    REGISTRY = {k: v for k, v in registry.items() if v is not None}


def get_client(tool_name: str) -> Optional[Any]:
    return REGISTRY.get(tool_name)


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Minimal call_tool that does NOT apply chatd's write-tool hooks."""
    client = REGISTRY.get(tool_name)
    if client is None:
        return {"error": f"tool not registered: {tool_name}"}
    try:
        return client.call_tool(tool_name, arguments)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


def has_tool(tool_name: str) -> bool:
    return tool_name in REGISTRY
