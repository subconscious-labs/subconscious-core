"""Shared MCP tool registry.

Owns tool schemas, MCP client instances, and MCP dispatch. Both
chat.api (at startup) and worker.tick (executor tool calls) read from
here.

Keep this module a leaf: no imports from other project modules.
"""
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("subconscious.tools.registry")

TOOLS: List[Dict] = []
REGISTRY: Dict[str, Any] = {}   # tool_name -> MCPClient
CLIENTS: List[Any] = []         # MCPClient instances


def reset(
    tools: List[Dict],
    registry: Dict[str, Any],
    clients: Optional[List[Any]] = None,
) -> None:
    """Replace registry state. Called once at startup by chat.api.

    Internal tools (registered with client=None) stay in TOOLS so they
    reach the LLM schema, but are excluded from REGISTRY - callers who
    may be asked to run either kind must check is_internal first (see
    tools.internal).
    """
    global TOOLS, REGISTRY, CLIENTS
    TOOLS = list(tools)
    REGISTRY = {k: v for k, v in registry.items() if v is not None}
    CLIENTS = list(clients or [])


def list_tools() -> List[Dict]:
    return TOOLS


def list_clients() -> List[Any]:
    return list(CLIENTS)


def get_client(tool_name: str) -> Optional[Any]:
    return REGISTRY.get(tool_name)


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call a registered MCP tool.

    Does not dispatch internal tools. Callers that may need both kinds
    must check tools.internal.is_internal first.
    """
    client = REGISTRY.get(tool_name)
    if client is None:
        return {"error": f"tool not registered: {tool_name}"}
    try:
        return client.call_tool(tool_name, arguments)
    except Exception as e:
        return {"error": str(e), "tool": tool_name}


def has_tool(tool_name: str) -> bool:
    return tool_name in REGISTRY
