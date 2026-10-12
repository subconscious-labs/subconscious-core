#!/usr/bin/env python3
"""memory.py — memory abstraction for chatd.

Layered memory model:
  L0   mempalace : identity.txt — always loaded
  L1   mempalace : Essential Story (wake-up context) — always loaded
  L1.5 chatd     : request-scoped sidecar — RAG recall (KG recall removed)
                   injected per request — always loaded
                   (disable with CHATD_LAYER_15_ENABLED=false)
  L2   mempalace : Room Recall (filtered retrieval) — when topic comes up
  L3   mempalace : Deep Search (full semantic query) — when explicitly asked

mempalace layers (L0, L1, L2, L3) are managed by mempalace itself.
chatd layer (L1.5) is a purely additive overlay.

Note: KG and FastMemory were removed in Stage 2. The wake-up block is now
assembled from local identity.txt plus a KG snippet fetched over MCP.
"""
import functools
import logging
import os
from pathlib import Path
from typing import Optional

from config import (
    MEMPALACE_PALACE_PATH,  # kept for backwards compat / logging
    MEMORY_KG_SNIPPET_LIMIT,
)

log = logging.getLogger("chatd.memory")

# ── Layer feature flags ───────────────────────────────────────────────────────

def _bool_env(name: str, default: bool = True) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val not in ("false", "0", "no", "off")


LAYER_15_ENABLED: bool = _bool_env("CHATD_LAYER_15_ENABLED", default=True)


# ── identity ──────────────────────────────────────────────────────────────────

def _read_identity() -> str:
    """Read identity.txt relative to this file."""
    identity_path = Path(__file__).parent / "identity.txt"
    try:
        return identity_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        log.warning("identity.txt read failed: %s", e)
        return ""


# ── KG snippet via MCP ────────────────────────────────────────────────────────

def _kg_snippet_via_mcp(limit: int = MEMORY_KG_SNIPPET_LIMIT) -> str:
    """Fetch a KG snippet via MCP. Returns "" on any failure."""
    try:
        import mcp_registry
    except Exception as e:
        log.debug("mcp_registry not available: %s", e)
        return ""

    if not mcp_registry.has_tool("mempalace_kg_timeline"):
        log.debug("mempalace_kg_timeline not registered, KG snippet empty")
        return ""

    try:
        result = mcp_registry.call_tool("mempalace_kg_timeline", {})
    except Exception as e:
        log.warning("mempalace_kg_timeline call failed: %s", e)
        return ""

    if not isinstance(result, dict):
        return ""

    timeline = result.get("timeline") or []
    if not isinstance(timeline, list):
        return ""

    lines = []
    for t in timeline[:limit]:
        s = t.get("subject") or "?"
        p = t.get("predicate") or "?"
        o = t.get("object") or "?"
        if t.get("current", True):
            lines.append(f"{s} {p} {o}")
    return "\n".join(lines)


# ── wake-up assembly ──────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _wakeup_cached() -> str:
    identity = _read_identity()
    kg_block = _kg_snippet_via_mcp()

    parts = [p for p in (identity, kg_block) if p]
    text = "\n\n".join(parts)
    log.info("FastMemory: wake-up assembled (%d chars)", len(text))
    return text


def wake_up() -> str:
    """Assemble the always-loaded system-prompt block (L0 + L1)."""
    return _wakeup_cached()


def invalidate() -> None:
    """Clear the wake-up cache. Called after mempalace write operations."""
    _wakeup_cached.cache_clear()
    log.info("FastMemory: cache invalidated")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def init() -> None:
    """Run startup health checks."""
    log.info("FastMemory: layer flags - L1.5=%s", LAYER_15_ENABLED)

    identity = _read_identity()
    if identity:
        log.info("FastMemory: identity.txt loaded (%d chars)", len(identity))
    else:
        log.warning(
            "FastMemory: identity.txt missing or empty — wake-up will be "
            "KG-only until it is created"
        )
