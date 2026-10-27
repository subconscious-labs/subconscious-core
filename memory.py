#!/usr/bin/env python3
"""memory.py - memory abstraction for chatd.

Layered memory model:
  L0   mempalace : identity.txt - always loaded
  L1   mempalace : Essential Story (wake-up context) - always loaded
  L1.5 chatd     : request-scoped sidecar - RAG recall (KG recall removed)
                   injected per request - always loaded
                   (disable with SUBCONSCIOUS_LAYER_15_ENABLED=false)
  L2   mempalace : Room Recall (filtered retrieval) - when topic comes up
  L3   mempalace : Deep Search (full semantic query) - when explicitly asked

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
    IDENTITY_PATH,
    _env,
)

log = logging.getLogger("chatd.memory")

LAYER_15_ENABLED: bool = _env("LAYER_15_ENABLED", "true").lower() == "true"


# ── identity ──────────────────────────────────────────────────────────────────

_DEFAULT_IDENTITY = """\
I am a local autonomous agent.

I have memory (identity, knowledge about the infrastructure, history of dialogues) and tools (MCP servers for diagnostics, communication, and working with memory). I work in interactive mode (dialogue with a person) and in the background (I independently explore the system).

My task is to notice events, investigate, remember findings, and keep the person informed. First, I observe, then I act.
"""


def _read_identity() -> str:
    """Read identity.txt from state. Create a minimal default if missing.

    The file is operator content, not repository content: it is never
    shipped with the code and never overwritten once created. A missing
    file is seeded with a generic default so the agent has a working
    sense of self on a fresh install. Read-only filesystems fall back
    to the in-memory default rather than crashing at startup.
    """
    path = Path(os.path.expanduser(IDENTITY_PATH))
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_DEFAULT_IDENTITY, encoding="utf-8")
            log.warning(
                "identity.txt not found, created default at %s - edit it "
                "to describe this installation", path,
            )
        return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        log.error(
            "identity.txt unavailable at %s: %s - using in-memory default",
            path, e,
        )
        return _DEFAULT_IDENTITY.strip()


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

@functools.lru_cache(maxsize=1)
def _identity_only_cached() -> str:
    return _read_identity()


def wake_up_identity_only() -> str:
    """Identity block without KG snippet. For phases that only need the
    project's self-description, not its knowledge graph."""
    return _identity_only_cached()


def invalidate() -> None:
    """Clear the wake-up caches. Called after mempalace write operations."""
    _wakeup_cached.cache_clear()
    _identity_only_cached.cache_clear()
    log.info("FastMemory: cache invalidated")


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def init() -> None:
    """Run startup health checks."""
    log.info("FastMemory: layer flags - L1.5=%s", LAYER_15_ENABLED)

    path = Path(os.path.expanduser(IDENTITY_PATH))
    log.info("FastMemory: identity path = %s", path)

    identity = _read_identity()
    if identity:
        log.info("FastMemory: identity loaded (%d chars)", len(identity))
    else:
        log.warning(
            "FastMemory: identity is empty - wake-up will be KG-only "
            "until content is added to %s", path,
        )
