"""internal_tools.py - in-process tools exposed to the chat LLM.

These are NOT MCP tools. They run in the same Python process as chatd and
read/write the background ledger directly. The LLM sees them as regular
tool_calls; the user sees results in chat.

Scope: chat only.
  - Registered in chatd.TOOLS so the chat model can call them.
  - NOT added to BG_TOOLS_ALLOWED, so the background worker never sees them.
  - background._execute_tool_call also filters by BG_TOOLS_ALLOWED, so even
    a hallucinated call from the executor is rejected.

Concurrency: all ledger writes acquire the same TickLock that run_tick
holds. Chat tries with a short timeout; if a tick is running, the tool
returns an error the user can retry.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("chatd.internal_tools")


# ── schemas ───────────────────────────────────────────────────────────────────

_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "subcortex_goals_list",
            "description": (
                "List background goals (the subconscious worker's ledger). "
                "Use this when the user asks what the background agent is "
                "working on, or wants to see open/blocked/done goals. "
                "Returns compact but complete entries - do NOT call "
                "subcortex_goals_get for goals already returned by list. Use "
                "get only when you need full history of a specific goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "pending", "active", "blocked",
                                 "done", "cancelled", "all"],
                        "description": "'open' = pending+active (default).",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["normal", "exploratory", "all"],
                        "description": "Filter by goal kind. Default 'all'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max goals. Default 20, max 100.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subcortex_goals_get",
            "description": (
                "Fetch full details of one goal, including history and "
                "last_result. Works for active and archived goals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Goal id, e.g. 'g-a1b2c3'.",
                    },
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subcortex_goals_create",
            "description": (
                "Create a new normal goal for the background worker. "
                "Use this when the user explicitly asks to queue an "
                "investigation or task for the subconscious. "
                "Title must be concrete and verifiable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Concrete goal title. Max 300 chars.",
                    },
                    "based_on": {
                        "type": "string",
                        "description": (
                            "Optional context - what prompted this goal "
                            "(user request, prior observation, etc.)."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subcortex_goals_cancel",
            "description": (
                "Cancel a goal that is no longer relevant. Writes a reason "
                "into the goal's history and moves it to cancelled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Single goal id (legacy)."
                    },
                    "goal_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple goal ids to cancel at once."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why cancelled. Goes into history.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subcortex_journal_tail",
            "description": (
                "Return the most recent entries from the background worker's "
                "journal (append-only tick log). Use this when the user asks "
                "what the background worker did recently, or when you need to "
                "understand why a goal was created or cancelled."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "How many entries. Default 20, max 100.",
                    },
                },
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "mcp_server_list",
            "description": (
                "List registered MCP servers with transport, PID, and "
                "command. Use before restart/reload to see current state."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_server_restart",
            "description": (
                "Restart an MCP server. Graceful by default: close session, "
                "SIGTERM, wait up to 5s, then SIGKILL if still alive. "
                "force=true skips the graceful path and sends SIGKILL "
                "immediately - last resort for hung servers.\n\n"
                "Side effects: the server briefly disappears from the tool "
                "list. Tool schema changes are NOT auto-applied - chatd "
                "restart is required if the server's tool set changed.\n\n"
                "Chat-only. Not available in background mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "MCP server name (from mcp_server_list)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "SIGKILL immediately, skip graceful path",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp_server_reload",
            "description": (
                "Ask a stdio MCP server to reload its configuration without "
                "a full restart. Sends SIGHUP, waits 2 seconds, verifies the "
                "server still responds.\n\n"
                "IMPORTANT: many servers do not implement SIGHUP usefully - "
                "some ignore it (nothing changes), some die on it (SIGHUP's "
                "default action is termination). This tool detects both cases "
                "and automatically restarts the server in response, so the "
                "server is never left dead.\n\n"
                "Possible outcomes:\n"
                "  status=reloaded               - SIGHUP worked, config reloaded\n"
                "  status=restarted_after_sighup - SIGHUP failed, server was "
                "restarted instead (config re-read from scratch)\n"
                "  status=failed                 - both failed, server may need "
                "manual intervention\n\n"
                "Chat-only. Not available in background mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "MCP server name",
                    },
                },
                "required": ["name"],
            },
        },
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _compact_goal(g: Dict[str, Any]) -> Dict[str, Any]:
    lr = g.get("last_result")
    if isinstance(lr, str) and len(lr) > 300:
        lr = lr[:300] + "…"
    return {
        "id":          g.get("id"),
        "kind":        g.get("kind"),
        "status":      g.get("status"),
        "attempts":    g.get("attempts", 0),
        "title":       g.get("title"),
        "created_at":  g.get("created_at"),
        "updated_at":  g.get("updated_at"),
        "last_result": lr,
    }


def _read_archive_all() -> List[Dict[str, Any]]:
    from subconscious.worker import tick as background
    path = background._goals_archive_path()
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        log.warning("[internal] archive read failed: %s", e)
    return out


def _acquire_ledger_lock(timeout_sec: float = 5.0):
    """Acquire TickLock with a short retry loop.

    Returns the lock on success, None if a tick is running. We intentionally
    reuse TickLock - it already serializes run_tick, and one lock for
    "background state writes" is simpler than two.
    """
    from subconscious.worker import tick as background
    deadline = time.time() + timeout_sec
    lock = background.TickLock(background._lock_path())
    while time.time() < deadline:
        if lock.acquire():
            return lock
        time.sleep(0.2)
    return None


# ── implementations ───────────────────────────────────────────────────────────

def _list_goals(args: Dict[str, Any]) -> Dict[str, Any]:
    from subconscious.worker import tick as background

    status_filter = (args.get("status") or "open").lower()
    kind_filter   = (args.get("kind") or "all").lower()
    limit         = int(args.get("limit") or 20)
    limit         = max(1, min(limit, 100))

    active   = background._read_goals().get("goals", [])
    archived = _read_archive_all()
    everything = active + archived

    def status_ok(g: Dict[str, Any]) -> bool:
        st = g.get("status")
        if status_filter == "all":
            return True
        if status_filter == "open":
            return st in ("pending", "active")
        return st == status_filter

    def kind_ok(g: Dict[str, Any]) -> bool:
        if kind_filter == "all":
            return True
        return g.get("kind") == kind_filter

    filtered = [g for g in everything if status_ok(g) and kind_ok(g)]
    filtered.sort(
        key=lambda g: g.get("updated_at") or g.get("created_at") or "",
        reverse=True,
    )
    filtered = filtered[:limit]

    return {
        "count": len(filtered),
        "goals": [_compact_goal(g) for g in filtered],
    }


def _get_goal(args: Dict[str, Any]) -> Dict[str, Any]:
    from subconscious.worker import tick as background

    goal_id = (args.get("goal_id") or "").strip()
    if not goal_id:
        return {"error": "goal_id is required"}

    g = background._find_goal(
        background._read_goals().get("goals", []), goal_id
    )
    if g is None:
        for entry in _read_archive_all():
            if entry.get("id") == goal_id:
                g = entry
                break
    if g is None:
        return {"error": f"goal not found: {goal_id}"}
    return g


def _create_goal(args: Dict[str, Any]) -> Dict[str, Any]:
    from subconscious.worker import tick as background

    title = (args.get("title") or "").strip()
    if not title:
        return {"error": "title is required"}
    if len(title) > 300:
        title = title[:300]

    based_on = args.get("based_on")
    if isinstance(based_on, str):
        based_on = based_on.strip()[:300] or None
    else:
        based_on = None

    lock = _acquire_ledger_lock()
    if lock is None:
        return {
            "error": "background tick is running; try again in a minute"
        }
    try:
        goals = background._read_goals()
        open_count = len(background._open_goals(goals.get("goals", [])))
        if open_count >= background.BG_MAX_OPEN_GOALS:
            return {
                "error": (
                    f"open-goal cap reached "
                    f"({open_count}/{background.BG_MAX_OPEN_GOALS})"
                )
            }
        new_g = background._new_goal(title, kind="normal")
        if based_on:
            new_g["based_on"] = based_on
        goals.setdefault("goals", []).append(new_g)
        background._increment_new_goals_counter()
        background._write_goals(goals)
        return {"created": new_g}
    finally:
        lock.release()


def _cancel_goal(args: Dict[str, Any]) -> Dict[str, Any]:
    from subconscious.worker import tick as background

    ids: List[str] = []
    single = (args.get("goal_id") or "").strip()
    if single:
        ids.append(single)
    multi = args.get("goal_ids") or []
    if isinstance(multi, list):
        for gid in multi:
            if isinstance(gid, str) and gid.strip():
                ids.append(gid.strip())

    if not ids:
        return {"error": "goal_id or goal_ids required"}

    reason = (args.get("reason") or "").strip()

    lock = _acquire_ledger_lock()
    if lock is None:
        return {"error": "background tick is running; try again in a minute"}
    try:
        goals = background._read_goals()
        now = _iso()
        cancelled: List[Dict[str, Any]] = []
        not_found: List[str] = []
        already_closed: List[Dict[str, str]] = []

        for gid in ids:
            target = background._find_goal(goals.get("goals", []), gid)
            if target is None:
                not_found.append(gid)
                continue
            if target.get("status") in ("done", "cancelled"):
                already_closed.append({
                    "id": gid, "status": target.get("status"),
                })
                continue
            target["status"] = "cancelled"
            target["updated_at"] = now
            target.setdefault("history", []).append({
                "ts":    now,
                "event": "cancelled",
                "by":    "user",
                "note":  reason or "(no reason given)",
            })
            cancelled.append(_compact_goal(target))

        if cancelled:
            background._write_goals(goals)

        return {
            "cancelled_count": len(cancelled),
            "cancelled":       cancelled,
            "not_found":       not_found,
            "already_closed":  already_closed,
        }
    finally:
        lock.release()


def _journal_tail(args: Dict[str, Any]) -> Dict[str, Any]:
    from subconscious.worker import tick as background
    from collections import deque

    n = int(args.get("n") or 20)
    n = max(1, min(n, 100))

    path = background._journal_path()
    if not path.exists():
        return {"count": 0, "entries": []}

    with path.open("r", encoding="utf-8") as f:
        tail = deque(f, maxlen=n)

    entries: List[Dict[str, Any]] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        entry: Dict[str, Any] = {
            "ts":     obj.get("ts"),
            "status": obj.get("status"),
            "req_id": obj.get("req_id"),
        }
        if "phase" in obj:
            entry["phase"] = obj["phase"]
        if "model" in obj:
            entry["model"] = obj["model"]

        dec = obj.get("decision")
        if isinstance(dec, dict):
            entry["decision"] = {
                "decision": dec.get("decision"),
                "reason":   (dec.get("reason") or "")[:200],
                "goal_id":  dec.get("goal_id"),
            }

        res = obj.get("result")
        if isinstance(res, dict):
            entry["outcome"] = res.get("outcome")
            entry["goal_id"] = res.get("goal_id")
            entry["tokens"]  = res.get("tokens")
            obs = res.get("observation")
            if isinstance(obs, str):
                entry["observation"] = obs[:300]

        if obj.get("status") == "skipped" and "reason" in obj:
            entry["reason"] = obj["reason"]

        entries.append(entry)

    return {"count": len(entries), "entries": entries}


# ── dispatcher ────────────────────────────────────────────────────────────────

def _find_mcp_client(name: str):
    from subconscious.chat import api as chatd
    for client in getattr(chatd, "_MCP_CLIENTS", []):
        if client.name == name:
            return client
    return None


def _mcp_server_list(args: Dict[str, Any]) -> Dict[str, Any]:
    from subconscious.chat import api as chatd

    clients = getattr(chatd, "_MCP_CLIENTS", [])
    servers = []
    for c in clients:
        entry: Dict[str, Any] = {
            "name": c.name,
            "transport": c.transport,
        }
        pid = getattr(c, "_pid", None)
        if c.transport == "stdio":
            entry["command"] = c.cmd
            entry["pid"] = pid
            if pid:
                # Liveness check: the stored _pid can be stale if the
                # process died outside our control (SIGHUP, OOM, kill).
                try:
                    os.kill(pid, 0)
                    entry["status"] = "zombie" if _is_zombie_external(pid) else "alive"
                except ProcessLookupError:
                    entry["status"] = "dead"
                except PermissionError:
                    entry["status"] = "unknown (no permission to probe)"
                except Exception as e:
                    log.warning(
                        "[internal] liveness probe for %s pid %s failed: %s: %s",
                        c.name, pid, type(e).__name__, e,
                    )
                    entry["status"] = f"unknown ({type(e).__name__}: {e})"
            else:
                entry["status"] = "no_pid"
        else:
            entry["url"] = c.url
            entry["status"] = "remote"
        servers.append(entry)
    return {"count": len(servers), "servers": servers}


def _is_zombie_external(pid: int) -> bool:
    """Same as mcp_client._is_zombie but local - avoid cross-module import."""
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


def _mcp_server_restart(args: Dict[str, Any]) -> Dict[str, Any]:
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "name is required"}
    force = bool(args.get("force", False))

    client = _find_mcp_client(name)
    if client is None:
        return {"error": f"MCP server not found: {name}"}

    ok = client.restart(force=force)
    if ok:
        return {
            "status": "restarted",
            "name": name,
            "force": force,
            "pid": getattr(client, "_pid", None),
            "note": (
                "Tool schemas are not auto-refreshed. If the server's tool "
                "set changed, restart chatd to rebuild the registry."
            ),
        }
    return {"error": f"restart failed for {name}"}


def _mcp_server_reload(args: Dict[str, Any]) -> Dict[str, Any]:
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "name is required"}

    client = _find_mcp_client(name)
    if client is None:
        return {"error": f"MCP server not found: {name}"}
    if client.transport != "stdio":
        return {"error": f"reload not supported for {client.transport} transport"}

    result = client.reload()
    result["name"] = name
    if result.get("status") == "failed":
        result["hint"] = (
            "Reload failed AND automatic restart failed. The server may be "
            "in an unrecoverable state. Try mcp_server_restart manually "
            "with force=true."
        )
    elif result.get("status") == "restarted_after_sighup":
        result["note"] = (
            "SIGHUP did not work (server either ignored it or died on it). "
            "The server was restarted instead - configuration is now "
            "re-read from scratch. Tool schemas are not auto-refreshed in "
            "chatd; restart chatd if the tool set changed."
        )
    return result


_DISPATCH = {
    "subcortex_goals_list":   _list_goals,
    "subcortex_goals_get":    _get_goal,
    "subcortex_goals_create": _create_goal,
    "subcortex_goals_cancel": _cancel_goal,
    "subcortex_journal_tail": _journal_tail,
    "mcp_server_list":        _mcp_server_list,
    "mcp_server_restart":     _mcp_server_restart,
    "mcp_server_reload":      _mcp_server_reload,
}


def get_schemas() -> List[Dict[str, Any]]:
    return list(_SCHEMAS)


def is_internal(name: str) -> bool:
    return name in _DISPATCH


def call_internal(name: str, args: Dict[str, Any]) -> Any:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown internal tool: {name}"}
    try:
        return fn(args or {})
    except Exception as e:
        log.error("[internal] %s failed: %s", name, e, exc_info=True)
        return {"error": str(e), "tool": name}
