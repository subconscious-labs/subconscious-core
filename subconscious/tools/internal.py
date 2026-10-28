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

from subconscious.events import store as events_store
from subconscious.runtime.config import (
    BG_MAX_OPEN_GOALS,
    TOOLS_ALLOWED,
    TOOLS_FILTER,
)
from subconscious.tools import registry
from subconscious.worker import ledger


log = logging.getLogger("subconscious.tools.internal")


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
                                 "needs-review", "done", "cancelled",
                                 "all"],
                        "description": (
                            "'open' = pending+active (default). "
                            "'needs-review' = awaiting operator via "
                            "subcortex_goals_resolve."
                        ),
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
                "Cancel a goal that is no longer relevant. Writes a "
                "reason into the goal's history and moves it to "
                "cancelled. With cascade=true, also cancels every goal "
                "descended from any of the target ids via parent_goal_id "
                "- useful when a parent goal was found to be a dead end "
                "and its sub-goals exist only to unblock it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "string",
                        "description": "Single goal id (legacy).",
                    },
                    "goal_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple goal ids to cancel at once.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why cancelled. Goes into history.",
                    },
                    "cascade": {
                        "type": "boolean",
                        "description": (
                            "Also cancel all descendants of any target "
                            "id via parent_goal_id. Default false."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subcortex_goals_resolve",
            "description": (
                "Resolve a goal in needs-review status. The assessor "
                "moved it there because the executor's 'done' claim "
                "was ambiguous (unclear) or repeatedly false "
                "(regressed at the limit). Actions: 'verified' - "
                "accept as done; 'cancelled' - give up; 'retry' - "
                "return to pending with counters reset for another "
                "attempt. Only affects goals with status='needs-review'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["verified", "cancelled", "retry"],
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why resolved. Goes into the goal's history."
                        ),
                    },
                },
                "required": ["goal_id", "action"],
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
            "name": "event_get",
            "description": (
                "Fetch a single event by id, including its payload, "
                "status, and linked goal. Use when the user refers to a "
                "specific event id, or when following up on an event "
                "mentioned in another tool's output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "Event id, e.g. 'ev-a1b2c3d4e5f6'.",
                    },
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "event_list",
            "description": (
                "List recent events from the ledger, optionally filtered "
                "by user, status, or goal. Use to see what has arrived "
                "recently, what is still pending, or what belongs to a "
                "specific investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Filter by user identity (email).",
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "pending", "processing", "processed",
                            "linked", "skipped",
                        ],
                    },
                    "goal_id": {
                        "type": "string",
                        "description": "Filter by linked goal.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max events. Default 20, max 200.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "event_link",
            "description": (
                "Attach an event to a goal. Idempotent - linking an "
                "already-linked event updates the goal_id. Use when a "
                "new event is a continuation of an existing investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "goal_id":  {"type": "string"},
                },
                "required": ["event_id", "goal_id"],
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
        "id":              g.get("id"),
        "kind":            g.get("kind"),
        "status":          g.get("status"),
        "attempts":        g.get("attempts", 0),
        "partials":        g.get("partials", 0),
        "regressed_count": g.get("regressed_count", 0),
        "title":           g.get("title"),
        "created_at":      g.get("created_at"),
        "updated_at":      g.get("updated_at"),
        "last_result":     lr,
        "parent_goal_id":  g.get("parent_goal_id"),
        "assess_verdict":  g.get("assess_verdict"),
    }


def _read_archive_all() -> List[Dict[str, Any]]:
    path = ledger.goals_archive_path()
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
    """Acquire the tick lock with a short retry loop.

    Returns the lock on success, None if a tick is running. We reuse the
    same lock that run_tick holds - one lock for "background state
    writes" is simpler than two.
    """
    deadline = time.time() + timeout_sec
    lock = ledger.TickLock(ledger.lock_path())
    while time.time() < deadline:
        if lock.acquire():
            return lock
        time.sleep(0.2)
    return None


# ── implementations ───────────────────────────────────────────────────────────

def _list_goals(args: Dict[str, Any]) -> Dict[str, Any]:
    status_filter = (args.get("status") or "open").lower()
    kind_filter   = (args.get("kind") or "all").lower()
    limit         = int(args.get("limit") or 20)
    limit         = max(1, min(limit, 100))

    active   = ledger.read_goals().get("goals", [])
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
    goal_id = (args.get("goal_id") or "").strip()
    if not goal_id:
        return {"error": "goal_id is required"}

    g = ledger.find_goal(
        ledger.read_goals().get("goals", []), goal_id
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
        goals = ledger.read_goals()
        open_count = len(ledger.open_goals(goals.get("goals", [])))
        if open_count >= BG_MAX_OPEN_GOALS:
            return {
                "error": (
                    f"open-goal cap reached "
                    f"({open_count}/{BG_MAX_OPEN_GOALS})"
                )
            }
        new_g = ledger.new_goal(title, kind="normal")
        if based_on:
            new_g["based_on"] = based_on
        goals.setdefault("goals", []).append(new_g)
        ledger.increment_new_goals_counter()
        ledger.write_goals(goals)
        return {"created": new_g}
    finally:
        lock.release()


def _cancel_goal(args: Dict[str, Any]) -> Dict[str, Any]:
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
    cascade = bool(args.get("cascade", False))

    lock = _acquire_ledger_lock()
    if lock is None:
        return {"error": "background tick is running; try again in a minute"}
    try:
        goals = ledger.read_goals()
        now = ledger.now_iso()
        cancelled: List[Dict[str, Any]] = []
        not_found: List[str] = []
        already_closed: List[Dict[str, str]] = []

        # Phase 1: resolve the set to cancel (targets + descendants).
        # Order preserved so history reads top-down. A descendant that
        # is also listed explicitly in ids appears only once.
        resolved: List[str] = []
        explicit = set(ids)
        for gid in ids:
            target = ledger.find_goal(goals.get("goals", []), gid)
            if target is None:
                not_found.append(gid)
                continue
            if gid not in resolved:
                resolved.append(gid)
            if cascade:
                for d in ledger.find_descendants(goals.get("goals", []), gid):
                    did = d.get("id")
                    if did and did not in resolved:
                        resolved.append(did)

        # Phase 2: apply cancel.
        for gid in resolved:
            target = ledger.find_goal(goals.get("goals", []), gid)
            if target is None:
                continue
            if target.get("status") in ("done", "cancelled"):
                already_closed.append({
                    "id": gid, "status": target.get("status"),
                })
                continue
            target["status"] = "cancelled"
            target["updated_at"] = now
            note = reason or "(no reason given)"
            if cascade and gid not in explicit:
                note = f"{note} (cascade)"
            target.setdefault("history", []).append({
                "ts":    now,
                "event": "cancelled",
                "by":    "user",
                "note":  note,
            })
            cancelled.append(_compact_goal(target))

        if cancelled:
            ledger.write_goals(goals)

        return {
            "cancelled_count": len(cancelled),
            "cancelled":       cancelled,
            "not_found":       not_found,
            "already_closed":  already_closed,
        }
    finally:
        lock.release()


def _resolve_goal(args: Dict[str, Any]) -> Dict[str, Any]:
    goal_id = (args.get("goal_id") or "").strip()
    action = (args.get("action") or "").strip()
    if not goal_id or not action:
        return {"error": "goal_id and action are required"}
    if action not in ("verified", "cancelled", "retry"):
        return {"error": f"invalid action: {action}"}

    reason = (args.get("reason") or "").strip()

    lock = _acquire_ledger_lock()
    if lock is None:
        return {"error": "background tick is running; try again in a minute"}
    try:
        goals = ledger.read_goals()
        target = ledger.find_goal(goals.get("goals", []), goal_id)
        if target is None:
            return {"error": f"goal not found: {goal_id}"}
        if target.get("status") != "needs-review":
            return {
                "error": (
                    f"goal {goal_id} has status "
                    f"{target.get('status')!r}, not 'needs-review'"
                )
            }

        now = ledger.now_iso()
        if action == "verified":
            target["status"] = "done"
        elif action == "cancelled":
            target["status"] = "cancelled"
        else:  # retry
            target["status"] = "pending"
            target["attempts"] = 0
            target["partials"] = 0
            target["regressed_count"] = 0

        target["updated_at"] = now
        target.setdefault("history", []).append({
            "ts": now,
            "event": f"resolved={action}",
            "by": "user",
            "note": reason or "(no reason given)",
        })
        ledger.write_goals(goals)
        return {"resolved": True, "id": goal_id, "action": action}
    finally:
        lock.release()


def _journal_tail(args: Dict[str, Any]) -> Dict[str, Any]:
    from collections import deque

    n = int(args.get("n") or 20)
    n = max(1, min(n, 100))

    path = ledger.journal_path()
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
    for client in registry.list_clients():
        if client.name == name:
            return client
    return None


def _mcp_server_list(args: Dict[str, Any]) -> Dict[str, Any]:
    clients = registry.list_clients()
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
    if not ok:
        return {
            "status": "restart_initiated",
            "name": name,
            "force": force,
            "note": (
                "Reconnect loop is running in the background. Retry "
                "mcp_server_restart after the backend is up; tools will "
                "register on the next successful restart."
            ),
        }

    # Refresh tool schemas for this server so they become visible
    # without restarting subconscious.
    try:
        server_tools = client.list_tools()
    except Exception as e:
        return {
            "status": "restarted",
            "name": name,
            "force": force,
            "pid": getattr(client, "_pid", None),
            "warning": f"restart ok but list_tools failed: {e}",
        }

    new_tools: List[Dict[str, Any]] = []
    new_registry: Dict[str, Any] = {}
    for t in server_tools:
        tn = getattr(t, "name", None)
        schema = (
            getattr(t, "inputSchema", None)
            or getattr(t, "input_schema", None)
        )
        desc = getattr(t, "description", "") or ""
        if not tn or not schema:
            continue
        if TOOLS_FILTER and tn not in TOOLS_ALLOWED:
            continue
        new_tools.append({
            "type": "function",
            "function": {
                "name": tn,
                "description": desc,
                "parameters": schema,
            },
        })
        new_registry[tn] = client

    registry.refresh_server(name, new_tools, new_registry)
    return {
        "status": "restarted",
        "name": name,
        "force": force,
        "pid": getattr(client, "_pid", None),
        "tools_count": len(new_tools),
        "tools": [t["function"]["name"] for t in new_tools],
    }


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


def _event_get(args: Dict[str, Any]) -> Dict[str, Any]:
    event_id = (args.get("event_id") or "").strip()
    if not event_id:
        return {"error": "event_id is required"}
    ev = events_store.get(event_id)
    if ev is None:
        return {"error": f"event not found: {event_id}"}
    return ev


def _event_list(args: Dict[str, Any]) -> Dict[str, Any]:
    limit = int(args.get("limit") or 20)
    limit = max(1, min(limit, 200))
    events = events_store.list_events(
        user_id=(args.get("user_id") or None),
        status=(args.get("status") or None),
        goal_id=(args.get("goal_id") or None),
        limit=limit,
    )
    return {"count": len(events), "events": events}


def _event_link(args: Dict[str, Any]) -> Dict[str, Any]:
    event_id = (args.get("event_id") or "").strip()
    goal_id = (args.get("goal_id") or "").strip()
    if not event_id or not goal_id:
        return {"error": "event_id and goal_id are required"}
    ok = events_store.link(event_id, goal_id)
    if not ok:
        return {"error": f"event not found: {event_id}"}
    return {"linked": True, "event_id": event_id, "goal_id": goal_id}


_DISPATCH = {
    "subcortex_goals_list":    _list_goals,
    "subcortex_goals_get":     _get_goal,
    "subcortex_goals_create":  _create_goal,
    "subcortex_goals_cancel":  _cancel_goal,
    "subcortex_goals_resolve": _resolve_goal,
    "subcortex_journal_tail":  _journal_tail,
    "event_get":               _event_get,
    "event_list":              _event_list,
    "event_link":              _event_link,
    "mcp_server_list":         _mcp_server_list,
    "mcp_server_restart":      _mcp_server_restart,
    "mcp_server_reload":       _mcp_server_reload,
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
