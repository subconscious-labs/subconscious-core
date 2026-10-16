"""background.py - background thinking (subconscious) for chatd.

Phases (one per tick):
  Planner    model without tools; selects skip | pick | new goal
  Executor   model with tools; runs the selected goal (tool loop)
  Reflector  model without tools; only on blocked goals: retry | cancel | keep

State lives entirely in $CHATD_BG_STATE_DIR:
  goals.json     - ledger (atomic rewrite)
  budget.json    - daily token/time counters
  journal.jsonl  - append-only tick log
  tick.lock      - flock, prevents overlapping ticks

Hard rules:
  - No writes to mempalace. Only memory.wake_up() (read-only) is used.
  - No session / RAG side effects.
  - Exploratory goals: local executor model, read-only tools only, one
    active at a time, TTL, daily cap.
  - Ledger rewritten atomically (tmp + os.replace).
  - Two-level timeouts: per phase, and per tick.

Token budget note:
  Ollama returns prompt_eval_count + eval_count; we use their sum.
  OpenRouter currently go through chatd's Ollama-shaped shim and do not
  expose usage, so his ticks count as 0 tokens. This is a known limitation
  to be fixed when a real usage channel is plumbed (PR D).

Threading note:
  Phase timeouts use daemon threads. If a phase times out, the worker thread
  keeps running until it finishes or dies with the process. This is a
  deliberate trade-off: we prefer a clean "timeout" result over complex
  cancellation that Python cannot do reliably anyway.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    BG_MODEL,
    BG_MODEL_EXECUTOR,
    BG_MODEL_PLANNER,
    BG_MODEL_REFLECTOR,
    BG_MODEL_THINK,  # noqa: F401 - declared for PR D
    BG_MAX_ATTEMPTS_PER_GOAL,
    BG_MAX_OPEN_GOALS,
    BG_MAX_TOOL_ROUNDS,
    BG_TOOL_RESULT_MAX_CHARS,
    BG_EXPLORATORY_SEED_MAX_CHARS,
    BG_OBSERVATION_SNIPPET_CHARS,
    BG_RECENT_OBSERVATIONS_LIMIT,
    BG_EXPLORATORY_PER_DAY,
    BG_EXPLORATORY_TTL_HOURS,
    BG_EXPLORATORY_MIN_INTERVAL_MIN,
    BG_PHASE_TIMEOUT,
    BG_PHASE_TIMEOUT_EXECUTOR,
    BG_PHASE_TIMEOUT_EXPLORATORY,
    BG_TICK_TIMEOUT,
    BG_DAILY_TOKEN_BUDGET,
    BG_DAILY_TIME_BUDGET_SEC,
    BG_CLOUD_PREFIXES,
    BG_CLOUD_DAILY_TOKEN_BUDGET,
    BG_DEGRADE_MODEL,
    BG_NUM_PREDICT,
    BG_SESSION_ID,
    BG_STATE_DIR,
    BG_TOOLS_ALLOWED,
    BG_MAX_NEW_GOALS_PER_DAY,
    BG_JOURNAL_MAX_MB,
    BG_JOURNAL_KEEP,
    BG_SKIP_WITH_OPEN_GOALS_STOPS_TICK,
    BG_PROMPT_PLANNER,
    BG_PROMPT_EXECUTOR,
    BG_PROMPT_EXPLORATORY,
    BG_PROMPT_REFLECTOR,
)

log = logging.getLogger("chatd.bg")

GOALS_VERSION = 1
BUDGET_VERSION = 1

# Read-only subset used for exploratory goals - intersection of any tool
# list with these suffixes. Keeps the "read-only" promise even if the
# operator accidentally adds write tools to BG_TOOLS_ALLOWED.
_READONLY_SUFFIXES = (
    "_search", "_query", "_status", "_list", "_summary",
    "_timeline", "_wake_up", "_stats", "_read", "_wings",
    "_rooms", "_taxonomy", "_peers", "_tunnels", "_hallways",
    "_drawer", "_drawers", "_duplicate", "_away", "_settings",
)

_DEFAULT_PLANNER_INSTRUCTIONS = """\
You are the planner of a background worker. Pick exactly one action.

You are shown:
- Recent observations from previous ticks (what you noticed).
- Currently open goals (pending or active).
- A hard cap on how many goals can be open at once.
- A daily limit on how many NEW normal goals you may create.

Return ONLY a JSON object, no prose, using one of these shapes:

  {"decision": "pick", "goal_id": "<id>", "reason": "..."}
  {"decision": "new",  "title": "...", "reason": "...", "based_on": "<ts>"}
  {"decision": "skip", "reason": "..."}

Rules:
- "pick"  - choose an existing pending goal that is most relevant now.
- "new"   - propose a NEW concrete goal. Only allowed if:
              * open goals < cap, AND
              * new goals created today < daily limit, AND
              * you can point to a specific observation via "based_on".
            The goal should FOLLOW UP on something you already observed.
            Never invent work out of thin air.
- "skip"  - nothing worth doing this tick.

Good "new" examples (following up on observations):
- "HTTP :80 is open globally"  -> "Check what service listens on port 80"
- "10 expired facts in KG"     -> "List the 10 expired KG facts and decide if any should be renewed"
- "wing 'user_family' has only 1 drawer" -> "Investigate why user_family wing is nearly empty"

Bad "new" examples (do NOT do this):
- "Check that ai00 is running" (vague, no basis in observations)
- "Verify backups work" (already covered by borgmatic observation)
- "Monitor disk space" (not based on any observation)

If you cannot point to a specific observation, return "skip".
"""

_DEFAULT_EXECUTOR_INSTRUCTIONS = """\
When you are done, produce ONLY a JSON object, no prose:

  {"outcome": "done" | "partial" | "failed",
   "observation": "one short paragraph",
   "goal_updates": {"status": "done" | "partial" | "blocked",
                    "last_result": "..."},
   "new_goals": [{"title": "..."}]}

Rules:
- "done"    - goal fully completed.
- "partial" - some progress, goal should stay open.
- "failed"  - could not make progress; attempts will be incremented.
- new_goals: only if you discovered something worth following up. May be [].
"""

_DEFAULT_EXPLORATORY_INSTRUCTIONS = """\
No open goals were selected. Your task: perform ONE concrete read-only
observation about the current infrastructure state.

You MUST:
1. Pick one read-only tool from the available tool list.
2. Call it with a sensible default query.
3. Read the result.
4. Report in one short paragraph what you observed.

You MUST call at least one tool before producing the final JSON. If the
tool returns nothing useful, say so - but the call still must happen.

When you are done, produce ONLY a JSON object, no prose:

  {"outcome": "done",
   "observation": "one short paragraph",
   "goal_updates": {"status": "done", "last_result": "..."},
   "new_goals": []}
"""

_DEFAULT_REFLECTOR_INSTRUCTIONS = """\
A goal has been repeatedly blocked. Decide what to do with it.

Return ONLY a JSON object, no prose:

  {"decision": "retry" | "cancel" | "keep_blocked", "reason": "..."}

- "retry"        - reset attempts and leave it pending for another try.
- "cancel"       - give up permanently.
- "keep_blocked" - leave it blocked; try again next tick.
"""

# Resolved prompts: file/env override wins, code default is the fallback.
_PLANNER_INSTRUCTIONS = BG_PROMPT_PLANNER or _DEFAULT_PLANNER_INSTRUCTIONS
_EXECUTOR_JSON_INSTRUCTIONS = BG_PROMPT_EXECUTOR or _DEFAULT_EXECUTOR_INSTRUCTIONS
_EXPLORATORY_INSTRUCTIONS = BG_PROMPT_EXPLORATORY or _DEFAULT_EXPLORATORY_INSTRUCTIONS
_REFLECTOR_INSTRUCTIONS = BG_PROMPT_REFLECTOR or _DEFAULT_REFLECTOR_INSTRUCTIONS

_EXPLORATORY_SEED_PREFERENCE = [
    # external signals first - actual changes in the environment
    "alerts_list",
    "alerts_summary",
    "monitor_query",
    "notes_search",
    "meta_list_servers",
    "meta_search",
    # memory introspection - useful but easy to loop on
    "mempalace_kg_timeline",
    "mempalace_status",
    "mempalace_kg_stats",
    "mempalace_list_wings",
    "mempalace_get_taxonomy",
    # rarely useful as seed (mostly counts and structural metadata)
    "mempalace_graph_stats",
    "mempalace_list_rooms",
    "mempalace_find_tunnels",
    "mempalace_mesh_peers",
    "mempalace_list_tunnels",
]


# ── paths ─────────────────────────────────────────────────────────────────────

def _state_dir() -> Path:
    p = Path(os.path.expanduser(BG_STATE_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _goals_path() -> Path:
    return _state_dir() / "goals.json"


def _budget_path() -> Path:
    return _state_dir() / "budget.json"


def _journal_path() -> Path:
    return _state_dir() / "journal.jsonl"


def _lock_path() -> Path:
    return _state_dir() / "tick.lock"


def _goals_archive_path() -> Path:
    return _state_dir() / "goals.archive.jsonl"


def _new_goals_counter_path() -> Path:
    return _state_dir() / "new_goals_counter.json"


def _append_archive(goal: Dict[str, Any]) -> None:
    """Append one done/cancelled goal to the archive (JSONL)."""
    try:
        with _goals_archive_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(goal, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[bg] goals archive append failed: %s", e)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _new_id() -> str:
    return "g-" + uuid.uuid4().hex[:6]


# ── atomic JSON I/O ───────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("[bg] failed to read %s: %s - using default", path, e)
        return default


# ── journal ───────────────────────────────────────────────────────────────────

def _rotate_journal_if_needed() -> None:
    """Rotate journal.jsonl if it exceeds BG_JOURNAL_MAX_MB."""
    if BG_JOURNAL_MAX_MB <= 0:
        return
    path = _journal_path()
    if not path.exists():
        return
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < BG_JOURNAL_MAX_MB:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rotated = path.with_name(f"journal.{stamp}.jsonl")
        os.rename(path, rotated)
        log.info("[bg] journal rotated: %s -> %s (%.1f MB)",
                 path.name, rotated.name, size_mb)
        if BG_JOURNAL_KEEP > 0:
            old = sorted(_state_dir().glob("journal.*.jsonl"))
            for p in old[:-BG_JOURNAL_KEEP]:
                try:
                    p.unlink()
                    log.debug("[bg] removed old journal: %s", p.name)
                except Exception:
                    pass
    except Exception as e:
        log.warning("[bg] journal rotation failed: %s", e)


def _append_journal(entry: Dict[str, Any]) -> None:
    try:
        _rotate_journal_if_needed()
        with _journal_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[bg] journal append failed: %s", e)


# ── lock ──────────────────────────────────────────────────────────────────────

class TickLock:
    def __init__(self, path: Path):
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            os.close(self._fd)
            self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


# ── budget ────────────────────────────────────────────────────────────────────

def _empty_budget() -> Dict[str, Any]:
    return {
        "version": BUDGET_VERSION,
        "date": _today_utc(),
        "tokens_used": 0,
        "time_used_sec": 0,
        "cloud_tokens_used": 0,
    }


def _read_budget() -> Dict[str, Any]:
    b = _read_json(_budget_path(), _empty_budget())
    if b.get("date") != _today_utc():
        b = _empty_budget()
        _atomic_write_json(_budget_path(), b)
    return b


def _budget_allows_new_tick() -> Optional[str]:
    """Return a reason string if a new tick is not allowed, else None."""
    if BG_DAILY_TOKEN_BUDGET <= 0 and BG_DAILY_TIME_BUDGET_SEC <= 0:
        return None
    b = _read_budget()
    if BG_DAILY_TOKEN_BUDGET > 0 and b["tokens_used"] >= BG_DAILY_TOKEN_BUDGET:
        return "daily_token_budget_exceeded"
    if BG_DAILY_TIME_BUDGET_SEC > 0 and b["time_used_sec"] >= BG_DAILY_TIME_BUDGET_SEC:
        return "daily_time_budget_exceeded"
    return None


def _budget_add_llm(tokens: int, is_cloud: bool) -> None:
    """Called after each LLM call. Updates token counters.

    Cloud tokens are always counted (so the operator can observe usage even
    without a budget); total tokens are also tracked for BG_DAILY_TOKEN_BUDGET.
    """
    if tokens <= 0:
        return
    b = _read_budget()
    b["tokens_used"] = int(b.get("tokens_used", 0)) + int(tokens)
    if is_cloud:
        b["cloud_tokens_used"] = int(b.get("cloud_tokens_used", 0)) + int(tokens)
    _atomic_write_json(_budget_path(), b)


def _budget_add_time(elapsed_sec: float) -> None:
    """Called once per tick. Updates the time counter."""
    if BG_DAILY_TIME_BUDGET_SEC <= 0:
        return
    b = _read_budget()
    b["time_used_sec"] = int(b.get("time_used_sec", 0)) + int(elapsed_sec)
    _atomic_write_json(_budget_path(), b)


def _is_cloud(model: str) -> bool:
    """Return True if *model* is a cloud model (prefix match)."""
    return any(model.startswith(p) for p in BG_CLOUD_PREFIXES)


def _resolve_model(model: str, phase: str) -> tuple[str, Optional[str]]:
    """Return (effective_model, degrade_reason).

    If *model* is a cloud model and the cloud daily token budget is
    exhausted, replace it with BG_DEGRADE_MODEL (or BG_MODEL). Otherwise
    return as-is with degrade_reason=None.
    """
    if not _is_cloud(model):
        return model, None
    if BG_CLOUD_DAILY_TOKEN_BUDGET <= 0:
        return model, None
    b = _read_budget()
    used = int(b.get("cloud_tokens_used", 0))
    if used < BG_CLOUD_DAILY_TOKEN_BUDGET:
        return model, None
    fallback = BG_DEGRADE_MODEL or BG_MODEL
    reason = (
        f"cloud budget exhausted ({used}/{BG_CLOUD_DAILY_TOKEN_BUDGET} tokens), "
        f"degraded {phase} to {fallback}"
    )
    log.info("[bg %s] %s", phase, reason)
    return fallback, reason




# ── phase timeout helper ──────────────────────────────────────────────────────

def _run_with_timeout(fn, timeout_sec: int, *args, **kwargs):
    """Run fn in a daemon thread, honour timeout_sec. Returns fn's result."""
    box: Dict[str, Any] = {"result": None, "error": None}

    def runner() -> None:
        try:
            box["result"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=runner, daemon=True, name="bg-phase")
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise TimeoutError(f"phase timed out after {timeout_sec}s")
    if box["error"] is not None:
        raise box["error"]
    return box["result"]


# ── goals ledger ──────────────────────────────────────────────────────────────

def _empty_goals() -> Dict[str, Any]:
    return {"version": GOALS_VERSION, "updated_at": _now_iso(), "goals": []}


def _read_goals() -> Dict[str, Any]:
    g = _read_json(_goals_path(), _empty_goals())
    if not isinstance(g.get("goals"), list):
        g["goals"] = []
    return g


def _write_goals(g: Dict[str, Any]) -> None:
    """Persist goals. Active/pending/blocked stay in goals.json;
    done/cancelled get appended to goals.archive.jsonl and removed
    from the active file.
    """
    active = []
    for goal in g.get("goals", []):
        if goal.get("status") in ("done", "cancelled"):
            _append_archive(goal)
        else:
            active.append(goal)
    g["goals"] = active
    g["updated_at"] = _now_iso()
    _atomic_write_json(_goals_path(), g)


def _read_new_goals_counter() -> Dict[str, Any]:
    c = _read_json(_new_goals_counter_path(), {"date": _today_utc(), "count": 0})
    if c.get("date") != _today_utc():
        c = {"date": _today_utc(), "count": 0}
    return c


def _increment_new_goals_counter() -> None:
    c = _read_new_goals_counter()
    c["count"] = int(c.get("count", 0)) + 1
    c["date"] = _today_utc()
    _atomic_write_json(_new_goals_counter_path(), c)


def _new_goals_created_today() -> int:
    return int(_read_new_goals_counter().get("count", 0))


def _read_recent_observations(limit: int = 12) -> List[Dict[str, Any]]:
    """Return the most recent observations from archive + active goals.

    Used to give the planner context about what it has noticed before.
    """
    items: List[Dict[str, Any]] = []

    archive_path = _goals_archive_path()
    if archive_path.exists():
        try:
            with archive_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-limit * 3:]
            for line in lines:
                try:
                    g = json.loads(line)
                except Exception:
                    continue
                if g.get("last_result"):
                    items.append(g)
        except Exception as e:
            log.debug("[bg] archive read failed: %s", e)

    for g in _read_goals().get("goals", []):
        if g.get("last_result"):
            items.append(g)

    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items[:limit]


def _open_goals(goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [g for g in goals if g.get("status") in ("pending", "active")]


def _active_exploratory(goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [g for g in goals
            if g.get("kind") == "exploratory"
            and g.get("status") in ("pending", "active")]


def _exploratory_created_today(goals: List[Dict[str, Any]]) -> int:
    prefix = _today_utc()[:10]
    return sum(
        1 for g in goals
        if g.get("kind") == "exploratory"
        and (g.get("created_at") or "").startswith(prefix)
    )


def _last_exploratory_timestamp(goals: List[Dict[str, Any]]) -> Optional[datetime]:
    """Return the most recent created_at among exploratory goals, or None."""
    latest: Optional[datetime] = None
    for g in goals:
        if g.get("kind") != "exploratory":
            continue
        created = g.get("created_at")
        if not created:
            continue
        try:
            ts = datetime.strptime(
                created[:19], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _exploratory_min_interval_ok(goals: List[Dict[str, Any]]) -> bool:
    """Return True if enough time has passed since the last exploratory."""
    if BG_EXPLORATORY_MIN_INTERVAL_MIN <= 0:
        return True
    last = _last_exploratory_timestamp(goals)
    if last is None:
        return True
    elapsed_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    return elapsed_min >= BG_EXPLORATORY_MIN_INTERVAL_MIN


def _new_goal(title: str, kind: str) -> Dict[str, Any]:
    now = _now_iso()
    return {
        "id": _new_id(),
        "kind": kind,
        "title": title.strip()[:300] or "(untitled)",
        "status": "pending",
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
        "last_result": None,
        "history": [{"ts": now, "event": "created", "by": "system"}],
    }


def _find_goal(goals: List[Dict[str, Any]], goal_id: str) -> Optional[Dict[str, Any]]:
    for g in goals:
        if g.get("id") == goal_id:
            return g
    return None


def _cancel_stale_exploratory(goals: List[Dict[str, Any]]) -> int:
    """Auto-cancel exploratory goals older than BG_EXPLORATORY_TTL_HOURS."""
    if BG_EXPLORATORY_TTL_HOURS <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=BG_EXPLORATORY_TTL_HOURS)
    changed = 0
    for g in goals:
        if g.get("kind") != "exploratory":
            continue
        if g.get("status") not in ("pending", "active"):
            continue
        try:
            created = datetime.strptime(
                g["created_at"][:19], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if created < cutoff:
            g["status"] = "cancelled"
            g["updated_at"] = _now_iso()
            g.setdefault("history", []).append({
                "ts": g["updated_at"],
                "event": "cancelled",
                "by": "system",
                "note": "exploratory TTL expired",
            })
            changed += 1
    return changed


# ── LLM primitive ─────────────────────────────────────────────────────────────

def _llm_call(
    messages: List[Dict[str, Any]],
    model: str,
    tools: Optional[List[Dict[str, Any]]] = None,
    num_predict: Optional[int] = None,
    response_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Single backend call. Returns {content, thinking, tool_calls, tokens, raw}."""
    import backends
    from config import DEFAULT_OPTIONS, THINKING

    options = dict(DEFAULT_OPTIONS)
    if num_predict is not None:
        options["num_predict"] = num_predict
    if response_format is not None:
        options["response_format"] = response_format
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if tools:
        body["tools"] = tools
    if not THINKING:
        body["think"] = False

    backend = backends.get_backend(model)
    resp = backend.chat_sync(body)

    msg = resp.get("message") or {}
    usage = resp.get("usage") or {}
    thinking = msg.get("thinking") or ""

    log.info(
        "[bg llm] model=%s finish=%s content=%d thinking=%d tool_calls=%d "
        "prompt_tokens=%s completion_tokens=%s",
        model,
        resp.get("done_reason", "?"),
        len(msg.get("content") or ""),
        len(thinking),
        len(msg.get("tool_calls") or []),
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )

    tokens = (
        int(usage.get("prompt_tokens") or 0)
        + int(usage.get("completion_tokens") or 0)
    )
    if tokens > 0:
        _budget_add_llm(tokens, _is_cloud(model))
    return {
        "content":    msg.get("content") or "",
        "thinking":   thinking,
        "tool_calls": msg.get("tool_calls") or [],
        "tokens":     tokens,
        "raw":        resp,
    }


def _system_block() -> str:
    import memory
    return memory.wake_up()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from text; tolerate ```json fences."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            return None
        candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


# ── tools filtering for executor ──────────────────────────────────────────────

def _allowed_tools_for(kind: str) -> List[Dict[str, Any]]:
    """Return the tools list visible to the Executor for this goal kind."""
    import chatd  # local import to avoid circular import at module load

    all_tools = getattr(chatd, "TOOLS", []) or []
    allowed_names = {t for t in BG_TOOLS_ALLOWED if t}

    selected: List[Dict[str, Any]] = []
    for t in all_tools:
        fn = (t.get("function") or {})
        name = fn.get("name")
        if not name or name not in allowed_names:
            continue
        if kind == "exploratory" and not name.endswith(_READONLY_SUFFIXES):
            continue
        selected.append(t)
    return selected


def _truncate_tool_result(result: Any, limit: Optional[int] = None) -> str:
    if limit is None:
        limit = BG_TOOL_RESULT_MAX_CHARS
    try:
        serialized = json.dumps(result, ensure_ascii=False)
    except Exception:
        serialized = str(result)
    if limit <= 0 or len(serialized) <= limit:
        return serialized
    return (
        serialized[:limit]
        + ' ... [truncated, result was '
        + str(len(serialized))
        + ' chars]'
    )


def _execute_tool_call(tc: Dict[str, Any]) -> Any:
    import chatd

    name = tc.get("name") or "unknown"
    args = tc.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    # Defense in depth: the tool schema list already excludes internal tools
    # (not in BG_TOOLS_ALLOWED), but if the model hallucinates one, reject.
    if name not in BG_TOOLS_ALLOWED:
        log.warning("[bg tool] rejected unknown tool: %r args=%r", name, args)
        return {"error": f"tool not allowed in background: {name}"}

    try:
        return chatd.call_tool(name, args)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "tool": name}


# ── Planner ───────────────────────────────────────────────────────────────────

def _read_recent_normal_goals(limit: int = 5) -> List[Dict[str, Any]]:
    items = []
    archive_path = _goals_archive_path()
    if archive_path.exists():
        try:
            with archive_path.open("r", encoding="utf-8") as f:
                tail = deque(f, maxlen=limit * 3)
                for line in tail:
                    try:
                        g = json.loads(line)
                    except Exception:
                        continue
                    if g.get("kind") == "normal":
                        items.append(g)
        except Exception:
            pass
    for g in _read_goals().get("goals", []):
        if g.get("kind") == "normal":
            items.append(g)
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items[:limit]


def _read_recent_archive(limit: int = 30) -> List[Dict[str, Any]]:
    """Return the last `limit` entries from goals.archive.jsonl.

    Used by exploratory accounting, which must see completed goals too —
    otherwise daily cap and min-interval reset every time an exploratory
    finishes and moves to the archive.
    """
    items: List[Dict[str, Any]] = []
    path = _goals_archive_path()
    if not path.exists():
        return items
    try:
        with path.open("r", encoding="utf-8") as f:
            tail = deque(f, maxlen=limit)
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        pass
    return items


def _planner() -> Dict[str, Any]:
    model = BG_MODEL_PLANNER or BG_MODEL
    model, degrade_reason = _resolve_model(model, "planner")
    goals = _read_goals()
    open_list = _open_goals(goals["goals"])
    cap = BG_MAX_OPEN_GOALS

    # Recent observations for context
    observations = _read_recent_observations(limit=BG_RECENT_OBSERVATIONS_LIMIT)
    obs_lines = []
    for o in observations:
        ts = (o.get("updated_at") or o.get("created_at") or "?")[:16]
        kind = o.get("kind", "?")
        res = (o.get("last_result") or "").replace("\n", " ").strip()
        if len(res) > BG_OBSERVATION_SNIPPET_CHARS:
            res = res[:BG_OBSERVATION_SNIPPET_CHARS] + "..."
        obs_lines.append(f"- [{ts}] ({kind}) {res}")
    obs_block = "\n".join(obs_lines) if obs_lines else "(no observations yet)"

    made_today = _new_goals_created_today()

    # Recent normal goals — to prevent duplicate proposals. We look at
    # the last 5 created normal goals regardless of status, because a
    # completed investigation shouldn't be repeated.
    recent_normal = _read_recent_normal_goals(limit=5)
    recent_titles = "\n".join(
        f"- {g.get('title', '?')}" for g in recent_normal
    ) or "(none)"

    listing_lines = []

    for g in open_list:
        listing_lines.append(
            f"- id={g['id']} kind={g['kind']} attempts={g['attempts']} "
            f"title={g['title']!r}"
        )
    listing = "\n".join(listing_lines) if listing_lines else "(none)"

    user_text = (
        f"Recent observations:\n{obs_block}\n\n"
        f"Recently created normal goals (do NOT duplicate these):\n{recent_titles}\n\n"
        f"Open goals: {len(open_list)} / {cap}\n"
        f"{listing}\n\n"
        f"New normal goals created today: {made_today} / {BG_MAX_NEW_GOALS_PER_DAY}\n\n"
        f"Return your JSON decision."
    )

    messages = [
        {"role": "system", "content": _system_block()},
        {"role": "system", "content": _PLANNER_INSTRUCTIONS},
        {"role": "user", "content": user_text},
    ]

    resp = _llm_call(messages, model, tools=None,
                     num_predict=BG_NUM_PREDICT)
    _c = resp["content"] or ""
    _r = resp.get("thinking") or ""
    log.info("[bg planner] content(%d)=%r", len(_c), _c[:400])
    log.info("[bg planner] thinking(%d)=%r", len(_r), _r[:400])

    if not _c.strip():
        _append_journal({
            "ts": _now_iso(),
            "phase": "planner",
            "diagnostic": "empty_content",
            "content_len": 0,
            "thinking_len": len(_r),
            "thinking": _r[:2000],
        })

    text_for_parse = resp["content"] or ""
    if not text_for_parse.strip() and _r.strip():
        log.info("[bg planner] content empty, falling back to thinking")
        text_for_parse = _r
    decision = _extract_json(text_for_parse) or {}

    d = decision.get("decision")

    if d == "new":
        if len(open_list) >= cap:
            log.info("[bg planner] 'new' rejected: open cap reached")
            decision = {"decision": "skip", "reason": "open-goal cap reached"}
            d = "skip"
        elif made_today >= BG_MAX_NEW_GOALS_PER_DAY:
            log.info("[bg planner] 'new' rejected: daily new-goal limit reached")
            decision = {"decision": "skip",
                        "reason": "daily new-goal limit reached"}
            d = "skip"
        elif not decision.get("based_on"):
            log.info("[bg planner] 'new' rejected: missing 'based_on'")
            decision = {"decision": "skip",
                        "reason": "new goal without based_on"}
            d = "skip"
        elif not decision.get("title"):
            log.info("[bg planner] 'new' rejected: empty title")
            decision = {"decision": "skip", "reason": "new goal without title"}
            d = "skip"

    if d == "pick" and not _find_goal(goals["goals"], decision.get("goal_id", "")):
        log.info("[bg planner] 'pick' with unknown goal_id - downgrading to skip")
        decision = {"decision": "skip", "reason": "unknown goal_id"}
        d = "skip"
    if d not in ("pick", "new", "skip"):
        decision = {"decision": "skip", "reason": "invalid decision"}
        d = "skip"

    return {
        "decision": decision,
        "model": model,
        "tokens": resp["tokens"],
        "degraded": degrade_reason,
    }


# ── Executor ──────────────────────────────────────────────────────────────────

def _exploratory_seed_tool(tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick one read-only tool to call before invoking the LLM.

    Small models on slow hardware often skip tool calls entirely when the
    list is long and the prompt is dense. Pre-seeding a single call gives
    the model a concrete result to interpret instead of an instruction it
    can ignore.
    """
    by_name = {(t.get("function") or {}).get("name"): t for t in tools}
    for name in _EXPLORATORY_SEED_PREFERENCE:
        if name in by_name:
            return by_name[name]
    return tools[0] if tools else None


def _seed_state_path() -> Path:
    return _state_dir() / "seed_rotation.json"


def _tool_needs_args(tool: Dict[str, Any]) -> bool:
    """Return True if the tool has any required parameters.

    Exploratory seed tools are called with empty arguments. Any tool with
    required parameters would fail immediately. We filter those out instead
    of relying on a hand-maintained preference list.
    """
    fn = tool.get("function") or {}
    params = fn.get("parameters") or {}
    if not isinstance(params, dict):
        return False
    required = params.get("required") or []
    return bool(required)


def _pick_seed_tool(tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Rotate through read-only tools that can be called with no arguments.

    Preference list is used for ordering: preferred tools go first, the
    rest follow. Tools with required parameters are excluded entirely,
    because calling them with {} would just return an error.
    """
    if not tools:
        return None

    # Exclude tools that need arguments - they will fail with {}.
    callable_now = [t for t in tools if not _tool_needs_args(t)]
    if not callable_now:
        log.warning("[bg executor] no no-arg tools available for exploratory")
        return None

    by_name = {(t.get("function") or {}).get("name"): t for t in callable_now}
    preferred = [by_name[n] for n in _EXPLORATORY_SEED_PREFERENCE if n in by_name]
    preferred_names = {n for n in _EXPLORATORY_SEED_PREFERENCE if n in by_name}
    rest = [
        t for t in callable_now
        if (t.get("function") or {}).get("name") not in preferred_names
    ]
    available = preferred + rest

    state = _read_json(_seed_state_path(), {"last_index": -1})
    idx = (int(state.get("last_index", -1)) + 1) % len(available)
    _atomic_write_json(_seed_state_path(), {"last_index": idx})

    picked = available[idx]
    log.info("[bg executor] exploratory seed rotation: idx=%d/%d tool=%s",
             idx, len(available),
             (picked.get("function") or {}).get("name", "?"))
    return picked


def _executor_tool_loop(
    messages: List[Dict[str, Any]],
    model: str,
    tools: List[Dict[str, Any]],
    max_rounds: int,
) -> tuple[int, int]:
    """Run tool rounds in-place on `messages`.

    Returns (tokens_total, tool_call_count).
    """
    tokens_total = 0
    tool_call_count = 0
    for round_num in range(max_rounds):
        resp = _llm_call(messages, model, tools=tools, num_predict=BG_NUM_PREDICT)
        tokens_total += resp["tokens"]
        tool_calls = resp["tool_calls"]
        if not tool_calls:
            messages.append({
                "role": "assistant",
                "content": resp["content"],
            })
            break

        assistant_msg: Dict[str, Any] = {
            "role":       "assistant",
            "content":    resp["content"],
            "tool_calls": tool_calls,
        }
        if resp.get("thinking"):
            assistant_msg["thinking"] = resp["thinking"]
        messages.append(assistant_msg)

        for tc in tool_calls:
            result = _execute_tool_call(tc)
            tool_call_count += 1
            if result is None:
                result = {"result": None, "note": "tool returned no data"}
            truncated = _truncate_tool_result(result, BG_TOOL_RESULT_MAX_CHARS)
            log.info("[bg executor] round %d tool result: %d chars (after cap)",
                     round_num, len(truncated))
            messages.append({
                "role":    "tool",
                "content": truncated,
            })
        log.info("[bg executor] round %d: %d tool call(s)",
                 round_num, len(tool_calls))
    return tokens_total, tool_call_count


def _executor(goal: Dict[str, Any]) -> Dict[str, Any]:
    model = BG_MODEL_EXECUTOR or BG_MODEL
    model, degrade_reason = _resolve_model(model, "executor")
    kind = goal.get("kind", "normal")
    tools = _allowed_tools_for(kind)

    # ── Exploratory: single LLM call, no tool loop ──────────────────────
    # On slow hardware a tool loop of 2-3 rounds costs 10+ minutes and
    # never fits the phase budget. Exploratory does exactly one
    # deterministic read-only tool call (seed), then asks the model to
    # interpret the result. One LLM call total.
    if kind == "exploratory":
        if not tools:
            return {
                "model": model, "tokens": 0, "tool_calls": 0,
                "outcome": "failed",
                "observation": "no read-only tools available",
                "new_status": "blocked", "new_goals": [],
            }
        seed = _pick_seed_tool(tools)
        if seed is None:
            return {
                "model": model, "tokens": 0, "tool_calls": 0,
                "outcome": "failed",
                "observation": "no no-arg read-only tools available",
                "new_status": "blocked", "new_goals": [],
            }
        seed_name = (seed.get("function") or {}).get("name", "?")
        log.info("[bg executor] exploratory seed tool: %s", seed_name)
        seed_result = _execute_tool_call({
            "name": seed_name,
            "arguments": {},
        })
        if seed_result is None:
            seed_result = {"result": None, "note": "tool returned no data"}
        seed_str = _truncate_tool_result(
            seed_result, BG_EXPLORATORY_SEED_MAX_CHARS
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _EXPLORATORY_INSTRUCTIONS},
            {"role": "user", "content":
                f"I ran the read-only tool {seed_name} with empty arguments.\n"
                f"Result:\n{seed_str}\n\n"
                f"Interpret this result in one short paragraph. If it suggests "
                f"a concrete follow-up, put it in new_goals; otherwise leave "
                f"new_goals empty. Produce ONLY the JSON summary."
            },
        ]
        tokens_total = 0
        try:
            resp = _llm_call(messages, model, tools=None,
                             num_predict=BG_NUM_PREDICT)
            tokens_total = resp["tokens"]
            parsed = _extract_json(resp["content"]) or {}
        except Exception as e:
            log.warning("[bg executor] exploratory LLM call failed: %s", e)
            parsed = {}

        outcome = parsed.get("outcome") if parsed.get("outcome") in (
            "done", "partial", "failed"
        ) else "done"
        useful = parsed.get("useful")
        if not isinstance(useful, bool):
            useful = None
        observation = str(parsed.get("observation") or "").strip()
        if not observation:
            observation = "(empty observation)"
        new_goals = []
        raw = parsed.get("new_goals") or []
        if isinstance(raw, list):
            for item in raw[:3]:
                if isinstance(item, dict):
                    t = str(item.get("title") or "").strip()
                    if t:
                        new_goals.append(t[:300])

        title = (observation[:80].rstrip() or "(exploratory tick)") \
            if observation else "(exploratory tick)"

        return {
            "model": model, "tokens": tokens_total, "tool_calls": 1,
            "outcome": outcome,
            "observation": observation,
            "new_status": "done",
            "new_goals": new_goals,
            "title": title,
            "useful": useful,
            "degraded": degrade_reason,
        }

    # ── Normal goal: tool loop + summary ────────────────────────────────
    user_text = (
        f"Goal: {goal['title']}\n"
        f"Attempt: {goal['attempts'] + 1} of {BG_MAX_ATTEMPTS_PER_GOAL}\n"
        f"Previous result: {goal.get('last_result') or '(none)'}\n\n"
        f"Work the goal using the available tools, then produce the JSON "
        f"summary as instructed."
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _system_block()},
        {"role": "system", "content": _EXECUTOR_JSON_INSTRUCTIONS},
        {"role": "user", "content": user_text},
    ]

    tokens_total = 0
    tool_calls_made = 0
    try:
        t, c = _executor_tool_loop(
            messages, model, tools, BG_MAX_TOOL_ROUNDS
        )
        tokens_total += t
        tool_calls_made = c
    except Exception as e:
        log.warning("[bg executor] tool loop failed: %s", e)

    messages.append({
        "role": "user",
        "content": (
            "Now produce ONLY the JSON summary object. Do NOT call any "
            "tools. If you need more information, describe what you would "
            "need in the observation field - do not call tools."
        ),
    })
    try:
        # Keep tools in the payload to preserve the DeepSeek prefix cache.
        # The tool schema block is part of the cache key; stripping it here
        # resets cache_hit to just the base system prompt, forcing a full
        # re-evaluation of the accumulated tool-loop history.
        resp = _llm_call(messages, model, tools=tools, num_predict=BG_NUM_PREDICT)
        tokens_total += resp["tokens"]

        # If the model disobeys the instruction and calls tools, fall back
        # to the old behaviour (tools stripped) for one clean summary round.
        if resp.get("tool_calls"):
            log.info(
                "[bg executor] summary call returned %d tool_calls; "
                "retrying with tools stripped",
                len(resp["tool_calls"]),
            )
            resp = _llm_call(messages, model, tools=None,
                             num_predict=BG_NUM_PREDICT)
            tokens_total += resp["tokens"]

        parsed = _extract_json(resp["content"]) or {}
    except Exception as e:
        log.warning("[bg executor] summary call failed: %s", e)
        parsed = {}

    outcome = parsed.get("outcome") if parsed.get("outcome") in (
        "done", "partial", "failed"
    ) else "failed"

    observation = str(parsed.get("observation") or "").strip()
    upd = parsed.get("goal_updates") or {}
    new_status = upd.get("status") if upd.get("status") in (
        "done", "partial", "blocked"
    ) else ("done" if outcome == "done" else
            "pending" if outcome == "partial" else "blocked")

    new_goals_raw = parsed.get("new_goals") or []
    new_goals: List[str] = []
    if isinstance(new_goals_raw, list):
        for item in new_goals_raw[:3]:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                if title:
                    new_goals.append(title[:300])

    return {
        "model": model,
        "tokens": tokens_total,
        "tool_calls": tool_calls_made,
        "outcome": outcome,
        "observation": observation,
        "new_status": new_status,
        "new_goals": new_goals,
        "degraded": degrade_reason,
    }


# ── Reflector ─────────────────────────────────────────────────────────────────

def _reflector(goal: Dict[str, Any]) -> Dict[str, Any]:
    model = BG_MODEL_REFLECTOR or BG_MODEL
    model, degrade_reason = _resolve_model(model, "reflector")
    history_lines = []
    for h in (goal.get("history") or [])[-10:]:
        history_lines.append(
            f"- {h.get('ts','?')} {h.get('event','?')} "
            f"{h.get('by','')} {h.get('note','')}".rstrip()
        )
    history = "\n".join(history_lines) or "(empty)"

    user_text = (
        f"Goal id: {goal['id']}\n"
        f"Title: {goal['title']}\n"
        f"Attempts: {goal['attempts']}\n"
        f"Last result: {goal.get('last_result') or '(none)'}\n"
        f"History:\n{history}\n\n"
        f"Return your JSON decision."
    )

    messages = [
        {"role": "system", "content": _system_block()},
        {"role": "system", "content": _REFLECTOR_INSTRUCTIONS},
        {"role": "user", "content": user_text},
    ]

    try:
        resp = _llm_call(messages, model, tools=None, num_predict=BG_NUM_PREDICT)
    except Exception as e:
        log.warning("[bg reflector] call failed: %s", e)
        return {"decision": "keep_blocked", "reason": f"reflector error: {e}",
                "model": model, "tokens": 0}

    text_for_parse = resp["content"] or ""
    if not text_for_parse.strip():
        _r = resp.get("thinking") or ""
        if _r.strip():
            log.info("[bg reflector] content empty, falling back to thinking")
            text_for_parse = _r
    parsed = _extract_json(text_for_parse) or {}
    d = parsed.get("decision")
    if d not in ("retry", "cancel", "keep_blocked"):
        d = "keep_blocked"

    return {"decision": d, "reason": parsed.get("reason", ""),
            "model": model, "tokens": resp["tokens"],
            "degraded": degrade_reason}


# ── ledger update helpers ─────────────────────────────────────────────────────

def _apply_goal_update(g: Dict[str, Any], status: str, last_result: Optional[str]) -> None:
    now = _now_iso()
    g["status"] = status
    g["updated_at"] = now
    if last_result is not None:
        g["last_result"] = last_result
    g.setdefault("history", []).append(
        {"ts": now, "event": f"status={status}", "by": "executor"}
    )


def _apply_failure(g: Dict[str, Any], observation: str) -> None:
    g["attempts"] = int(g.get("attempts", 0)) + 1
    now = _now_iso()
    g["updated_at"] = now
    if observation:
        g["last_result"] = observation
    if g["attempts"] >= BG_MAX_ATTEMPTS_PER_GOAL:
        g["status"] = "blocked"
        event = "blocked"
    else:
        g["status"] = "pending"
        event = "failed"
    g.setdefault("history", []).append(
        {"ts": now, "event": event, "by": "executor"}
    )


# ── run_tick ──────────────────────────────────────────────────────────────────

def _tick_impl(payload: Dict[str, Any], req_id: str) -> Dict[str, Any]:
    started = time.time()
    tick_deadline = started + BG_TICK_TIMEOUT

    _append_journal({
        "req_id": req_id,
        "ts": _now_iso(),
        "session": BG_SESSION_ID,
        "payload": payload,
        "status": "started",
    })

    # Ledger hygiene: TTL for exploratory goals.
    goals = _read_goals()
    cancelled = _cancel_stale_exploratory(goals["goals"])
    if cancelled:
        _write_goals(goals)
        log.info("[bg %s] cancelled %d stale exploratory goal(s)",
                 req_id, cancelled)

    # ── Auto-resume pending exploratory before invoking the planner ────
    # If a previous exploratory tick timed out or failed, its goal is left
    # in `pending`. Do not spend time on the planner only for it to return
    # `skip` and then discover there's already a pending exploratory. Take
    # the existing goal directly - this is deterministic and saves a full
    # planner round.
    goals = _read_goals()
    existing_expl = _active_exploratory(goals["goals"])

    if existing_expl:
        picked_goal = existing_expl[0]
        picked_goal["status"] = "active"
        picked_goal["updated_at"] = _now_iso()
        picked_goal.setdefault("history", []).append({
            "ts": picked_goal["updated_at"],
            "event": "status=active",
            "by": "system",
            "note": "auto-resume",
        })
        _write_goals(goals)
        _append_journal({
            "req_id": req_id,
            "ts": _now_iso(),
            "phase": "planner",
            "model": "(auto-resume)",
            "decision": {"decision": "pick", "goal_id": picked_goal["id"],
                         "reason": "auto-resume pending exploratory"},
        })
        tokens_used = 0
        log.info("[bg %s] auto-resuming exploratory %s (attempts=%d)",
                 req_id, picked_goal["id"], picked_goal.get("attempts", 0))
    else:
        # Short-circuit: if there is nothing useful the planner could do,
        # do not spend 2.5 minutes on it. The planner can only act on
        # pending open goals or create a new exploratory - if neither is
        # possible, skip outright.
        goals_now = _read_goals()
        open_now = _open_goals(goals_now["goals"])
        combined = goals_now["goals"] + _read_recent_archive()
        made_today = _exploratory_created_today(combined)
        if not open_now and made_today >= BG_EXPLORATORY_PER_DAY:
            log.info("[bg %s] short-circuit: no open goals, exploratory cap "
                     "reached (%d/%d)", req_id, made_today,
                     BG_EXPLORATORY_PER_DAY)
            _append_journal({
                "req_id": req_id, "ts": _now_iso(),
                "status": "skipped",
                "reason": "no_open_goals_and_exploratory_cap_reached",
            })
            return {
                "req_id": req_id,
                "status": "skipped",
                "reason": "no_open_goals_and_exploratory_cap_reached",
            }

        # ── Planner ────────────────────────────────────────────────────
        planner_result = _run_with_timeout(_planner, BG_PHASE_TIMEOUT)
        tokens_used = planner_result["tokens"]
        decision = planner_result["decision"]
        _append_journal({
            "req_id": req_id,
            "ts": _now_iso(),
            "phase": "planner",
            "model": planner_result["model"],
            "degraded": planner_result.get("degraded"),
            "decision": decision,
        })
        log.info("[bg %s] planner: %s", req_id, decision.get("decision"))

        goals = _read_goals()
        picked_goal: Optional[Dict[str, Any]] = None
        d = decision.get("decision")

        if d == "new":
            open_count = len(_open_goals(goals["goals"]))
            if open_count >= BG_MAX_OPEN_GOALS:
                log.info("[bg %s] new-goal blocked at ledger stage", req_id)
            else:
                g = _new_goal(decision.get("title", ""), kind="normal")
                if decision.get("based_on"):
                    g["based_on"] = decision["based_on"]
                goals["goals"].append(g)
                g["status"] = "active"
                g["updated_at"] = _now_iso()
                g.setdefault("history", []).append(
                    {"ts": g["updated_at"], "event": "status=active",
                     "by": "planner"}
                )
                picked_goal = g
                _increment_new_goals_counter()
                _write_goals(goals)

        elif d == "pick":
            g = _find_goal(goals["goals"], decision.get("goal_id", ""))
            if g is not None:
                g["status"] = "active"
                g["updated_at"] = _now_iso()
                g.setdefault("history", []).append(
                    {"ts": g["updated_at"], "event": "status=active",
                     "by": "planner"}
                )
                picked_goal = g
                _write_goals(goals)

    # Remaining tick budget
    remaining = int(max(30, tick_deadline - time.time()))
    if remaining <= 0:
        raise TimeoutError("tick deadline exceeded after planner")

    # If planner skipped but open goals exist, end the tick. The goal
    # the planner refused to pick won't magically become pickable by
    # running an exploratory step; and exploratory tends to introspect.
    if (picked_goal is None
            and BG_SKIP_WITH_OPEN_GOALS_STOPS_TICK
            and decision.get("decision") == "skip"):
        goals_now = _read_goals()
        open_now = _open_goals(goals_now["goals"])
        if open_now:
            log.info(
                "[bg %s] planner skipped with %d open goals — stopping",
                req_id, len(open_now),
            )
            _append_journal({
                "req_id": req_id, "ts": _now_iso(),
                "status": "skipped",
                "reason": "planner_skipped_with_open_goals",
                "open_count": len(open_now),
            })
            return {
                "req_id": req_id,
                "status": "skipped",
                "reason": "planner_skipped_with_open_goals",
                "open_count": len(open_now),
            }

    # ── Executor - runs picked goal, or explores read-only ─────────────
    if picked_goal is None:
        goals = _read_goals()
        combined = goals["goals"] + _read_recent_archive()
        made_today = _exploratory_created_today(combined)
        if made_today >= BG_EXPLORATORY_PER_DAY:
            log.info("[bg %s] exploratory cap reached (%d/%d)",
                     req_id, made_today, BG_EXPLORATORY_PER_DAY)
            return {
                "req_id": req_id,
                "status": "skipped",
                "reason": "exploratory_cap_reached",
            }
        if not _exploratory_min_interval_ok(combined):
            log.info("[bg %s] exploratory min interval not elapsed "
                     "(%d min required)",
                     req_id, BG_EXPLORATORY_MIN_INTERVAL_MIN)
            return {
                "req_id": req_id,
                "status": "skipped",
                "reason": "exploratory_min_interval",
            }
        # Note: no need to check _active_exploratory here - if there was an
        # active one, we would have auto-resumed it above.
        exp_goal = _new_goal("(exploratory tick)", kind="exploratory")
        exp_goal["status"] = "active"
        exp_goal.setdefault("history", []).append(
            {"ts": exp_goal["updated_at"], "event": "status=active",
             "by": "planner"}
        )
        goals["goals"].append(exp_goal)
        _write_goals(goals)
        picked_goal = exp_goal

    if picked_goal["kind"] == "exploratory":
        phase_timeout = min(remaining, BG_PHASE_TIMEOUT_EXPLORATORY)
    else:
        phase_timeout = min(remaining, BG_PHASE_TIMEOUT_EXECUTOR)

    try:
        executor_result = _run_with_timeout(
            _executor, phase_timeout, picked_goal
        )
    except TimeoutError as e:
        log.warning("[bg %s] executor timeout: %s", req_id, e)
        goals = _read_goals()
        g = _find_goal(goals["goals"], picked_goal["id"])
        if g is not None:
            _apply_failure(g, "executor timeout")
            _write_goals(goals)
        _append_journal({
            "req_id": req_id, "ts": _now_iso(),
            "phase": "executor", "status": "timeout", "error": str(e),
        })
        raise

    # Exploratory that produced zero tool calls is not "done" - it did
    # nothing. Downgrade it so it goes back into the ledger as pending
    # (or blocked after BG_MAX_ATTEMPTS_PER_GOAL) instead of silently
    # filling goals.json with empty entries.
    if (executor_result.get("tool_calls", 0) == 0
            and executor_result["outcome"] in ("done", "partial")):
        # An executor that produced no tool calls did not execute the goal.
        # For exploratory this is always wrong; for normal goals it means
        # the model answered from memory and likely hallucinated. Either
        # way, mark as failed so attempts increment and Reflector can act.
        log.info("[bg %s] executor made no tool calls (%s) - marking as failed",
                 req_id, picked_goal["kind"])
        executor_result["outcome"] = "failed"
        if not executor_result["observation"]:
            executor_result["observation"] = \
                f"executor made no tool calls ({picked_goal['kind']})"

    tokens_used += executor_result["tokens"]
    _append_journal({
        "req_id": req_id,
        "ts": _now_iso(),
        "phase": "executor",
        "model": executor_result["model"],
        "degraded": executor_result.get("degraded"),
        "outcome": executor_result["outcome"],
        "tool_calls": executor_result.get("tool_calls", 0),
        "tokens": executor_result["tokens"],
        "useful": executor_result.get("useful"),
    })

    # Persist executor result
    goals = _read_goals()
    g = _find_goal(goals["goals"], picked_goal["id"])
    reflector_needed = False
    if g is not None:
        if executor_result["outcome"] == "failed":
            _apply_failure(g, executor_result["observation"])
            if g["status"] == "blocked":
                reflector_needed = True
        else:
            if picked_goal["kind"] == "exploratory" and executor_result.get("title"):
                g["title"] = executor_result["title"]
            _apply_goal_update(
                g,
                executor_result["new_status"] if executor_result["new_status"]
                in ("done", "partial") else "done",
                executor_result["observation"],
            )

        open_count = len(_open_goals(goals["goals"]))
        added = 0
        for title in executor_result["new_goals"]:
            if open_count + added >= BG_MAX_OPEN_GOALS:
                log.info("[bg %s] dropping extra new_goals (cap reached)",
                         req_id)
                break
            goals["goals"].append(_new_goal(title, kind="normal"))
            added += 1

        _write_goals(goals)

    # Reflector - only on blocked goal with attempts >= cap.
    if reflector_needed and g is not None:
        remaining = int(max(30, tick_deadline - time.time()))
        if remaining > 0:
            try:
                refl = _run_with_timeout(
                    _reflector, min(remaining, BG_PHASE_TIMEOUT), g
                )
            except TimeoutError as e:
                log.warning("[bg %s] reflector timeout: %s", req_id, e)
                refl = {"decision": "keep_blocked", "reason": "timeout",
                        "model": BG_MODEL_REFLECTOR or BG_MODEL,
                        "tokens": 0}

            tokens_used += refl["tokens"]
            _append_journal({
                "req_id": req_id, "ts": _now_iso(),
                "phase": "reflector", "model": refl["model"],
                "degraded": refl.get("degraded"),
                "decision": refl["decision"], "tokens": refl["tokens"],
            })

            goals = _read_goals()
            g = _find_goal(goals["goals"], picked_goal["id"])
            if g is not None:
                now = _now_iso()
                if refl["decision"] == "retry":
                    g["attempts"] = 0
                    g["status"] = "pending"
                elif refl["decision"] == "cancel":
                    g["status"] = "cancelled"
                else:
                    g["status"] = "blocked"
                g["updated_at"] = now
                g.setdefault("history", []).append({
                    "ts": now,
                    "event": f"reflector={refl['decision']}",
                    "by": "reflector",
                    "note": refl.get("reason", ""),
                })
                _write_goals(goals)

    elapsed = time.time() - started
    _budget_add_time(elapsed)

    result = {
        "req_id": req_id,
        "status": "ok",
        "goal_id": picked_goal["id"],
        "goal_kind": picked_goal["kind"],
        "outcome": executor_result["outcome"],
        "observation": executor_result["observation"],
        "tokens": tokens_used,
        "elapsed_ms": int(elapsed * 1000),
        "useful": executor_result.get("useful"),
    }
    _append_journal({
        "req_id": req_id, "ts": _now_iso(),
        "status": "done", "result": result,
    })
    log.info("[bg %s] tick done: goal=%s outcome=%s tokens=%d elapsed=%dms",
             req_id, picked_goal["id"], executor_result["outcome"],
             tokens_used, int(elapsed * 1000))
    return result


def run_tick(payload: Dict[str, Any], req_id: str) -> Dict[str, Any]:
    """One tick. Returns a JSON-serialisable result dict."""
    lock = TickLock(_lock_path())
    if not lock.acquire():
        log.info("[bg %s] tick skipped - previous tick still running", req_id)
        _append_journal({
            "req_id": req_id, "ts": _now_iso(),
            "status": "skipped", "reason": "tick_already_running",
        })
        return {"req_id": req_id, "status": "skipped",
                "reason": "tick_already_running"}

    try:
        reason = _budget_allows_new_tick()
        if reason:
            log.info("[bg %s] tick skipped - %s", req_id, reason)
            _append_journal({
                "req_id": req_id, "ts": _now_iso(),
                "status": "skipped", "reason": reason,
            })
            return {"req_id": req_id, "status": "skipped", "reason": reason}

        return _tick_impl(payload, req_id)

    except TimeoutError as e:
        log.warning("[bg %s] tick timeout: %s", req_id, e)
        _append_journal({
            "req_id": req_id, "ts": _now_iso(),
            "status": "timeout", "error": str(e),
        })
        return {"req_id": req_id, "status": "timeout", "error": str(e)}

    except Exception as e:
        log.error("[bg %s] tick error: %s", req_id, e, exc_info=True)
        _append_journal({
            "req_id": req_id, "ts": _now_iso(),
            "status": "error", "error": str(e),
        })
        raise

    finally:
        lock.release()
