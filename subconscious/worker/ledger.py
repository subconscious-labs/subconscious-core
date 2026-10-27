"""Goal ledger, budget counters, journal, and tick lock.

Owns all persistent state under BG_STATE_DIR. tick.py orchestrates the
phases (planner, executor, reflector); ledger owns the files. Public
names (no leading underscore) - this is a module API, not a set of
module-private helpers.

Keep this module free of imports from subconscious.chat or
subconscious.tools: it is a leaf that everything else can depend on.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from subconscious.runtime.config import (
    BG_STATE_DIR,
    BG_DAILY_TOKEN_BUDGET,
    BG_DAILY_TIME_BUDGET_SEC,
    BG_CLOUD_DAILY_TOKEN_BUDGET,
    BG_CLOUD_PREFIXES,
    BG_DEGRADE_MODEL,
    BG_MODEL,
    BG_EXPLORATORY_TTL_HOURS,
    BG_EXPLORATORY_MIN_INTERVAL_MIN,
    BG_JOURNAL_MAX_MB,
    BG_JOURNAL_KEEP,
)

log = logging.getLogger("subconscious.worker.ledger")

GOALS_VERSION = 1
BUDGET_VERSION = 1


# ── paths ─────────────────────────────────────────────────────────────────────

def state_dir() -> Path:
    p = Path(os.path.expanduser(BG_STATE_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def goals_path() -> Path:
    return state_dir() / "goals.json"


def budget_path() -> Path:
    return state_dir() / "budget.json"


def journal_path() -> Path:
    return state_dir() / "journal.jsonl"


def lock_path() -> Path:
    return state_dir() / "tick.lock"


def goals_archive_path() -> Path:
    return state_dir() / "goals.archive.jsonl"


def new_goals_counter_path() -> Path:
    return state_dir() / "new_goals_counter.json"


def seed_state_path() -> Path:
    return state_dir() / "seed_rotation.json"


# ── time / ids ────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def new_id() -> str:
    return "g-" + uuid.uuid4().hex[:6]


# ── atomic JSON I/O ───────────────────────────────────────────────────────────

def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("failed to read %s: %s - using default", path, e)
        return default


# ── journal ───────────────────────────────────────────────────────────────────

def rotate_journal_if_needed() -> None:
    if BG_JOURNAL_MAX_MB <= 0:
        return
    path = journal_path()
    if not path.exists():
        return
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < BG_JOURNAL_MAX_MB:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rotated = path.with_name(f"journal.{stamp}.jsonl")
        os.rename(path, rotated)
        log.info("journal rotated: %s -> %s (%.1f MB)",
                 path.name, rotated.name, size_mb)
        if BG_JOURNAL_KEEP > 0:
            old = sorted(state_dir().glob("journal.*.jsonl"))
            for p in old[:-BG_JOURNAL_KEEP]:
                try:
                    p.unlink()
                    log.debug("removed old journal: %s", p.name)
                except Exception:
                    pass
    except Exception as e:
        log.warning("journal rotation failed: %s", e)


def append_journal(entry: Dict[str, Any]) -> None:
    try:
        rotate_journal_if_needed()
        with journal_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("journal append failed: %s", e)


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

def empty_budget() -> Dict[str, Any]:
    return {
        "version": BUDGET_VERSION,
        "date": today_utc(),
        "tokens_used": 0,
        "time_used_sec": 0,
        "cloud_tokens_used": 0,
    }


def read_budget() -> Dict[str, Any]:
    b = read_json(budget_path(), empty_budget())
    if b.get("date") != today_utc():
        b = empty_budget()
        atomic_write_json(budget_path(), b)
    return b


def budget_allows_new_tick() -> Optional[str]:
    """Return a reason string if a new tick is not allowed, else None."""
    if BG_DAILY_TOKEN_BUDGET <= 0 and BG_DAILY_TIME_BUDGET_SEC <= 0:
        return None
    b = read_budget()
    if BG_DAILY_TOKEN_BUDGET > 0 and b["tokens_used"] >= BG_DAILY_TOKEN_BUDGET:
        return "daily_token_budget_exceeded"
    if BG_DAILY_TIME_BUDGET_SEC > 0 and b["time_used_sec"] >= BG_DAILY_TIME_BUDGET_SEC:
        return "daily_time_budget_exceeded"
    return None


def budget_add_llm(tokens: int, is_cloud_model: bool) -> None:
    if tokens <= 0:
        return
    b = read_budget()
    b["tokens_used"] = int(b.get("tokens_used", 0)) + int(tokens)
    if is_cloud_model:
        b["cloud_tokens_used"] = int(b.get("cloud_tokens_used", 0)) + int(tokens)
    atomic_write_json(budget_path(), b)


def budget_add_time(elapsed_sec: float) -> None:
    if BG_DAILY_TIME_BUDGET_SEC <= 0:
        return
    b = read_budget()
    b["time_used_sec"] = int(b.get("time_used_sec", 0)) + int(elapsed_sec)
    atomic_write_json(budget_path(), b)


def is_cloud(model: str) -> bool:
    return any(model.startswith(p) for p in BG_CLOUD_PREFIXES)


def resolve_model(model: str, phase: str) -> tuple[str, Optional[str]]:
    if not is_cloud(model):
        return model, None
    if BG_CLOUD_DAILY_TOKEN_BUDGET <= 0:
        return model, None
    b = read_budget()
    used = int(b.get("cloud_tokens_used", 0))
    if used < BG_CLOUD_DAILY_TOKEN_BUDGET:
        return model, None
    fallback = BG_DEGRADE_MODEL or BG_MODEL
    reason = (
        f"cloud budget exhausted ({used}/{BG_CLOUD_DAILY_TOKEN_BUDGET} tokens), "
        f"degraded {phase} to {fallback}"
    )
    log.info("[%s] %s", phase, reason)
    return fallback, reason


# ── goals ledger ──────────────────────────────────────────────────────────────

def empty_goals() -> Dict[str, Any]:
    return {"version": GOALS_VERSION, "updated_at": now_iso(), "goals": []}


def read_goals() -> Dict[str, Any]:
    g = read_json(goals_path(), empty_goals())
    if not isinstance(g.get("goals"), list):
        g["goals"] = []
    return g


def append_archive(goal: Dict[str, Any]) -> None:
    try:
        with goals_archive_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(goal, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("goals archive append failed: %s", e)


def write_goals(g: Dict[str, Any]) -> None:
    """Persist goals. Active/pending/blocked stay in goals.json;
    done/cancelled get appended to goals.archive.jsonl and removed
    from the active file.
    """
    active = []
    for goal in g.get("goals", []):
        if goal.get("status") in ("done", "cancelled"):
            append_archive(goal)
        else:
            active.append(goal)
    g["goals"] = active
    g["updated_at"] = now_iso()
    atomic_write_json(goals_path(), g)


def read_new_goals_counter() -> Dict[str, Any]:
    c = read_json(new_goals_counter_path(), {"date": today_utc(), "count": 0})
    if c.get("date") != today_utc():
        c = {"date": today_utc(), "count": 0}
    return c


def increment_new_goals_counter() -> None:
    c = read_new_goals_counter()
    c["count"] = int(c.get("count", 0)) + 1
    c["date"] = today_utc()
    atomic_write_json(new_goals_counter_path(), c)


def new_goals_created_today() -> int:
    return int(read_new_goals_counter().get("count", 0))


def open_goals(goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [g for g in goals if g.get("status") in ("pending", "active")]


def active_exploratory(goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [g for g in goals
            if g.get("kind") == "exploratory"
            and g.get("status") in ("pending", "active")]


def exploratory_created_today(goals: List[Dict[str, Any]]) -> int:
    prefix = today_utc()[:10]
    return sum(
        1 for g in goals
        if g.get("kind") == "exploratory"
        and (g.get("created_at") or "").startswith(prefix)
    )


def last_exploratory_timestamp(goals: List[Dict[str, Any]]) -> Optional[datetime]:
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


def exploratory_min_interval_ok(goals: List[Dict[str, Any]]) -> bool:
    if BG_EXPLORATORY_MIN_INTERVAL_MIN <= 0:
        return True
    last = last_exploratory_timestamp(goals)
    if last is None:
        return True
    elapsed_min = (datetime.now(timezone.utc) - last).total_seconds() / 60.0
    return elapsed_min >= BG_EXPLORATORY_MIN_INTERVAL_MIN


def new_goal(title: str, kind: str) -> Dict[str, Any]:
    now = now_iso()
    return {
        "id": new_id(),
        "kind": kind,
        "title": title.strip()[:300] or "(untitled)",
        "status": "pending",
        "attempts": 0,
        "created_at": now,
        "updated_at": now,
        "last_result": None,
        "history": [{"ts": now, "event": "created", "by": "system"}],
    }


def find_goal(goals: List[Dict[str, Any]], goal_id: str) -> Optional[Dict[str, Any]]:
    for g in goals:
        if g.get("id") == goal_id:
            return g
    return None


def cancel_stale_exploratory(goals: List[Dict[str, Any]]) -> int:
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
            g["updated_at"] = now_iso()
            g.setdefault("history", []).append({
                "ts": g["updated_at"],
                "event": "cancelled",
                "by": "system",
                "note": "exploratory TTL expired",
            })
            changed += 1
    return changed


# ── recent-history readers (planner + exploratory accounting) ────────────────

def read_recent_observations(limit: int = 12) -> List[Dict[str, Any]]:
    """Recent observations from archive + active goals (planner context)."""
    items: List[Dict[str, Any]] = []

    archive_path = goals_archive_path()
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
            log.debug("archive read failed: %s", e)

    for g in read_goals().get("goals", []):
        if g.get("last_result"):
            items.append(g)

    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items[:limit]


def read_recent_normal_goals(limit: int = 5) -> List[Dict[str, Any]]:
    items = []
    archive_path = goals_archive_path()
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
    for g in read_goals().get("goals", []):
        if g.get("kind") == "normal":
            items.append(g)
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items[:limit]


def read_recent_archive(limit: int = 30) -> List[Dict[str, Any]]:
    """Last `limit` entries from goals.archive.jsonl.

    Used by exploratory accounting, which must see completed goals too -
    otherwise daily cap and min-interval reset every time an exploratory
    finishes and moves to the archive.
    """
    items: List[Dict[str, Any]] = []
    path = goals_archive_path()
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
