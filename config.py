#!/usr/bin/env python3
"""Central configuration module.

All runtime settings are read from environment variables so that the project
can be installed anywhere without modifying source files.  Copy .env.example
to .env (or set Environment= in the systemd unit) and adjust as needed.
"""
import configparser
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

_env_log = logging.getLogger("chatd.config")
_env_warned: set = set()


MCP_ENV_PREFIXES = ("SUBCONSCIOUS_MCP_", "CHATD_MCP_")


def _env(name: str, default: str = "") -> str:
    """Read env with backward-compat:
       SUBCONSCIOUS_* > CHATD_* > default.
    Warns once per name when falling back to CHATD_*."""
    new = os.environ.get(f"SUBCONSCIOUS_{name}")
    if new is not None:
        return new
    old = os.environ.get(f"CHATD_{name}")
    if old is not None:
        if name not in _env_warned:
            _env_warned.add(name)
            _env_log.warning(
                "CHATD_%s is deprecated; use SUBCONSCIOUS_%s",
                name, name,
            )
        return old
    return default


# ── Thinking mode ─────────────────────────────────────────────────────────────
THINKING: bool = _env("THINKING", "false").lower() == "true"

# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS_FILTER: bool = _env("TOOLS_FILTER", "false").lower() == "true"

_default_allowed = (
    "mempalace_status,mempalace_search,mempalace_add_drawer,"
    "mempalace_kg_query,mempalace_kg_add"
)
TOOLS_ALLOWED: Set[str] = set(_env("TOOLS_ALLOWED", _default_allowed).split(","))

# ── Tool rounds ───────────────────────────────────────────────────────────────
MAX_TOOL_ROUNDS: int = int(_env("MAX_TOOL_ROUNDS", "5"))

# ── History window ────────────────────────────────────────────────────────────
# Number of user+assistant pairs to keep when trimming history before sending
# to the backend.  Increase when using a model with a large context window.
MAX_HISTORY_TURNS: int = int(_env("MAX_HISTORY_TURNS", "20"))

# ── Sampling ──────────────────────────────────────────────────────────────────
DEFAULT_OPTIONS: Dict[str, Any] = {
    "num_predict":    int(_env("NUM_PREDICT",    "768")),
    "num_ctx":        int(_env("NUM_CTX",        "8192")),
    "temperature":    float(_env("TEMPERATURE",  "0.3")),
    "top_p":          float(_env("TOP_P",        "0.9")),
    "top_k":          int(_env("TOP_K",          "40")),
    "repeat_penalty": float(_env("REPEAT_PENALTY", "1.05")),
}

# ── MemPalace ─────────────────────────────────────────────────────────────────
MEMPALACE_PALACE_PATH: str = os.environ.get(
    "MEMPALACE_PALACE_PATH", "~/.local/share/mempalace"
)

MEMPALACE_KG_PATH: str = os.environ.get(
    "MEMPALACE_KG_PATH", "~/.mempalace/knowledge_graph.sqlite3"
)

MEMPALACE_WRITE_TOOLS: Set[str] = {
    "mempalace_kg_add",
    "mempalace_add_drawer",
}

# ── Session ───────────────────────────────────────────────────────────────────
CHATD_SESSION_DIR: str = _env("SESSION_DIR", "~/.local/share/chatd/sessions")

# ── External RAG store ────────────────────────────────────────────────────────
RAG_ENABLED: bool = _env("RAG_ENABLED", "true").lower() == "true"
RAG_DB_PATH: str = _env("RAG_DB_PATH", "~/.local/share/chatd/rag.sqlite3")
RAG_EMBED_MODEL: str = _env("RAG_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K: int = int(_env("RAG_TOP_K", "4"))
RAG_MAX_CHARS_PER_CHUNK: int = int(_env("RAG_MAX_CHARS_PER_CHUNK", "4000"))
RAG_MIN_SCORE: float = float(_env("RAG_MIN_SCORE", "0.25"))

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_MODELS: List[str] = [
    m.strip()
    for m in os.environ.get("OPENROUTER_API_MODELS", "").split(",")
    if m.strip()
]

# ── Events ────────────────────────────────────────────────────────────────────
# Optional shared secret for /api/event.
# If set, caller must pass Authorization: Bearer <token>.
# If empty - endpoint is open (only for trusted internal networks).
CHATD_EVENT_TOKEN: str = _env("EVENT_TOKEN", "")

# Model used to process incoming events (can differ from chat model).
CHATD_EVENT_MODEL: str = _env("EVENT_MODEL", "")

# ── MCP auto-discovery prefix ─────────────────────────────────────────────────
MCP_ENV_PREFIX: str = "CHATD_MCP_"

# ── Tool description overrides ────────────────────────────────────────────────
TOOL_OVERRIDE: bool = _env("TOOL_OVERRIDE", "false").lower() == "true"

def _load_tool_descriptions() -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    bundled = Path(__file__).parent / "tool_descriptions.ini"
    paths = [
        str(bundled),
        "/etc/chatd/tool_descriptions.ini",
        os.path.expanduser("~/.config/chatd/tool_descriptions.ini"),
        _env("TOOL_DESCRIPTIONS", ""),
    ]
    cfg.read([p for p in paths if p])
    return {
        key: val.strip()
        for section in cfg.sections()
        for key, val in cfg.items(section)
    }


TOOL_DESCRIPTION_OVERRIDES: Dict[str, str] = _load_tool_descriptions()

DEEPSEEK_API: str = _env("DEEPSEEK_API", "https://api.deepseek.com")

DEEPSEEK_API_MODELS = [
    m.strip()
    for m in _env(
        "DEEPSEEK_MODELS",
        "deepseek-flash",
    ).split(",")
    if m.strip()
]

# ── Background thinking (subconscious) ───────────────────────────────────────
# Disabled by default - enable per-installation if you want a periodic
# background worker ("subconscious") that observes and proposes, but never
# writes to canonical memory.
BG_ENABLED: bool = _env("BG_ENABLED", "false").lower() == "true"

# Optional bearer token. If set, /api/tick requires Authorization: Bearer <token>.
# Keep separate from CHATD_EVENT_TOKEN - different triggers, different secrets.
BG_TOKEN: str = _env("BG_TOKEN", "")

# Where the background worker keeps its own state (journal, ledger, locks).
# Never place this inside mempalace or session storage.
BG_STATE_DIR: str = _env("BG_STATE_DIR", "/var/lib/subconscious/bg")

# Session id used for background ticks - isolated from user chat sessions.
BG_SESSION_ID: str = _env("BG_SESSION_ID", "background")

# Model used by the background worker. Same prefix routing as chat models
# (e.g. "qwen3:8b" → Ollama, "deepseek/deepseek-flash" → DeepSeek).
BG_MODEL: str = _env("BG_MODEL", "qwen3.5:9b")

# Max output tokens per background tick. Background answers are short by
# design; keep this well below the chat default to avoid 30-minute ticks
# on CPU-only hosts.
BG_NUM_PREDICT: int = int(_env("BG_NUM_PREDICT", "256"))

# Max tool rounds per tick (reserved for Planner/Executor - next PR).
BG_MAX_TOOL_ROUNDS: int = int(_env("BG_MAX_TOOL_ROUNDS", "5"))

# Tools visible to the background worker - read-only subset.
# Filter is applied when building the tool list for /api/tick.
BG_TOOLS_ALLOWED: Set[str] = set(
    _env(
        "BG_TOOLS_ALLOWED",
        # mempalace
        "mempalace_search,mempalace_kg_query,mempalace_status,"
        "mempalace_kg_timeline,"
        # external signals (via lazy-mcp)
        "meta_list_servers,meta_search,meta_call,meta_describe_tool,"
        # direct MCP where available
        "alerts_list,alerts_summary,monitor_query,notes_search",
    ).split(",")
)

# ── Background thinking - phases (PR C) ──────────────────────────────────────
# Per-phase model routing. If a phase var is empty, BG_MODEL is used as
# fallback - this preserves backwards compatibility with PR B+.
BG_MODEL_PLANNER: str = _env("BG_MODEL_PLANNER", "")
BG_MODEL_EXECUTOR: str = _env("BG_MODEL_EXECUTOR", "")
BG_MODEL_REFLECTOR: str = _env("BG_MODEL_REFLECTOR", "")
# Reserved for PR D (escalation on repeated failure). Declared now, unused.
BG_MODEL_THINK: str = _env("BG_MODEL_THINK", "")

# Per-phase timeouts (seconds). Executor gets more room for its tool loop.
BG_PHASE_TIMEOUT: int = int(_env("BG_PHASE_TIMEOUT", "600"))
BG_PHASE_TIMEOUT_EXECUTOR: int = int(_env("BG_PHASE_TIMEOUT_EXECUTOR", "1500"))
# Per-phase timeout for exploratory ticks. Exploratory does one deterministic
# seed tool call and one LLM call - no tool loop - so it needs less headroom
# than a normal executor phase.
BG_PHASE_TIMEOUT_EXPLORATORY: int = int(_env("BG_PHASE_TIMEOUT_EXPLORATORY", "600"))
# Whole-tick timeout - everything above should fit inside this.
BG_TICK_TIMEOUT: int = int(_env("BG_TICK_TIMEOUT", "3600"))

# Goal ledger limits.
BG_MAX_ATTEMPTS_PER_GOAL: int = int(_env("BG_MAX_ATTEMPTS_PER_GOAL", "3"))
# Maximum number of goals in status pending or active. blocked/done/cancelled
# do not count. Once reached, Planner cannot create new goals and Executor
# drops any new_goals from its output.
BG_MAX_OPEN_GOALS: int = int(_env("BG_MAX_OPEN_GOALS", "5"))

# Exploratory goals (created by Executor when Planner returns skip).
#  - local executor model only
#  - read-only tools only
#  - one active at a time
#  - TTL: auto-cancelled if not closed within N hours
#  - hard cap per day
BG_EXPLORATORY_PER_DAY: int = int(_env("BG_EXPLORATORY_PER_DAY", "3"))
BG_EXPLORATORY_TTL_HOURS: int = int(_env("BG_EXPLORATORY_TTL_HOURS", "24"))
# Minimum minutes between two exploratory ticks. Prevents back-to-back
# introspections when the planner returns skip repeatedly.
# 0 = no limit.
BG_EXPLORATORY_MIN_INTERVAL_MIN: int = int(
    _env("BG_EXPLORATORY_MIN_INTERVAL_MIN", "90")
)

# Daily budget. 0 = no limit. Token counting relies on Ollama's
# prompt_eval_count + eval_count; OpenAI-compatible backends currently do
# not surface token usage through chatd, so their ticks count as 0.
BG_DAILY_TOKEN_BUDGET: int = int(_env("BG_DAILY_TOKEN_BUDGET", "0"))
BG_DAILY_TIME_BUDGET_SEC: int = int(_env("BG_DAILY_TIME_BUDGET_SEC", "0"))

# Daily cap on new normal goals created by the planner. Prevents the 4B
# model from inventing goals indefinitely when nothing changes.
BG_MAX_NEW_GOALS_PER_DAY: int = int(_env("BG_MAX_NEW_GOALS_PER_DAY", "3"))

# Journal rotation. Rotate journal.jsonl when it exceeds this size (MB).
# 0 disables rotation.
BG_JOURNAL_MAX_MB: int = int(_env("BG_JOURNAL_MAX_MB", "10"))

# How many rotated journal files to keep. 0 = keep all.
BG_JOURNAL_KEEP: int = int(_env("BG_JOURNAL_KEEP", "5"))

# If planner returns skip while open goals exist, end the tick without
# running an exploratory step. Exploratory is for "nothing to do at all",
# not for "couldn't decide which goal to pick". Without this, a pending
# goal the planner won't touch causes endless introspection loops.
BG_SKIP_WITH_OPEN_GOALS_STOPS_TICK: bool = (
    _env("BG_SKIP_WITH_OPEN_GOALS_STOPS_TICK", "true").lower() == "true"
)

# ── Background prompts (layered override) ────────────────────────────────────
# Each background prompt has a bundled default in ./prompts/<name>.txt.
# Installations can override per-host or per-user without touching code:
#
#   1. prompts/<name>.txt                     (bundled default, in repo)
#   2. /etc/chatd/prompts/<name>.txt          (system override)
#   3. ~/.config/chatd/prompts/<name>.txt     (user override)
#   4. CHATD_BG_PROMPT_<NAME> env var         (runtime override)
#
# Later sources override earlier ones. Empty result means "use code default",
# which is the same as the bundled file at the time of release.

def _load_bg_prompt(name: str) -> str:
    paths = [
        Path(__file__).parent / "prompts" / f"{name}.txt",
        Path("/etc/chatd/prompts") / f"{name}.txt",
        Path(os.path.expanduser("~/.config/chatd/prompts")) / f"{name}.txt",
    ]
    for p in paths:
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    return text
            except Exception:
                continue
    env_key = f"CHATD_BG_PROMPT_{name.upper()}"
    return os.environ.get(env_key, "")


BG_PROMPT_PLANNER: str = _load_bg_prompt("planner")
BG_PROMPT_EXECUTOR: str = _load_bg_prompt("executor")
BG_PROMPT_EXPLORATORY: str = _load_bg_prompt("exploratory")
BG_PROMPT_REFLECTOR: str = _load_bg_prompt("reflector")
