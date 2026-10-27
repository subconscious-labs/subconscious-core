#!/usr/bin/env python3
"""Central configuration module.

All runtime settings are read from environment variables so that the project
can be installed anywhere without modifying source files.  Copy .env.example
to .env (or set Environment= in the systemd unit) and adjust as needed.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

log = logging.getLogger("subconscious.runtime.config")
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
            log.warning(
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

# Which MCP servers are allowed to push notifications/event.
# Empty set = none (default). Use "*" to allow all.
MCP_EVENT_ALLOWED: Set[str] = set(
    s.strip() for s in _env("MCP_EVENT_ALLOWED", "").split(",") if s.strip()
)

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
SESSION_DIR: str = _env("SESSION_DIR", "/var/lib/subconscious/sessions")

# ── Identity ──────────────────────────────────────────────────────────────────
# Operator content, not repository content. Auto-created from a generic
# default on first start if missing; existing files are never overwritten.
IDENTITY_PATH: str = _env(
    "IDENTITY_PATH", "/var/lib/subconscious/identity.txt"
)

# ── External RAG store ────────────────────────────────────────────────────────
RAG_ENABLED: bool = _env("RAG_ENABLED", "true").lower() == "true"
RAG_DB_PATH: str = _env("RAG_DB_PATH", "/var/lib/subconscious/rag.sqlite3")
RAG_EMBED_MODEL: str = _env("RAG_EMBED_MODEL", "nomic-embed-text")
RAG_TOP_K: int = int(_env("RAG_TOP_K", "4"))
RAG_MAX_CHARS_PER_CHUNK: int = int(_env("RAG_MAX_CHARS_PER_CHUNK", "4000"))
RAG_MIN_SCORE: float = float(_env("RAG_MIN_SCORE", "0.25"))

# Number of KG triples loaded into the wake-up block (L1). Affects prefill
# on every chat and every background tick.
MEMORY_KG_SNIPPET_LIMIT: int = int(
    _env("MEMORY_KG_SNIPPET_LIMIT", "60")
)

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
EVENT_TOKEN: str = _env("EVENT_TOKEN", "")

# Model used to process incoming events (can differ from chat model).
EVENT_MODEL: str = _env("EVENT_MODEL", "")

# ── Event ledger ──────────────────────────────────────────────────────────────
# SQLite store for queryable event history. Distinct from the append-only
# events.jsonl audit trail (kept for tail -f and postmortems).
EVENTS_DB_PATH: str = _env(
    "EVENTS_DB_PATH", "/var/lib/subconscious/events.sqlite3"
)

# Default user id when no X-OpenWebUI-User-Email header is present.
# Empty means "no user context" - events still get recorded, just without
# a user_id. Set this on single-user installs to enable cross-channel
# identity for non-WebUI sources.
USER_DEFAULT_ID: str = _env("USER_DEFAULT_ID", "")

# -- DeepSeek API --------------------------------------------------------------

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

# ── Cloud model budget + soft degrade ────────────────────────────────────
# Any model whose name starts with one of these prefixes is "cloud" for the
# purpose of the cloud daily token budget.
BG_CLOUD_PREFIXES: Set[str] = set(
    p.strip() for p in _env("BG_CLOUD_PREFIXES", "deepseek/,or/").split(",")
    if p.strip()
)

# Daily cloud token budget. 0 = unlimited. When the limit is reached,
# cloud models are replaced with a local fallback for the rest of the day.
BG_CLOUD_DAILY_TOKEN_BUDGET: int = int(
    _env("BG_CLOUD_DAILY_TOKEN_BUDGET", "0")
)

# Model to use when cloud budget is exhausted. If empty, BG_MODEL is used
# (which should be a local model).
BG_DEGRADE_MODEL: str = _env("BG_DEGRADE_MODEL", "qwen3.5:9b")

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

# ── Truncation limits ─────────────────────────────────────────────────────
# Tool results and text snippets are capped before being fed back to the
# model, to keep prefill bounded on CPU-only hosts. Values are characters,
# not tokens.

# Max size of a tool result fed back to the background executor.
# Anything larger gets truncated with a "... [truncated]" marker.
BG_TOOL_RESULT_MAX_CHARS: int = int(
    _env("BG_TOOL_RESULT_MAX_CHARS", "3000")
)

# Max size of the seed tool result in an exploratory tick. Separate from
# BG_TOOL_RESULT_MAX_CHARS because exploratory does a single LLM call and
# the seed result is all it sees.
BG_EXPLORATORY_SEED_MAX_CHARS: int = int(
    _env("BG_EXPLORATORY_SEED_MAX_CHARS", "4000")
)

# Max size of one observation entry in the planner's recent-observations
# block. 12 observations × 250 chars ≈ 3 KB of prefill per planner call.
BG_OBSERVATION_SNIPPET_CHARS: int = int(
    _env("BG_OBSERVATION_SNIPPET_CHARS", "250")
)

# Max age (hours) for an observation to be considered by the planner.
# Older observations are still in the ledger but not shown to the model:
# they describe a past state and the planner otherwise resurrects
# already-fixed problems as new goals. 0 = no filter.
BG_OBSERVATION_MAX_AGE_HOURS: float = float(
    _env("BG_OBSERVATION_MAX_AGE_HOURS", "24")
)

# How many recent observations to show the planner.
BG_RECENT_OBSERVATIONS_LIMIT: int = int(
    _env("BG_RECENT_OBSERVATIONS_LIMIT", "12")
)

# Max size of a tool result fed back to the chat model. Looser than the
# background limit - the user is waiting and full results are more useful.
# 0 = no cap (not recommended).
CHAT_TOOL_RESULT_MAX_CHARS: int = int(
    _env("CHAT_TOOL_RESULT_MAX_CHARS", "8000")
)

# ── Background prompts (layered override) ────────────────────────────────────
# Each background prompt has a bundled default in ./prompts/<name>.txt.
# Installations can override per-host or per-user without touching code:
#
#   1. prompts/<name>.txt                     (bundled default, in repo)
#   2. /etc/subconscious/prompts/<name>.txt          (system override)
#   3. ~/.config/subconscious/prompts/<name>.txt     (user override)
#   4. SUBCONSCIOUS_BG_PROMPT_<NAME> env var  (runtime override)
#
# Later sources override earlier ones. Empty result means "use code default",
# which is the same as the bundled file at the time of release.

def _load_bg_prompt(name: str) -> str:
    paths = [
        Path(__file__).resolve().parent.parent / "worker" / "prompts" / f"{name}.txt",
        Path("/etc/subconscious/prompts") / f"{name}.txt",
        Path(os.path.expanduser("~/.config/subconscious/prompts")) / f"{name}.txt",
    ]
    for p in paths:
        if p.exists():
            try:
                text = p.read_text(encoding="utf-8").strip()
                if text:
                    log.info("prompt %s loaded from %s (%d chars)", name, p, len(text))
                    return text
            except Exception:
                continue
    env_val = _env(f"BG_PROMPT_{name.upper()}", "")
    if env_val:
        log.info("prompt %s loaded from env (%d chars)", name, len(env_val))
    else:
        log.info("prompt %s using code default", name)
    return env_val


# Wake-up mode for background phases.
#   "none"     - no wake-up block at all (default for planner/reflector)
#   "identity" - identity.txt only, no KG snippet
#   "full"     - full wake-up (identity + KG)
#
# The background worker rarely needs the model's identity/KG context:
# planner sees observations and goals, reflector sees one goal's history.
# Wake-up is ~10k chars per LLM call; stripping it saves 15–20% of
# tokens on every tick.
BG_WAKEUP_MODE: str = _env("BG_WAKEUP_MODE", "none")

# Per-phase overrides. Empty string means "use BG_WAKEUP_MODE".
BG_WAKEUP_MODE_PLANNER: str = _env("BG_WAKEUP_MODE_PLANNER", "")
BG_WAKEUP_MODE_EXECUTOR: str = _env("BG_WAKEUP_MODE_EXECUTOR", "")
BG_WAKEUP_MODE_REFLECTOR: str = _env("BG_WAKEUP_MODE_REFLECTOR", "")
BG_WAKEUP_MODE_EXPLORATORY: str = _env("BG_WAKEUP_MODE_EXPLORATORY", "")

BG_PROMPT_PLANNER: str = _load_bg_prompt("planner")
BG_PROMPT_EXECUTOR: str = _load_bg_prompt("executor")
BG_PROMPT_EXPLORATORY: str = _load_bg_prompt("exploratory")
BG_PROMPT_REFLECTOR: str = _load_bg_prompt("reflector")
