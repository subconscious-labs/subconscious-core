# subconscious-core

Autonomous agent runtime for your infrastructure. One process, one
thinking loop, memory that survives restarts.

**Docs:** [ADR-001](docs/adr/ADR-001-thinking-loop.md) ·
[ADR-002](docs/adr/ADR-002-package-layout.md) ·
[ADR-003](docs/adr/ADR-003-memory.md) ·
[ADR-004](docs/adr/ADR-004-deferred.md) ·
[PROTOCOL](docs/PROTOCOL.md) · [OPERATIONS](docs/OPERATIONS.md)

## What it is

An application, not a framework. One agent that notices events,
investigates (read-only), remembers findings, asks before acting.

Typical deployment:

```
you  ──chat──►  subconscious-core  ──►  LLM (local or cloud)
                       │
                       ├──►  MCP tools  (monitor, alerts, notes, ssh, git…)
                       ├──►  memory     (KG, RAG, sessions, events)
                       └──►  worker     (planner → executor → reflector)
                                             ↑
                                        systemd timer
```

## Four conceptual zones

| Zone | Responsibility | Module |
|---|---|---|
| **runtime** | data, config, budgets, lifecycle | `subconscious/runtime/` |
| **chat** | interfaces outward (HTTP, session) | `subconscious/chat/` |
| **tools** | bindings (LLM backends, MCP client, tool routing) | `subconscious/tools/` |
| **worker** | background thinking (planner, executor, reflector) | `subconscious/worker/` |

Memory and events are shared services:

| Service | Responsibility | Module |
|---|---|---|
| **memory** | wake-up, RAG, hot/cold tiers | `subconscious/memory/` |
| **events** | event ledger and session key extraction | `subconscious/events/` |

The `cortex / thalamus / subcortex` terminology used elsewhere in docs
is conceptual, not navigational. Module paths are functional. See
[ADR-002](docs/adr/ADR-002-package-layout.md).

## Features

- **Chat interface** - NDJSON streaming, tool loop, layered memory
  injection, Open WebUI compatible.
- **Background worker** - planner → executor → reflector, goal ledger
  with attempts and partials, self-healing on restart.
- **Event loop** - one thinking lock per turn; XMPP, email, WebUI,
  alerts all arrive as events with priority and correlation.
- **Layered memory** - identity (L0), wake-up (L1), request-scoped
  RAG sidecar (L1.5). Hot/cold tiering planned (ADR-003).
- **MCP tool dispatch** - discover, route, reconnect, restart servers.
  Push notifications received as events.
- **Multi-backend** - Ollama, DeepSeek, OpenRouter, ONNX. Selected by
  model name prefix.

## Requirements

- Python 3.11+
- An LLM endpoint: Ollama (local), DeepSeek API, or OpenRouter
- Optional: MCP servers for the tools you want the agent to use
- Optional: MemPalace for the cold memory tier

## Quick start

```sh
git clone https://github.com/subconscious-labs/subconscious-core
cd subconscious-core

python3 -m venv venv
. venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
$EDITOR .env
```

Minimum to get a chat running:

```sh
OLLAMA_API=http://127.0.0.1:11434
SUBCONSCIOUS_MCP_HTTP_LAZY_URL=http://127.0.0.1:3333/mcp
SUBCONSCIOUS_MCP_HTTP_LAZY_TOKEN=<token>
```

Run:

```sh
python -m subconscious.chat.api
```

Or via systemd - see `subconscious.service.example`.

## Layout

```
subconscious/
  runtime/       config, defaults, paths
  chat/          Flask routes, session, prompt assembly, streaming
  tools/         MCP client, tool registry, internal tools
  worker/        planner/executor/reflector, goal ledger, prompts
  memory/        wake-up, RAG, hot/cold tiers (stubs)
  events/        event ledger (SQLite)
  backends/      ollama, deepseek, openrouter, onnx
```

## Configuration

Layered lookup: code defaults < `/etc/subconscious/` <
`~/.config/subconscious/` < env vars. Later sources win.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_API` | `http://127.0.0.1:11434` | Ollama endpoint |
| `SUBCONSCIOUS_MCP_HTTP_<NAME>_URL` | *(unset)* | HTTP MCP server |
| `SUBCONSCIOUS_MCP_HTTP_<NAME>_TOKEN` | *(unset)* | Bearer token |
| `SUBCONSCIOUS_MCP_STDIO_<NAME>` | *(unset)* | stdio MCP command |
| `SUBCONSCIOUS_MAX_TOOL_ROUNDS` | `12` | Max tool rounds per chat |
| `SUBCONSCIOUS_MAX_HISTORY_TURNS` | `20` | History window (turns) |
| `SUBCONSCIOUS_RAG_ENABLED` | `true` | Enable L1.5 RAG |
| `SUBCONSCIOUS_BG_ENABLED` | `false` | Enable background worker |
| `SUBCONSCIOUS_BG_MODEL` | `qwen3.5:9b` | Fallback model |
| `SUBCONSCIOUS_BG_MODEL_PLANNER` | *(unset)* | Override planner |
| `SUBCONSCIOUS_BG_MODEL_EXECUTOR` | *(unset)* | Override executor |
| `SUBCONSCIOUS_BG_MODEL_REFLECTOR` | *(unset)* | Override reflector |
| `SUBCONSCIOUS_BG_OBSERVATION_MAX_AGE_HOURS` | `24` | Observation freshness filter |
| `SUBCONSCIOUS_BG_MAX_OPEN_GOALS` | `5` | Open goals cap |
| `SUBCONSCIOUS_BG_CLOUD_DAILY_TOKEN_BUDGET` | `0` | Cloud token limit (0=unlimited) |
| `SUBCONSCIOUS_EVENT_MODEL` | *(unset)* | Model for incoming events |
| `SUBCONSCIOUS_EVENT_TOKEN` | *(unset)* | Bearer for `/api/event` |

Full list: `.env.example`.

**Backward compatibility:** `CHATD_*` env vars continue to work as
fallbacks; deprecation warning on first use.

## Backends

Routing by model prefix:

| Prefix | Backend | Notes |
|---|---|---|
| *(none)* | Ollama | local over HTTP |
| `or/` | OpenRouter | OpenAI-compatible |
| `deepseek/` | DeepSeek | API with prefix caching |
| `onnx/` | ONNX Runtime | embed only |

Add a backend by implementing `BackendProtocol` in
`subconscious/backends/base.py`.

## Memory layers

| Layer | Source | Content | When |
|---|---|---|---|
| L0 | MemPalace | identity | always |
| L1 | MemPalace | KG timeline | always |
| L1.5 | subconscious | RAG recall + (planned) `user_context` | per request |
| L2 | MemPalace | room recall | on topic |
| L3 | MemPalace | deep search | on ask |

L1.5 (RAG) is injected into the **last user message**, not the system
block, to preserve prefix cache. See [PROTOCOL](docs/PROTOCOL.md#cache-preservation-rules).

## Background thinking

Enable and configure in `/etc/subconscious/subconscious.env`:

```sh
SUBCONSCIOUS_BG_ENABLED=true
SUBCONSCIOUS_BG_TOKEN=<random-secret>
SUBCONSCIOUS_BG_MODEL=deepseek/deepseek-flash
SUBCONSCIOUS_BG_MODEL_PLANNER=deepseek/deepseek-flash
SUBCONSCIOUS_BG_MODEL_EXECUTOR=deepseek/deepseek-flash
SUBCONSCIOUS_BG_MODEL_REFLECTOR=deepseek/deepseek-flash
```

Trigger manually:

```sh
curl -X POST http://127.0.0.1:5001/api/tick \
  -H "Authorization: Bearer $SUBCONSCIOUS_BG_TOKEN" \
  -d '{}'
```

Or enable the hourly timer:

```sh
systemctl enable --now subconscious-tick.timer
```

State lives in `/var/lib/subconscious/bg/`:

- `goals.json` - active goals
- `goals.archive.jsonl` - completed and cancelled
- `journal.jsonl` - append-only tick log
- `budget.json` - daily token and time counters
- `tick.lock` - flock, prevents overlapping ticks

See [OPERATIONS](docs/OPERATIONS.md) for runbooks and when to enable
the timer.

## Tests

```sh
pytest -v
```

Notable:

- `tests/test_no_legacy_imports.py` - guards against post-move flat
  module imports.
- `tests/test_background.py` - goal ledger, JSON extraction, plan
  behaviour.
- `tests/test_events.py` - event store, priority, status transitions.
- `tests/test_user_identity.py` - user_id propagation, RAG filter,
  session state.

## Related projects

Part of [`subconscious-labs`](https://github.com/subconscious-labs):

- [lazy-mcp](https://github.com/subconscious-labs/lazy-mcp) - lazy MCP proxy
- [gonnx](https://github.com/subconscious-labs/gonnx) - ONNX runtime
- [tts-api](https://github.com/subconscious-labs/tts-api) - TTS MCP server
- [dav-mcp](https://github.com/subconscious-labs/dav-mcp) - CalDAV/CardDAV MCP
- [time-mcp](https://github.com/subconscious-labs/time-mcp) - time MCP

## Status

Early. Core works; docs describe where it is going. See
[todo.md](todo.md) for current state and roadmap.

## License

MIT
