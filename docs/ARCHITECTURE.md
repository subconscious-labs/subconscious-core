# subconscious - архитектура

## Четыре зоны

Один Python-пакет `subconscious`, четыре подпакета. Каждый - одна
ответственность.

### core/ - данные и состояние

Что хранится, где, в каком формате. Не логика.

- `memory/` - KG (SQLite), vector store (sqlite-vec), identity, layers.
- `rag.py` - RAG-sidecar (L1.5).
- `session.py` - сессии чата (JSONL chatlog + JSON markers).
- `ledger.py` - goals.json + goals.archive.jsonl + journal.jsonl.
- `budget.py` - budget.json + new_goals_counter.json.
- `config.py` - layered конфиг (env > file > default).

### cortex/ - интерфейсы

Кто разговаривает с внешним миром. Всё, что видит пользователь или
другой агент.

- `api.py` - Flask app, `/api/chat`, `/api/tick`, `/api/event`.
- `a2a.py` - Agent Card, Task mapping (PR D).
- `cli.py` - команды `subconscious init`, `subconscious tick`, `subconscious status`.
- `event.py` - обработчик входящих событий.

**Чат - это cortex, не ядро.** Чат использует core, thalamus, subcortex,
но сам по себе не суть проекта.

### thalamus/ - связки

Роутер между нами и внешним миром.

- `backends/` - Ollama, OpenRouter, DeepSeek, ONNX.
- `mcp/client.py` - stdio-клиент к MCP-серверам.
- `mcp/router.py` - namespacing, фильтры, категории инструментов.
- `mcp/config.py` - загрузчик `mcp_servers.yaml`.
- `tools.py` - фильтр «что видит chat / executor / exploratory».

### subcortex/ - фоновое мышление

Главная фича. То, чего нет в других chatd-подобных проектах.

- `tick.py` - orchestration одного тика.
- `planner.py` - выбор цели / создание / skip.
- `executor.py` - tool loop выполнения цели.
- `reflector.py` - разбор blocked-целей.
- `assessor.py` - оценка done-целей (PR D).
- `prompts/` - planner.txt, executor.txt, exploratory.txt, reflector.txt.

## Поток данных

```
systemd timer
  → HTTP POST /api/tick
    → cortex/api.py
      → subcortex/tick.py
        → subcortex/planner.py  → core/ledger, core/memory, thalamus/backends
        → subcortex/executor.py → thalamus/mcp, thalamus/backends
        → subcortex/reflector.py → thalamus/backends
        → subcortex/assessor.py → thalamus/backends (PR D)
        → core/ledger (persist)
        → core/budget (accounting)
```

```
пользователь
  → HTTP POST /api/chat
    → cortex/api.py
      → core/memory.wake_up()
      → core/rag.retrieve()
      → thalamus/backends (streaming)
      → thalamus/mcp (tool calls)
      → core/session (persist)
```

## Зависимости (направление)

```
cortex    →  core, subcortex, thalamus
subcortex →  core, thalamus
thalamus  →  core (только конфиг)
core      →  ничего внутри проекта
```

**Никаких обратных зависимостей.** core не знает про subcortex. thalamus
не знает про cortex. Если появляется обратная - это ошибка дизайна.

## State

Всё мутабельное - в `$SUBCONSCIOUS_STATE_DIR` (по умолчанию
`~/.local/share/subconscious/`):

- `bg/` - цели, journal, budget, seed_rotation, new_goals_counter.
- `sessions/` - JSONL чатлоги.
- `rag.sqlite3` - RAG-индекс.
- `kg.sqlite3` - свой KG (PR E).
- `vectors.sqlite3` - свой vector store (PR E).

**Никогда** в `/opt/<svc>/`, никогда в home сервиса.
```

## Filesystem layout

Code and state are separated.

```
/opt/subconscious/          ← code and venv (read-only for the service)
├── venv/
├── subconscious/           ← Python package
├── prompts/
├── docs/
├── tests/
├── pyproject.toml
└── ...

/etc/subconscious/          ← configuration (not touched on upgrade)
├── subconscious.env        ← env vars
├── tool_descriptions.ini   ← tool overrides
└── prompts/                ← prompt overrides

/var/lib/subconscious/      ← mutable state
├── bg/                     ← background worker state
├── sessions/               ← chat session logs
├── rag.sqlite3             ← RAG index
├── kg.sqlite3              ← own KG (PR E)
└── vectors.sqlite3         ← own vector store (PR E)

/var/lib/mempalace/         ← mempalace palace (external)

/run/subconscious/          ← runtime (tmpfs, cleared on boot)
├── subconscious.sock       ← UDS (future)
└── subconscious.pid

/usr/local/bin/
├── subconscious            ← CLI wrapper
└── subconscious-tick       ← tick wrapper
```

**Principles:**

1. **Code in `/opt`, state in `/var/lib`.** Upgrading code never touches
   data; backing up data never drags code.
2. **Config in `/etc`.** Not touched on upgrade. Edit here, not in repo.
3. **No `.local/share` inside `/opt`.** State belongs in `/var/lib`.
4. **HOME = `/var/lib/subconscious`.** HOME is state, not code.
5. **Runtime in `/run`.** UDS, pid, ephemeral locks.
6. **Logs in journald.** No `/var/log/subconscious/`.

### MCP transports

Two transports are supported, chosen by env prefix:

- `SUBCONSCIOUS_MCP_STDIO_<NAME>=<command>` - spawn the MCP server as a
  subprocess, talk over stdin/stdout. This is the default for local tools.
- `SUBCONSCIOUS_MCP_HTTP_<NAME>_URL=<url>` (+ optional `_TOKEN=<bearer>`) -
  connect to a remote MCP server over HTTP. Used for services that already
  run as daemons (e.g. mempalace in Docker).
