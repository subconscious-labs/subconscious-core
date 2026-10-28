# ADR-002: Package layout

Status: Accepted
Date: 2026-10-27
Deciders: Nikita Popov

## Context

Modules lived flat in the repository root: `chatd.py`, `background.py`,
`memory.py`, `rag.py`, `session.py`, `mcp_client.py`, `mcp_registry.py`,
`internal_tools.py`, `config.py`, plus a `backends/` package. Total
~5000 lines in `chatd.py` alone.

Consequences:
- Cyclic imports between `background.py` and `chatd.py`, worked around
  with deferred imports inside functions.
- Untestable `chatd.py`: it composes HTTP routes, prompt assembly,
  tool dispatch, session state, and streaming in one file.
- Any new feature fights the file size.
- Operator cannot infer from a filename what it does; `chatd` is a
  historical name from the original Hollama-backed design.

## Decision

Move everything into a single `subconscious/` package, named by
function not by metaphor.

```
subconscious/
  runtime/     config, lifecycle, budget
  chat/        HTTP interface, session, prompt assembly
  tools/       MCP client, registry, internal tools
  worker/      tick orchestration, prompts, goal ledger
  memory/      wake-up, RAG, and (stub) hot/cold tiers
  events/      event ledger (SQLite)
  backends/    LLM backends (ollama, deepseek, openrouter, onnx)
```

Conceptual terms from the project's language (`cortex`, `thalamus`,
`subcortex`) remain in documentation and in README as explanation.
They are not module paths. Anybody reading `subconscious/tools/` knows
it dispatches tools; nobody would guess that `thalamus/` is the tool
bus without a map.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `runtime/config.py` | environment, defaults, path resolution |
| `chat/api.py` | Flask routes, chat/generate/event/tick handlers |
| `chat/session.py` | per-chat session state, prompt cache, chatlog |
| `tools/mcp_client.py` | MCP transport (stdio, HTTP), reconnect, notifications |
| `tools/registry.py` | tool schemas + client instances (single owner) |
| `tools/internal.py` | in-process tools (goals, events, mcp control) |
| `worker/tick.py` | planner, executor, reflector, LLM primitive |
| `worker/ledger.py` | goal ledger, budget, journal, tick lock |
| `worker/prompts/*.txt` | per-phase prompts (operator-overridable) |
| `memory/wakeup.py` | L0+L1 wake-up assembly, KG snippet |
| `memory/rag.py` | L1.5 RAG store (SQLite) |
| `memory/hot/` | (stub) curated working set, local embeddings |
| `memory/cold/` | (stub) MemPalace adapter |
| `events/store.py` | event ledger (SQLite) |
| `backends/*.py` | LLM backends |

### Leaf-module rule

`runtime/config.py`, `tools/registry.py`, `worker/ledger.py`,
`events/store.py`, and `memory/rag.py` are **leaves**: no imports from
other project modules except `runtime.config`. This preserves the
dependency DAG:

```
chat/api       worker/tick       memory/wakeup
    |               |                 |
    +-------+-------+---------+-------+
            |                 |
        tools/registry    worker/ledger
            |                 |
            +----+------------+
                 |
            events/store
                 |
          runtime/config
```

Cycles are structurally impossible. No deferred imports for cycle
avoidance; deferred imports are allowed only for laziness (backend
initialization, optional dependencies) and must be commented as such.

## Consequences

### Positive

- One file, one responsibility. Tests can target each module.
- New features (assess phase, memory/hot) add files, not lines to a
  monolith.
- README can describe the four conceptual zones and map them to
  actual module paths.
- No more `import chatd` from `background`.

### Negative

- The move broke every existing import once. Two deferred imports were
  missed in the first pass (`backends/onnx.py`, `tools/mcp_client.py`)
  and only surfaced in production. Mitigation: see
  `tests/test_no_legacy_imports.py`.

## Migration

Executed in two commits:

1. **Move + import rewrite.** `git mv`, no logic changes. Systemd unit
   updated to `subconscious.chat.api:app`.
2. **Cycle break.** Extract `worker/ledger.py` from `worker/tick.py`,
   move MCP client ownership to `tools/registry.py`. `worker/tick`
   imports `tools.registry` and `tools.internal`, never `chat.api`.

## Guard

`tests/test_no_legacy_imports.py` greps for any
`^\s*(from|import) (config|backends|mcp_client|mcp_registry|internal_tools|background|memory|rag|session)`
anywhere under `subconscious/`. This catches the whole class of bug
introduced by the move, whether top-level or deferred.

## References

- `docs/adr/ADR-001-thinking-loop.md`
- `tests/test_no_legacy_imports.py`
