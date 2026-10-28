# ADR-001: Thinking loop

Status: Accepted
Date: 2026-10-27
Deciders: Nikita Popov

## Context

Events arrive from multiple sources: WebUI chat, XMPP, email, alerts,
background ticks. Before this ADR, each event ran in its own daemon
thread. Consequences observed in production:

- N parallel thinking loops. Two events from the same XMPP sender could
  interleave answers.
- No context carried across events. The agent could not say "I asked
  you about this five minutes ago".
- No awareness of what the agent is currently doing. The planner
  re-derived goals from stale observations.
- Cache invalidation on every concurrent event. DeepSeek prefix cache
  dropped to near zero during bursts.

## Mental model

The agent has one thought at a time. Events queue. Between thoughts the
agent decides: continue with a related event, or wait for the next one.
Memory is bounded; archive is unlimited.

This is not an actor model. It is deliberately simpler: one long-lived
process, one lock, one queue, one loop. Persistence and observability
are more valuable than concurrency for a single-agent deployment.

## Five conceptual axes

Independent. Conflating them caused every bug that motivated this ADR.

1. **Person** (`user_id`) - who, crosses channels.
2. **Channel** (`session_key`) - where, per-channel dialogue history.
3. **Event** (`event_id`) - what arrived.
4. **Goal** (`goal_id`) - what the agent is doing about it.
5. **Tool scope** (`persona`) - what the agent can do.

Conceptual only. Implementation reuses existing entities:

- Channel -> session (exists)
- Goal -> goal ledger (exists, extended with `partials`, `correlation_id`)
- Event -> `events.sqlite3` (new)
- Person -> `user_id` on session turns and events (new)
- Tool scope -> persona filter (deferred, not required for single-user)

## Decisions

### D1. Every input is an event

WebUI message, XMPP message, email, alert, timer - same struct:

```
{event_id, ts, source, user_id, event_type, priority, status,
 goal_id, correlation_id, payload, processed_at}
```

Priority scale (smaller = more urgent): 0 critical, 1 interactive,
2 normal, 3 background.

### D2. One thinking lock, held per turn

Not per tick. Between turns the next event can enter. Interactive
channels preempt background by arriving in the queue with higher
priority; the background executor finishes its current turn, not its
whole tick.

### D3. Event dispatch policy

- Interactive (webui, xmpp, email): immediate.
- Critical: immediate, ahead of interactive.
- Normal / background (alerts, monitor, housekeeping): queued for the
  next tick.
- Critical never interrupts mid-LLM. Between tool rounds only, via a
  `should_interrupt()` callback that returns True when a critical event
  is pending. Currently not implemented; critical events are handled
  next turn.

### D4. Relevance between turns: rules only

correlation_id match -> user_id match with waiting-user goal ->
user_id match -> other. No embeddings. Embedding-based selection is
deferred; rules are deterministic, zero-latency, and adequate for a
single-user deployment.

### D5. Goal state machine (extended)

```
pending -> active -> done | failed | blocked | cancelled
                   -> waiting-user -> active | cancelled
```

New fields: `correlation_id` (optional, from event payload),
`partials` (count of partial completions, distinct from `attempts`).

`partial` is an executor self-assessment, not a goal status. A partial
outcome keeps the goal `pending` and increments `partials`. After
`BG_MAX_ATTEMPTS_PER_GOAL` partials the goal becomes `blocked`, and
Reflector fires.

### D6. `user_context` (planned, not implemented)

Bounded block (500-1000 chars) injected into the last user message,
like RAG. Contents: open goals, waiting-user flags, recent topics.
Cache-friendly by construction - never inserted into the stable
prefix. Deferred until `memory/hot` MVP exists.

### D7. Storage layout

| Data | Format | Reason |
|---|---|---|
| `journal.jsonl` | JSONL | append-only tick log |
| `<channel>.jsonl` | JSONL | append-only dialogue |
| `goals.archive.jsonl` | JSONL | append-only archive |
| `goals.json` | JSON atomic | bounded (< 10 open), infrequent rewrite |
| `budget.json` | JSON atomic | 1 write per day |
| `rag.sqlite3` | SQLite | vectors, similarity search |
| `events.sqlite3` | SQLite | mutable status, indexed filters |

Rule: append-only -> JSONL. Mutable state -> SQLite3 (stdlib, no ORM,
no migration framework). One new file, not a new service.

### D8. Rotation: bounded visibility, unlimited archive

- Dialogue: `MAX_HISTORY_TURNS` in prompt, file untouched.
- `events.sqlite3`: no rotation yet. SQLite scales to millions of rows;
  add archival when the table actually grows.
- Goals: done/cancelled archived by `write_goals`. Add TTL for resolved
  as future work.
- `user_context` (when implemented): hard char limit.

The invariant: **we lose visibility, never data**. Everything removed
from the prompt is retrievable (RAG similarity, goal ledger by id,
archive by grep).

### D9. Cache preservation

RAG, `user_context`, and event payload live in the last user message.
Stable prefix (system + history) never changes per-turn unless the
channel itself changes. DeepSeek prefix cache hit measured at 90-99%
on dialogue, ~70% on tool loops.

### D10. Deferred background

When a tick is blocked by the interactive lock, write a marker file
(`bg.tick.pending`). The interactive turn checks the marker at turn
end, runs the tick immediately after releasing the lock, removes the
marker. Not implemented yet - current behaviour is "next scheduled
tick picks it up".

### D11. KISS / UNIX constraints

- No new service. Same process, same loop.
- No database server. SQLite3 is stdlib.
- No abstraction until the second concrete use case.
- Each step is independently testable and independently revertable.
- Reuse existing primitives: `TickLock`, journal rotation, session log,
  goal ledger.

## Consequences

### Positive

- Deterministic ordering. Events from one channel never interleave.
- Preemption works: interactive messages do not wait for a long
  background tick.
- Persistence across restart: the orphan scan closes `started` entries
  older than `BG_TICK_TIMEOUT` at process start.
- Cache hit rate stays high because dialogue is per-channel and
  `user_context` (when implemented) is injected per-request, not into
  the prefix.

### Negative

- No parallelism. A single slow tool call blocks the whole loop.
- A stuck LLM call blocks all events until `BG_TICK_TIMEOUT`.
- Selective receive (waiting for a specific event while letting others
  through) is not supported by design. The async pattern (`waiting-user`
  state + next tick) covers the same use case.

## Deferred (not decided here)

- Resume hint explicit field (read `goal.history` instead).
- Embedding-based relevance between turns (rules first).
- Mid-LLM interrupt (between tool rounds only).
- Tool scope per persona beyond existing filter.
- Multi-user tool isolation (MCP per credential).
- Attachments ingress and egress (see ADR-004).
- A2A blocking wait with correlation_id (see ADR-004).
- `event_peek` non-blocking queue check (see ADR-004).

## References

- `docs/PROTOCOL.md` - normalized backend shape, event types, DSML.
- `docs/adr/ADR-002-package-layout.md` - module layout implementing
  this ADR.
- `docs/adr/ADR-003-memory.md` - memory tiering (hot / cold).
- `docs/adr/ADR-004-deferred.md` - consolidated deferred decisions.
