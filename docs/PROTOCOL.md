# Protocol

Wire formats for LLM backends and event transport. This document is
normative: code must match it. When code and document disagree, one of
them is a bug.

## Normalized backend shape

All backends (`ollama`, `deepseek`, `openrouter`, `onnx`) translate to
and from a single internal shape. Callers see only the internal shape.

### Chat chunk (streaming)

```json
{
  "model": "deepseek/deepseek-flash",
  "created_at": "2026-10-27T14:18:16.869Z",
  "message": {
    "role": "assistant",
    "content": "text delta",
    "thinking": "reasoning delta",
    "tool_calls": [
      {"id": "call_abc", "name": "meta_search", "arguments": {"query": "..."}}
    ]
  },
  "done": false
}
```

Rules:

- `message.thinking` is optional, present when the backend produces
  reasoning. Omitting it (empty string) is equivalent to absent.
- `message.tool_calls` is a **flat** array. Not OpenAI's nested
  `{function: {name, arguments}}` form. Backends translate.
- `arguments` is always a parsed object on output, never a JSON string.
- `content` is text, never a list. Multimodal content is deferred
  (ADR-004).

### Chat done chunk

```json
{
  "model": "deepseek/deepseek-flash",
  "created_at": "2026-10-27T14:18:33.123Z",
  "message": {"role": "assistant", "content": ""},
  "done": true,
  "done_reason": "stop",
  "usage": {
    "prompt_tokens": 16663,
    "completion_tokens": 234,
    "reasoning_tokens": 0,
    "cache_hit_tokens": 16512,
    "cache_miss_tokens": 151
  }
}
```

`done_reason` values:

- `stop` - normal completion.
- `length` - hit `num_predict` limit.
- `tool_calls` - model wants to call tools; caller must run them and
  continue.
- `error` - backend reported error; message content is diagnostic.

`usage` is optional per-backend but all backends fill it when
available. `cache_hit_tokens` and `cache_miss_tokens` are
backend-specific (currently DeepSeek) and may be zero elsewhere.

### Chat sync response

Same shape as a single done chunk with populated `content`,
`thinking`, and `tool_calls`. No streaming intermediate.

## Event transport

### MCP notification

MCP servers push events via `notifications/message`:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/message",
  "params": {
    "level": "info",
    "logger": "event",
    "data": {
      "type": "xmpp_message",
      "source": "xmpp",
      "payload": {"text": "npopov@polyserv.xyz: hello"}
    }
  }
}
```

The `logger` field MUST be `"event"` for the notification to be
treated as an application event. Anything else is ignored (MCP spec
allows arbitrary logging).

`data.payload` may be a dict or a bare string. If a bare string, it is
preserved as `{"text": "<string>"}` for downstream processing.

### HTTP event ingress

`POST /api/event`:

```json
{
  "type": "alert",
  "source": "monitoring",
  "payload": {"severity": "critical", "service": "service1"},
  "user_id": "npopov@polyserv.xyz",
  "model": "deepseek/deepseek-flash",
  "options": {"num_predict": 1024}
}
```

Fields `model`, `options`, `user_id` are optional. Missing values
fall back to `EVENT_MODEL` and header extraction.

### Event ledger (SQLite)

Schema (see `subconscious/events/store.py`):

```
events(
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  source TEXT NOT NULL,
  user_id TEXT,
  event_type TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 2,
  status TEXT NOT NULL DEFAULT 'pending',
  goal_id TEXT,
  correlation_id TEXT,
  payload TEXT NOT NULL,   -- JSON
  processed_at TEXT
)
```

Indexes on `(status, priority, ts)`, `goal_id`, `correlation_id`,
`user_id`.

Priority constants: 0 critical, 1 interactive, 2 normal, 3 background.

Status transitions: `pending → processing → processed | linked |
skipped`. `linked` sets `goal_id`.

## Session storage

Per-channel JSONL. One file per `session_key`
(`xmpp:dm:npopov@...`, `webui:<chat-id>`, `email:<thread-id>`).

```
{"type": "human", "message": {"role": "human", "content": "..."}, "user_id": "npopov@..."}
{"type": "assistant", "message": {"role": "assistant", "content": "..."}}
```

Lines appended, never rewritten. `MAX_HISTORY_TURNS` bounds what is
loaded into the prompt, not what is stored.

## RAG store (SQLite)

```
rag_chunks(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,     -- e.g. "session:xmpp:dm:npopov@..."
  chunk TEXT NOT NULL,
  embedding TEXT NOT NULL,  -- JSON float array
  added_at TEXT NOT NULL,
  user_id TEXT              -- nullable, legacy rows
)
```

Retrieval filters by `user_id` when provided: exact match OR `NULL`
(legacy, pre-migration). Without `user_id`, no filter (pre-migration
behaviour).

## DSML filter

DeepSeek Flash emits DSML tool-call markup as plain text when tools
were stripped from the payload. Marker:

```
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="foo">
...
</｜｜DSML｜｜ calls>
```

The filter buffers from the first `｜` until newline; drops lines
containing the marker; flushes buffered content at stream end. Buffer
cap 8192 chars protects against a stray `｜` in user text stalling
the stream.

## Thinking channel

Reasoning is delivered via `message.thinking` on each chunk. It is:

- **Displayed** to the client, one chunk at a time.
- **Dropped** from history before the next backend call. Reasoning is
  display-only; re-feeding it bloats context, and DeepSeek rejects
  `reasoning_content` on input in some API versions.
- **Never** used for tool-call logic. Tool calls come from
  `message.tool_calls`.

## Cache preservation rules

Three invariants keep DeepSeek prefix cache hit rate high:

1. **System prompt stable per session.** `L0+L1` (identity + KG) is
   assembled once per session and cached. Invalidate on memory writes.
2. **RAG in last user message.** Per-request context (RAG recall,
   future `user_context`) is appended to the last user message, not
   to the system block. The stable prefix (`system + history`) never
   changes between consecutive turns.
3. **History append-only.** Turns are appended, never rewritten.
   `build_model_messages` trims to `MAX_HISTORY_TURNS` but does not
   modify entries.

Violations of these rules collapse cache hit to zero. Observed drop
from 99% to 43% when RAG was injected into the system block (fixed
2026-10-26).

## Error handling

Backends must **not** raise on transient errors during streaming.
Instead, yield an error chunk with `done_reason: "error"` and
diagnostic content. Callers log and continue. Exceptions are reserved
for programming errors.

Retry/reconnect: `MCPClient` handles reconnect with exponential
backoff. LLM backends do not retry; a failed call surfaces as an
error chunk and the caller decides.

## References

- `backends/base.py` - `Backend` protocol docstring.
- `docs/adr/ADR-001-thinking-loop.md`
