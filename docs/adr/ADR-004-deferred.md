# ADR-004: Consolidated deferred decisions

Status: Accepted
Date: 2026-10-27
Deciders: Nikita Popov

## Purpose

This ADR collects decisions that were discussed, evaluated, and
deliberately deferred. Recording them here prevents re-litigating the
same questions every time they come up. Each entry explains what was
decided *not to do now* and why.

## Attachments (ingress and egress)

**Status:** deferred until after 2.9c and PR D. Possibly after
memory/hot MVP.

**Ingress.** Users attach files via Open WebUI (multimodal `content`
array), XMPP (XEP-0066 OOB, XEP-0363 HTTP Upload), email (MIME parts),
or MCP events (URL or base64 in payload).

**Current behaviour:** not supported. The code paths that handle
user content (`_last_user_text`, `build_model_messages`,
`ensure_system_prompt`, `rag.retrieve`) assume `content` is a string.
A multimodal payload currently causes `AttributeError` on `.strip()`.
This is a latent crash; see "Defensive guard" below.

**Egress.** Agent produces an artifact (report, downloaded file,
screenshot) and delivers it. Model returns only `attachment_id`; the
transport fetches from storage. WebUI: download endpoint. XMPP:
XEP-0363 with OOB URL. Email: MIME attach.

**Why deferred:**

1. All paths go through `chat/api.py`, which is split by 2.9c.
2. Attachments are a new memory entity; design depends on
   `memory/hot` schema.
3. Vision requires backend routing by model capability (DeepSeek Flash
   does not support images).
4. Nothing is blocked. The agent works without it.

**MVP scope when implemented:**

- Phase 1: text documents (PDF, MD, TXT, CSV). Extract → truncate →
  prepend to user message → RAG-index. Content shape unchanged; the
  model sees a string. ~2-3 days.
- Phase 2: images. Route by model capability; DeepSeek Flash gets an
  explicit "model does not support images" skip.
- Phase 3: audio (whisper), video (last).

**Storage:** `/var/lib/subconscious/attachments/<session>/<uuid>.<ext>`
plus metadata (sha256, mime, source, size). Reference by id, not
inline base64.

**Cache:** strip images from history after the first turn, fetch
on-demand via `read_attachment` tool. Base64 in history is catastrophic
for prefill.

**Defensive guard (to add in 2.9c):** in `_last_user_text`,
`build_model_messages`, `ensure_system_prompt`, check
`isinstance(content, str)`. If list, log warning, extract text-parts,
skip non-text. ~10 lines, prevents crash from Open WebUI which sends
multimodal shape by default even when no files are attached.

## `event_peek` (non-blocking queue check)

**Status:** deferred, low priority.

**Idea:** `event_peek(filters, limit=1)` returns pending events without
consuming them. Semantically distinct from `event_list` (history): peek
asks "what is in the queue right now, before I take my next action".

**Use case:** exploratory mode, before starting investigation X, check
if there is a pending alert about X.

**Why deferred:** current `event_list(status="pending")` covers the
functional need; the peek/consume distinction has no consumer yet.
~20 lines when needed.

## Blocking `wait_event` (rejected)

**Status:** explicitly rejected, not just deferred.

**Idea:** model calls `wait_event(timeout=300)`, blocks until a
matching event arrives, returns it. Requeue non-matching events.

**Rejected because:**

- Holds `TickLock` for the entire wait. Interactive messages wait 5
  minutes. Direct violation of the preemption model in ADR-001.
- Model cannot "wait and think" simultaneously. `wait_event` is a
  `sleep()` with a trigger.
- "Expected or not → requeue" is selective receive. Without persistent
  actor state, requeue loses the reason for the wait.
- The async pattern (`waiting-user` goal state + next tick) already
  solves the same problem and better: lock released, state persisted
  in goal ledger, interactive preemption works, parallel waits possible
  on multiple goals.

**When `wait_event` would make sense:** in a reactive actor model
(BEAM/Erlang style) where each event spawns an actor and pattern
matching is native. We are not there and migrating is not justified.

**Replaced by:**
- `waiting-user` goal state (implemented).
- `event_peek` (deferred, above).
- A2A wait with `correlation_id` (below, deferred until PR D).

## A2A blocking wait with correlation_id

**Status:** deferred until PR D (Agent-to-Agent protocol).

**Idea:** agent sends a task to another agent with a `correlation_id`,
then calls `wait_for_reply(correlation_id, timeout)`. Blocking or
polling. Different semantics from event loop: waiting for a specific
reply, not any event.

**Why deferred:** A2A is not implemented yet. When it is, `wait_*`
becomes a natural primitive inside the A2A client, not an event loop
tool. The distinction matters: event loop is "any relevant input";
A2A is "this specific reply".

## Assess phase (in PR D)

**Status:** planned for PR D, not deferred.

**Scope:** after Executor returns `done`, a separate Assess phase
evaluates whether the goal was actually accomplished. Closes the
StrangeLoop: currently Executor claims done, nobody verifies.

**Related:** `parent_goal_id` for goal lineage, addressing the
sub-goals-of-sub-goals tree observed on 2026-10-27.

## Token usage for OpenRouter

**Status:** deferred, blocked on access.

The code already extracts `usage` from `chat_sync` (added 2026-10-26).
Cannot be verified without live `or/` traffic through the backend. Kept
in backlog until access to test.

## Stale-goal detection

**Status:** deferred, ~15 lines when needed.

**Problem:** `g-6e883d` and `g-98354f` on 2026-10-27 sat pending for
two hours because the planner never picked them. Without a planner
pick, `partials` does not grow, `blocked` does not trigger, Reflector
does not fire. The goal is unreachable by any state-transition path.

**Idea:** TTL for normal goals (analogous to `BG_EXPLORATORY_TTL_HOURS`).
Pending > N hours without `updated_at` change → cancel or mark stale.

**Why deferred:** the immediate case was resolved by manual cancel.
The mechanism is worth adding but not before it repeats with a real
goal.

## Capability-gap detection

**Status:** deferred, design uncertain.

**Problem:** executor observes "I need X but X is not in the tool list"
and marks partial. The same observation repeats on the next tick.

**Idea:** if the same missing-capability appears across partials of
the same goal, escalate directly rather than waiting for
`BG_MAX_ATTEMPTS_PER_GOAL`. Parsing observation is fragile; a more
reliable signal might be a separate field in the executor JSON schema.

**Why deferred:** requires prompt and schema change. Wait until a
second real case appears.

## `wing_elsa` (canonical wing list)

**Status:** open question, small.

On 2026-10-27 the agent wrote a diary entry with `wing="wing_elsa"`
without operator approval. The identity document (section "Wings —
canonical set") lists the allowed wings and does not include
`wing_elsa`.

Two options:

- Add `wing_elsa` to canonical list (if intended).
- Tighten the diary prompt to require an existing wing and refuse
  otherwise.

Not urgent. On production diary writes the agent will keep inventing
wings until fixed.

## `_EVENT_DIRECTIVE` refinement

**Status:** observation from 2026-10-27, might reoccur.

A simple "Проверка xmpp" message triggered 25 tool-loop rounds
before `MAX_TOOL_ROUNDS` was reduced from 150 to 12. With 12 rounds
the same message takes 3 rounds. The directive still says "do NOT just
describe what you would do - actually call the tools", which pushes
toward action even for pings. If the round limit is ever raised again,
or if a future model is less obedient, the over-investigation pattern
returns.

**Possible fix:** add to the directive: "plain human pings, test
messages, and acknowledgments get a one-line reply and stop. Concrete
triggers (alert, error, named task) still drive the full loop."

**Why deferred:** current behaviour under `MAX_TOOL_ROUNDS=12` is
correct. Change the prompt only if the pattern reappears.

## References

- `docs/adr/ADR-001-thinking-loop.md`
- `docs/adr/ADR-002-package-layout.md`
- `docs/adr/ADR-003-memory.md`
