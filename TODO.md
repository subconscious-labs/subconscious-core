# subconscious-core: status and roadmap

Updated: 2026-10-27

## Done

### Infrastructure

- Repository `chatd` -> `subconscious-core`.
- Env prefix `CHATD_*` -> `SUBCONSCIOUS_*` with `_env()` fallback.
- systemd: `subconscious.service`, `subconscious-tick.timer`.
- State moved to `/var/lib/subconscious/`.

### Package layout (2.9a, 2.9b)

- Flat modules moved into `subconscious/{runtime,chat,tools,worker,memory,events,backends}/`.
- `worker/ledger.py` extracted from `worker/tick.py`.
- `tools/registry.py` owns MCP client list; `worker/tick` and
  `tools/internal` no longer import `chat.api`.
- No deferred imports for cycle avoidance; leaf-module rule enforced.
- Guard test: `tests/test_no_legacy_imports.py`.

### Backend normalization

- Single internal shape (flat `tool_calls`, `thinking`, `done_reason`,
  `usage`) across ollama, deepseek, openrouter, onnx.
- DeepSeek cache-hit and cache-miss reported in `usage`.
- Thinking dropped from history before backend call (DeepSeek rejects
  `reasoning_content` on input).
- DSML filter for DeepSeek Flash when tools stripped.
- Documented in `docs/PROTOCOL.md`.

### Events

- `events.sqlite3` with indexed queries.
- Priority scale (critical / interactive / normal / background).
- `event_get`, `event_list`, `event_link` internal tools.
- `session_key` extraction from XMPP payloads (JID, MUC, self-echo
  protection).
- Per-session lock, deduplication by message id.
- Event turns persisted to session log and RAG.

### Identity and memory

- `identity.txt` moved to `/var/lib/subconscious/identity.txt`,
  auto-created with generic default if missing.
- `user_id` propagated through session turn and RAG. Filters by user
  on retrieval; legacy chunks visible during migration.
- KG snippet fallback via `meta_call` when mempalace is behind a proxy.
- KG snippet fetched on wake-up; wake-up grown from 777 (identity only)
  to ~6360 chars (identity + KG).
- Cache invalidation on write tools called directly or via `meta_call`
  (detected by `tool_ref`).

### Background worker (subcortex)

- Planner / executor / reflector with per-phase models.
- Goal ledger with `partials` counter distinct from `attempts`.
- `partial` outcome keeps goal pending; `blocked` after partial cap
  routes through `_apply_failure` and Reflector.
- Self-healing: `close_orphaned_ticks` at startup, closes `started`
  entries older than `BG_TICK_TIMEOUT`.
- Age filter on observations: `BG_OBSERVATION_MAX_AGE_HOURS=24`.
  Planner does not resurrect already-fixed problems.
- Prompt source logged at WARNING level (survives startup before
  `logging.basicConfig`).
- Executor prompt forbids creating `new_goals` to unblock a partial.
- `MAX_TOOL_ROUNDS` reduced from 150 to 12; background tick rounds
  = 3.

### Documentation

- `docs/adr/ADR-001-thinking-loop.md`
- `docs/adr/ADR-002-package-layout.md`
- `docs/adr/ADR-003-memory.md`
- `docs/adr/ADR-004-deferred.md`
- `docs/PROTOCOL.md`
- `docs/OPERATIONS.md`

## Open now

### Immediate

- **`KG_SNIPPET_TIMEOUT`**: cap KG fetch at 10s to prevent 60s stall
  on first event after restart when mempalace is cold. ~20 lines in
  `memory/wakeup.py` + `config.py`.
- **Defensive guard for multimodal content**: `isinstance(content,
  str)` check in `_last_user_text`, `build_model_messages`,
  `ensure_system_prompt`. Prevents crash when Open WebUI sends
  multimodal shape by default. ~10 lines. Include in 2.9c.
- **Stage 2.9c**: split `chat/api.py` into `chat/{api,stream,
  tools_loop,prompt,event}.py` + tests (see
  `tests/README.md` plan).

### Backlog (small, well-scoped)

- `partials=0` on Reflector `retry` (currently only `attempts`
  resets; after retry the goal quickly accumulates partials again).
- Stale-goal detection: pending > N hours without movement → cancel
  or mark stale (ADR-004).
- Capability-gap detection: same missing-capability across partials →
  escalate directly (ADR-004).
- `event_peek` non-blocking queue check (ADR-004).
- `wing_elsa` in canonical wing list, or tighten diary prompt
  (ADR-004).
- `_EVENT_DIRECTIVE` rule for simple pings (ADR-004).
- RAG deduplication: do not inject chunks from current session when
  similarity is very high and the same source is already in history.

### Medium (PR D)

- Assess phase after Executor: verify that a `done` goal was actually
  accomplished.
- `parent_goal_id` for goal lineage, addressing sub-goals-of-sub-goals
  trees.
- Token usage for OpenRouter (code exists, needs live test).
- Minimal A2A: Agent Card + Task mapping.

### Large (PR E, after PR D)

- `memory/hot` MVP: SQLite + sqlite-vec + FTS5 + RRF + forgetting
  + escalate + warmup + rebuild.
- Invariant: forgetting = demotion, never deletion.
- Escalation rate in log from day one.
- ADR for hot-tier design (already drafted as ADR-003).

### Backlog (no date)

- Attachments ingress/egress (ADR-004, PR G).
- A2A blocking wait with correlation_id (ADR-004).
- Argument-level tool gating with shadow-first rollout and one-use
  permits.
- PR F: multi-agent, workspace reservation, full A2A.
- DAG of goals with dependencies.
- CLI & Skills: safe MCP proxy for shell + lazy-load instructions.
- `soothe-nano` for code editing.
- Plugin system.

## When to enable the timer

The `subconscious-tick.timer` is **disabled by default** on dev
installs and during the current debugging cycle. Enable it when:

1. **Before PR D.** Assess evaluates accumulated done goals; without a
   regular tick there is nothing to assess.
2. **Before memory/hot MVP.** Decay and consolidation are tested over
   time; a manual tick cannot validate them.
3. **Before leaving the system unattended.** Autonomous observation
   requires a regular tick.
4. **Before multi-user.** Background observations must run for all
   users, not just when the operator triggers a manual tick.

Not a reason to enable:
- "I want to see exploratory work" (2-3 manual ticks show the same).
- "I want to test cloud budget degrade" (rare event, manual triggers
  find it).
- "I want to test orphan self-healing" (manual: restart mid-tick and
  check log).

## References

- `docs/adr/ADR-001-thinking-loop.md` - architecture.
- `docs/adr/ADR-002-package-layout.md` - module layout.
- `docs/adr/ADR-003-memory.md` - memory tiering decision.
- `docs/adr/ADR-004-deferred.md` - deferred decisions.
- `docs/PROTOCOL.md` - wire formats.
- `docs/OPERATIONS.md` - runbooks.
