# ADR-003: Memory tiering

Status: Accepted
Date: 2026-10-27
Deciders: Nikita Popov

## Context

Two candidate memory engines were evaluated for the hot tier:

- **MemPalace** (existing cold tier): full palace with wings, rooms,
  halls, tunnels, MCP server with 29 tools, drawers as markdown files,
  KG in SQLite. Slow but comprehensive.
- **Mnémlet** (external project, considered as alternative): self-hosted
  memory engine, exponential decay with frequency weighting, nightly
  consolidation, hybrid BM25 + vector via ChromaDB, 16 MCP tools,
  ~450 MB RAM baseline. Design that explicitly forgets.

A third option was evaluated and chosen: **write our own hot tier**.

## Decision

### Two tiers, one invariant

**Cold tier: MemPalace.** Full raw archive, source of truth for audit.
No new data is written here by the hot tier; it is read from.

**Hot tier: built in-house** under `subconscious/memory/hot/`. Curated
working set of what is "in front of the agent's eyes" right now.
Rebuildable from cold tier at any time.

**Invariant: forgetting = demotion, never deletion.** Before removing
anything from hot, verify it exists in cold. If it doesn't, it is not
removed. This single rule protects against silent data loss, which is
the worst possible failure mode for agent memory.

### Rejected alternatives

**Mnémlet as hot tier.** Considered and rejected on 2026-10-27. Reasons:

1. **Different semantics.** Mnémlet owns forgetting and consolidation.
   Adopting it means accepting its policy as opaque. For a single-agent
   deployment with a known ontology, an opaque policy is a liability,
   not a feature.
2. **Heavy footprint.** ChromaDB and a separate server process, ~450 MB
   RAM. Our hot tier fits in tens of MB (SQLite + sqlite-vec).
3. **Deployment mismatch.** Mnémlet SDK has no auth; production use
   requires REST or MCP with `X-Mnemlet-Key`. Either the hot tier
   becomes a REST hop (not hot), or the deployment runs without auth
   (not acceptable).
4. **Integration cost.** Adapter between Mnémlet schema and MemPalace
   schema, ongoing tracking of upstream changes. Comparable to writing
   our own MVP but with less control.

**Mnémlet's design ideas kept.** Two patterns from Mnémlet informed the
design of our hot tier:
- Forgetting as a first-class operation, not TTL-based eviction.
- Every removal has a receipt (audit log of why something was demoted).

### Hot tier MVP scope

Deliberately small. Phase 1 (implementation PR):

- **Storage:** SQLite + `sqlite-vec` for vectors, FTS5 for keyword.
- **Retrieval:** hybrid BM25 + vector via Reciprocal Rank Fusion (RRF).
- **Scoring:** recency × relevance × pin. Frequency weighting deferred.
- **Eviction:** score-based with pins excluded. Pins are identity,
  active goals, and operator-marked items.
- **Consolidation:** deduplication only. Clustering deferred.
- **Escalation:** on miss or explicit deep-intent, read through to
  MemPalace; cache the result in hot.
- **Warmup:** prefetch from MemPalace at session start.
- **Rebuild:** `rebuild()` reconstructs hot from cold. Doubles as
  invariant test.

### Deferred from MVP

- Frequency weighting in decay.
- Nightly consolidation beyond deduplication.
- LLM-based extraction or summarisation.
- Clustering.
- LLM-generated morning briefing.
- MCP wrapper around the hot tier (internal tools suffice).

## Consequences

### Positive

- Full control over the forgetting policy. Tuned for one scenario, not
  general case.
- Fits in the same process. No separate server, no MCP hop on the hot
  path.
- Invariant enforced by code (`escalate` must succeed before `demote`
  proceeds).
- If hot tier fails or is rebuilt, cold tier is intact. The failure
  mode is degraded context, not data loss.

### Negative

- We maintain our own memory engine.
- No community contribution. Bug fixes come from us.
- No upstream improvements in retrieval algorithms.

### Mitigation

- MVP is small (~500 lines estimated), well within maintenance budget
  for a single-agent project.
- Fallback: if MVP exceeds 2 weeks or quality is unacceptable, Mnémlet
  can be adopted as hot tier without touching cold tier. Tests and
  contract designed during MVP remain useful.

## Cache considerations

- Hot tier data enters the prompt via `user_context` in the **last
  user message** (like RAG), never in the stable prefix.
- `wake_up()` (L0+L1) is cached per session, invalidated on memory
  writes.
- `user_context` is bounded by char count (500-1000), not by token
  budget; it must not crowd the dialogue.

## Metrics

From day one of the hot tier, log:

- **escalation_rate**: fraction of queries where hot missed and cold
  was consulted. Target: below 20% for steady-state work.
- **eviction_reason**: recency vs relevance vs capacity vs pin.
- **consolidation_count**: how many items per cycle, how many merged.
- **false_hit_rate**: hot returned a result but it was not relevant.
  Measured via operator feedback and query similarity analysis.
- **rebuild_duration**: rebuild from cold, seconds. Should stay under
  the time to process one dialogue turn.

The escalation rate is the single most important number: it tells you
whether the "95% in front of the eyes" hypothesis is true for your
actual workload.

## References

- `subconscious/memory/hot/README.md` (to be created in MVP PR)
- `docs/adr/ADR-001-thinking-loop.md`
- `docs/adr/ADR-004-deferred.md`
