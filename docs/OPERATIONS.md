# Operations

Runbooks, backups, incident response. For a single-agent deployment on
one host.

## Paths

| What | Path |
|---|---|
| Code | `/opt/subconscious/` |
| Virtualenv | `/opt/subconscious/venv/` |
| State | `/var/lib/subconscious/` |
| Config | `/etc/subconscious/subconscious.env` |
| Systemd unit | `/etc/systemd/system/subconscious.service` |
| Timer | `/etc/systemd/system/subconscious-tick.timer` |
| Journal (systemd) | `journalctl -u subconscious` |
| Identity | `/var/lib/subconscious/identity.txt` |
| Session logs | `/var/lib/subconscious/history/` |
| RAG | `/var/lib/subconscious/rag.sqlite3` |
| Events | `/var/lib/subconscious/events.sqlite3` |
| BG state | `/var/lib/subconscious/bg/` |

## Common tasks

### Restart after code change

```sh
cd /opt/subconscious
git pull
. venv/bin/activate
pip install -e .
systemctl restart subconscious
journalctl -u subconscious -n 30 --no-pager
```

A restart mid-tick leaves an orphaned `started` entry in
`bg/journal.jsonl`. It is closed at the next startup by
`ledger.close_orphaned_ticks`.

### Trigger a tick manually

```sh
curl -s -X POST http://127.0.0.1:5001/api/tick \
  -H "Authorization: Bearer $SUBCONSCIOUS_BG_TOKEN" \
  -d '{}' | jq .
```

Fires all phases (planner, executor, reflector) as one tick. See
`tail -f /var/lib/subconscious/bg/journal.jsonl` for live output.

### Change the identity

Identity lives in `/var/lib/subconscious/identity.txt`. Edit in place.
Takes effect on next wake-up (new session or after a memory write that
invalidates the cache). To force immediate effect:

```sh
# from chat or /api/chat: trigger any mempalace write tool
# or restart:
systemctl restart subconscious
```

### Reset KG snippet cache

If wake-up shows `777 chars` (identity only, no KG), the KG fetch
timed out or failed. To retry without a full restart, trigger any
memory write from chat (e.g. `mempalace_diary_write` with a test
entry, then `mempalace_delete_drawer`). This invalidates the wake-up
cache; the next request re-fetches KG.

### Inspect the goal ledger

```sh
cat /var/lib/subconscious/bg/goals.json | jq .
cat /var/lib/subconscious/bg/goals.archive.jsonl | tail -5 | jq .
```

From chat, use `subcortex_goals_list` (open goals) or
`subcortex_journal_tail` (recent tick activity).

### Cancel a stuck goal

From chat:

```
Отмени цель g-XXXXXX — причина: <why>
```

Or via internal tool directly.

## Backups

Minimum:

- `/var/lib/subconscious/history/` - dialogue logs (source of truth)
- `/var/lib/subconscious/rag.sqlite3` - RAG index (rebuildable from
  history, but expensive)
- `/var/lib/subconscious/events.sqlite3` - event ledger
- `/var/lib/subconscious/bg/` - goal ledger and journal
- `/var/lib/subconscious/identity.txt` - operator content
- `/etc/subconscious/subconscious.env` - config with secrets

Not backed up (regenerated on start): `__pycache__`, `.pytest_cache`,
`.gunicorn/`, `bg/tick.lock`, `bg/seed_rotation.json`.

MemPalace backup is separate; see MemPalace docs.

## Incident response

### Agent unresponsive

1. `systemctl status subconscious` - is the process alive?
2. `journalctl -u subconscious -n 100 --no-pager` - look for
   `Traceback`, `ERROR`, or the last activity timestamp.
3. `cat /var/lib/subconscious/bg/journal.jsonl | tail -5` - is there
   an open `started` entry?
4. If process alive but stuck: `systemctl restart subconscious`. The
   orphan scan closes the open tick on startup.

### Background worker silent for hours

1. `systemctl status subconscious-tick.timer` - is the timer enabled
   and active?
2. `tail -5 /var/lib/subconscious/bg/journal.jsonl` - when was the
   last tick? If hours ago and timer active, ticks are being skipped.
3. Common cause: `systemctl stop` during deploy, timer not restarted.
   `systemctl enable --now subconscious-tick.timer`.

### Events not being processed

1. `sqlite3 /var/lib/subconscious/events.sqlite3 "SELECT status, COUNT(*) FROM events GROUP BY status;"` - are events stuck in `pending`?
2. `journalctl -u subconscious | grep "event received"` - did the
   notification reach chatd?
3. If events arrive but stay pending: check `process_event` path. If
   they arrive and disappear: check session log for recorded turns.

### Cache hit rate dropped

Look at `[deepseek chat_sync]` log lines. `cache_hit` should be high.
Causes of drop:

- RAG injected into system block (should be last user message).
- System prompt changed between turns (identity edit, KG change).
- History rewritten instead of appended.
- Concurrent sessions sharing cache namespace (should not happen;
  sessions are per-channel).

### Memory (KG) empty in wake-up

See "Reset KG snippet cache" above. If repeated timeouts: check
`journalctl -u subconscious | grep "call_tool.*timed out"`. The
mempalace or lazy-mcp proxy is slow; `KG_SNIPPET_TIMEOUT` (deferred
fix) would cap this.

## Upgrade procedure

1. Backup state (see Backups).
2. `cd /opt/subconscious && git pull`
3. `. venv/bin/activate && pip install -e .`
4. Review migrations if any (there are none currently; schema changes
   are `CREATE TABLE IF NOT EXISTS` or in-place `ALTER TABLE`).
5. `systemctl restart subconscious`
6. Smoke: send a chat message, verify answer and no errors in journal.

Rollback: `git checkout <previous-tag>`, reinstall, restart. State
files are forward-compatible (no downgrades of schema).

## Monitoring

Minimum signals to watch:

- `subconscious.service` active and restart count stable.
- `subconscious-tick.timer` next run within expected interval.
- Recent journal entries without `ERROR` or `Traceback`.
- `events.sqlite3` pending count low (near zero in steady state).
- `bg/goals.json` open count below `BG_MAX_OPEN_GOALS`.
- DeepSeek `cache_hit` ratio above 70% on dialogue.

Nice to have (not implemented):
- Prometheus exporter on `:5001/metrics`.
- Budget gauge (cloud tokens used / daily limit).
- Escalation rate (from memory/hot, once implemented).
