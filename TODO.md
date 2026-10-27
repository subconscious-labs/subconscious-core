# subconscious: что сделано и что осталось

## ✅ Сделано и проверено

### Инфраструктура и переименование (Stage 1)
- Репозиторий переименован `chatd` → `subconscious-core`.
- Env-префикс `CHATD_*` → `SUBCONSCIOUS_*` с fallback через `_env()` в `config.py`.
- Systemd: `subconscious.service`, `subconscious-tick.timer` работают.
- State переехал: `/opt/chatd/.local/share/chatd/` → `/var/lib/subconscious/`.

### Stage 2 (1–7)
- **2.1–2.4:** FS-раскладка, пользователь `subconscious`, перенос state, конфиг.
- **2.5:** `mcp_client.py` умеет HTTP-транспорт. mempalace подключён через HTTP.
- **2.6:** `memory.py` без FastMemory и прямого KG - wake-up через MCP.
- **2.7:** `chatd.py` без интерцептора `mempalace_kg_query`.

### PR C.5 - внутренние инструменты
- `internal_tools.py`: `subcortex_goals_list/get/create/cancel`, `subcortex_journal_tail`.
- Bulk cancel по списку id.
- Регистрируются в `chatd.TOOLS`, недоступны фону через `BG_TOOLS_ALLOWED`.
- Из чата управляют целями - проверено.

### Полировка planner/executor/reflector
- `BG_SKIP_WITH_OPEN_GOALS_STOPS_TICK`, `BG_EXPLORATORY_MIN_INTERVAL_MIN`.
- `useful` / `useful_reason` в exploratory observations.
- Промпты в `prompts/*.txt` усилены.
- `_read_recent_archive` - exploratory-учёт видит завершённые цели.

### DeepSeek Flash интеграция
- `chat_sync` через стриминг (Flash в non-streaming обрывает reasoning).
- `BG_NUM_PREDICT` работает (нужно 2048+ для reasoning).
- `response_format="json"` включён для planner/reflector.

### Cloud budget + soft degrade
- `SUBCONSCIOUS_BG_CLOUD_PREFIXES`, `_CLOUD_DAILY_TOKEN_BUDGET`, `_DEGRADE_MODEL`.
- `_resolve_model()` деградирует фазу на локальную при исчерпании.
- `degraded` в journal.

### Search fix
- ONNX-модель примонтирована в mempalace-контейнер.
- `mempalace_search`: 0.10s (было 40s таймаут).

### Truncation limits
- Все через env: `BG_TOOL_RESULT_MAX_CHARS`, `BG_EXPLORATORY_SEED_MAX_CHARS`, `BG_OBSERVATION_SNIPPET_CHARS`, `BG_RECENT_OBSERVATIONS_LIMIT`, `CHAT_TOOL_RESULT_MAX_CHARS`, `MEMORY_KG_SNIPPET_LIMIT`.

---

## ✅ Сделано в этой сессии (Stage 2.75–2.85)

### Протокол: нормализация бэкендов
- Единый внутренний shape для всех бэкендов:
  - `message.thinking` (было `thinking` / `reasoning_content`)
  - `message.tool_calls` - **плоский** `[{id, name, arguments}]` (было вложенное `{function: {name, arguments}}`)
  - `done_reason` - во всех бэкендах (было только у некоторых)
  - `usage` - объект с `prompt_tokens`/`completion_tokens`/`reasoning_tokens`/`cache_hit_tokens`/`cache_miss_tokens`
- `backends/ollama.py` переписан с нормализацией в обе стороны.
- `backends/deepseek.py` переписан: принимает flat, эмитит `thinking` + `usage`; cache hit/miss в логе.
- `backends/openrouter.py`: извлечение `usage` и `finish_reason` из `chat_sync` (было хардкод `"stop"` и ноль токенов), `reasoning_content` → `thinking`.
- `backends/base.py` - docstring с описанием нормализованного shape.

### Протокол: типизированные события
- Поле `event` верхнего уровня в NDJSON. Типы:
  - `thinking` - `{delta}`
  - `tool_call` - `{id, name, arguments}`
  - `tool_result` - `{id, name, status, content}`
  - `error`, `warning` - зарезервированы
- События **не сохраняются** в историю клиента (display-only).
- Хелперы `make_event_chunk`, `make_done_chunk`, `make_thinking_chunk` в `chatd.py`.
- `/api/generate` пробрасывает `event` и `usage`.
- `event.type` - строковый дискриминатор, расширяемо без версионирования.

### Протокол: dual-channel thinking для Hollama
- Hollama читает `message.thinking` (Ollama-native), поэтому thinking эмитится **двумя каналами**:
  - `message.thinking` - для Hollama
  - `event.type=thinking` - для event-aware клиентов
- В `build_model_messages` thinking удаляется из истории assistant-сообщений перед backend (DeepSeek отклоняет `reasoning_content` в input на части версий API).

### Протокол: DSML-фильтр
- `DSMLFilter` - stateful: буферизует от `｜` до newline, дропает строку с маркером.
- `_strip_dsml` (одноразовый) и `_contains_dsml` для non-streaming путей.
- Fallback-инструкция `"no tools, plain text only"` для последнего раунда (DeepSeek Flash иначе эмитит DSML-текст).
- `think_remapper.py` - **удалён**, его функция поглощена event-каналом.

### Cache-aware транспорт
- **RAG-инъекция переехала из system в последнее user-сообщение** - ключевая оптимизация.
  - `ensure_system_prompt` копирует messages и дописывает RAG-блок к последнему user.
  - `clean_user` параметр в `chat_stream_generator` - сессия и RAG-индекс получают **оригинальный** user-текст, а не аугментированный.
  - `chat()` захватывает `original_user` до инъекции.
- **DeepSeek stream usage** - `stream_options={"include_usage": True}`, единый done=True чанк на `[DONE]` с usage/tool_calls/done_reason.
- **Summary в фоне сохраняет tools** - не сбрасывает prefix cache; fallback на tools=None если модель не подчиняется.
- Результат в цифрах:
  - Чат: 90–99% cache_hit (было ~0)
  - Фон: ~70% за тик (по архитектуре ниже - уникальный контент на каждом шаге)
  - Summary: 9728/9904 hit (было 3584/4853)

### Killer bug: фон не мог вызывать инструменты
- `background._execute_tool_call` читал **старый вложенный** формат `tc.get("function")`.
- После нормализации бэкендов `fn` был пуст → `name="unknown"` → `BG_TOOLS_ALLOWED` отбрасывал всё.
- Симптом: все tool-результаты `52 chars`, planner честно писал `tool not allowed in background: unknown`.
- Исправлено в `_execute_tool_call` и в exploratory seed path.
- Добавлен `log.warning` при reject.
- **Побочный эффект:** 4 воркер-цели про «пустой allowlist / guard / unknown tool» были созданы на основе этого симптома и отменены вручную после фикса.

### Косметика и защита
- `_tc_signature` и `run_tool_loop` - плоский формат.
- `build_model_messages` чистит legacy `[→ name]` и `<think>…</think>` для старых сессий.
- `/tags` дедуплицирует по `model`/`name` - если одна модель пришла и из Ollama, и из OpenRouter, остаётся одна запись.
- `_run_with_timeout` - задокументированный trade-off (daemon thread продолжает бежать после таймаута).

---

## ⚠️ Не доделано

### P0 - было в backlog до этой сессии

1. **`mcp_client.py`** - двойной префикс `MCP_ENV_PREFIXES` **сделан** (см. ниже). Осталось: убедиться, что нигде не осталось `MCP_ENV_PREFIX` в единственном числе - grep по репо.
2. **`memory.py`** - `LAYER_15_ENABLED` через `_env()` - **не проверял в этой сессии**, вероятно ещё актуально.
3. **`config.py`** - мёртвая `MCP_ENV_PREFIX`, `_load_bg_prompt` через `_env` - **не проверял**.
4. **`.env.example`** - три места (`SUBCONSCIOUS_STDIO_MCP_*` порядок слов, `mempalace_wake_up` в allowlist, комментарии про `BG_DEGRADE_MODEL`) - **не проверял**.

Из них **пункт 1 сделан в этой сессии:**
- `discover_mcp_servers()` обходит `MCP_ENV_PREFIXES` в порядке приоритета.
- `key[len(prefix):]` работает корректно для любого из префиксов.
- Старший префикс маскирует младший (SUBCONSCIOUS > CHATD).

### P1 - может подождать

5. **`_run_with_timeout`** - daemon thread остаётся живым при таймауте. Задокументировано.
6. **Concurrency:** `run_tick` держит `TickLock` всю длительность тика. Из чата `subcortex_goals_*` берут тот же lock с таймаутом 5s. В PR D - разделить на `tick.lock` и `ledger.lock`.
7. **`num_predict=380000`** - вернуть к 8192. DeepSeek клампит, но полагаться нельзя.
8. **OpenRouter `chat_sync`** - extraction `usage` добавлена в этой сессии (см. выше), но эффект не проверялся без трафика через `or/`.

### P2 - дизайн-долг, извлечённый из этой сессии

9. **Failure-lineage целей** - failed-цель плодит цель про причину провала. За 4 итерации одна проблема размножается в 5 целей. Держит только `BG_MAX_OPEN_GOALS=5`. Три идеи на PR D:
   - **Supersede**: заменять похожую открытую цель вместо создания рядом.
   - **Failure-lineage counter**: при done/cancelled родителя автоматически снимать производные.
   - **Stale by `based_on`**: помечать цель устаревшей, если observation, на которой она основана, больше не соответствует реальности.
10. **`_read_recent_observations`** - окно 12 непустых `last_result` воскрешает уже починенный баг. После fix симптома planner ещё 1–2 тика может видеть старые наблюдения и пытаться расследовать несуществующую проблему.
11. **Write-tools для воркера** - цели типа KG-дедупа (`kg_supersede`, `kg_invalidate`) не выполнимы в read-only воркере. Три варианта:
    - **Propose-then-confirm**: воркер создаёт patch-цель, ждёт апрува человеком.
    - **Narrow write-allowlist**: 2–3 инструмента с audit-логом.
    - **`BG_WRITE_ENABLED=false/true`**: при `false` write-цели помечаются `requires_manual` вместо failed.
12. **Серверная история** - клиент шлёт `chat_id + new_message`, chatd восстанавливает историю из session. Решает и кэш после tool-сессий (сейчас 13696/77123 - падение до системного префикса, потому что Hollama не сохраняет tool_calls), и открывает дорогу к своему клиенту.

---

## 📋 Stage 2.8 и 2.9 (плановые)

### Stage 2.8 - systemd переименование
- Симлинки старых имён.
- Проверка `SUBCONSCIOUS_*` без fallback-warning'ов.
- Env-переменные в `/etc/subconscious/subconscious.env` под новыми именами.

**Оценка:** 1 час.

### Stage 2.9 - реструктуризация в пакет `subconscious/`
- `subconscious/{core,cortex,thalamus,subcortex}/`
- `git mv`:
  - `chatd.py` → `subconscious/cortex/api.py`
  - `background.py` → `subconscious/subcortex/tick.py`
  - `memory.py`, `rag.py`, `session.py`, `config.py` → `subconscious/core/`
  - `mcp_client.py` → `subconscious/thalamus/mcp/client.py`
  - `mcp_registry.py` → `subconscious/core/registry.py`
  - `backends/` → `subconscious/thalamus/backends/`
- Разорвать циклический `subcortex → cortex` (`import chatd` → `from subconscious.core import registry`).
- Пути к `prompts/` и `tool_descriptions.ini` через `Path(__file__).resolve().parent.parent.parent`.
- Shim-файлы в корне, `pyproject.toml`, `pip install -e .`.
- Systemd unit: `subconscious.cortex.api:app`.

**Оценка:** этап 1 - 40 мин, этап 2 - 20 мин.

---

## 🎯 Roadmap после 2.9

### PR D - замыкание StrangeLoop
- **Assess-фаза** после Executor (оценка done-целей).
- **`parent_goal_id`** - lineage целей + чистка по failure-lineage.
- **Token usage для OpenRouter** - сделано, нужен smoke.
- **A2A: Agent Card + Task mapping** (минимум).
- **Свой KG на SQLite** (часть PR E).

### PR E - своя память
- Свой KG (~200 строк), `sqlite-vec`, свой identity loader.
- Mempalace за MCP как источник старых данных.
- `mcp_servers.yaml` для thalamus.

### PR F - multi-agent
- Worker pool, workspace reservation.
- A2A полный: Message, Artifact, push notifications.
- Multi-host.

### Backlog
- DAG целей с зависимостями.
- LangGraph-подобное планирование.
- soothe-nano для редактирования кода.
- Плагины.

---

## 📝 Документы в репо

- `docs/GOALS.md` - манифест проекта.
- `docs/ARCHITECTURE.md` - четыре зоны, FS-раскладка, внешние сервисы.
- `docs/RENAME.md` - план переименования.

**Не хватает:**
- `docs/adr/` - три ADR (backends, memory, mcp).
- `docs/OPERATIONS.md` - бэкапы, runbooks.
- **`docs/PROTOCOL.md`** - спецификация нормализованного shape + event-канал. Появилось в этой сессии, надо зафиксировать письменно.
- Обновить `README.md` - местами ещё старое название.

---

## 🏁 Следующий шаг

Три параллельных трека:

1. **Закрыть оставшиеся P0 из backlog** - проверить `memory.py`, `config.py`, `.env.example` на три оставшихся пункта (нужен grep по репо).
2. **Дописать `docs/PROTOCOL.md`** - нормализованный shape, event-типы, dual-channel thinking, DSML-фильтр. Сейчас это живёт только в коде.
3. **Stage 2.8 → 2.9** - переименование systemd, потом реструктуризация в пакет.

После этого - `pre-2.9` тег и старт PR D.
