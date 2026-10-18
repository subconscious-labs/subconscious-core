# Переименование chatd → subconscious-core

## Принцип

Всё делается с обратной совместимостью. Прод работает во время
переименования. Ни один шаг не требует downtime дольше рестарта.

## Что меняется

| Было | Станет | Совместимость |
|---|---|---|
| repo `chatd` | repo `subconscious-core` | GitHub redirect |
| package `chatd` | package `subconscious` | `import chatd` shim (1–2 релиза) |
| env `CHATD_*` | env `SUBCONSCIOUS_*` | читаются оба, warning на старое |
| `chatd.py` | `subconscious/cortex/api.py` | - |
| `background.py` | `subconscious/subcortex/tick.py` | `import background` shim |
| `/etc/chatd/` | `/etc/subconscious/` | оба читаются |
| `~/.local/share/chatd/` | `~/.local/share/subconscious/` | миграционный скрипт |
| systemd `chatd.service` | `subconscious.service` | symlink |
| systemd `chatd-tick.timer` | `subconscious-tick.timer` | symlink |
| `/usr/local/bin/chatd-tick` | `/usr/local/bin/subconscious-tick` | symlink |
| `chatd.service.txt` | `subconscious.service.example` | - |
| `/api/chat` и т.д. | **не меняются** | публичный API |

## Env backward compat

Хелпер в `config.py`:

    def _env(name: str, default: str = "") -> str:
        new = os.environ.get(f"SUBCONSCIOUS_{name}")
        if new is not None:
            return new
        old = os.environ.get(f"CHATD_{name}")
        if old is not None:
            log.warning("CHATD_%s deprecated, use SUBCONSCIOUS_%s", name, name)
            return old
        return default

Все существующие `os.environ.get("CHATD_...")` заменяются на `_env("...")`.

Срок жизни shim'ов - до v2.0 (примерно год).

## Порядок работ

1. **GitHub rename.** `chatd` → `subconscious-core`. Redirect появится автоматически.
2. **Python package rename.** `chatd.py` → `subconscious/cortex/api.py`, `background.py` → `subconscious/subcortex/tick.py`.
3. **Shim'ы.** `chatd.py` в корне содержит `from subconscious.cortex.api import app`, `background.py` - `from subconscious.subcortex.tick import *`.
4. **Env helper.** `_env()` в config.py + замена по коду.
5. **systemd units.** Новые файлы + symlinks для старых имён.
6. **State dir migration.** Скрипт `subconscious migrate-state`, переносит `~/.local/share/chatd/` → `~/.local/share/subconscious/` (symlink на старое для совместимости).
7. **README.** Новое имя, ссылка на `docs/GOALS.md`.
8. **Version bump.** `1.0.0` → `2.0.0`.

## Что НЕ делаем в этом PR

- Не трогаем структуру `goals.json`.
- Не трогаем URL endpoints.
- Не разбиваем `background.py` на файлы (это PR D).
- Не меняем mempalace-зависимость (это PR E).

## Оценка

4–6 часов работы. Не переписывание, а механические переименования + shim'ы.
Риск низкий. Откатывается по git.

## Проверка

После переименования должны работать:

- `systemctl status chatd.service` (symlink)
- `curl /api/chat` (endpoint не менялся)
- `curl /api/tick` (endpoint не менялся)
- env vars `CHATD_*` (с warning)
- env vars `SUBCONSCIOUS_*` (новые)
- journal по адресу `~/.local/share/subconscious/bg/journal.jsonl` (или symlink)
