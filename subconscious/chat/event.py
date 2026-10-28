"""Event processing.

Normalizes inbound events, extracts session key and user id, and
dispatches into the same tool loop as chat (non-streaming). Duplicate
events are dropped per session; events from the same session are
serialized with a per-session lock.

Session key parsing is best-effort: unknown sources fall back to
stateless processing. An event without a recoverable session key is
still processed, just without dialogue history.

Note on run_tool_loop: it calls call_tool from tools_io, which imports
process_event from this module. Both imports are deferred, so the
module graph is acyclic at load time. See tools_io docstring.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import traceback
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from subconscious.chat import session as sess
from subconscious.chat.prompt import ensure_system_prompt
from subconscious.chat.tools_loop import (
    EVENT_FINAL_ROUND_NUM_PREDICT,
    run_tool_loop,
)
from subconscious.events import store as events_store
from subconscious.memory import rag
from subconscious.runtime.config import (
    EVENT_MODEL,
    MAX_HISTORY_TURNS,
    USER_DEFAULT_ID,
    _env,
)

log = logging.getLogger("subconscious.chat.event")


# ── state ────────────────────────────────────────────────────────────────────
#
# Deduplication: bounded per-session buffers of seen message ids.
# Locks: per-session, acquired during the LLM call only. Both dicts
# grow unbounded with the number of distinct sessions; for a
# single-agent deployment that number is small.

_SEEN_EVENT_IDS: Dict[str, "deque[str]"] = {}
_SEEN_EVENT_IDS_GUARD = threading.Lock()
_SEEN_EVENT_IDS_MAX = 200

_SESSION_LOCKS: Dict[str, threading.Lock] = {}
_SESSION_LOCKS_GUARD = threading.Lock()


# ── session key ──────────────────────────────────────────────────────────────

def _extract_session_key(
    event_source: str,
    event_payload: Dict[str, Any],
) -> Optional[str]:
    """Extract a stable session key from an event payload.

    Priority:
      1. Explicit `session_key` field in the payload - any MCP source can
         opt in by setting it, without chatd needing to know the source.
      2. XMPP-specific JID-based key for dm and groupchat events.

    Returns a lowercase key like "xmpp:dm:user@domain" or None when no
    session can be determined (the event will be processed stateless).
    """
    explicit = event_payload.get("session_key")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    if not event_source.startswith("xmpp"):
        return None

    from_jid = None
    for key in ("from", "from_jid", "sender", "from_bare", "jid"):
        val = event_payload.get(key)
        if isinstance(val, str) and val.strip():
            from_jid = val.strip()
            break

    if not from_jid:
        text = event_payload.get("text")
        if isinstance(text, str):
            m = re.match(
                r"^([^\s@:]+@[^\s:/]+)(?:/[^\s:]+)?\s*:\s",
                text,
            )
            if m:
                from_jid = m.group(1)

    if not from_jid:
        return None

    bare = from_jid.split("/", 1)[0].lower()
    if not bare or "@" not in bare:
        return None

    self_jid = _env("XMPP_SELF_JID", "").strip().lower()
    if self_jid:
        self_bare = self_jid.split("/", 1)[0]
        if bare == self_bare:
            return None

    is_muc = (
        event_payload.get("type") == "groupchat"
        or bool(event_payload.get("room"))
        or bool(event_payload.get("muc"))
    )
    if is_muc:
        return f"xmpp:muc:{bare}"
    return f"xmpp:dm:{bare}"


def _extract_user_id(flask_request) -> Optional[str]:
    """Extract the user identity from request headers.

    Open WebUI, when ENABLE_FORWARD_USER_INFO_HEADERS=true, sends:
      X-OpenWebUI-User-Email, X-OpenWebUI-User-Id, X-OpenWebUI-User-Name.
    Email is preferred because it matches the XMPP JID form
    (person@domain) and makes cross-channel identity possible without a
    mapping table. Falls back to USER_DEFAULT_ID for single-user
    installs.
    """
    email = flask_request.headers.get("X-OpenWebUI-User-Email")
    if email and email.strip():
        return email.strip().lower()
    uid = flask_request.headers.get("X-OpenWebUI-User-Id")
    if uid and uid.strip():
        return uid.strip().lower()
    if USER_DEFAULT_ID:
        return USER_DEFAULT_ID.strip().lower()
    return None


def _priority_for_event(event_source: str, payload: Dict[str, Any]) -> int:
    """Classify event priority from source and payload.

    Critical alerts bypass the queue and are picked first. Interactive
    channels (webui, xmpp, email) go next. Alerts and monitors are
    normal. Everything else (ticks, housekeeping) is background.
    """
    if payload.get("severity") == "critical":
        return events_store.PRIORITY_CRITICAL
    if event_source in ("webui", "xmpp", "email"):
        return events_store.PRIORITY_INTERACTIVE
    if event_source == "alert" or event_source.startswith("monitor"):
        return events_store.PRIORITY_NORMAL
    return events_store.PRIORITY_BACKGROUND


def _extract_event_human_text(payload: Dict[str, Any]) -> str:
    """Pull the human-readable portion of an event payload.

    Used as the `user` side of the session log entry, so the JSONL shows
    the actual message text rather than a JSON envelope. Falls back to ""
    when the payload is a pure system event with no human content.
    """
    for key in ("body", "message", "text", "content"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


# ── deduplication and locking ────────────────────────────────────────────────

def _is_duplicate_event(session_key: str, payload: Dict[str, Any]) -> bool:
    """Return True if this event id was already processed for this session.

    Best-effort: the buffer lives in memory and is lost on restart. That
    is acceptable - dedupe protects against the transport re-delivering
    the same notification, not against a deliberate replay days later.
    """
    msg_id = None
    for key in ("id", "message_id", "stanza_id"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            msg_id = val.strip()
            break
    if not msg_id:
        return False

    with _SEEN_EVENT_IDS_GUARD:
        buf = _SEEN_EVENT_IDS.get(session_key)
        if buf is None:
            buf = deque(maxlen=_SEEN_EVENT_IDS_MAX)
            _SEEN_EVENT_IDS[session_key] = buf
        if msg_id in buf:
            return True
        buf.append(msg_id)
    return False


def _get_session_lock(session_key: str) -> threading.Lock:
    """Return (and lazily create) the per-session lock.

    The dict grows unbounded, but for a single-agent deployment the
    number of distinct sessions is small. If this ever becomes a
    problem, swap for an LRU with eviction on lock release.
    """
    with _SESSION_LOCKS_GUARD:
        lock = _SESSION_LOCKS.get(session_key)
        if lock is None:
            lock = threading.Lock()
            _SESSION_LOCKS[session_key] = lock
        return lock


# ── history and audit ────────────────────────────────────────────────────────

def _load_session_history(
    session: "sess.Session",
    limit_turns: int = MAX_HISTORY_TURNS,
) -> List[Dict[str, Any]]:
    """Load the last N turns from a session's JSONL chatlog.

    Returns messages in OpenAI format: [{role, content}, ...].
    Only human/assistant messages with non-empty content are returned.
    The chatlog format is produced by Session.append_chatlog():
      {"type": "human",     "message": {"role": "human",     "content": "..."}}
      {"type": "assistant", "message": {"role": "assistant", "content": "..."}}
    """
    try:
        path = session._jsonl_path()
    except Exception as e:
        log.warning("[event] cannot resolve session path: %s", e)
        return []

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        log.warning("[event] failed to read session log %s: %s", path, e)
        return []

    max_lines = max(2, limit_turns * 2)
    lines = lines[-max_lines:]

    messages: List[Dict[str, Any]] = []
    for line in lines:
        try:
            entry = json.loads(line)
        except Exception:
            continue
        msg = entry.get("message") or {}
        role = msg.get("role")
        content = msg.get("content") or ""
        if not isinstance(content, str) or not content.strip():
            continue
        if role in ("human", "user"):
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
    return messages


def _log_event_journal(
    req_id: str,
    event_type: str,
    event_source: str,
    payload: Dict[str, Any],
    answer: Optional[str] = None,
) -> None:
    """Append event and its outcome to a dedicated journal file.

    Human-readable audit trail, separate from the SQLite ledger. Kept
    for tail -f and postmortems; the SQLite ledger is the queryable
    source of truth.
    """
    try:
        log_file = Path("/var/lib/subconscious/events.jsonl")
        log_file.parent.mkdir(parents=True, exist_ok=True)

        from subconscious.chat.util import now_iso
        entry = {
            "req_id": req_id,
            "ts": now_iso(),
            "type": event_type,
            "source": event_source,
            "payload": payload,
            "status": "done" if answer is not None else "started",
        }
        if answer is not None:
            entry["answer"] = answer

        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[event] failed to write journal: %s", e)


# ── dispatch ─────────────────────────────────────────────────────────────────

def process_event(
    event_type: str,
    event_source: str,
    event_payload: Any,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    req_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """Common event dispatch logic used by /api/event and MCP notifications.

    If the event identifies a session (XMPP JID, explicit session_key),
    the event is processed inside that session's dialogue: history is
    loaded, the event becomes the latest user turn, and the answer is
    recorded back. Otherwise it is processed stateless.
    """
    req_id = req_id or uuid.uuid4().hex[:8]
    model = model or EVENT_MODEL or "qwen3:8b"

    # options is left None; merge_options lives in chat.api and would
    # pull the whole api module into the event loop. Callers pass a
    # fully formed dict or nothing.
    opts = dict(options or {})

    if not isinstance(event_payload, (dict, list)):
        payload_dict: Dict[str, Any] = {"text": str(event_payload)}
    elif isinstance(event_payload, list):
        payload_dict = {"items": event_payload}
    else:
        payload_dict = event_payload

    session_key = _extract_session_key(event_source, payload_dict)

    log.info(
        "[%s] session_key probe: source=%r payload_keys=%s session_key=%r",
        req_id, event_source, sorted(payload_dict.keys())[:10], session_key,
    )

    if session_key and _is_duplicate_event(session_key, payload_dict):
        log.info("[%s] event deduplicated: session=%s", req_id, session_key)
        _log_event_journal(req_id, event_type, event_source, payload_dict)
        return req_id

    log.info(
        "[%s] system event: type=%s source=%s model=%s session=%s",
        req_id, event_type, event_source, model,
        session_key or "(stateless)",
    )

    event_id = events_store.append(
        source=event_source,
        event_type=event_type,
        payload=payload_dict,
        user_id=user_id,
        priority=_priority_for_event(event_source, payload_dict),
        correlation_id=payload_dict.get("correlation_id"),
    )
    log.info(
        "[%s] event recorded: id=%s priority=%d user_id=%s",
        req_id, event_id,
        _priority_for_event(event_source, payload_dict),
        user_id or "-",
    )

    _log_event_journal(req_id, event_type, event_source, payload_dict)

    session = sess.get_session(session_key, user_id=user_id) if session_key else None

    history: List[Dict[str, Any]] = []
    if session is not None:
        history = _load_session_history(session, MAX_HISTORY_TURNS)
        log.info(
            "[%s] loaded %d history message(s) for session %s",
            req_id, len(history), session_key,
        )

    event_text = (
        f"[SYSTEM EVENT] type={event_type} source={event_source}\n"
        f"{json.dumps(payload_dict, ensure_ascii=False, indent=2)}"
        f"{_EVENT_DIRECTIVE}"
    )

    messages: List[Dict[str, Any]] = list(history)
    messages.append({"role": "user", "content": event_text})

    human_text = _extract_event_human_text(payload_dict)
    original_user = human_text or event_text

    messages, layer_sizes = ensure_system_prompt(
        messages, session=session, user_id=user_id,
    )

    def _run():
        lock = _get_session_lock(session_key) if session_key else None
        if lock is not None:
            lock.acquire()
        try:
            answer = run_tool_loop(
                model, messages, opts, layer_sizes, req_id,
                final_num_predict=EVENT_FINAL_ROUND_NUM_PREDICT,
            )
            if session is not None and answer:
                try:
                    sess.record_turn(
                        session, original_user, answer, user_id=user_id,
                    )
                    rag.index_turn(
                        source=f"session:{session.session_id}",
                        user=original_user,
                        assistant=answer,
                        user_id=user_id,
                    )
                except Exception as e:
                    log.warning(
                        "[%s] failed to persist event turn to session %s: %s",
                        req_id, session_key, e,
                    )
            _log_event_journal(
                req_id, event_type, event_source, payload_dict, answer,
            )
            events_store.mark(event_id, status="processed")
            log.info("[%s] event worker done, answer_len=%d", req_id, len(answer))
        except Exception:
            log.error(
                "[%s] event worker error:\n%s", req_id, traceback.format_exc(),
            )
        finally:
            if lock is not None:
                lock.release()

    t = threading.Thread(target=_run, daemon=True, name=f"event-{req_id}")
    t.start()
    return req_id


# EVENT_DIRECTIVE lives here because process_event needs it. It used to
# live in chat/api.py; nothing else referenced it. Directive text is
# unchanged from the current api.py version.
_EVENT_DIRECTIVE = (
    "\n\n"
    "This is an automated system event - there is no human in the conversation.\n"
    "\n"
    "STEP 1 - Acknowledge immediately:\n"
    "  Call send_message (or the appropriate notification tool) RIGHT NOW to inform\n"
    "  the user that you received this event. If the event requires action, briefly\n"
    "  say what you are about to do. If nothing needs to be done - say so and stop.\n"
    "\n"
    "STEP 2 - Act (only if action is needed):\n"
    "  Call the required diagnostic or action tools to handle the event.\n"
    "  Do NOT just describe what you would do - actually call the tools.\n"
    "\n"
    "STEP 3 - Report result:\n"
    "  After all action tools are done, call send_message again to report\n"
    "  what was done and what was found.\n"
    "\n"
    "STEP 4 - Persist (only for important events):\n"
    "  If this event is an alert, a security issue, or an anomaly, save a brief\n"
    "  summary to long-term memory using `mempalace_add_drawer` or `mempalace_kg_add`,\n"
    "  so you can recall it later if asked.\n"
    "\n"
    "Do NOT skip STEP 1. Do NOT write a plain-text summary without calling send_message."
)
