#!/usr/bin/env python3
"""session.py - per-chat session state.

Each session is identified by the chatId that hollama sends in every
request.  State is persisted as an append-only JSONL chatlog so the
conversation can be ingested by ``mempalace mine --mode convos`` without
any extra conversion step.

The rolling-summary / compression layer has been removed: context
management is handled entirely by the GUI (it sends the full messages[]
array on every request) and by build_model_messages() which caps history
at 20 user+assistant pairs before forwarding to the backend.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from subconscious.runtime.config import SESSION_DIR

log = logging.getLogger("chatd.session")

_SESSIONS: Dict[str, "Session"] = {}


@dataclass
class Session:
    session_id: str

    # ── system prompt cache (L0+L1 only; L1.5 is always per-request) ─────────
    # Avoids rebuilding the static wake-up block on every HTTP request.
    # Invalidated when a mempalace write-tool is called.
    _base_prompt_cache: Optional[str] = field(default=None, repr=False)

    def get_cached_base_prompt(self) -> Optional[str]:
        """Return cached L0+L1 wake-up string, or None if not yet built."""
        return self._base_prompt_cache

    def set_cached_base_prompt(self, prompt: str) -> None:
        """Store the assembled L0+L1 wake-up string for this session."""
        self._base_prompt_cache = prompt
        log.debug("[session:%s] base prompt cached (%d chars)", self.session_id, len(prompt))

    def invalidate_prompt_cache(self) -> None:
        """Drop the cached base prompt (call after mempalace write ops)."""
        if self._base_prompt_cache is not None:
            self._base_prompt_cache = None
            log.debug("[session:%s] base prompt cache invalidated", self.session_id)

    # ── persistence ──────────────────────────────────────────────────────────

    def _dir(self) -> Path:
        d = Path(os.path.expanduser(SESSION_DIR))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _json_path(self) -> Path:
        return self._dir() / f"{self.session_id}.json"

    def _jsonl_path(self) -> Path:
        return self._dir() / f"{self.session_id}.jsonl"

    def save(self) -> None:
        """Touch the JSON marker so the session directory entry exists."""
        data = {
            "session_id": self.session_id,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path = self._json_path()
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.debug("[session:%s] saved marker to %s", self.session_id, path)
        except Exception as e:
            log.warning("[session:%s] save failed (path=%s): %s", self.session_id, path, e)

    def append_chatlog(self, user: str, assistant: str) -> None:
        """Append one turn to the persistent JSONL chatlog - never truncates."""
        log_path = self._jsonl_path()
        lines = [
            json.dumps({"type": "human",
                        "message": {"role": "human", "content": user}},
                       ensure_ascii=False),
            json.dumps({"type": "assistant",
                        "message": {"role": "assistant", "content": assistant}},
                       ensure_ascii=False),
        ]
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            log.warning("[session:%s] chatlog append failed: %s", self.session_id, e)

    @classmethod
    def load(cls, session_id: str) -> "Session":
        """Load session from disk marker, or return a blank one."""
        path = Path(os.path.expanduser(SESSION_DIR)) / f"{session_id}.json"
        if path.exists():
            log.info("[session:%s] loaded from %s", session_id, path)
        else:
            log.debug("[session:%s] no file at %s, starting fresh", session_id, path)
        return cls(session_id=session_id)


def get_session(session_id: str) -> Session:
    """Return an in-memory Session, loading from disk on first access."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = Session.load(session_id)
    return _SESSIONS[session_id]


def invalidate_all_prompt_caches() -> None:
    """Drop base prompt cache on every active session.

    Called after mempalace write operations so the next request
    rebuilds the wake-up block from fresh data.
    """
    for s in _SESSIONS.values():
        s.invalidate_prompt_cache()
    log.debug("[session] invalidated base prompt cache on %d sessions", len(_SESSIONS))


def record_turn(session: Session, user_msg: str, assistant_msg: str) -> None:
    """Append a Q/A pair to the JSONL chatlog and touch the JSON marker."""
    session.append_chatlog(user_msg, assistant_msg)
    session.save()
    log.debug(
        "[session:%s] turn recorded (user=%d chars, assistant=%d chars)",
        session.session_id, len(user_msg), len(assistant_msg),
    )
