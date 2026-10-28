"""Prompt assembly and payload construction.

ensure_system_prompt: L0+L1 wake-up + L1.5 RAG injection into the last
user message (cache-friendly placement).
build_model_messages: sanitize history before sending to backend.
make_ollama_payload: package model, messages, options, tools for the
backend call.

All functions are stateless. Depends on memory, rag, session, and the
text helpers.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from subconscious.chat import session as sess
from subconscious.chat.text_utils import (
    _extract_text_from_content,
    _last_user_text,
    _strip_dsml,
)
from subconscious.memory import wakeup as memory
from subconscious.memory import rag
from subconscious.runtime.config import (
    MAX_HISTORY_TURNS,
    THINKING,
)
from subconscious.tools import registry as mcp_registry

log = logging.getLogger("subconscious.chat.prompt")


def ensure_system_prompt(
    messages: List[Dict[str, Any]],
    session: Optional[sess.Session] = None,
    user_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assemble system prompt and inject request-scoped RAG context.

    Layout sent to the backend:
      [system: L0+L1 wake-up]
      [history ...]
      [user: L1.5 RAG sidecar + original user text]

    RAG is deliberately placed in the last user message, not the system
    block. DeepSeek (and other OpenAI-compatible providers) cache stable
    prefixes server-side; putting the per-request RAG block at the very
    end maximises the cached prefix between consecutive turns.

    Returns (messages, layer_sizes) where layer_sizes maps layer name to
    char count: L0+L1, L1.5_kg, L1.5_rag.
    """
    resolved_uid = (session.user_id if session else None) or user_id

    if session is not None:
        base_prompt = session.get_cached_base_prompt()
        if base_prompt is None:
            base_prompt = memory.wake_up()
            session.set_cached_base_prompt(base_prompt)
        else:
            log.debug(
                "[syspr] session %s: reusing cached base prompt (%d chars)",
                session.session_id, len(base_prompt),
            )
    else:
        base_prompt = memory.wake_up()

    layer_sizes: Dict[str, int] = {
        "L0+L1":   len(base_prompt),
        "L1.5_kg": 0,
        "L1.5_rag": 0,
    }

    msgs: List[Dict[str, Any]] = [dict(m) for m in (messages or [])]

    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {"role": "system", "content": base_prompt}
    else:
        msgs.insert(0, {"role": "system", "content": base_prompt})

    original_user = _last_user_text(msgs)
    if original_user:
        rag_ctx = rag.retrieve(original_user, user_id=resolved_uid)
        if rag_ctx:
            layer_sizes["L1.5_rag"] = len(rag_ctx)
            log.debug(
                "[sidecar] injecting %d-char RAG recall into last user message",
                len(rag_ctx),
            )
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i].get("role") == "user":
                    msgs[i]["content"] = (
                        f"# Relevant past context\n{rag_ctx}\n\n"
                        f"---\n\n{original_user}"
                    )
                    break

    return msgs, layer_sizes


def build_model_messages(
    raw_messages: List[Dict],
    max_history_turns: int = MAX_HISTORY_TURNS,
) -> List[Dict]:
    """Sanitise messages for the backend.

    - Strip fields unknown to Ollama (chatId, id, createdAt, updatedAt …).
    - Remove <think>…</think> blocks and legacy [→ tool] markers from
      previous assistant turns.
    - Preserve tool_calls (flat) and thinking on assistant messages.
    - Keep system message as-is.
    - Limit history to *max_history_turns* user+assistant pairs.
    """
    system: Optional[Dict] = None
    turns: List[Dict] = []

    for m in raw_messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        content = _extract_text_from_content(m.get("content"))
        if role == "assistant":
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()
            content = re.sub(r"\[→\s*[^\]]+\]\s*", "", content).strip()
            content = _strip_dsml(content)

        if role == "system":
            system = {"role": "system", "content": content}
            continue

        entry: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
        turns.append(entry)

    if len(turns) > max_history_turns * 2:
        cutoff = len(turns) - max_history_turns * 2
        while cutoff < len(turns) and turns[cutoff].get("role") == "tool":
            cutoff += 1
        turns = turns[cutoff:]

    result = []
    if system:
        result.append(system)
    result.extend(turns)
    return result


def _log_payload_sizes(
    req_id: str,
    messages: List[Dict],
    tools_count: int,
    layer_sizes: Optional[Dict[str, int]] = None,
) -> None:
    sizes: Dict[str, int] = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    for m in messages:
        role = m.get("role", "other")
        content = m.get("content") or ""
        sizes[role] = sizes.get(role, 0) + len(content)
    total = sum(sizes.values())

    if layer_sizes:
        log.debug(
            "[%s] system layers: L0+L1=%d L1.5_kg=%d L1.5_rag=%d  total=%d",
            req_id,
            layer_sizes.get("L0+L1", 0),
            layer_sizes.get("L1.5_kg", 0),
            layer_sizes.get("L1.5_rag", 0),
            sizes.get("system", 0),
        )
    log.debug(
        "[%s] payload sizes: system=%d user=%d assistant=%d tool=%d  total_chars=%d  tools=%d",
        req_id,
        sizes.get("system", 0),
        sizes.get("user", 0),
        sizes.get("assistant", 0),
        sizes.get("tool", 0),
        total,
        tools_count,
    )


def make_ollama_payload(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    stream: bool,
) -> Dict[str, Any]:
    tools = mcp_registry.list_tools()
    payload: Dict[str, Any] = {
        "model":    model,
        "messages": messages,
        "stream":   stream,
        "options":  options,
    }
    if tools:
        payload["tools"] = tools
    if not THINKING:
        payload["think"] = False
    return payload
