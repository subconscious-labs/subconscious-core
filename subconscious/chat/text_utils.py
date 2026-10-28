"""Text and content helpers for chat.

Includes the DSML filter (DeepSeek Flash tool-call markup that arrives
as plain text when tools are stripped), the multimodal content guard
(Open WebUI sends content as an array even when no files are attached),
and the tool-result truncator.

Stateless helpers; no state, no logging beyond debug. Safe to import
from anywhere in the chat package without creating cycles.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from subconscious.runtime.config import CHAT_TOOL_RESULT_MAX_CHARS

log = logging.getLogger("subconscious.chat.text_utils")


# ── DSML filter ──────────────────────────────────────────────────────────────
#
# DeepSeek Flash emits tool-call markup as plain text when tools are
# stripped from the payload (last round of the tool loop, or forced
# final answer). The markup looks like:
#
#   <｜｜DSML｜｜ calls>
#   <｜｜DSML｜｜ invoke name="foo">
#   ...
#   </｜｜DSML｜｜ calls>
#
# In streaming mode the marker can be split across chunks, so a
# per-chunk regex won't catch it. The filter buffers from the first
# `｜` until the next newline, then decides: if the buffered line
# contains the marker, drop it entirely; otherwise emit it unchanged.

_DSML_MARKER = "｜｜DSML｜｜"
_DSML_TRIGGER = "｜"


class DSMLFilter:
    """Stateful stripper for DeepSeek DSML tool-call markup."""

    MARKER = _DSML_MARKER
    TRIGGER = _DSML_TRIGGER
    BUFFER_CAP = 8192

    def __init__(self) -> None:
        self._buf = ""
        self._in_buffer = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        out: List[str] = []
        for ch in text:
            if not self._in_buffer:
                if ch == self.TRIGGER:
                    self._in_buffer = True
                    self._buf = ch
                else:
                    out.append(ch)
            else:
                self._buf += ch
                if ch == "\n":
                    if self.MARKER not in self._buf:
                        out.append(self._buf)
                    self._buf = ""
                    self._in_buffer = False
                elif len(self._buf) > self.BUFFER_CAP:
                    out.append(self._buf)
                    self._buf = ""
                    self._in_buffer = False
        return "".join(out)

    def flush(self) -> str:
        """Return anything still buffered at stream end."""
        if not self._in_buffer:
            return ""
        leftover = self._buf
        self._buf = ""
        self._in_buffer = False
        if self.MARKER in leftover:
            return ""
        return leftover


def _strip_dsml(text: str) -> str:
    """One-shot strip for non-streaming content (whole string at once)."""
    if not text or _DSML_TRIGGER not in text:
        return text
    f = DSMLFilter()
    return (f.feed(text) + f.flush()).strip()


def _contains_dsml(text: str) -> bool:
    return bool(text) and _DSML_MARKER in text


# ── content extraction ───────────────────────────────────────────────────────

def _extract_text_from_content(content: Any) -> str:
    """Return plain text from a message content field.

    Handles both shapes:
      - str: as-is.
      - list: OpenAI multimodal content array ([{"type": "text",
        "text": "..."}, {"type": "image_url", ...}]). Text parts joined
        with newlines; non-text parts (images, audio) dropped with a
        warning. This is a defensive guard, not attachment support:
        full ingress is deferred (ADR-004). Without it, an Open WebUI
        request with its default multimodal shape crashes downstream
        callers that assume str.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        log.warning(
            "unexpected content type %s, treating as empty",
            type(content).__name__,
        )
        return ""

    text_parts: List[str] = []
    dropped = 0
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "text":
                txt = part.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
            elif ptype is not None:
                dropped += 1
        elif isinstance(part, str):
            text_parts.append(part)
    if dropped:
        log.warning(
            "dropped %d non-text content part(s) - attachments are not "
            "supported yet, see ADR-004",
            dropped,
        )
    return "\n".join(text_parts)


def _last_user_text(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _extract_text_from_content(m.get("content"))
    return ""


# ── tool result truncation ───────────────────────────────────────────────────

def _truncate_chat_tool_result(result: Any) -> str:
    """Cap a tool result before feeding it back to the chat model."""
    if CHAT_TOOL_RESULT_MAX_CHARS <= 0:
        return json.dumps(result, ensure_ascii=False)
    try:
        serialized = json.dumps(result, ensure_ascii=False)
    except Exception:
        serialized = str(result)
    if len(serialized) <= CHAT_TOOL_RESULT_MAX_CHARS:
        return serialized
    return (
        serialized[:CHAT_TOOL_RESULT_MAX_CHARS]
        + ' ... [truncated, result was '
        + str(len(serialized))
        + ' chars]'
    )
