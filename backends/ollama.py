"""backends/ollama.py - Ollama local inference backend.

Wire format: Ollama NDJSON over HTTP.
Embed: POST /api/embed.

Emits normalized internal chunks (see backends.base docstring):
  {"model", "created_at"?, "message": {"role", "content", "thinking"?,
   "tool_calls"?}, "done", "done_reason"?, "usage"?}

Ollama's native `message.thinking` is already the right name — passthrough.
tool_calls are flattened from {function: {name, arguments}} to
{id, name, arguments}.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Generator, List

import requests

OLLAMA_API: str = os.environ.get("OLLAMA_API", "http://127.0.0.1:11434")

log = logging.getLogger("chatd.backends.ollama")

_session: requests.Session = requests.Session()


def _make_tc_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


def _normalize_tool_calls(tcs: List[Dict]) -> List[Dict]:
    """Nested Ollama tool_calls → flat internal shape."""
    out: List[Dict] = []
    for tc in tcs:
        fn = tc.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({
            "id":        tc.get("id") or _make_tc_id(),
            "name":      fn.get("name", ""),
            "arguments": args,
        })
    return out


def _denormalize_tool_calls(tcs: List[Dict]) -> List[Dict]:
    """Flat internal tool_calls → nested Ollama shape for outgoing payload."""
    out: List[Dict] = []
    for tc in tcs:
        args = tc.get("arguments") or {}
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        out.append({
            "id":   tc.get("id") or _make_tc_id(),
            "type": "function",
            "function": {
                "name":      tc.get("name", ""),
                "arguments": args,
            },
        })
    return out


def _normalize_message(msg: Dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "role":    "assistant",
        "content": msg.get("content") or "",
    }
    if msg.get("thinking"):
        out["thinking"] = msg["thinking"]
    if msg.get("tool_calls"):
        out["tool_calls"] = _normalize_tool_calls(msg["tool_calls"])
    return out


def _normalize_response(raw: Dict) -> Dict[str, Any]:
    msg = raw.get("message") or {}
    out: Dict[str, Any] = {
        "model":   raw.get("model", ""),
        "message": _normalize_message(msg),
        "done":    bool(raw.get("done", False)),
    }
    if raw.get("created_at"):
        out["created_at"] = raw["created_at"]
    if out["done"]:
        out["done_reason"] = raw.get("done_reason") or "stop"
        prompt_tokens = int(raw.get("prompt_eval_count") or 0)
        eval_tokens   = int(raw.get("eval_count") or 0)
        if prompt_tokens or eval_tokens:
            out["usage"] = {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": eval_tokens,
            }
    return out


def _prepare_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Outgoing: convert flat internal tool_calls back to Ollama nested shape
    and map `thinking` back to whatever Ollama expects on input.

    Ollama reads prior assistant turns as-is; it does not require
    tool_calls on historical assistant messages to be re-serialised in any
    particular way — but we normalise anyway for round-trip safety.
    """
    prepared = dict(payload)
    msgs_in = payload.get("messages") or []
    msgs_out: List[Dict] = []
    for m in msgs_in:
        m2 = dict(m)
        tcs = m2.get("tool_calls")
        if tcs:
            m2["tool_calls"] = _denormalize_tool_calls(tcs)
        msgs_out.append(m2)
    prepared["messages"] = msgs_out
    return prepared


class OllamaBackend:
    """Ollama local runtime. Default backend for all unprefixed models."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        body = _prepare_payload(payload)
        r = _session.post(
            f"{OLLAMA_API}/api/chat",
            json=body, timeout=3600, stream=True,
        )
        r.raise_for_status()
        with r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("[ollama] bad JSON line: %r", line[:120])
                    continue
                norm = _normalize_response(raw)
                yield (json.dumps(norm, ensure_ascii=False) + "\n").encode("utf-8")

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = _prepare_payload(payload)
        r = _session.post(
            f"{OLLAMA_API}/api/chat",
            json=body, timeout=3600,
        )
        r.raise_for_status()
        return _normalize_response(r.json())

    def embed(self, text: str, model: str) -> List[float]:
        r = _session.post(
            f"{OLLAMA_API}/api/embed",
            json={"model": model, "input": text},
            timeout=120,
        )
        r.raise_for_status()
        embeddings = r.json().get("embeddings") or []
        if not embeddings:
            raise RuntimeError("Ollama embed returned no embeddings")
        return embeddings[0]
