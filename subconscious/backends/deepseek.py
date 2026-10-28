"""backends/deepseek.py - DeepSeek inference backend.

Wire format: OpenAI-compatible /chat/completions over HTTPS (SSE).
Embed: not supported.

Internal chunk shape (see backends.base docstring):
  {"model", "message": {"role", "content", "thinking"?,
   "tool_calls"?}, "done", "done_reason"?, "usage"?}

Mapping:
  DeepSeek streaming delta.reasoning_content  →  message.thinking
  DeepSeek non-stream usage                   →  chunk.usage
  DeepSeek finish_reason                      →  chunk.done_reason
  Internal flat tool_calls                    ↔  OpenAI nested tool_calls
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Generator, List

import requests

from subconscious.runtime.config import DEEPSEEK_API

log = logging.getLogger("subconscious.backends.deepseek")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

_PREFIX = "deepseek/"
_session = requests.Session()


def _strip(model: str) -> str:
    return model[len(_PREFIX):] if model.startswith(_PREFIX) else model


def _make_tc_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


# ── outgoing: internal flat → OpenAI nested ──────────────────────────────────

def _to_openai(payload: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    """Convert internal payload to OpenAI wire format.

    Input assistant messages carry FLAT tool_calls:
      {"id": "...", "name": "foo", "arguments": {...}}
    Input assistant messages carry `thinking` (not `reasoning_content`).
    """
    raw_messages: List[Dict] = payload.get("messages", [])
    converted: List[Dict] = []
    pending_tc_ids: List[str] = []
    pending_tc_index: int = 0

    for m in raw_messages:
        role = m.get("role")

        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                openai_tcs = []
                pending_tc_ids = []
                pending_tc_index = 0
                for tc in tcs:
                    name = tc.get("name", "")
                    args = tc.get("arguments") or {}
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    tc_id = tc.get("id") or _make_tc_id()
                    pending_tc_ids.append(tc_id)
                    openai_tcs.append({
                        "id":   tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                entry: Dict[str, Any] = {
                    "role":       "assistant",
                    "content":    m.get("content") or "",
                    "tool_calls": openai_tcs,
                }
                if m.get("thinking"):
                    entry["reasoning_content"] = m["thinking"]
                converted.append(entry)
            else:
                pending_tc_ids = []
                pending_tc_index = 0
                entry = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("thinking"):
                    entry["reasoning_content"] = m["thinking"]
                converted.append(entry)

        elif role == "tool":
            if pending_tc_index < len(pending_tc_ids):
                tc_id = pending_tc_ids[pending_tc_index]
                pending_tc_index += 1
                converted.append({
                    "role":         "tool",
                    "tool_call_id": tc_id,
                    "content":      m.get("content") or "",
                })
            else:
                # Orphaned tool message. DeepSeek rejects the whole payload
                # with HTTP 400 if this reaches the wire. Drop it instead
                # of fabricating a tool_call_id - the model never saw a
                # matching assistant call, so the result is meaningless.
                log.warning(
                    "[deepseek] dropping orphan tool message (%d chars); "
                    "history was likely cut mid-tool-sequence",
                    len(m.get("content") or ""),
                )

        else:
            converted.append({"role": role, "content": m.get("content") or ""})

    body: Dict[str, Any] = {
        "model":    _strip(payload["model"]),
        "messages": converted,
        "stream":   stream,
    }
    if stream:
        # Ask OpenAI-compatible providers (DeepSeek included) to send a
        # trailing usage-only chunk before [DONE]. Without this, no token
        # accounting is available on the streaming path.
        body["stream_options"] = {"include_usage": True}

    options = payload.get("options") or {}
    if options.get("response_format") == "json":
        body["response_format"] = {"type": "json_object"}

    if payload.get("tools"):
        body["tools"] = payload["tools"]
        body["tool_choice"] = "auto"

    for src, dst in (("temperature", "temperature"),
                     ("top_p", "top_p"),
                     ("num_predict", "max_tokens")):
        if src in options:
            body[dst] = options[src]

    return body


# ── streaming tool-call accumulator ──────────────────────────────────────────

def _merge_tc(acc: List[Dict], deltas: List[Dict]) -> List[Dict]:
    for delta in deltas:
        idx = delta.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
        entry = acc[idx]
        if delta.get("id") and not entry["id"]:
            entry["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name") and not entry["function"]["name"]:
            entry["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]
    return acc


def _finalise_tc(acc: List[Dict]) -> List[Dict]:
    """Accumulated OpenAI-style → flat internal shape."""
    result: List[Dict] = []
    for entry in acc:
        fn = entry.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        result.append({
            "id":        entry.get("id") or _make_tc_id(),
            "name":      fn.get("name", ""),
            "arguments": args,
        })
    return result


# ── stream: SSE → normalized NDJSON ──────────────────────────────────────────

def _sse_to_ndjson(response, model: str) -> Generator[bytes, None, None]:
    """DeepSeek SSE (OpenAI delta) → normalized internal NDJSON chunks.

    Emits exactly one done=True chunk, at [DONE], carrying the final
    done_reason, tool_calls (if any) and usage (if stream_options enabled).
    Content and thinking deltas are emitted as done=False chunks as they
    arrive. Never emits an intermediate done=True - chatd breaks on the
    first one and would miss the trailing usage chunk.
    """
    tc_acc: List[Dict] = []
    usage_acc: Dict[str, Any] = {}
    finish_reason: str = "stop"

    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").removeprefix("data: ")

        if line.strip() == "[DONE]":
            out: Dict[str, Any] = {
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "tool_calls" if tc_acc else finish_reason,
            }
            if tc_acc:
                out["message"]["tool_calls"] = _finalise_tc(tc_acc)
            if usage_acc:
                r_tokens = int(
                    (usage_acc.get("completion_tokens_details") or {})
                    .get("reasoning_tokens") or 0
                )
                out["usage"] = {
                    "prompt_tokens":     int(usage_acc.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage_acc.get("completion_tokens") or 0),
                    "cache_hit_tokens":  int(usage_acc.get("prompt_cache_hit_tokens") or 0),
                    "cache_miss_tokens": int(usage_acc.get("prompt_cache_miss_tokens") or 0),
                    "completion_tokens_details": {
                        "reasoning_tokens": r_tokens,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                }
                log.info(
                    "[deepseek stream] prompt_tokens=%s completion_tokens=%s "
                    "reasoning_tokens=%s cache_hit=%s cache_miss=%s",
                    usage_acc.get("prompt_tokens"),
                    usage_acc.get("completion_tokens"),
                    (usage_acc.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                    usage_acc.get("prompt_cache_hit_tokens"),
                    usage_acc.get("prompt_cache_miss_tokens"),
                )
            yield (json.dumps(out, ensure_ascii=False) + "\n").encode()
            return

        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Usage arrives on its own trailing chunk (choices empty).
        if chunk.get("usage"):
            usage_acc = chunk["usage"]

        choice = (chunk.get("choices") or [{}])[0]
        delta  = choice.get("delta") or {}
        finish = choice.get("finish_reason")
        if finish:
            finish_reason = finish

        if delta.get("reasoning_content"):
            yield (json.dumps({
                "model": model,
                "message": {
                    "role":     "assistant",
                    "content":  "",
                    "thinking": delta["reasoning_content"],
                },
                "done": False,
            }, ensure_ascii=False) + "\n").encode()

        if delta.get("tool_calls"):
            _merge_tc(tc_acc, delta["tool_calls"])
            continue

        content = delta.get("content") or ""
        if not content:
            continue

        yield (json.dumps({
            "model":   model,
            "message": {"role": "assistant", "content": content},
            "done":    False,
        }, ensure_ascii=False) + "\n").encode()


# ── backend ──────────────────────────────────────────────────────────────────

class DeepSeekBackend:
    def _headers(self) -> Dict[str, str]:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type":  "application/json",
        }

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """DeepSeek truncates reasoning in non-streaming mode when thinking
        runs long, leaving content empty. Streaming does not have this limit.
        We always stream and accumulate the result."""
        body = _to_openai(payload, stream=True)
        r = _session.post(
            f"{DEEPSEEK_API}/chat/completions",
            headers=self._headers(), json=body, timeout=600, stream=True,
        )
        if not r.ok:
            log.error("[deepseek] HTTP %s body: %s", r.status_code, r.text[:1000])
        r.raise_for_status()

        content_acc = ""
        reasoning_acc = ""
        tc_acc: List[Dict] = []
        finish_reason = "stop"
        usage_acc: Dict[str, Any] = {}

        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8").removeprefix("data: ")
            if line.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            choice = (chunk.get("choices") or [{}])[0]
            delta  = choice.get("delta") or {}

            if delta.get("reasoning_content"):
                reasoning_acc += delta["reasoning_content"]
            if delta.get("tool_calls"):
                _merge_tc(tc_acc, delta["tool_calls"])
                continue
            if delta.get("content"):
                content_acc += delta["content"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if chunk.get("usage"):
                usage_acc = chunk["usage"]

        log.info(
            "[deepseek chat_sync] finish=%s content=%d reasoning=%d "
            "prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s "
            "cache_hit=%s cache_miss=%s",
            finish_reason,
            len(content_acc),
            len(reasoning_acc),
            usage_acc.get("prompt_tokens"),
            usage_acc.get("completion_tokens"),
            (usage_acc.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            usage_acc.get("prompt_cache_hit_tokens"),
            usage_acc.get("prompt_cache_miss_tokens"),
        )

        result: Dict[str, Any] = {
            "model":       payload["model"],
            "message":     {"role": "assistant", "content": content_acc},
            "done":        True,
            "done_reason": finish_reason,
        }
        if tc_acc:
            result["message"]["tool_calls"] = _finalise_tc(tc_acc)
            if finish_reason == "stop":
                result["done_reason"] = "tool_calls"
        if reasoning_acc:
            result["message"]["thinking"] = reasoning_acc
        if usage_acc:
            result["usage"] = {
                "prompt_tokens":     int(usage_acc.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_acc.get("completion_tokens") or 0),
                "reasoning_tokens":  int(
                    (usage_acc.get("completion_tokens_details") or {})
                    .get("reasoning_tokens") or 0
                ),
                "cache_hit_tokens":  int(usage_acc.get("prompt_cache_hit_tokens") or 0),
                "cache_miss_tokens": int(usage_acc.get("prompt_cache_miss_tokens") or 0),
            }

        return result

    def chat_stream(self, payload: Dict[str, Any]) -> Generator[bytes, None, None]:
        body = _to_openai(payload, stream=True)
        r = _session.post(
            f"{DEEPSEEK_API}/chat/completions",
            headers=self._headers(), json=body, timeout=600, stream=True,
        )
        if not r.ok:
            log.error("[deepseek] HTTP %s body: %s", r.status_code, r.text[:2000])
        r.raise_for_status()
        yield from _sse_to_ndjson(r, payload["model"])

    def embed(self, text: str, model: str):
        raise RuntimeError("DeepSeekBackend does not support embed.")
