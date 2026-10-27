"""backends/openrouter.py - OpenRouter inference backend.

Wire format: OpenAI /chat/completions over HTTPS.
Embed: not supported (raises RuntimeError).
       Use OllamaBackend.embed() for RAG - it is always available locally.

Model name convention: strip "or/" prefix before sending to OpenRouter.
Example: "or/mistralai/mistral-7b-instruct:free" -> "mistralai/mistral-7b-instruct:free"

tool_call_id contract
---------------------
OpenAI requires that every tool_call in an assistant message carries a
unique `id`, and that the subsequent `tool` message echoes that id as
`tool_call_id`.  chatd uses a simple internal format that does NOT carry
ids through the message list, so we generate/preserve ids here:

  _finalise_tool_calls()  - keeps the `id` collected during SSE streaming
  _to_openai()            - converts assistant/tool messages to OpenAI wire
                            format, injecting generated ids where missing
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import HTTPError

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_BASE: str = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_PREFIX: str = os.environ.get("OPENROUTER_PREFIX", "or/")

log = logging.getLogger("chatd.backends.openrouter")

_PREFIX: str = OPENROUTER_PREFIX

_model_cache: Dict[str, Dict[str, Any]] = {}
_model_cache_loaded: bool = False

# Persistent session - reuses TCP+TLS connections across tool loop rounds.
_session: requests.Session = requests.Session()


def _strip(model: str) -> str:
    return model[len(_PREFIX):] if model.startswith(_PREFIX) else model


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nikita-popov/chatd",
    }


def _load_model_cache() -> None:
    global _model_cache, _model_cache_loaded
    if _model_cache_loaded:
        return
    try:
        r = _session.get(
            f"{OPENROUTER_API_BASE}/models",
            headers=_headers(),
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        models = data.get("data") or []
        _model_cache = {m["id"]: m for m in models if m.get("id")}
        _model_cache_loaded = True
        log.info("[openrouter] model cache loaded: %d models", len(_model_cache))
    except Exception as exc:
        log.warning("[openrouter] failed to load model cache: %s", exc)


def fetch_model_info(model_with_prefix: str) -> Optional[Dict[str, Any]]:
    _load_model_cache()
    model_id = _strip(model_with_prefix)
    result = _model_cache.get(model_id)
    if result is None:
        log.warning("[openrouter] model not found in catalog: %s", model_id)
    return result


def supports_tools(model_with_prefix: str) -> bool:
    _load_model_cache()
    model_id = _strip(model_with_prefix)
    info = _model_cache.get(model_id)
    if info is None:
        return True
    params: List[str] = info.get("supported_parameters") or []
    return "tools" in params


def _make_tc_id() -> str:
    """Generate a short tool_call id compatible with OpenAI format."""
    return "call_" + uuid.uuid4().hex[:16]


def _to_openai(payload: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    """Convert Ollama-format payload to OpenAI wire format.

    Accepts FLAT tool_calls on input:  {"id": ..., "name": ..., "arguments": {...}}
    Emits NESTED tool_calls to OpenAI: {"id": ..., "type": "function",
                                        "function": {"name": ..., "arguments": "..."}}

    Key invariant: every assistant message with tool_calls has an id, and
    the following tool messages echo those ids as tool_call_id.
    """
    model = payload["model"]
    opts = payload.get("options", {})

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
                    # FLAT shape on input
                    name = tc.get("name", "")
                    args = tc.get("arguments") or {}
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    tc_id = tc.get("id") or _make_tc_id()
                    pending_tc_ids.append(tc_id)
                    openai_tcs.append({
                        "id":       tc_id,
                        "type":     "function",
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
            else:
                tc_id = _make_tc_id()
                log.warning("[openrouter] tool message without matching assistant tool_call_id")
            converted.append({
                "role":         "tool",
                "tool_call_id": tc_id,
                "content":      m.get("content") or "",
            })

        else:
            converted.append({"role": role, "content": m.get("content") or ""})

    result: Dict[str, Any] = {
        "model":    _strip(model),
        "messages": converted,
        "stream":   stream,
    }
    if "num_predict" in opts:
        result["max_tokens"] = opts["num_predict"]
    if "temperature" in opts:
        result["temperature"] = opts["temperature"]
    if "top_p" in opts:
        result["top_p"] = opts["top_p"]
    if payload.get("tools"):
        if supports_tools(model):
            result["tools"] = payload["tools"]
            result["tool_choice"] = "auto"
        else:
            log.debug(
                "[openrouter] stripping tools from payload: %s does not support function calling",
                _strip(model),
            )
    return result


def _merge_tool_call_deltas(acc: List[Dict], deltas: List[Dict]) -> List[Dict]:
    """Accumulate streaming tool_call delta fragments by index.

    OpenAI streaming sends tool_calls across multiple chunks:
      chunk 1: [{index:0, id:"call_x", function:{name:"foo", arguments:""}}]
      chunk 2: [{index:0, function:{arguments:'{"k":'}}]
      chunk 3: [{index:0, function:{arguments:'"v"}'}}]
    """
    for delta in deltas:
        idx = delta.get("index", 0)
        while len(acc) <= idx:
            acc.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        entry = acc[idx]
        if delta.get("id") and not entry["id"]:
            entry["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name") and not entry["function"]["name"]:
            entry["function"]["name"] = fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]
    return acc


def _finalise_tool_calls(acc: List[Dict]) -> List[Dict]:
    """Parse accumulated argument strings; emit FLAT tool_calls.

    Returns: [{"id": "call_...", "name": "foo", "arguments": {...}}]
    """
    result = []
    for entry in acc:
        fn = entry.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            log.warning("[openrouter] failed to parse tool_call arguments: %r", args_str[:200])
            args = {}
        result.append({
            "id":        entry.get("id") or _make_tc_id(),
            "name":      fn.get("name", ""),
            "arguments": args,
        })
    return result


def _extract_or_error(response: requests.Response) -> str:
    try:
        body = response.json()
        err = body.get("error") or {}
        msg = err.get("message") or ""
        code = err.get("code")
        return f"{msg} (code={code})" if code else msg or response.text[:200]
    except Exception:
        return response.text[:200]


def _or_response_meta(r: requests.Response) -> str:
    parts = []
    provider = r.headers.get("X-OR-Provider") or r.headers.get("x-or-provider")
    gen_id   = r.headers.get("X-OR-Generation-ID") or r.headers.get("x-or-generation-id")
    if provider:
        parts.append(f"provider={provider}")
    if gen_id:
        parts.append(f"generation={gen_id}")
    return " ".join(parts)


def _log_http_error(exc: HTTPError) -> str:
    r = exc.response
    status = r.status_code if r is not None else 0
    detail = _extract_or_error(r) if r is not None else str(exc)

    extras: List[str] = []
    if r is not None:
        if status == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                extras.append(f"Retry-After={retry_after}s")
        meta = _or_response_meta(r)
        if meta:
            extras.append(meta)

    extra_str = " | " + ", ".join(extras) if extras else ""
    msg = f"[openrouter] HTTP {status}{extra_str}: {detail}"
    if status in (429, 503):
        log.warning(msg)
    else:
        log.error(msg)
    return detail


def _error_chunk(model: str, detail: str, status: int) -> bytes:
    return (json.dumps({
        "model": model,
        "message": {"role": "assistant", "content": f"[OpenRouter error {status}] {detail}"},
        "done": True,
        "done_reason": "error",
    }, ensure_ascii=False) + "\n").encode()


def _sse_to_ndjson(
    response: requests.Response, model: str
) -> Generator[bytes, None, None]:
    """Convert OpenAI SSE stream to normalized internal chunks.

    Emits:
      - thinking deltas as message.thinking
      - text deltas as message.content
      - tool_calls (flat) on the final done chunk
    """
    _tc_acc: List[Dict] = []

    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8").removeprefix("data: ")
        if line.strip() == "[DONE]":
            out: Dict[str, Any] = {
                "model":   model,
                "message": {"role": "assistant", "content": ""},
                "done":    True,
                "done_reason": "stop",
            }
            if _tc_acc:
                finalised = _finalise_tool_calls(_tc_acc)
                out["message"]["tool_calls"] = finalised
                out["done_reason"] = "tool_calls"
                log.debug(
                    "[openrouter] assembled %d tool_call(s) at [DONE]: %s",
                    len(finalised),
                    [tc["name"] for tc in finalised],
                )
            yield (json.dumps(out, ensure_ascii=False) + "\n").encode()
            return

        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        choice = (chunk.get("choices") or [{}])[0]
        delta  = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        # thinking delta → отдельный нормализованный чанк
        if delta.get("reasoning_content"):
            out = {
                "model": model,
                "message": {
                    "role":     "assistant",
                    "content":  "",
                    "thinking": delta["reasoning_content"],
                },
                "done": False,
            }
            yield (json.dumps(out, ensure_ascii=False) + "\n").encode()

        if delta.get("tool_calls"):
            _merge_tool_call_deltas(_tc_acc, delta["tool_calls"])
            continue

        content = delta.get("content") or ""
        if not content and not finish:
            continue

        out = {
            "model":   model,
            "message": {"role": "assistant", "content": content},
            "done":    finish is not None,
        }
        if finish:
            out["done_reason"] = finish
            if _tc_acc:
                finalised = _finalise_tool_calls(_tc_acc)
                out["message"]["tool_calls"] = finalised
                log.debug(
                    "[openrouter] assembled %d tool_call(s) at finish: %s",
                    len(finalised),
                    [tc["name"] for tc in finalised],
                )
        yield (json.dumps(out, ensure_ascii=False) + "\n").encode()


class OpenRouterBackend:
    """OpenRouter cloud runtime. Selected for models prefixed with 'or/'."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        model = payload["model"]
        try:
            r = _session.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                headers=_headers(),
                json=_to_openai(payload, stream=True),
                timeout=3600, stream=True,
            )
            r.raise_for_status()
        except HTTPError as exc:
            detail = _log_http_error(exc)
            status = exc.response.status_code if exc.response is not None else 0
            yield _error_chunk(model, detail, status)
            return
        yield from _sse_to_ndjson(r, model)

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = _session.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                headers=_headers(),
                json=_to_openai(payload, stream=False),
                timeout=3600,
            )
            r.raise_for_status()
        except HTTPError as exc:
            detail = _log_http_error(exc)
            status = exc.response.status_code if exc.response is not None else 0
            raise RuntimeError(f"OpenRouter HTTP {status}: {detail}") from exc

        data   = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg    = choice.get("message") or {}
        usage  = data.get("usage") or {}

        result: Dict[str, Any] = {
            "model":   payload["model"],
            "message": {"role": "assistant", "content": msg.get("content") or ""},
            "done":    True,
            "done_reason": choice.get("finish_reason") or "stop",
        }
        if msg.get("reasoning_content"):
            result["message"]["thinking"] = msg["reasoning_content"]
        if msg.get("tool_calls"):
            tcs = []
            for tc in msg["tool_calls"]:
                fn   = tc.get("function") or {}
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tcs.append({
                    "id":        tc.get("id") or _make_tc_id(),
                    "name":      fn.get("name", ""),
                    "arguments": args,
                })
            result["message"]["tool_calls"] = tcs
            if result["done_reason"] == "stop":
                result["done_reason"] = "tool_calls"
        if usage:
            result["usage"] = {
                "prompt_tokens":     int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens":  int(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                ),
                "cache_hit_tokens":  int(usage.get("prompt_cache_hit_tokens") or 0),
                "cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens") or 0),
            }
        return result

    def embed(self, text: str, model: str) -> List[float]:
        raise RuntimeError(
            "OpenRouterBackend does not support embed. "
            "Configure RAG_EMBED_MODEL on a local Ollama instance."
        )
