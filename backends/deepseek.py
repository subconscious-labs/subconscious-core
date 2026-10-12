import json
import logging
import os
import requests
import uuid
from typing import Any, Dict, Generator, List

from config import DEEPSEEK_API

log = logging.getLogger("chatd.backends.deepseek")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

_PREFIX = "deepseek/"
_session = requests.Session()


def _strip(model: str) -> str:
    return model[len(_PREFIX):] if model.startswith(_PREFIX) else model


def _make_tc_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


def _to_openai(payload: Dict[str, Any], stream: bool) -> Dict[str, Any]:
    """Convert Ollama-format payload to OpenAI wire format with proper tool ids."""
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
                    fn = tc.get("function") or {}
                    args = fn.get("arguments") or {}
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    tc_id = tc.get("id") or _make_tc_id()
                    pending_tc_ids.append(tc_id)
                    openai_tcs.append({
                        "id":   tc_id,
                        "type": "function",
                        "function": {
                            "name":      fn.get("name", ""),
                            "arguments": args,
                        },
                    })
                entry: Dict[str, Any] = {
                    "role":       "assistant",
                    "content":    m.get("content") or "",
                    "tool_calls": openai_tcs,
                }
                if m.get("reasoning_content"):
                    entry["reasoning_content"] = m["reasoning_content"]
                converted.append(entry)
            else:
                # No tool_calls - DO NOT emit the key at all (empty [] is rejected)
                pending_tc_ids = []
                pending_tc_index = 0
                entry = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("reasoning_content"):
                    entry["reasoning_content"] = m["reasoning_content"]
                converted.append(entry)

        elif role == "tool":
            if pending_tc_index < len(pending_tc_ids):
                tc_id = pending_tc_ids[pending_tc_index]
                pending_tc_index += 1
            else:
                tc_id = _make_tc_id()
                log.warning("[deepseek] tool message without matching assistant tool_call_id")
            converted.append({
                "role":         "tool",
                "tool_call_id": tc_id,
                "content":      m.get("content") or "",
            })

        else:
            converted.append({"role": role, "content": m.get("content") or ""})

    body: Dict[str, Any] = {
        "model":    _strip(payload["model"]),
        "messages": converted,
        "stream":   stream,
    }

    # Force JSON output for planner/reflector-like calls where the caller
    # expects a strict JSON object. DeepSeek Flash always emits reasoning
    # and may otherwise wrap its JSON in prose.
    options = payload.get("options") or {}
    if options.get("response_format") == "json":
        body["response_format"] = {"type": "json_object"}

    if payload.get("tools"):
        body["tools"] = payload["tools"]
        body["tool_choice"] = "auto"

    options = payload.get("options") or {}
    for src, dst in (("temperature", "temperature"),
                     ("top_p", "top_p"),
                     ("num_predict", "max_tokens")):
        if src in options:
            body[dst] = options[src]

    return body


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
    result = []
    for entry in acc:
        fn = entry.get("function") or {}
        args_str = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        result.append({
            "id":       entry.get("id") or _make_tc_id(),
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    return result


def _sse_to_ndjson(response, model: str) -> Generator[bytes, None, None]:
    """DeepSeek SSE (OpenAI delta) → Ollama NDJSON chunks."""
    tc_acc: List[Dict] = []
    reasoning_acc: str = ""

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
            if tc_acc:
                out["message"]["tool_calls"] = _finalise_tc(tc_acc)
            if reasoning_acc:
                out["message"]["reasoning_content"] = reasoning_acc
            yield (json.dumps(out, ensure_ascii=False) + "\n").encode()
            return

        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue

        choice = (chunk.get("choices") or [{}])[0]
        delta  = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        if delta.get("reasoning_content"):
            reasoning_acc += delta["reasoning_content"]

        if delta.get("tool_calls"):
            _merge_tc(tc_acc, delta["tool_calls"])
            continue

        content = delta.get("content") or ""

        out = {
            "model":   model,
            "message": {"role": "assistant", "content": content},
            "done":    finish is not None,
        }
        if finish:
            out["done_reason"] = finish
            if tc_acc:
                out["message"]["tool_calls"] = _finalise_tc(tc_acc)
            if reasoning_acc:
                out["message"]["reasoning_content"] = reasoning_acc
        yield (json.dumps(out, ensure_ascii=False) + "\n").encode()


class DeepSeekBackend:
    def _headers(self) -> Dict[str, str]:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        return {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type":  "application/json",
        }

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """DeepSeek Flash truncates reasoning in non-streaming mode when
        thinking runs long, leaving content empty. Streaming does not have
        this limit. We always stream and accumulate the result."""
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
            delta = choice.get("delta") or {}

            if delta.get("reasoning_content"):
                reasoning_acc += delta["reasoning_content"]
            if delta.get("tool_calls"):
                _merge_tc(tc_acc, delta["tool_calls"])
                continue
            if delta.get("content"):
                content_acc += delta["content"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            # DeepSeek sends usage in the last chunk before [DONE].
            if chunk.get("usage"):
                usage_acc = chunk["usage"]

        log.info("[deepseek chat_sync] finish=%s content=%d reasoning=%d "
                 "prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s",
                 finish_reason,
                 len(content_acc),
                 len(reasoning_acc),
                 usage_acc.get("prompt_tokens"),
                 usage_acc.get("completion_tokens"),
                 (usage_acc.get("completion_tokens_details") or {}).get("reasoning_tokens"))

        result: Dict[str, Any] = {
            "model":       payload["model"],
            "message":     {"role": "assistant", "content": content_acc},
            "done":        True,
            "done_reason": finish_reason,
        }

        if tc_acc:
            result["message"]["tool_calls"] = _finalise_tc(tc_acc)
        if reasoning_acc:
            result["message"]["reasoning_content"] = reasoning_acc

        return result

    def chat_stream(self, payload: Dict[str, Any]) -> Generator[bytes, None, None]:
        body = _to_openai(payload, stream=True)
        r = _session.post(
            f"{DEEPSEEK_API}/chat/completions",
            headers=self._headers(), json=body, timeout=600, stream=True,
        )
        if not r.ok:
            # Log the body - this is where your 400 detail lives
            log.error("[deepseek] HTTP %s body: %s", r.status_code, r.text[:2000])
        r.raise_for_status()
        yield from _sse_to_ndjson(r, payload["model"])

    def embed(self, text: str, model: str):
        raise RuntimeError("DeepSeekBackend does not support embed.")
