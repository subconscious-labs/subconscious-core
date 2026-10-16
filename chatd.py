#!/usr/bin/env python3
import atexit
import hashlib
import json
import logging
import os
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from requests.exceptions import ReadTimeout, RequestException
from flask import Flask, request, Response, jsonify, stream_with_context

import backends
import background
import internal_tools
import memory
import rag
import session as sess
from config import (
    THINKING,
    TOOLS_FILTER,
    TOOLS_ALLOWED,
    DEFAULT_OPTIONS,
    MAX_TOOL_ROUNDS,
    MAX_HISTORY_TURNS,
    MEMPALACE_WRITE_TOOLS,
    TOOL_OVERRIDE,
    TOOL_DESCRIPTION_OVERRIDES,
    CHAT_TOOL_RESULT_MAX_CHARS,
    CHATD_EVENT_TOKEN,
    CHATD_EVENT_MODEL,
    OPENROUTER_API_MODELS,
    DEEPSEEK_API,
    DEEPSEEK_API_MODELS,
    BG_ENABLED,
    BG_TOKEN,
)
from backends.ollama import OLLAMA_API
from backends.openrouter import fetch_model_info, OPENROUTER_PREFIX
from mcp_client import MCPClient, discover_mcp_servers


# ── logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatd")

# ── app ────────────────────────────────────────────────────────────────────────

VERSION = "2.0.0"

TOOLS: List[Dict] = []
TOOL_REGISTRY: Dict[str, MCPClient] = {}

# All started MCP clients - used for clean shutdown via atexit.
_MCP_CLIENTS: List[MCPClient] = []

app = Flask(__name__)
app.json.ensure_ascii = False

# How much to boost num_predict for tool follow-up rounds.
TOOL_ROUND_NUM_PREDICT_BOOST = 1024

# Extra tokens guaranteed for the final answer round (no tool_calls).
EVENT_FINAL_ROUND_NUM_PREDICT = 2048

# Maximum seconds to wait for the background event worker before logging a warning.
EVENT_TOOL_LOOP_TIMEOUT = 120

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
    "Do NOT skip STEP 1. Do NOT write a plain-text summary without calling send_message."
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _tc_signature(tc_list: List[Dict]) -> str:
    """Stable hash of tool_calls, including arguments.

    Expects FLAT internal form:
      [{"id": "...", "name": "...", "arguments": {...}}]
    """
    parts = []
    for tc in tc_list:
        name = tc.get("name") or ""
        args = tc.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args, sort_keys=True, ensure_ascii=False)
        parts.append(f"{name}:{args}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


def make_chunk(model: str, content: str, done: bool = False) -> bytes:
    obj: Dict[str, Any] = {
        "model":    model,
        "created_at": now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        obj["done_reason"] = "stop"
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def make_keepalive(model: str) -> bytes:
    return make_chunk(model, "")


def make_event_chunk(
    model: str,
    event_type: str,
    payload: Dict[str, Any],
) -> bytes:
    """Typed event chunk. Clients MUST NOT append to message history."""
    obj: Dict[str, Any] = {
        "model":      model,
        "created_at": now_iso(),
        "message":    {"role": "assistant", "content": ""},
        "event":      {"type": event_type, **payload},
        "done":       False,
    }
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def make_thinking_chunk(model: str, delta: str) -> bytes:
    """Thinking delta, dual-channel.

    Emits `message.thinking` for Ollama-native clients (Hollama reads
    this field for Qwen3-style thinking). Also emits
    `event.type == "thinking"` for event-aware clients.
    """
    obj: Dict[str, Any] = {
        "model":      model,
        "created_at": now_iso(),
        "message": {
            "role":     "assistant",
            "content":  "",
            "thinking": delta,
        },
        "event": {"type": "thinking", "delta": delta},
        "done": False,
    }
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def make_done_chunk(
    model: str,
    done_reason: str = "stop",
    usage: Optional[Dict[str, Any]] = None,
) -> bytes:
    obj: Dict[str, Any] = {
        "model":       model,
        "created_at":  now_iso(),
        "message":     {"role": "assistant", "content": ""},
        "done":        True,
        "done_reason": done_reason,
    }
    if usage:
        obj["usage"] = usage
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _log_yield(req_id: str, label: str, data: bytes) -> bytes:
    """Log a yielded chunk and return it unchanged."""
    log.debug("[%s] yield %-12s %d bytes", req_id, label, len(data))
    return data


def proxy_get(path: str) -> Response:
    r = requests.get(f"{OLLAMA_API}{path}", timeout=600)
    return Response(
        r.content,
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


def proxy_post(path: str, json_body: Dict[str, Any]) -> Response:
    r = requests.post(f"{OLLAMA_API}{path}", json=json_body, timeout=600, stream=True)
    return Response(
        r.iter_content(chunk_size=None),
        status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/x-ndjson"),
    )


def merge_options(client_options: Optional[Dict]) -> Dict[str, Any]:
    opts = dict(DEFAULT_OPTIONS)
    if client_options:
        opts.update(client_options)
    return opts


def extract_chat_id(flask_request) -> Optional[str]:
    """Extract chat session ID from the Referer header.

    Hollama sends requests from /sessions/<id> - the browser automatically
    includes this URL as the Referer header, so we parse it from there.
    Falls back to scanning messages for a legacy chatId field.
    """
    referer = flask_request.headers.get("Referer", "")
    if referer:
        m = re.search(r"/sessions/([^/?#]+)", referer)
        if m:
            return m.group(1)

    messages = flask_request.get_json(silent=True) or {}
    for msg in (messages.get("messages") or []):
        chat_id = msg.get("chatId")
        if chat_id is not None:
            return str(chat_id)

    return None


def _last_user_text(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "")
    return ""


def _check_event_token() -> Optional[Tuple[Response, int]]:
    """Return (Response, status) tuple if CHATD_EVENT_TOKEN is set and Bearer doesn't match."""
    if not CHATD_EVENT_TOKEN:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "missing Authorization header"}), 401
    token = auth.removeprefix("Bearer ").strip()
    if token != CHATD_EVENT_TOKEN:
        return jsonify({"error": "invalid token"}), 401
    return None


# ── system prompt assembly ─────────────────────────────────────────────────────

def ensure_system_prompt(
    messages: List[Dict[str, Any]],
    session: Optional[sess.Session] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Assemble the always-loaded memory block and prepend it as system message.

    L0+L1 (wake-up base) is cached per session - rebuilt only on the first
    request of each session, or after a mempalace write operation invalidates
    the cache.  L1.5 (KG recall + RAG sidecar) is always computed per-request
    because it depends on the current user message.

    Returns (messages, layer_sizes) where layer_sizes maps layer name to char count:
      L0+L1     mempalace identity + wake-up base (Essential Story included)
      L1.5_kg   KG recall sidecar
      L1.5_rag  RAG recall sidecar

    Note: L0.5g (global summary) is intentionally omitted - MemPalace wake-up
    already covers the same information.
    """
    # L0+L1: use session cache when available
    if session is not None:
        base_prompt = session.get_cached_base_prompt()
        if base_prompt is None:
            base_prompt = memory.wake_up()
            session.set_cached_base_prompt(base_prompt)
        else:
            log.debug("[syspr] session %s: reusing cached base prompt (%d chars)",
                      session.session_id, len(base_prompt))
    else:
        base_prompt = memory.wake_up()

    layer_sizes: Dict[str, int] = {
        "L0+L1":   len(base_prompt),
        "L1.5_kg": 0,
        "L1.5_rag": 0,
    }

    # L1.5: always per-request - KG and RAG depend on the current question
    last_user = _last_user_text(messages)
    if last_user:
        rag_ctx = rag.retrieve(last_user)
        if rag_ctx:
            layer_sizes["L1.5_rag"] = len(rag_ctx)
            log.debug("[sidecar] injecting %d-char RAG recall into system prompt", len(rag_ctx))
            base_prompt = f"{base_prompt}\n\n# Relevant past context\n{rag_ctx}"

    system_prompt = base_prompt

    if not isinstance(messages, list) or not messages:
        return [{"role": "system", "content": system_prompt}], layer_sizes
    if messages[0].get("role") != "system":
        result = [{"role": "system", "content": system_prompt}] + messages
    else:
        result = [{"role": "system", "content": system_prompt}] + messages[1:]
    return result, layer_sizes


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
        content = m.get("content", "") or ""
        if role == "assistant":
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            # legacy cleanup — old sessions may contain [→ name] markers
            content = re.sub(r"\[→\s*[^\]]+\]\s*", "", content).strip()
            content = _strip_dsml(content)

        if role == "system":
            system = {"role": "system", "content": content}
            continue

        entry: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            # NOTE: thinking/reasoning is deliberately dropped from history.
            # Reasoning is display-only; re-feeding it bloats context and
            # DeepSeek rejects reasoning_content in input on some API
            # versions. Backends therefore never see it here.
        turns.append(entry)

    if len(turns) > max_history_turns * 2:
        turns = turns[-(max_history_turns * 2):]

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
    payload: Dict[str, Any] = {
        "model":    model,
        "messages": messages,
        "stream":   stream,
        "options":  options,
    }
    if TOOLS:
        payload["tools"] = TOOLS
    if not THINKING:
        payload["think"] = False
    return payload


# ── tool loop helper ──────────────────────────────────────────────────────────

_DSML_MARKER = "｜｜DSML｜｜"
_DSML_TRIGGER = "｜"


class DSMLFilter:
    """Stateful stripper for DeepSeek DSML tool-call markup.

    DSML arrives as plain text inside `message.content` when DeepSeek Flash
    wants to call a tool but tools were stripped from the payload. The
    markup looks like:

        <｜｜DSML｜｜ calls>
        <｜｜DSML｜｜ invoke name="foo">
        ...
        </｜｜DSML｜｜ calls>

    In streaming mode the marker can be split across chunks, so a per-chunk
    regex won't catch it. This filter buffers from the first `｜` until the
    next newline, then decides: if the buffered line contains the marker,
    drop it entirely; otherwise emit it unchanged.

    A safety cap (BUFFER_CAP) flushes the buffer if no newline arrives, so
    a stray `｜` in user text cannot stall the stream.
    """

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


def run_tool_loop(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    layer_sizes: Dict[str, int],
    req_id: str,
    final_num_predict: Optional[int] = None,
) -> str:
    """Run the non-streaming tool loop and return the final assistant text."""
    prev_tc_sig: Optional[str] = None
    rep = 0
    resp: Dict = {}
    backend = backends.get_backend(model)

    for i in range(MAX_TOOL_ROUNDS):
        round_options = _options_for_round(options, i)
        built = build_model_messages(messages)

        is_last_round = (i == MAX_TOOL_ROUNDS - 1)
        if is_last_round:
            log.info("[%s] tool_loop round %d: tools stripped - forcing final answer",
                     req_id, i)
            # Mutate the LOCAL built list, not `messages` — callers rely on
            # _last_user_text(messages) after we return, and appending a
            # synthetic user turn would corrupt that.
            built.append({
                "role": "user",
                "content": (
                    "You have reached the tool-call limit. Provide your final "
                    "answer NOW using plain text only. Do NOT emit tool calls, "
                    "XML, or any special markup. A concise natural-language "
                    "summary of what you found is required."
                ),
            })

        payload = make_ollama_payload(model, built, round_options, stream=False)

        if is_last_round:
            payload.pop("tools", None)

        _log_payload_sizes(req_id, built, len(TOOLS), layer_sizes if i == 0 else None)
        log.debug("[%s] tool_loop round %d", req_id, i)

        resp       = backend.chat_sync(payload)
        msg        = resp.get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            content = _strip_dsml((msg.get("content") or "").strip())
            if not content and final_num_predict:
                log.debug(
                    "[%s] empty answer on round %d, retrying with num_predict=%d",
                    req_id, i, final_num_predict,
                )
                boosted = dict(round_options)
                boosted["num_predict"] = final_num_predict
                payload2 = make_ollama_payload(model, built, boosted, stream=False)
                resp2 = backend.chat_sync(payload2)
                content2 = _strip_dsml(
                    ((resp2.get("message") or {}).get("content") or "").strip()
                )
                if content2:
                    return content2

            break

        tc_names = [tc.get("name") for tc in tool_calls]
        log.info("[%s] tool_calls: %s", req_id, tc_names)

        current_sig = _tc_signature(tool_calls)
        if current_sig == prev_tc_sig:
            rep += 1
            if rep >= 3:
                break
        else:
            rep = 0
        prev_tc_sig = current_sig

        assistant_msg: Dict[str, Any] = {
            "role":       "assistant",
            "content":    msg.get("content") or "",
            "tool_calls": tool_calls,
        }

        if msg.get("thinking"):
            assistant_msg["thinking"] = msg["thinking"]

        messages.append(assistant_msg)

        for tc in tool_calls:
            name = tc.get("name") or "unknown"
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = call_tool(name, args)
            if isinstance(result, dict) and "error" in result:
                log.warning("[%s] tool %s error: %s", req_id, name, result["error"])
            messages.append({
                "role":    "tool",
                "content": _truncate_chat_tool_result(result),
            })

    final_text = (resp.get("message") or {}).get("content") or ""
    return _strip_dsml(final_text).strip()


# ── CORS ───────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/chat",  methods=["OPTIONS"])
@app.route("/api/event", methods=["OPTIONS"])
@app.route("/api/tick",  methods=["OPTIONS"])
@app.route("/chat",      methods=["OPTIONS"])
def options_chat():
    return Response(status=204)


# ── tools ──────────────────────────────────────────────────────────────────────

def load_tools():
    tools    = []
    registry = {}
    clients  = []

    for server_name, cfg in discover_mcp_servers().items():
        client = MCPClient(cfg, name=server_name)
        try:
            client.start()
        except Exception as e:
            log.error("[tools] %s: failed to start MCP server: %s", server_name, e)
            continue

        clients.append(client)

        try:
            server_tools = client.list_tools()
        except Exception as e:
            log.error("[tools] %s: failed to list tools: %s", server_name, e)
            continue
        log.info("[tools] %s: %d tools loaded", server_name, len(server_tools))

        for t in server_tools:
            name         = getattr(t, "name", None)
            description  = getattr(t, "description", "") or ""
            input_schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)

            if not name or not input_schema:
                log.warning("[tools] %s: skipping tool without name/schema: %r", server_name, t)
                continue

            if TOOL_OVERRIDE and name in TOOL_DESCRIPTION_OVERRIDES:
                description = TOOL_DESCRIPTION_OVERRIDES.get(name, description)

            registry[name] = client

            if TOOLS_FILTER:
                if name not in TOOLS_ALLOWED:
                    log.debug("[tools] hiding from model context: %s", name)
                    continue

            tools.append({
                "type": "function",
                "function": {
                    "name":        name,
                    "description": description,
                    "parameters":  input_schema,
                },
            })
            log.debug("[tools] registered: %s", name)

    # Internal tools (in-process, not MCP) — chat-only.
    for schema in internal_tools.get_schemas():
        fn = schema.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        tools.append(schema)
        registry[name] = None      # marker: not an MCP client
        log.debug("[tools] registered internal: %s", name)

    return tools, registry, clients


def _stop_all_mcp_clients() -> None:
    for client in _MCP_CLIENTS:
        try:
            client.stop()
        except Exception as e:
            log.debug("[mcp] atexit stop error: %s", e)


def init_tools():
    global TOOLS, TOOL_REGISTRY, _MCP_CLIENTS
    log.info("[tools] initializing MCP tools...")
    TOOLS, TOOL_REGISTRY, _MCP_CLIENTS = load_tools()
    import mcp_registry
    mcp_registry.reset(TOOLS, TOOL_REGISTRY)
    atexit.register(_stop_all_mcp_clients)
    log.info("[tools] total: %d tools in registry", len(TOOLS))


def call_tool(name: str, arguments: Dict[str, Any]) -> Any:
    """Invoke a tool by name."""
    if internal_tools.is_internal(name):
        log.debug("[tool] calling internal %s with args: %s", name, arguments)
        result = internal_tools.call_internal(name, arguments)
        if name in ("subcortex_goals_create", "subcortex_goals_cancel"):
            # Ledger changed — goal list may be embedded in cached wake-up.
            sess.invalidate_all_prompt_caches()
        return result

    client = TOOL_REGISTRY.get(name)
    if client is None:
        log.error("[tool] unknown tool requested: %s", name)
        return {"error": f"Unknown tool: {name}", "tool": name, "args": arguments}

    log.debug("[tool] calling %s with args: %s", name, arguments)
    try:
        result = client.call_tool(name, arguments)
    except Exception as exc:
        log.error("[tool] %s raised exception: %s", name, exc)
        return {"error": str(exc), "tool": name, "args": arguments}

    log.debug("[tool] %s result: %s", name, str(result)[:300])

    if isinstance(result, dict) and result.get("success") is False:
        err_msg = result.get("error") or "unknown MCP error"
        log.warning("[tool] %s returned MCP error: %s", name, err_msg)
        return {
            "error":       err_msg,
            "tool":        name,
            "args":        arguments,
            "mcp_result":  result,
        }

    if name in MEMPALACE_WRITE_TOOLS:
        memory.invalidate()
        sess.invalidate_all_prompt_caches()

    return result


# ── augmented /api/tags ────────────────────────────────────────────────────────

def _or_model_to_tag(model_name: str) -> Optional[Dict[str, Any]]:
    info = fetch_model_info(model_name)
    if info is None:
        return None

    context_length: int = info.get("context_length") or 0
    architecture: str   = (info.get("architecture") or {}).get("modality") or "text"
    description: str    = info.get("description") or ""

    return {
        "name":        model_name,
        "model":       model_name,
        "modified_at": now_iso(),
        "size":        context_length,
        "digest":      "",
        "details": {
            "family":         "openrouter",
            "families":       ["openrouter"],
            "format":         "api",
            "parameter_size": description[:64] if description else "",
            "quantization_level": architecture,
        },
    }


def _build_or_tags() -> List[Dict[str, Any]]:
    result = []
    for model_name in OPENROUTER_API_MODELS:
        entry = _or_model_to_tag(model_name)
        if entry:
            result.append(entry)
            log.debug("[tags] OR model added: %s", model_name)
        else:
            log.warning("[tags] OR model skipped (fetch failed): %s", model_name)
    return result


def _build_deepseek_tags() -> List[Dict[str, Any]]:
    return [
        {
            "name": f"deepseek/{model}",
            "model": f"deepseek/{model}",
            "modified_at": now_iso(),
            "size": 0,
            "digest": "deepseek-api",
            "details": {
                "parent_model": "",
                "format": "api",
                "family": "deepseek",
                "families": ["deepseek"],
                "parameter_size": "API",
                "quantization_level": "remote",
            },
        }
        for model in DEEPSEEK_API_MODELS
    ]


# ── routes ──────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/api")
def root():
    return Response("Ollama is running", status=200, mimetype="text/plain")


@app.get("/version")
@app.get("/api/version")
def version():
    r = requests.get(f"{OLLAMA_API}/api/version")
    ollama_version = r.json()['version']
    log.debug("ollama version: %s", ollama_version)
    return jsonify({"chatd": f"{VERSION}", "ollama": f"{ollama_version}"})


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "tools": len(TOOLS)})


@app.post("/pull")
@app.post("/api/pull")
def pull():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.error("pull: bad JSON: %s", e)
        return jsonify({"error": "invalid JSON"}), 400

    model_name = payload.get("name") or payload.get("model") or ""
    log.debug("[pull] fake success for model %s", model_name)

    def _stream() -> Generator[bytes, None, None]:
        chunk = {
            "status":    "success",
            "digest":    "",
            "total":     1,
            "completed": 1,
            "model":     model_name,
        }
        yield (json.dumps(chunk, ensure_ascii=False) + "\n").encode("utf-8")

    return Response(
        _stream(),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/show")
@app.post("/api/show")
def show_model():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.error("show: bad JSON: %s", e)
        return jsonify({"error": "invalid JSON"}), 400

    model_name = payload.get("name") or payload.get("model")
    if not model_name:
        return jsonify({"error": "missing model name"}), 400

    if model_name.startswith(OPENROUTER_PREFIX):
        info = fetch_model_info(model_name)
        if info is None:
            return jsonify({"error": f"model not found: {model_name}"}), 404

        description = info.get("description") or ""
        context_len = info.get("context_length") or 0
        arch = (info.get("architecture") or {}).get("modality") or "text"

        result = {
            "model": model_name,
            "modified_at": now_iso(),
            "size": context_len,
            "digest": "",
            "details": {
                "format": "api",
                "family": "openrouter",
                "families": ["openrouter"],
                "parameter_size": description[:64],
                "quantization_level": arch,
            },
            "capabilities": ["chat", "tools"],
        }
        return jsonify(result)

    if model_name.startswith("deepseek/"):
        model_name = model_name.removeprefix("deepseek/")

        return jsonify({
            "license": "DeepSeek API",
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": {
                "parent_model": "",
                "format": "api",
                "family": "deepseek",
                "families": ["deepseek"],
                "parameter_size": "API",
                "quantization_level": "remote",
            },
            "model_info": {
                "general.architecture": "deepseek",
                "general.basename": model_name,
            },
        })

    try:
        r = requests.post(
            f"{OLLAMA_API}/api/show",
            json={"name": model_name},
            timeout=600,
        )
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )
    except Exception as e:
        log.error("[show] proxy failed: %s", e)
        return jsonify({"error": str(e)}), 502


@app.get("/tags")
@app.get("/api/tags")
def tags():
    r = requests.get(f"{OLLAMA_API}/api/tags", timeout=600)

    try:
        data = r.json()
    except Exception:
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )

    if not r.ok:
        return jsonify(data), r.status_code

    ollama_entries = list(data.get("models") or [])
    or_entries = _build_or_tags()
    deepseek_entries = _build_deepseek_tags()

    merged, seen = [], set()
    for entry in ollama_entries + or_entries + deepseek_entries:
        key = entry.get("model") or entry.get("name")
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
    data["models"] = merged

    log.info(
        "[tags] ollama=%d or=%d deepseek=%d total=%d",
        len(ollama_entries), len(or_entries), len(deepseek_entries), len(merged),
    )
    return jsonify(data)


def _options_for_round(options: Dict[str, Any], round_num: int) -> Dict[str, Any]:
    if round_num == 0:
        return options
    boosted = dict(options)
    base = boosted.get("num_predict", 512)
    boosted["num_predict"] = base + TOOL_ROUND_NUM_PREDICT_BOOST
    log.debug("tool round %d: num_predict %d -> %d", round_num, base, boosted["num_predict"])
    return boosted


def chat_stream_generator(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    session: Optional[sess.Session],
    layer_sizes: Dict[str, int],
    req_id: str = "-",
) -> Generator[bytes, None, None]:
    prev_tool_calls: Optional[List[str]] = None
    repeat_count = 0

    last_user_msg = _last_user_text(messages)

    data = make_keepalive(model)
    log.debug("[%s] yield %-12s %d bytes", req_id, "keepalive/0", len(data))
    yield data

    final_assistant_content = ""
    backend = backends.get_backend(model)

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info("[%s] stream round %d, messages=%d", req_id, round_num, len(messages))

            round_options = _options_for_round(options, round_num)
            built_messages = build_model_messages(messages)
            payload = make_ollama_payload(model, built_messages, round_options, stream=True)

            is_last_round = (round_num == MAX_TOOL_ROUNDS - 1)
            if is_last_round:
                payload.pop("tools", None)
                log.info("[%s] round %d: tools stripped - forcing final answer",
                         req_id, round_num)

            _log_payload_sizes(req_id, built_messages, len(TOOLS),
                               layer_sizes if round_num == 0 else None)
            log.debug(
                "[%s] -> backend payload (messages=%d, tools=%d, options=%s, think=%s)",
                req_id, len(built_messages), len(TOOLS), round_options, THINKING,
            )

            last_tool_calls: Optional[List[Dict]] = None
            last_thinking: str = ""
            last_content: str = ""
            last_content_raw: str = ""
            dsml_filter = DSMLFilter()
            chunk_count = 0
            stream_done_reason = "stop"
            stream_usage: Optional[Dict[str, Any]] = None

            try:
                stream = backend.chat_stream(payload)
            except Exception as e:
                log.error("[%s] backend stream init failed: %s", req_id, e)
                yield make_chunk(model, f"\n[Ошибка соединения: {e}]\n", done=True)
                return

            try:
                for line in stream:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(
                            line.decode("utf-8") if isinstance(line, bytes) else line
                        )
                    except json.JSONDecodeError as e:
                        log.warning("[%s] bad JSON line: %s | %r", req_id, e, line[:120])
                        continue

                    chunk_count += 1
                    msg = chunk.get("message") or {}
                    done = chunk.get("done", False)

                    thinking = msg.get("thinking") or ""
                    content = msg.get("content") or ""

                    if thinking:
                        last_thinking += thinking
                        yield _log_yield(
                            req_id, f"think/{chunk_count}",
                            make_thinking_chunk(model, thinking),
                        )

                    if msg.get("tool_calls"):
                        log.info("[%s] tool_calls: %s", req_id,
                                 [c.get("name") for c in msg["tool_calls"]])
                        last_tool_calls = msg["tool_calls"]

                    if content and not msg.get("tool_calls"):
                        last_content_raw += content
                        cleaned = dsml_filter.feed(content)
                        if cleaned:
                            last_content += cleaned
                            yield _log_yield(
                                req_id, f"chunk/{chunk_count}",
                                make_chunk(model, cleaned, done=False),
                            )

                    if done:
                        stream_done_reason = chunk.get("done_reason", "stop")
                        stream_usage = chunk.get("usage")
                        log.info(
                            "[%s] round %d done, chunks=%d, tool_calls=%s, "
                            "content_len=%d, done_reason=%s",
                            req_id, round_num, chunk_count,
                            bool(last_tool_calls), len(last_content),
                            stream_done_reason,
                        )
                        break

            except RequestException as e:
                log.error("[%s] network error during stream: %s", req_id, e)
                yield make_chunk(
                    model,
                    f"\n[Ошибка сети: {type(e).__name__}: {e}]\n",
                    done=True,
                )
                return

            tail = dsml_filter.flush()
            if tail:
                last_content += tail
                yield _log_yield(
                    req_id, "chunk/tail",
                    make_chunk(model, tail, done=False),
                )

            log.debug("[%s] exited stream, last_tool_calls=%s",
                      req_id, bool(last_tool_calls))

            if not last_tool_calls:
                final_assistant_content = last_content
                if not final_assistant_content and _contains_dsml(last_content_raw):
                    log.warning(
                        "[%s] DSML-only response, no text after strip", req_id,
                    )
                    final_assistant_content = (
                        "[Не удалось получить текстовый ответ: модель "
                        "попыталась вызвать инструменты после исчерпания "
                        "лимита раундов. Увеличьте SUBCONSCIOUS_MAX_TOOL_ROUNDS "
                        "или переформулируйте запрос.]"
                    )
                yield _log_yield(
                    req_id, "done",
                    make_done_chunk(model, stream_done_reason, stream_usage),
                )
                log.info("[%s] stream complete, no more tool rounds", req_id)
                break

            current_sig = _tc_signature(last_tool_calls)
            prev_sig = _tc_signature(prev_tool_calls) if prev_tool_calls else None

            if current_sig == prev_sig:
                repeat_count += 1
                if repeat_count >= 3:
                    log.warning("[%s] tool loop detected (%s x%d), aborting",
                                req_id, current_sig, repeat_count)
                    break
            else:
                repeat_count = 0
            prev_tool_calls = last_tool_calls

            # assistant turn — flat tool_calls + thinking
            assistant_entry: Dict[str, Any] = {
                "role":       "assistant",
                "content":    last_content,
                "tool_calls": last_tool_calls,
            }
            if last_thinking:
                assistant_entry["thinking"] = last_thinking
            messages.append(assistant_entry)

            # tool_call + tool_result as events
            for tc in last_tool_calls:
                tool_name = tc.get("name") or "unknown"
                tool_args = tc.get("arguments") or {}
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:16]}"

                yield _log_yield(
                    req_id, "tool/call",
                    make_event_chunk(model, "tool_call", {
                        "id":        call_id,
                        "name":      tool_name,
                        "arguments": tool_args,
                    }),
                )

                log.info("[%s] executing tool: %s(%s)", req_id, tool_name, tool_args)
                tool_result = call_tool(tool_name, tool_args)

                status = "error" if (
                    isinstance(tool_result, dict) and "error" in tool_result
                ) else "ok"
                result_str = _truncate_chat_tool_result(tool_result)

                yield _log_yield(
                    req_id, "tool/result",
                    make_event_chunk(model, "tool_result", {
                        "id":      call_id,
                        "name":    tool_name,
                        "status":  status,
                        "content": result_str,
                    }),
                )

                messages.append({
                    "role":    "tool",
                    "content": result_str,
                })

            yield _log_yield(req_id, "keepalive/tr", make_keepalive(model))
            last_tool_calls = None

        # Fallback: if we never produced final text, force one more call without tools
        if not final_assistant_content:
            log.info("[%s] no final content after loop - forcing final answer without tools",
                     req_id)
            # Explicit instruction: models that ignored the tool-strip hint
            # (DeepSeek Flash in particular) fall back to DSML text otherwise.
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the tool-call limit. Provide your final "
                    "answer NOW using plain text only. Do NOT emit tool calls, "
                    "XML, or any special markup. A concise natural-language "
                    "summary of what you found is required."
                ),
            })
            final_built = build_model_messages(messages)
            final_payload = make_ollama_payload(model, final_built, options, stream=True)
            final_payload.pop("tools", None)

            try:
                dsml_final = DSMLFilter()
                final_stream = backend.chat_stream(final_payload)
                for line in final_stream:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(
                            line.decode("utf-8") if isinstance(line, bytes) else line
                        )
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message") or {}
                    thinking = msg.get("thinking") or ""
                    content = msg.get("content") or ""

                    if thinking:
                        yield _log_yield(
                            req_id, "final/think",
                            make_thinking_chunk(model, thinking),
                        )

                    if content and not msg.get("tool_calls"):
                        cleaned = dsml_final.feed(content)
                        if cleaned:
                            final_assistant_content += cleaned
                            yield _log_yield(
                                req_id, "final/chunk",
                                make_chunk(model, cleaned, done=False),
                            )

                    if chunk.get("done"):
                        yield _log_yield(
                            req_id, "final/done",
                            make_done_chunk(
                                model,
                                chunk.get("done_reason", "stop"),
                                chunk.get("usage"),
                            ),
                        )
                        break

                tail = dsml_final.flush()
                if tail:
                    final_assistant_content += tail
                    yield _log_yield(
                        req_id, "final/tail",
                        make_chunk(model, tail, done=False),
                    )

            except Exception as e:
                log.error("[%s] final answer failed: %s", req_id, e)
                fallback = f"\n[final answer failed: {e}]\n"
                final_assistant_content = fallback
                yield make_chunk(model, fallback, done=True)

        log.debug("[%s] generator exiting normally", req_id)

    except Exception as e:
        log.error("[%s] unhandled exception: %s", req_id, traceback.format_exc())
        yield make_chunk(model, f"\n[internal error: {e}]\n", done=True)

    finally:
        if session and last_user_msg and final_assistant_content:
            log.debug(
                "[%s] [finally] saving turn: session=%s user=%d chars assistant=%d chars",
                req_id, session.session_id,
                len(last_user_msg), len(final_assistant_content),
            )
            sess.record_turn(session, last_user_msg, final_assistant_content)
            rag.index_turn(
                source=f"session:{session.session_id}",
                user=last_user_msg,
                assistant=final_assistant_content,
            )
        else:
            log.debug(
                "[%s] [finally] skipping save: session=%s user_empty=%s answer_empty=%s",
                req_id,
                session.session_id if session else "None",
                not last_user_msg,
                not final_assistant_content,
            )


@app.post("/chat")
@app.post("/api/chat")
def chat():
    req_id = uuid.uuid4().hex[:8]

    try:
        original_payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.error("[%s] failed to parse request JSON: %s", req_id, e)
        return jsonify({"error": "invalid JSON"}), 400

    model       = original_payload.get("model") or "qwen3:8b"
    messages    = original_payload.get("messages") or []
    options     = merge_options(original_payload.get("options"))
    want_stream = original_payload.get("stream", True)

    chat_id = extract_chat_id(request)
    session = sess.get_session(chat_id) if chat_id else None

    log.info("[%s] chat_id=%s", req_id, chat_id)

    messages, layer_sizes = ensure_system_prompt(messages, session=session)

    log.info("[%s] POST /api/chat model=%s stream=%s messages=%d options=%s",
             req_id, model, want_stream, len(messages), options)

    if not want_stream:
        try:
            answer = run_tool_loop(model, messages, options, layer_sizes, req_id)

            if session and answer:
                last_user = _last_user_text(messages)
                if last_user:
                    log.debug(
                        "[%s] non-stream: saving turn session=%s user=%d chars answer=%d chars",
                        req_id, session.session_id, len(last_user), len(answer),
                    )
                    sess.record_turn(session, last_user, answer)
                    rag.index_turn(
                        source=f"session:{session.session_id}",
                        user=last_user,
                        assistant=answer,
                    )

            log.info("[%s] non-stream done, answer_len=%d", req_id, len(answer))
            return jsonify({"model": model, "answer": answer})

        except ReadTimeout:
            log.error("[%s] backend timeout", req_id)
            return jsonify({"error": "backend timeout"}), 504
        except Exception as e:
            log.error("[%s] non-stream error: %s", req_id, traceback.format_exc())
            return jsonify({"error": str(e)}), 502

    log.info("[%s] starting stream generator", req_id)
    return Response(
        stream_with_context(
            chat_stream_generator(model, messages, options, session, layer_sizes, req_id)
        ),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


@app.post("/generate")
@app.post("/api/generate")
def generate():
    req_id = uuid.uuid4().hex[:8]

    try:
        original_payload = request.get_json(force=True, silent=False)
    except Exception as e:
        log.error("[%s] generate: failed to parse request JSON: %s", req_id, e)
        return jsonify({"error": "invalid JSON"}), 400

    model       = original_payload.get("model") or "qwen3:8b"
    prompt      = original_payload.get("prompt") or ""
    system_text = original_payload.get("system") or ""
    want_stream = original_payload.get("stream", True)
    options     = merge_options(original_payload.get("options"))

    session = None

    messages: List[Dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt})

    log.info(
        "[%s] POST /api/generate model=%s stream=%s prompt_len=%d",
        req_id, model, want_stream, len(prompt),
    )

    messages, layer_sizes = ensure_system_prompt(messages, session=session)

    if not want_stream:
        try:
            answer = run_tool_loop(model, messages, options, layer_sizes, req_id)
            log.info("[%s] generate non-stream done, answer_len=%d", req_id, len(answer))
            return jsonify({
                "model":      model,
                "created_at": now_iso(),
                "response":   answer,
                "done":       True,
            })
        except ReadTimeout:
            log.error("[%s] generate backend timeout", req_id)
            return jsonify({"error": "backend timeout"}), 504
        except Exception as e:
            log.error("[%s] generate non-stream error: %s", req_id, traceback.format_exc())
            return jsonify({"error": str(e)}), 502

    def generate_stream():
        inner = chat_stream_generator(model, messages, options, session, layer_sizes, req_id)
        for raw in inner:
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                yield raw
                continue

            msg = obj.get("message") or {}
            out = {
                "model":      obj.get("model") or model,
                "created_at": obj.get("created_at") or now_iso(),
                "response":   msg.get("content") or "",
                "done":       obj.get("done", False),
            }
            if "event" in obj:
                out["event"] = obj["event"]
                out["response"] = ""
            if obj.get("done"):
                out["done_reason"] = obj.get("done_reason", "stop")
                if "usage" in obj:
                    out["usage"] = obj["usage"]
            yield (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")

    log.info("[%s] starting generate stream", req_id)
    return Response(
        stream_with_context(generate_stream()),
        content_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control":     "no-cache",
            "Transfer-Encoding": "chunked",
        },
    )


# ── /api/event ──────────────────────────────────────────────────────────────────────

@app.post("/api/event")
def system_event():
    req_id = uuid.uuid4().hex[:8]

    denied = _check_event_token()
    if denied:
        return denied

    try:
        ev = request.get_json(force=True, silent=False) or {}
    except Exception as e:
        log.error("[%s] event: bad JSON: %s", req_id, e)
        return jsonify({"error": "invalid JSON"}), 400

    event_type   = ev.get("type", "event")
    event_source = ev.get("source", "unknown")
    if "payload" in ev:
        event_payload = ev["payload"]
    else:
        event_payload = {k: v for k, v in ev.items()
                         if k not in ("type", "source", "model", "options")}
    model   = ev.get("model") or CHATD_EVENT_MODEL or "qwen3:8b"
    options = merge_options(ev.get("options"))

    log.info(
        "[%s] system event: type=%s source=%s model=%s",
        req_id, event_type, event_source, model,
    )

    event_text = (
        f"[SYSTEM EVENT] type={event_type} source={event_source}\n"
        f"{json.dumps(event_payload, ensure_ascii=False, indent=2)}"
        f"{_EVENT_DIRECTIVE}"
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": event_text}]

    # /api/event has no session context - session=None, wake-up not cached
    messages, layer_sizes = ensure_system_prompt(messages, session=None)

    def _run():
        try:
            answer = run_tool_loop(
                model, messages, options, layer_sizes, req_id,
                final_num_predict=EVENT_FINAL_ROUND_NUM_PREDICT,
            )
            log.info("[%s] event worker done, answer_len=%d", req_id, len(answer))
        except Exception:
            log.error("[%s] event worker error:\n%s", req_id, traceback.format_exc())

    t = threading.Thread(target=_run, daemon=True, name=f"event-{req_id}")
    t.start()

    return jsonify({"ok": True, "req_id": req_id}), 202


# ── /api/tick ──────────────────────────────────────────────────────────────────

@app.post("/api/tick")
def bg_tick():
    """Background thinking trigger (single-turn preview).

    Returns:
      404 - background disabled (CHATD_BG_ENABLED=false)
      401 - CHATD_BG_TOKEN set and bearer doesn't match
      200 - {ok, req_id, result}
      500 - internal error inside background.run_tick
    """
    req_id = uuid.uuid4().hex[:8]

    if not BG_ENABLED:
        return jsonify({"error": "background disabled"}), 404

    if BG_TOKEN:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing Authorization header"}), 401
        token = auth.removeprefix("Bearer ").strip()
        if token != BG_TOKEN:
            return jsonify({"error": "invalid token"}), 401

    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}

    log.info("[%s] POST /api/tick", req_id)

    try:
        result = background.run_tick(payload, req_id)
    except Exception as e:
        log.error("[%s] tick failed: %s", req_id, traceback.format_exc())
        return jsonify({"error": str(e), "req_id": req_id}), 500

    return jsonify({"ok": True, "req_id": req_id, "result": result}), 200


init_tools()
memory.init()
rag.init()

if __name__ == "__main__":
    log.info("started")
    app.run(host="0.0.0.0", port=5001, debug=False)
