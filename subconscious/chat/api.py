"""HTTP interface.

Flask routes for chat, event ingestion, model listing, and background
tick. All business logic lives in sibling modules; this file wires them
to HTTP endpoints.

Route inventory:
  /                     - health probe (Ollama compat)
  /version /api/version - version info
  /ps /api/ps           - empty running-models (Ollama compat)
  /health /api/health   - internal health
  /pull /api/pull       - fake pull success (Ollama compat)
  /show /api/show       - model details (OpenRouter, DeepSeek, Ollama)
  /tags /api/tags       - merged model list
  /chat /api/chat       - chat completion (stream or sync)
  /generate /api/generate - text completion (stream or sync)
  /api/event            - event ingress
  /api/tick             - background worker trigger

See docs/adr/ADR-002-package-layout.md for module boundaries.
"""
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import ReadTimeout
from flask import Flask, request, Response, jsonify, stream_with_context

from subconscious.backends.ollama import OLLAMA_API
from subconscious.backends.openrouter import fetch_model_info, OPENROUTER_PREFIX
from subconscious.chat import session as sess
from subconscious.chat.event import (
    _extract_user_id,
    process_event,
)
from subconscious.chat.prompt import ensure_system_prompt
from subconscious.chat.stream import chat_stream_generator
from subconscious.chat.text_utils import _last_user_text
from subconscious.chat.tools_io import init_tools
from subconscious.chat.tools_loop import run_tool_loop
from subconscious.chat.util import now_iso
from subconscious.events import store as events_store
from subconscious.memory import rag
from subconscious.memory import wakeup as memory
from subconscious.runtime.config import (
    BG_ENABLED,
    BG_TOKEN,
    DEEPSEEK_API_MODELS,
    DEFAULT_OPTIONS,
    EVENT_TOKEN,
    OPENROUTER_API_MODELS,
)
from subconscious.tools import registry as mcp_registry
from subconscious.worker import tick as background


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("subconscious.chat.api")

VERSION = "2.0.0"

app = Flask(__name__)
app.json.ensure_ascii = False


# ── helpers ──────────────────────────────────────────────────────────────────

def merge_options(client_options: Optional[Dict]) -> Dict[str, Any]:
    opts = dict(DEFAULT_OPTIONS)
    if client_options:
        opts.update(client_options)
    return opts


def extract_chat_id(flask_request) -> Optional[str]:
    """Extract chat session ID from headers or Referer.

    Priority:
      1. X-OpenWebUI-Chat-Id   - Open WebUI forwards this when
                                  ENABLE_FORWARD_USER_INFO_HEADERS=true.
      2. /sessions/<id> in Referer - Hollama's convention.
      3. Legacy chatId field in messages[] - oldest fallback.
    """
    owi_chat = flask_request.headers.get("X-OpenWebUI-Chat-Id")
    if owi_chat and owi_chat.strip():
        return owi_chat.strip()

    referer = flask_request.headers.get("Referer", "")
    if referer:
        import re as _re
        m = _re.search(r"/sessions/([^/?#]+)", referer)
        if m:
            return m.group(1)

    messages = flask_request.get_json(silent=True) or {}
    for msg in (messages.get("messages") or []):
        chat_id = msg.get("chatId")
        if chat_id is not None:
            return str(chat_id)

    return None


def _check_event_token() -> Optional[Tuple[Response, int]]:
    """Return (Response, status) if EVENT_TOKEN is set and Bearer does not match."""
    if not EVENT_TOKEN:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "missing Authorization header"}), 401
    token = auth.removeprefix("Bearer ").strip()
    if token != EVENT_TOKEN:
        return jsonify({"error": "invalid token"}), 401
    return None


# ── CORS ─────────────────────────────────────────────────────────────────────

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


# ── augmented model listing (OpenRouter, DeepSeek) ───────────────────────────

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


# ── routes ──────────────────────────────────────────────────────────────────

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


@app.get("/ps")
@app.get("/api/ps")
def ps():
    """Ollama-compatible running-models list. We don't track loaded models."""
    return jsonify({"models": []})


@app.get("/health")
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "tools": len(mcp_registry.list_tools())})


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

    def _stream():
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

        return jsonify({
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
        })

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
    user_id = _extract_user_id(request)
    session = sess.get_session(chat_id, user_id=user_id) if chat_id else None

    log.info("[%s] chat_id=%s user_id=%s", req_id, chat_id, user_id)

    original_user = _last_user_text(messages)

    messages, layer_sizes = ensure_system_prompt(
        messages, session=session, user_id=user_id,
    )

    log.info(
        "[%s] POST /api/chat model=%s stream=%s messages=%d options=%s",
        req_id, model, want_stream, len(messages), options,
    )

    if not want_stream:
        try:
            answer = run_tool_loop(model, messages, options, layer_sizes, req_id)

            if session and answer and original_user:
                log.debug(
                    "[%s] non-stream: saving turn session=%s "
                    "user=%d chars answer=%d chars",
                    req_id, session.session_id,
                    len(original_user), len(answer),
                )
                sess.record_turn(
                    session, original_user, answer, user_id=user_id,
                )
                rag.index_turn(
                    source=f"session:{session.session_id}",
                    user=original_user,
                    assistant=answer,
                    user_id=user_id,
                )

            log.info("[%s] non-stream done, answer_len=%d", req_id, len(answer))
            return jsonify({
                "model":       model,
                "created_at":  now_iso(),
                "message":     {"role": "assistant", "content": answer},
                "done":        True,
                "done_reason": "stop",
            })

        except ReadTimeout:
            log.error("[%s] backend timeout", req_id)
            return jsonify({"error": "backend timeout"}), 504
        except Exception as e:
            log.error(
                "[%s] non-stream error: %s", req_id, e, exc_info=True,
            )
            return jsonify({"error": str(e)}), 502

    log.info("[%s] starting stream generator", req_id)
    return Response(
        stream_with_context(
            chat_stream_generator(
                model, messages, options, session, layer_sizes, req_id,
                clean_user=original_user,
                user_id=user_id,
            )
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
            answer = run_tool_loop(
                model, messages, options, layer_sizes, req_id,
            )
            log.info(
                "[%s] generate non-stream done, answer_len=%d",
                req_id, len(answer),
            )
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
            log.error(
                "[%s] generate non-stream error: %s",
                req_id, e, exc_info=True,
            )
            return jsonify({"error": str(e)}), 502

    def generate_stream():
        inner = chat_stream_generator(
            model, messages, options, session, layer_sizes, req_id,
        )
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


# ── /api/event ──────────────────────────────────────────────────────────────

@app.post("/api/event")
def system_event():
    denied = _check_event_token()
    if denied:
        return denied
    try:
        ev = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    event_type   = ev.get("type", "event")
    event_source = ev.get("source", "unknown")
    if "payload" in ev:
        event_payload = ev["payload"]
    else:
        event_payload = {
            k: v for k, v in ev.items()
            if k not in ("type", "source", "model", "options")
        }

    user_id = ev.get("user_id") or _extract_user_id(request)

    req_id = process_event(
        event_type, event_source, event_payload,
        model=ev.get("model"), options=ev.get("options"),
        user_id=user_id,
    )
    return jsonify({"ok": True, "req_id": req_id}), 202


# ── /api/tick ────────────────────────────────────────────────────────────────

@app.post("/api/tick")
def bg_tick():
    """Background thinking trigger (single-turn preview).

    Returns:
      404 - background disabled (SUBCONSCIOUS_BG_ENABLED=false)
      401 - SUBCONSCIOUS_BG_TOKEN set and bearer doesn't match
      200 - {ok, req_id, result}
      500 - internal error inside worker.run_tick
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
        log.error("[%s] tick failed: %s", req_id, e, exc_info=True)
        return jsonify({"error": str(e), "req_id": req_id}), 500

    return jsonify({"ok": True, "req_id": req_id, "result": result}), 200


# ── startup ─────────────────────────────────────────────────────────────────

init_tools()
memory.init()
rag.init()
events_store.init()

if __name__ == "__main__":
    log.info("started")
    app.run(host="0.0.0.0", port=5001, debug=False)
