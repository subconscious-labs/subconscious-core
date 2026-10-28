"""Streaming chat generator and NDJSON chunk builders.

chat_stream_generator: drive the model through tool rounds, yielding
NDJSON chunks (content, thinking, done) to the HTTP response.
Chunk builders: make_chunk, make_keepalive, make_thinking_chunk,
make_done_chunk. _log_yield is a debugging wrapper.

Note on call_tool: same reasoning as tools_loop - deferred import to
avoid a cycle through chat.api.
"""
from __future__ import annotations

import json
import logging
import traceback
import uuid
from typing import Any, Dict, Generator, List, Optional

from requests.exceptions import RequestException

from subconscious import backends
from subconscious.chat import session as sess
from subconscious.chat.prompt import (
    _log_payload_sizes,
    build_model_messages,
    make_ollama_payload,
)
from subconscious.chat.text_utils import (
    DSMLFilter,
    _contains_dsml,
    _last_user_text,
    _strip_dsml,
    _truncate_chat_tool_result,
)
from subconscious.chat.tools_loop import _options_for_round
from subconscious.chat.util import _tc_signature, now_iso
from subconscious.memory import rag
from subconscious.runtime.config import (
    MAX_TOOL_ROUNDS,
    THINKING,
)

log = logging.getLogger("subconscious.chat.stream")


def make_chunk(model: str, content: str, done: bool = False) -> bytes:
    obj: Dict[str, Any] = {
        "model":      model,
        "created_at": now_iso(),
        "message":    {"role": "assistant", "content": content},
        "done":       done,
    }
    if done:
        obj["done_reason"] = "stop"
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def make_keepalive(model: str) -> bytes:
    return make_chunk(model, "")


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
        "done": False,
    }
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def make_done_chunk(
    model: str,
    done_reason: str = "stop",
    usage: Optional[Dict[str, Any]] = None,
    durations: Optional[Dict[str, int]] = None,
) -> bytes:
    obj: Dict[str, Any] = {
        "model":       model,
        "created_at":  now_iso(),
        "message":     {"role": "assistant", "content": ""},
        "done":        True,
        "done_reason": done_reason,
    }

    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        obj["prompt_eval_count"] = prompt_tokens
        obj["eval_count"] = completion_tokens

        details = usage.get("completion_tokens_details") or {}
        r_tokens = details.get("reasoning_tokens", 0)
        if r_tokens:
            obj["reasoning_tokens"] = r_tokens

        obj["usage"] = usage

        if not durations:
            p_ns = prompt_tokens * 5_000_000
            e_ns = completion_tokens * 15_000_000
            durations = {
                "total_ns": p_ns + e_ns,
                "load_ns": 0,
                "prompt_eval_ns": p_ns,
                "eval_ns": e_ns,
            }

    if durations:
        obj["total_duration"] = durations.get("total_ns", 0)
        obj["load_duration"] = durations.get("load_ns", 0)
        obj["prompt_eval_duration"] = durations.get("prompt_eval_ns", 0)
        obj["eval_duration"] = durations.get("eval_ns", 0)

    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _log_yield(req_id: str, label: str, data: bytes) -> bytes:
    """Log a yielded chunk and return it unchanged."""
    log.debug("[%s] yield %-12s %d bytes", req_id, label, len(data))
    return data


def chat_stream_generator(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    session: Optional[sess.Session],
    layer_sizes: Dict[str, int],
    req_id: str = "-",
    clean_user: str = "",
    user_id: Optional[str] = None,
) -> Generator[bytes, None, None]:
    # Deferred import: tools_io imports chat.api, which will import this
    # module. Importing at module level creates a cycle.
    from subconscious.chat.tools_io import call_tool
    from subconscious.tools import registry as mcp_registry

    prev_tool_calls: Optional[List[str]] = None
    repeat_count = 0

    # For session log and RAG index we want the ORIGINAL user text,
    # without the RAG sidecar ensure_system_prompt appended.
    last_user_msg = clean_user or _last_user_text(messages)

    final_assistant_content = ""
    backend = backends.get_backend(model)

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            log.info(
                "[%s] stream round %d, messages=%d",
                req_id, round_num, len(messages),
            )

            round_options = _options_for_round(options, round_num)
            built_messages = build_model_messages(messages)
            payload = make_ollama_payload(
                model, built_messages, round_options, stream=True
            )

            is_last_round = (round_num == MAX_TOOL_ROUNDS - 1)
            if is_last_round:
                payload.pop("tools", None)
                log.info(
                    "[%s] round %d: tools stripped - forcing final answer",
                    req_id, round_num,
                )

            _log_payload_sizes(
                req_id, built_messages, len(mcp_registry.list_tools()),
                layer_sizes if round_num == 0 else None,
            )
            log.debug(
                "[%s] -> backend payload (messages=%d, tools=%d, "
                "options=%s, think=%s)",
                req_id, len(built_messages),
                len(mcp_registry.list_tools()), round_options, THINKING,
            )

            last_tool_calls: Optional[List[Dict]] = None
            last_thinking: str = ""
            last_content: str = ""
            last_content_raw: str = ""
            dsml_filter = DSMLFilter()
            chunk_count = 0
            stream_done_reason = "stop"
            stream_usage: Optional[Dict[str, Any]] = None
            stream_durations: Optional[Dict[str, int]] = None

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
                        log.warning(
                            "[%s] bad JSON line: %s | %r", req_id, e, line[:120]
                        )
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
                        log.info(
                            "[%s] tool_calls: %s", req_id,
                            [c.get("name") for c in msg["tool_calls"]],
                        )
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
                        stream_durations = chunk.get("durations")
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

            log.debug(
                "[%s] exited stream, last_tool_calls=%s",
                req_id, bool(last_tool_calls),
            )

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
                    make_done_chunk(
                        model, stream_done_reason, stream_usage, stream_durations
                    ),
                )
                log.info("[%s] stream complete, no more tool rounds", req_id)
                break

            current_sig = _tc_signature(last_tool_calls)
            prev_sig = _tc_signature(prev_tool_calls) if prev_tool_calls else None

            if current_sig == prev_sig:
                repeat_count += 1
                if repeat_count >= 3:
                    log.warning(
                        "[%s] tool loop detected (%s x%d), aborting",
                        req_id, current_sig, repeat_count,
                    )
                    break
            else:
                repeat_count = 0
            prev_tool_calls = last_tool_calls

            assistant_entry: Dict[str, Any] = {
                "role":       "assistant",
                "content":    last_content,
                "tool_calls": last_tool_calls,
            }
            if last_thinking:
                assistant_entry["thinking"] = last_thinking
            messages.append(assistant_entry)

            for tc in last_tool_calls:
                tool_name = tc.get("name") or "unknown"
                tool_args = tc.get("arguments") or {}
                call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:16]}"

                log.info(
                    "[%s] executing tool: %s(%s)", req_id, tool_name, tool_args,
                )
                tool_result = call_tool(tool_name, tool_args)

                result_str = _truncate_chat_tool_result(tool_result)

                messages.append({
                    "role":    "tool",
                    "content": result_str,
                })

            last_tool_calls = None

        if not final_assistant_content:
            log.info(
                "[%s] no final content after loop - "
                "forcing final answer without tools", req_id,
            )
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
            final_payload = make_ollama_payload(
                model, final_built, options, stream=True
            )
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
                                chunk.get("durations"),
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
                "[%s] [finally] saving turn: session=%s user=%d chars "
                "assistant=%d chars",
                req_id, session.session_id,
                len(last_user_msg), len(final_assistant_content),
            )
            sess.record_turn(
                session, last_user_msg, final_assistant_content,
                user_id=user_id,
            )
            rag.index_turn(
                source=f"session:{session.session_id}",
                user=last_user_msg,
                assistant=final_assistant_content,
                user_id=user_id,
            )
        else:
            log.debug(
                "[%s] [finally] skipping save: session=%s user_empty=%s "
                "answer_empty=%s",
                req_id,
                session.session_id if session else "None",
                not last_user_msg,
                not final_assistant_content,
            )
