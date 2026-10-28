"""Non-streaming tool loop.

run_tool_loop: drive the model through a sequence of tool rounds until
it produces a final answer, or until MAX_TOOL_ROUNDS is exhausted
(last round strips tools, forces plain-text final answer).

_options_for_round: increase num_predict on tool follow-up rounds so
the model has room for tool-call JSON plus a text summary.

Note on call_tool: it lives in tools_io, which imports chat/api for
session invalidation on mempalace writes. Importing it here at module
level would create a cycle. Import inside run_tool_loop instead - the
call happens once per tool round, and the cost is a dict lookup after
the first import.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from subconscious import backends
from subconscious.chat.prompt import (
    _log_payload_sizes,
    build_model_messages,
    make_ollama_payload,
)
from subconscious.chat.text_utils import (
    _contains_dsml,
    _strip_dsml,
    _truncate_chat_tool_result,
)
from subconscious.chat.util import _tc_signature
from subconscious.runtime.config import MAX_TOOL_ROUNDS

log = logging.getLogger("subconscious.chat.tools_loop")

# How much to boost num_predict for tool follow-up rounds.
TOOL_ROUND_NUM_PREDICT_BOOST = 1024

# Extra tokens guaranteed for the final answer round (no tool_calls).
EVENT_FINAL_ROUND_NUM_PREDICT = 2048


def _options_for_round(options: Dict[str, Any], round_num: int) -> Dict[str, Any]:
    if round_num == 0:
        return options
    boosted = dict(options)
    base = boosted.get("num_predict", 512)
    boosted["num_predict"] = base + TOOL_ROUND_NUM_PREDICT_BOOST
    log.debug(
        "tool round %d: num_predict %d -> %d",
        round_num, base, boosted["num_predict"],
    )
    return boosted


def run_tool_loop(
    model: str,
    messages: List[Dict],
    options: Dict[str, Any],
    layer_sizes: Dict[str, int],
    req_id: str,
    final_num_predict: Optional[int] = None,
) -> str:
    """Run the non-streaming tool loop and return the final assistant text."""
    # Deferred import: tools_io imports chat.api, which will import this
    # module. Importing at module level creates a cycle.
    from subconscious.chat.tools_io import call_tool
    from subconscious.tools import registry as mcp_registry

    prev_tc_sig: Optional[str] = None
    rep = 0
    resp: Dict = {}
    backend = backends.get_backend(model)

    for i in range(MAX_TOOL_ROUNDS):
        round_options = _options_for_round(options, i)
        built = build_model_messages(messages)

        is_last_round = (i == MAX_TOOL_ROUNDS - 1)
        if is_last_round:
            log.info(
                "[%s] tool_loop round %d: tools stripped - "
                "forcing final answer", req_id, i,
            )
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

        _log_payload_sizes(
            req_id, built, len(mcp_registry.list_tools()),
            layer_sizes if i == 0 else None,
        )
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
