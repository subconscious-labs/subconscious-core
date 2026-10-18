"""backends/base.py - Backend Protocol (structural interface).

Chat contract:
  chat_stream(payload) -> Iterator[bytes]
    Yields normalized NDJSON chunks (see below) as UTF-8 bytes.

  chat_sync(payload) -> dict
    Returns a single normalized response dict.

  embed(text, model) -> list[float]

Normalized chunk shape (all backends MUST emit this):
  {
    "model": str,
    "created_at": str,               # optional
    "message": {
      "role": "assistant",
      "content": str,                # text delta / full text
      "thinking": str,               # optional; reasoning delta / full
      "tool_calls": [                # optional; only when complete
        {"id": str, "name": str, "arguments": dict}
      ],
    },
    "done": bool,
    "done_reason": str,              # on done: stop|length|tool_calls|error
    "usage": {                       # on done, if provider reports it
      "prompt_tokens": int, "completion_tokens": int,
      "reasoning_tokens": int,
      "cache_hit_tokens": int, "cache_miss_tokens": int,
    },
  }

payload is always an Ollama-format dict:
  {"model": str, "messages": [...], "options": {...}, "tools": [...], "stream": bool}

Incoming messages may carry `thinking` on assistant turns - backends map
it to whatever their provider expects (e.g. DeepSeek uses `reasoning_content`).
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """Structural interface for a chatd inference backend."""

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        """Yield Ollama-format NDJSON chunks as bytes."""
        ...

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a single Ollama-format response dict."""
        ...

    def embed(self, text: str, model: str) -> List[float]:
        """Return an embedding vector for *text* using *model*.

        Raise RuntimeError('embed not supported') if unsupported.
        """
        ...
