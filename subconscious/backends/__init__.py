"""backends/__init__.py - backend registry and router.

Selects a backend instance based on the model name prefix.
Add entries to _REGISTRY to register new backends.

Model routing rules (checked in order):
  "or/"       prefix  -> OpenRouter
  "onnx/"     prefix  -> ONNX
  "deepseek/" prefix  -> DeepSeek
  (default)           -> Ollama
"""
from __future__ import annotations

from typing import Dict

from subconscious.backends.base import Backend


def _make_registry() -> Dict[str, Backend]:
    from subconscious.backends.ollama import OllamaBackend
    from subconscious.backends.openrouter import OpenRouterBackend
    from subconscious.backends.onnx import OnnxBackend
    from subconscious.backends.deepseek import DeepSeekBackend

    return {
        "or/":   OpenRouterBackend(),
        "onnx/": OnnxBackend(),
        "deepseek/": DeepSeekBackend(),
        "": OllamaBackend(),  # default / fallback
    }


_registry: Dict[str, Backend] | None = None


def _get_registry() -> Dict[str, Backend]:
    global _registry
    if _registry is None:
        _registry = _make_registry()
    return _registry


def get_backend(model: str) -> Backend:
    """Return the backend instance responsible for *model*."""
    reg = _get_registry()
    for prefix, backend in reg.items():
        if prefix and model.startswith(prefix):
            return backend
    return reg[""]  # Ollama default
