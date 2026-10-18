"""backends/onnx.py - ONNX Runtime inference backend.

IMPORTANT: This backend is intentionally minimal and serves as a
reference implementation for non-LLM models (classifiers, encoders,
rerankers). It is NOT a general-purpose LLM runner.

For full LLM inference via ONNX, consider:
  - microsoft/onnxruntime-genai  (https://github.com/microsoft/onnxruntime-genai)
  - optimum[onnxruntime]         (https://huggingface.co/docs/optimum/onnxruntime/overview)

---- Setup for a sentence-encoder (embed-only model) ----

  1. Install dependencies:
       pip install onnxruntime numpy tokenizers

  2. Export a model to ONNX (example: all-MiniLM-L6-v2):
       pip install optimum[exporters] sentence-transformers
       optimum-cli export onnx \\
           --model sentence-transformers/all-MiniLM-L6-v2 \\
           --task feature-extraction \\
           ./models/all-MiniLM-L6-v2/

  3. Set environment variables:
       SUBCONSCIOUS_ONNX_MODEL_DIR=./models/all-MiniLM-L6-v2
       SUBCONSCIOUS_ONNX_TOKENIZER_DIR=./models/all-MiniLM-L6-v2

  4. Enable as the RAG embed backend:
       SUBCONSCIOUS_RAG_BACKEND=onnx

  5. For chat: not supported by default. Implement chat_stream / chat_sync
     below if your ONNX model is a seq2seq or decoder-only LLM.
     See microsoft/onnxruntime-genai for a production-grade approach.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, List

from config import _env

log = logging.getLogger("chatd.backends.onnx")

ONNX_MODEL_DIR: str = _env("ONNX_MODEL_DIR", "")
ONNX_TOKENIZER_DIR: str = _env("ONNX_TOKENIZER_DIR", ONNX_MODEL_DIR)

_session = None
_tokenizer = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        model_path = os.path.join(ONNX_MODEL_DIR, "model.onnx")
        _session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return _session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from tokenizers import Tokenizer
        tok_path = os.path.join(ONNX_TOKENIZER_DIR, "tokenizer.json")
        _tokenizer = Tokenizer.from_file(tok_path)
    return _tokenizer


class OnnxBackend:
    """ONNX Runtime backend.

    Default use-case: sentence encoder / reranker for RAG embed().
    chat_stream and chat_sync are not implemented for generic ONNX models
    - override in a subclass for LLM-style ONNX models.
    """

    def chat_stream(
        self, payload: Dict[str, Any]
    ) -> Generator[bytes, None, None]:
        raise RuntimeError(
            "OnnxBackend.chat_stream is not implemented. "
            "For LLM inference via ONNX, use microsoft/onnxruntime-genai "
            "and subclass OnnxBackend."
        )

    def chat_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError(
            "OnnxBackend.chat_sync is not implemented. "
            "For LLM inference via ONNX, use microsoft/onnxruntime-genai "
            "and subclass OnnxBackend."
        )

    def embed(self, text: str, model: str) -> List[float]:
        """Mean-pooled sentence embedding via ONNX encoder model."""
        import numpy as np

        tokenizer = _get_tokenizer()
        session   = _get_session()

        enc = tokenizer.encode(text)
        input_ids      = np.array([enc.ids],            dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = session.run(None, {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        # Mean pooling over token dimension (output[0] shape: [1, seq_len, hidden])
        hidden = outputs[0][0]  # [seq_len, hidden]
        mask   = np.array(enc.attention_mask, dtype=np.float32)
        pooled = (hidden * mask[:, None]).sum(0) / mask.sum()
        norm   = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.tolist()
