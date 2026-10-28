"""Small utilities shared across chat modules.

now_iso: ISO 8601 UTC timestamp with microsecond precision.
_tc_signature: stable hash of a tool_calls list, used for repeat
detection in tool loops (three identical consecutive signatures means
the model is stuck and the loop aborts).

Stateless. No dependencies on project modules beyond stdlib.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


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
