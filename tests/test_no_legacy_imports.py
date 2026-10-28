"""Guard: no flat-module imports after the package move.

Two imports were missed in the first pass and only caught in production:
  - backends/onnx.py: "from config import _env" at top level
  - tools/mcp_client.py: "from config import MCP_EVENT_ALLOWED" inside
    an async notification handler, so only fires on real MCP events.

A grep-based test is cheap, needs no mocks, and catches the whole
class: any import of a name that used to be a top-level flat module is
by definition wrong after the move.
"""
import re
from pathlib import Path


_LEGACY = re.compile(
    r"^\s*(?:from|import)\s+("
    r"config|backends|mcp_client|mcp_registry|internal_tools|"
    r"background|memory|rag|session"
    r")(?:\s|$|\.|,)"
)


def test_no_legacy_flat_module_imports():
    root = Path(__file__).resolve().parent.parent / "subconscious"
    hits = []
    for py in root.rglob("*.py"):
        for lineno, line in enumerate(
            py.read_text(encoding="utf-8").splitlines(), 1
        ):
            if _LEGACY.match(line):
                hits.append(
                    f"{py.relative_to(root.parent)}:{lineno}: {line.strip()}"
                )
    assert not hits, (
        "legacy flat-module imports found (these modules moved under "
        "subconscious/ during the package refactor):\n" + "\n".join(hits)
    )
