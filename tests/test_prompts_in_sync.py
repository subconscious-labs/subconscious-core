"""Guard: .example and .txt prompts must be byte-identical.

The two files serve different purposes: .example is tracked in git and
ships with the repo; .txt is gitignored and created on the host either
by the deploy step (`cp *.example *.txt`) or by a local edit. Their
content must match - a divergence means a local edit that was not
synced back to .example (silent loss on next deploy) or a shipped
default that never reached the host (fresh install behaviour gap).

Skipped on fresh installs where .txt has not yet been created.
"""
from pathlib import Path

import pytest

_PROMPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "subconscious" / "worker" / "prompts"
)

_PROMPT_NAMES = [
    "planner", "executor", "exploratory", "reflector", "assessor",
]


@pytest.mark.parametrize("name", _PROMPT_NAMES)
def test_prompt_in_sync(name):
    example = _PROMPTS_DIR / f"{name}.example"
    txt = _PROMPTS_DIR / f"{name}.txt"

    if not example.exists():
        pytest.skip(f"{name}.example missing - repo state is broken")
    if not txt.exists():
        pytest.skip(
            f"{name}.txt missing - fresh install, created by deploy cp"
        )

    ex = example.read_text(encoding="utf-8").strip()
    tx = txt.read_text(encoding="utf-8").strip()

    assert ex == tx, (
        f"{name}.example and {name}.txt have diverged.\n"
        f"Diff: `diff subconscious/worker/prompts/{name}.example "
        f"subconscious/worker/prompts/{name}.txt`\n"
        f"Fix: edit {name}.example (canonical), then "
        f"`cp {name}.example {name}.txt`."
    )
