"""Unit tests for ensure_system_prompt.

The function assembles the L0+L1 wake-up, caches it on the session,
and injects request-scoped RAG context into the last user message.
The RAG block must not touch the system message: it lives at the very
end of the payload, not in the prefix, to preserve the provider's
prefix cache.

wake_up() and rag.retrieve() are patched per test. A fake session
stands in for Session - only get_cached_base_prompt and
set_cached_base_prompt are used by the function.

The fake session does not implement user_id resolution; that lives
on the real Session and is tested separately in test_user_identity.
"""
import pytest

from subconscious.chat import prompt
from subconscious.chat.prompt import ensure_system_prompt


class FakeSession:
    """Minimal stand-in for chat.session.Session."""

    def __init__(self, session_id="s1", user_id=None, cached=None):
        self.session_id = session_id
        self.user_id = user_id
        self._cached = cached

    def get_cached_base_prompt(self):
        return self._cached

    def set_cached_base_prompt(self, prompt_str):
        self._cached = prompt_str


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def wake_up_spy(monkeypatch):
    """Patch wake_up to return a fixed string and count invocations."""
    calls = []

    def fake_wake_up():
        calls.append(1)
        return "IDENTITY+K GBLOCK"

    monkeypatch.setattr(prompt.memory, "wake_up", fake_wake_up)
    return calls


@pytest.fixture
def rag_spy(monkeypatch):
    """Patch rag.retrieve to return a fixed string and record queries."""
    calls = []

    def fake_retrieve(query, user_id=None):
        calls.append({"query": query, "user_id": user_id})
        return "RECALLED_CONTEXT"

    monkeypatch.setattr(prompt.rag, "retrieve", fake_retrieve)
    return calls


@pytest.fixture
def rag_miss(monkeypatch):
    """Patch rag.retrieve to return None (no matches)."""
    def fake_retrieve(query, user_id=None):
        return None

    monkeypatch.setattr(prompt.rag, "retrieve", fake_retrieve)


# ── assembly ────────────────────────────────────────────────────────────────

def test_system_inserted_when_absent(wake_up_spy, rag_miss):
    msgs = [{"role": "user", "content": "hi"}]
    out, layers = ensure_system_prompt(msgs, session=None)

    assert out[0] == {"role": "system", "content": "IDENTITY+K GBLOCK"}
    assert out[1]["role"] == "user"
    assert layers["L0+L1"] == len("IDENTITY+K GBLOCK")
    assert layers["L1.5_rag"] == 0


def test_system_overwritten_when_present(wake_up_spy, rag_miss):
    """A pre-existing system message (client's own preamble) is replaced
    entirely by the wake-up. Client-side system blocks are not merged."""
    msgs = [
        {"role": "system", "content": "client preamble"},
        {"role": "user", "content": "hi"},
    ]
    out, _ = ensure_system_prompt(msgs, session=None)

    assert out[0]["content"] == "IDENTITY+K GBLOCK"
    assert "client preamble" not in out[0]["content"]


def test_rag_injected_into_last_user(wake_up_spy, rag_spy):
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second question"},
    ]
    out, layers = ensure_system_prompt(msgs, session=None)

    # Only the last user message is augmented.
    assert out[1]["content"] == "first"
    assert out[2]["content"] == "ok"
    augmented = out[3]["content"]
    assert "RECALLED_CONTEXT" in augmented
    assert "second question" in augmented
    # RAG text appears before the original user text.
    assert augmented.index("RECALLED_CONTEXT") < augmented.index("second question")
    assert layers["L1.5_rag"] == len("RECALLED_CONTEXT")


def test_system_not_touched_by_rag(wake_up_spy, rag_spy):
    """The cache-preservation invariant: RAG lives in the last user
    message, never in the system block."""
    msgs = [{"role": "user", "content": "question"}]
    out, _ = ensure_system_prompt(msgs, session=None)

    assert out[0]["content"] == "IDENTITY+K GBLOCK"
    assert "RECALLED_CONTEXT" not in out[0]["content"]


def test_no_user_message_skips_rag(wake_up_spy, rag_spy):
    msgs = [{"role": "system", "content": "sys"}]
    out, layers = ensure_system_prompt(msgs, session=None)

    assert rag_spy == []
    assert layers["L1.5_rag"] == 0


# ── session cache ──────────────────────────────────────────────────────────

def test_cache_miss_calls_wake_up(wake_up_spy, rag_miss):
    sess = FakeSession(cached=None)
    msgs = [{"role": "user", "content": "hi"}]
    out, _ = ensure_system_prompt(msgs, session=sess)

    assert len(wake_up_spy) == 1
    assert sess.get_cached_base_prompt() == "IDENTITY+K GBLOCK"
    assert out[0]["content"] == "IDENTITY+K GBLOCK"


def test_cache_hit_skips_wake_up(wake_up_spy, rag_miss):
    sess = FakeSession(cached="CACHED_BASE")
    msgs = [{"role": "user", "content": "hi"}]
    out, layers = ensure_system_prompt(msgs, session=sess)

    assert wake_up_spy == []
    assert out[0]["content"] == "CACHED_BASE"
    assert layers["L0+L1"] == len("CACHED_BASE")


def test_no_session_calls_wake_up_every_time(wake_up_spy, rag_miss):
    msgs = [{"role": "user", "content": "hi"}]
    ensure_system_prompt(msgs, session=None)
    ensure_system_prompt(msgs, session=None)

    assert len(wake_up_spy) == 2


# ── user_id resolution ─────────────────────────────────────────────────────

def test_user_id_passed_to_rag_from_session(wake_up_spy, rag_spy):
    """Session user_id wins over the explicit parameter - the session
    was set up with the correct identity at creation time."""
    sess = FakeSession(cached="BASE", user_id="from-session@x.y")
    msgs = [{"role": "user", "content": "hi"}]
    ensure_system_prompt(msgs, session=sess, user_id="from-param@a.b")

    assert rag_spy[0]["user_id"] == "from-session@x.y"


def test_user_id_passed_to_rag_when_no_session(wake_up_spy, rag_spy):
    msgs = [{"role": "user", "content": "hi"}]
    ensure_system_prompt(msgs, session=None, user_id="explicit@x.y")

    assert rag_spy[0]["user_id"] == "explicit@x.y"


def test_no_user_id_when_neither_provided(wake_up_spy, rag_spy):
    msgs = [{"role": "user", "content": "hi"}]
    ensure_system_prompt(msgs, session=None)

    assert rag_spy[0]["user_id"] is None


# ── multimodal content ────────────────────────────────────────────────────

def test_multimodal_user_content_handled(wake_up_spy, rag_spy):
    """Open WebUI sends content as an array. RAG must receive the text
    part, not crash on list.strip()."""
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
    ]
    out, _ = ensure_system_prompt(msgs, session=None)

    assert rag_spy[0]["query"] == "what is this"
    # System remains plain string.
    assert isinstance(out[0]["content"], str)


# ── structural invariants ─────────────────────────────────────────────────

def test_caller_messages_not_mutated(wake_up_spy, rag_spy):
    """The input list must not be modified; a copy is made."""
    original_user = {"role": "user", "content": "hi"}
    msgs = [original_user]
    ensure_system_prompt(msgs, session=None)

    # Caller still sees one message, unchanged.
    assert msgs == [{"role": "user", "content": "hi"}]
    assert original_user["content"] == "hi"
