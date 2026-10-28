"""Unit tests for session key extraction, dedupe, and priority.

Session key parsing is best-effort: unknown shapes must fall back to
None (stateless processing) rather than raise. Dedupe and priority
must be deterministic and side-effect-free across test runs.
"""
import pytest

from subconscious.chat import event as ev
from subconscious.events import store as events_store


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Reset module-level dedupe buffers and locks before and after each
    test. Prevents cross-test contamination."""
    ev._SEEN_EVENT_IDS.clear()
    ev._SESSION_LOCKS.clear()
    yield
    ev._SEEN_EVENT_IDS.clear()
    ev._SESSION_LOCKS.clear()


# ── session key extraction ───────────────────────────────────────────────────

def test_explicit_session_key_wins():
    assert ev._extract_session_key(
        "webui", {"session_key": "My-Custom-Key"}
    ) == "my-custom-key"


def test_explicit_session_key_strips_whitespace():
    assert ev._extract_session_key(
        "webui", {"session_key": "  custom  "}
    ) == "custom"


def test_xmpp_dm_simple():
    assert ev._extract_session_key(
        "xmpp", {"from": "user@domain"}
    ) == "xmpp:dm:user@domain"


def test_xmpp_dm_strips_resource():
    assert ev._extract_session_key(
        "xmpp", {"from": "user@domain/phone"}
    ) == "xmpp:dm:user@domain"


def test_xmpp_muc_by_type():
    assert ev._extract_session_key(
        "xmpp", {"from": "room@conference/nick", "type": "groupchat"}
    ) == "xmpp:muc:room@conference"


def test_xmpp_muc_by_room_field():
    assert ev._extract_session_key(
        "xmpp", {"from": "room@conference/nick", "room": "room@conference"}
    ) == "xmpp:muc:room@conference"


def test_xmpp_self_jid_ignored(monkeypatch):
    """Echo protection: the agent's own JID must return None so its own
    outgoing messages are not re-processed."""
    monkeypatch.setenv("SUBCONSCIOUS_XMPP_SELF_JID", "elsa@ai00")
    assert ev._extract_session_key(
        "xmpp", {"from": "elsa@ai00/laptop"}
    ) is None


def test_xmpp_self_jid_resource_stripped(monkeypatch):
    """Comparison is against bare JID; resource on either side is stripped."""
    monkeypatch.setenv("SUBCONSCIOUS_XMPP_SELF_JID", "elsa@ai00/laptop")
    assert ev._extract_session_key(
        "xmpp", {"from": "elsa@ai00"}
    ) is None


def test_xmpp_fallback_from_text_prefix():
    """Servers that flatten metadata into text produce
    'user@domain: body'. Sender is recovered from the prefix."""
    assert ev._extract_session_key(
        "xmpp", {"text": "user@domain: hello"}
    ) == "xmpp:dm:user@domain"


def test_xmpp_fallback_from_text_with_resource():
    assert ev._extract_session_key(
        "xmpp", {"text": "user@domain/phone: hello"}
    ) == "xmpp:dm:user@domain"


def test_xmpp_no_jid_returns_none():
    assert ev._extract_session_key("xmpp", {"text": "hello"}) is None


def test_xmpp_text_with_at_in_body_no_match():
    """'@' in the body must not be mis-parsed as a JID prefix."""
    assert ev._extract_session_key(
        "xmpp", {"text": "hello @user how are you"}
    ) is None


def test_non_xmpp_source_returns_none():
    assert ev._extract_session_key("webui", {"text": "hi"}) is None


def test_malformed_jid_no_at():
    assert ev._extract_session_key("xmpp", {"from": "nobody"}) is None


def test_empty_payload_returns_none():
    assert ev._extract_session_key("xmpp", {}) is None


# ── deduplication ────────────────────────────────────────────────────────────

def test_dedupe_first_call_false():
    assert ev._is_duplicate_event("s1", {"id": "m1"}) is False


def test_dedupe_second_call_true():
    ev._is_duplicate_event("s1", {"id": "m1"})
    assert ev._is_duplicate_event("s1", {"id": "m1"}) is True


def test_dedupe_per_session():
    """Same message id in different sessions is not a duplicate."""
    ev._is_duplicate_event("s1", {"id": "m1"})
    assert ev._is_duplicate_event("s2", {"id": "m1"}) is False


def test_dedupe_no_message_id():
    """Payloads without an id cannot be deduplicated; always False."""
    assert ev._is_duplicate_event("s1", {"body": "hi"}) is False
    assert ev._is_duplicate_event("s1", {"body": "hi"}) is False


def test_dedupe_alternate_id_fields():
    """message_id and stanza_id work like id."""
    ev._is_duplicate_event("s1", {"message_id": "x1"})
    assert ev._is_duplicate_event("s1", {"message_id": "x1"}) is True


def test_dedupe_buffer_bounded():
    """Buffer holds at most _SEEN_EVENT_IDS_MAX entries; oldest fall out."""
    limit = ev._SEEN_EVENT_IDS_MAX
    for i in range(limit + 10):
        ev._is_duplicate_event("s1", {"id": f"m{i}"})
    # Oldest ids (m0..m9) must have been evicted.
    assert ev._is_duplicate_event("s1", {"id": "m0"}) is False
    # Recent ids still present.
    assert ev._is_duplicate_event("s1", {"id": f"m{limit + 9}"}) is True


# ── priority classification ──────────────────────────────────────────────────

def test_priority_critical():
    assert ev._priority_for_event(
        "alert", {"severity": "critical"}
    ) == events_store.PRIORITY_CRITICAL


def test_priority_interactive_sources():
    for src in ("webui", "xmpp", "email"):
        assert ev._priority_for_event(src, {}) == \
            events_store.PRIORITY_INTERACTIVE


def test_priority_alert_and_monitor():
    assert ev._priority_for_event("alert", {}) == \
        events_store.PRIORITY_NORMAL
    assert ev._priority_for_event("monitor_ping", {}) == \
        events_store.PRIORITY_NORMAL


def test_priority_background_default():
    assert ev._priority_for_event("tick", {}) == \
        events_store.PRIORITY_BACKGROUND
    assert ev._priority_for_event("unknown_source", {}) == \
        events_store.PRIORITY_BACKGROUND


# ── human text extraction ────────────────────────────────────────────────────

def test_human_text_body_first():
    assert ev._extract_event_human_text(
        {"body": "from body", "text": "from text"}
    ) == "from body"


def test_human_text_falls_through_to_message():
    assert ev._extract_event_human_text(
        {"message": "msg", "text": "txt"}
    ) == "msg"


def test_human_text_empty_payload():
    assert ev._extract_event_human_text({}) == ""


def test_human_text_ignores_non_string():
    assert ev._extract_event_human_text({"body": 123}) == ""
