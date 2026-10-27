"""Minimal tests for the events store."""
import sys

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBCONSCIOUS_EVENTS_DB_PATH", str(tmp_path / "events.sqlite3"))
    for mod in list(sys.modules):
        if mod == "subconscious.runtime.config" or mod.startswith("subconscious.events"):
            del sys.modules[mod]
    from subconscious.events import store
    store.init()
    return store


def test_append_and_get(store):
    eid = store.append(
        source="xmpp",
        event_type="message",
        payload={"text": "hi"},
        user_id="a@b.c",
        priority=store.PRIORITY_INTERACTIVE,
    )
    assert eid.startswith("ev-")
    ev = store.get(eid)
    assert ev["source"] == "xmpp"
    assert ev["user_id"] == "a@b.c"
    assert ev["payload"] == {"text": "hi"}
    assert ev["status"] == "pending"


def test_mark_processed(store):
    eid = store.append(source="alert", event_type="alert", payload={})
    assert store.mark(eid, status="processed")
    ev = store.get(eid)
    assert ev["status"] == "processed"
    assert ev["processed_at"] is not None


def test_link_sets_goal(store):
    eid = store.append(source="alert", event_type="alert", payload={})
    assert store.link(eid, "g-abc")
    ev = store.get(eid)
    assert ev["status"] == "linked"
    assert ev["goal_id"] == "g-abc"


def test_next_pending_by_priority(store):
    store.append(source="alert", event_type="alert", payload={},
                 priority=store.PRIORITY_NORMAL)
    store.append(source="xmpp", event_type="message", payload={},
                 priority=store.PRIORITY_INTERACTIVE)
    nxt = store.next_pending()
    assert nxt["source"] == "xmpp"


def test_list_filters(store):
    store.append(source="xmpp", event_type="message", payload={},
                 user_id="a@b.c")
    store.append(source="alert", event_type="alert", payload={},
                 user_id="a@b.c")
    store.append(source="alert", event_type="alert", payload={},
                 user_id="x@y.z")

    by_user = store.list_events(user_id="a@b.c")
    assert len(by_user) == 2

    by_source = store.list_events()
    assert len(by_source) == 3
