"""Structural tests for _EVENT_DIRECTIVE.

The directive is a plain string that gets prepended to every system
event. It is consumed by the chat model, not by code, so we cannot
test model behaviour here - only that the directive contains the
specific phrases that encode the intended policy.

Regression tests for ADR-004: a plain "проверка xmpp" must short-
circuit before STEP 2-4, while concrete triggers still drive the full
cycle. If any of these assertions break, someone removed a sentence
that the model depends on to classify the event correctly.
"""

from subconscious.chat.event import _EVENT_DIRECTIVE


def test_directive_classifies_pings():
    """Item 4: pings must be told NOT to run STEP 2-4."""
    assert "classify the event BEFORE acting" in _EVENT_DIRECTIVE
    assert "Do NOT run STEP 2-4" in _EVENT_DIRECTIVE


def test_directive_lists_ping_examples():
    """The examples are what makes the instruction concrete enough for
    a model to act on. Without them it tends to treat every message as
    a potential investigation."""
    for example in ("проверка", "test", "ping", "привет", "ok"):
        assert example in _EVENT_DIRECTIVE


def test_directive_keeps_concrete_triggers():
    """Concrete triggers must still drive the full STEP 1-4 cycle -
    otherwise alerts and named tasks would get the same one-line
    treatment as pings."""
    assert "Concrete triggers" in _EVENT_DIRECTIVE
    assert "drive the full STEP 1-4" in _EVENT_DIRECTIVE


def test_directive_keeps_all_steps():
    """Classification is additive; STEP 1-4 must remain in the directive
    so non-ping events still get the full handling."""
    for step in ("STEP 1", "STEP 2", "STEP 3", "STEP 4"):
        assert step in _EVENT_DIRECTIVE


def test_directive_step1_unchanged():
    """The specific 'do NOT skip STEP 1' invariant from the original
    directive must survive the classification edit."""
    assert "Do NOT skip STEP 1" in _EVENT_DIRECTIVE


def test_directive_not_empty():
    """Sanity: the string is at least as long as before (~1.2 KB)."""
    assert len(_EVENT_DIRECTIVE) > 1200
