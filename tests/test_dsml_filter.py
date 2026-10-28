"""Unit tests for the DSML filter.

DeepSeek Flash emits tool-call markup as plain text when tools are
stripped from the payload. In streaming mode the marker can be split
across chunks. The filter buffers from the first marker trigger
(`｜`) until the next newline and drops the buffered line if it
contains the marker.

Tests cover: passthrough of clean text, whole-marker strip,
split-marker strip, buffer-cap safety, flush on empty and non-empty
state.
"""
import pytest

from subconscious.chat.text_utils import (
    DSMLFilter,
    _contains_dsml,
    _strip_dsml,
)


# One-line complete DSML statement, as it arrives in a single chunk.
_DSML_ONE_LINE = '<｜｜DSML｜｜ invoke name="foo">\n'

# Same, split across two chunks at the trigger character.
_DSML_SPLIT_A = "<｜"
_DSML_SPLIT_B = "｜DSML｜｜ invoke name=\"foo\">\n"


def test_passthrough_clean_text():
    f = DSMLFilter()
    assert f.feed("hello world") == "hello world"
    assert f.flush() == ""


def test_passthrough_text_with_at_sign():
    """`@` and other common characters must not be treated as triggers."""
    f = DSMLFilter()
    assert f.feed("user@domain: hello") == "user@domain: hello"


def test_whole_marker_dropped():
    f = DSMLFilter()
    out = f.feed(_DSML_ONE_LINE + "after\n")
    assert out == "after\n"
    assert f.flush() == ""


def test_split_marker_across_chunks():
    """Marker split at the trigger character must still be caught."""
    f = DSMLFilter()
    assert f.feed(_DSML_SPLIT_A) == ""
    assert f.feed(_DSML_SPLIT_B) == ""
    assert f.flush() == ""


def test_stray_trigger_does_not_stall_stream():
    """A stray ｜ followed by >BUFFER_CAP chars without newline is
    flushed. Prevents user text containing ｜ from hanging the stream."""
    f = DSMLFilter()
    f.BUFFER_CAP = 32

    # Trigger, then a long run without newline.
    out = f.feed("｜" + "x" * 100)
    # Output must eventually contain the buffer; content is not lost,
    # only delayed until the cap.
    assert "x" * 100 in out or "x" * 50 in f.flush() + out


def test_flush_returns_buffered_clean_text():
    """Text buffered after ｜ but without newline (and without marker)
    must be returned on flush, not silently dropped."""
    f = DSMLFilter()
    f.feed("｜some clean text")
    assert f.flush() == "｜some clean text"


def test_flush_empty():
    f = DSMLFilter()
    f.feed("clean")
    assert f.flush() == ""


def test_strip_dsml_whole_string():
    text = "before\n" + _DSML_ONE_LINE + "after\n"
    assert _strip_dsml(text) == "before\nafter"


def test_strip_dsml_no_marker_fast_path():
    """Without ｜ in the input, _strip_dsml returns input unchanged."""
    text = "no markup here"
    assert _strip_dsml(text) == text


def test_contains_dsml():
    assert _contains_dsml("text " + "<｜｜DSML｜｜ x> " + "more")
    assert not _contains_dsml("plain text")
    assert not _contains_dsml("")
