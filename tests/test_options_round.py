"""Unit tests for per-round option boosting."""
import pytest

from subconscious.chat.tools_loop import (
    TOOL_ROUND_NUM_PREDICT_BOOST,
    _options_for_round,
)


def test_round_zero_passthrough():
    opts = {"num_predict": 512, "temperature": 0.3}
    assert _options_for_round(opts, 0) == opts
    # Original dict must not be mutated.
    assert opts["num_predict"] == 512


def test_boost_on_later_rounds():
    opts = {"num_predict": 512, "temperature": 0.3}
    boosted = _options_for_round(opts, 1)
    assert boosted["num_predict"] == 512 + TOOL_ROUND_NUM_PREDICT_BOOST
    # Other keys unchanged.
    assert boosted["temperature"] == 0.3
    # Original not mutated.
    assert opts["num_predict"] == 512


def test_default_num_predict_when_missing():
    """When options has no num_predict, base for boost is 512."""
    boosted = _options_for_round({}, 2)
    assert boosted["num_predict"] == 512 + TOOL_ROUND_NUM_PREDICT_BOOST


def test_boost_accumulates_relative_to_original():
    """Each call computes from the original value, not from the previous
    call's result. Round 1 and round 5 both start from the same base."""
    opts = {"num_predict": 1000}
    r1 = _options_for_round(opts, 1)
    r5 = _options_for_round(opts, 5)
    assert r1["num_predict"] == r5["num_predict"]
    assert r1["num_predict"] == 1000 + TOOL_ROUND_NUM_PREDICT_BOOST
