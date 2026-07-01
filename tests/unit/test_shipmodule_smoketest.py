"""Regression tests for _shipmodule_smoketest.accumulate.

Guards against re-introducing the mutable-default-argument bug where successive
calls to accumulate() would share the same backing list.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _shipmodule_smoketest import accumulate


def test_accumulate_returns_value():
    result = accumulate(1)
    assert result == [1]


def test_accumulate_explicit_bucket():
    bucket = [0]
    result = accumulate(1, bucket)
    assert result == [0, 1]
    assert result is bucket


def test_accumulate_does_not_share_state_across_calls():
    """Regression: mutable default bucket=[] would cause calls to bleed into each other."""
    first = accumulate("a")
    second = accumulate("b")
    # If the mutable-default bug were present, second would be ["a", "b"]
    assert first == ["a"], f"Expected ['a'], got {first}"
    assert second == ["b"], f"Expected ['b'], got {second}"
    assert first is not second, "Separate calls must return independent lists"
