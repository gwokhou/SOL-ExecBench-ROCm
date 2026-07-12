from __future__ import annotations

import argparse

import pytest

from sol_execbench.core.arguments import none_if_requested, parse_bool


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("0", False)])
def test_parse_bool_accepts_common_values(value: str, expected: bool) -> None:
    assert parse_bool(value) is expected


def test_parse_bool_rejects_unknown_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_bool("enabled")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("NONE", None), ("value", "value")],
)
def test_none_if_requested_normalizes_requested_nulls(
    value: str | None, expected: str | None
) -> None:
    assert none_if_requested(value) == expected
