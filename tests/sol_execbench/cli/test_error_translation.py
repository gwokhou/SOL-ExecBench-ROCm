"""Tests for declarative CLI exception translation."""

from __future__ import annotations

import pytest

from sol_execbench.cli.error_translation import (
    CliErrorRule,
    translate_cli_errors,
)
from sol_execbench.cli.protocol import CliExitCode, CliFailure


def test_translation_preserves_protocol_fields_and_exception_chain() -> None:
    source = ValueError("invalid value")
    rule = CliErrorRule(
        exception_type=ValueError,
        code="invalid_fixture",
        exit_code=CliExitCode.EXECUTION,
        hint="Fix the fixture.",
        message=lambda error: f"wrapped: {error}",
        details=lambda error: {"kind": type(error).__name__},
    )

    with pytest.raises(CliFailure) as raised, translate_cli_errors(rule):
        raise source

    failure = raised.value
    assert failure.code == "invalid_fixture"
    assert failure.cli_exit_code is CliExitCode.EXECUTION
    assert failure.hint == "Fix the fixture."
    assert failure.details == {"kind": "ValueError"}
    assert str(failure) == "wrapped: invalid value"
    assert failure.__cause__ is source


def test_translation_uses_first_matching_rule() -> None:
    rules = (
        CliErrorRule(exception_type=FileNotFoundError, code="missing"),
        CliErrorRule(exception_type=OSError, code="io"),
    )
    with pytest.raises(CliFailure) as raised, translate_cli_errors(*rules):
        raise FileNotFoundError("gone")
    assert raised.value.code == "missing"


def test_translation_does_not_wrap_unknown_errors() -> None:
    error = RuntimeError("unexpected")
    with (
        pytest.raises(RuntimeError) as raised,
        translate_cli_errors(
            CliErrorRule(exception_type=ValueError, code="invalid")
        ),
    ):
        raise error
    assert raised.value is error
