"""Declarative translation of expected domain errors at CLI boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sol_execbench.cli.protocol import CliExitCode, CliFailure

ErrorTransform = Callable[[Exception], object]


@dataclass(frozen=True, slots=True, kw_only=True)
class CliErrorRule:
    """Map one explicit exception family to the stable CLI protocol."""

    exception_type: type[Exception] | tuple[type[Exception], ...]
    code: str
    exit_code: CliExitCode = CliExitCode.INPUT
    hint: str | None = None
    message: ErrorTransform | None = None
    details: ErrorTransform | None = None

    def matches(self, error: Exception) -> bool:
        """Return whether this rule owns an exception instance."""
        return isinstance(error, self.exception_type)

    def failure(self, error: Exception) -> CliFailure:
        """Build the stable failure while preserving custom projections."""
        message = self.message(error) if self.message else str(error)
        details = self.details(error) if self.details else None
        return CliFailure(
            str(message),
            code=self.code,
            exit_code=self.exit_code,
            details=details,
            hint=self.hint,
        )


@contextmanager
def translate_cli_errors(*rules: CliErrorRule) -> Iterator[None]:
    """Translate only explicitly declared exceptions, in rule order."""
    if not rules:
        yield
        return
    handled_types = tuple(
        exception_type
        for rule in rules
        for exception_type in (
            rule.exception_type
            if isinstance(rule.exception_type, tuple)
            else (rule.exception_type,)
        )
    )
    try:
        yield
    except handled_types as error:
        for rule in rules:
            if rule.matches(error):
                raise rule.failure(error) from error
        raise


__all__ = ["CliErrorRule", "translate_cli_errors"]
