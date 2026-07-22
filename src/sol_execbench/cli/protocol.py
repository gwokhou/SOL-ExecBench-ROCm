"""Stable response and exit-code primitives for the 2.0 command tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from sol_execbench.core.integrity.schema_versions import SCHEMA_VERSIONS

CLI_RESPONSE_SCHEMA_VERSION = SCHEMA_VERSIONS["cli_response"]
CLI_CONTRACT_SCHEMA_VERSION = SCHEMA_VERSIONS["cli_contract"]

EXIT_SUCCESS = 0
EXIT_RESULT_FAILED = 1
EXIT_INPUT = 2
EXIT_UNAVAILABLE = 3
EXIT_EXECUTION = 4


@dataclass(frozen=True)
class CliResult:
    data: Any = field(default_factory=dict)
    artifacts: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    exit_code: int = EXIT_SUCCESS


class CliFailure(click.ClickException):
    """A handled CLI failure with a stable machine-readable classification."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "input_error",
        exit_code: int = EXIT_INPUT,
        details: Any = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cli_exit_code = exit_code
        self.details = {} if details is None else details
        self.hint = hint


def artifact(path: Path, artifact_type: str) -> dict[str, Any]:
    return {"type": artifact_type, "path": str(path)}


def response_success(command: str, result: CliResult | None) -> dict[str, Any]:
    result = result or CliResult()
    return {
        "schema_version": CLI_RESPONSE_SCHEMA_VERSION,
        "ok": True,
        "command": command,
        "data": result.data,
        "artifacts": list(result.artifacts),
        "warnings": list(result.warnings),
    }


def response_failure(command: str, error: BaseException) -> dict[str, Any]:
    if isinstance(error, CliFailure):
        code = error.code
        details = error.details
        hint = error.hint
    elif isinstance(error, click.UsageError):
        code = "usage_error"
        details = {}
        hint = "Run the command with --help to inspect its accepted arguments."
    elif isinstance(error, click.ClickException):
        code = "input_error"
        details = {}
        hint = None
    else:
        code = "execution_error"
        details = {"exception_type": type(error).__name__}
        hint = None
    return {
        "schema_version": CLI_RESPONSE_SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "error": {
            "code": code,
            "message": str(error),
            "details": details,
            "hint": hint,
        },
    }


def output_format(ctx: click.Context | None = None) -> str:
    ctx = ctx or click.get_current_context()
    return str(ctx.find_root().params.get("output_format", "text"))
