"""Stable response and exit-code primitives for the 2.0 command tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import Field

from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    FrozenArtifactModel,
)
from sol_execbench.core.integrity.schema_versions import (
    CLI_RESPONSE_SCHEMA_VERSION,
)


class CliExitCode(IntEnum):
    """Stable process exit codes for the command-line protocol."""

    SUCCESS = 0
    RESULT_FAILED = 1
    INPUT = 2
    UNAVAILABLE = 3
    EXECUTION = 4


@dataclass(frozen=True)
class CliResult:
    """Structured command result before text or JSON rendering."""

    data: Any = field(default_factory=dict)
    artifacts: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    exit_code: CliExitCode = CliExitCode.SUCCESS


class CliFailure(click.ClickException):
    """A handled CLI failure with a stable machine-readable classification."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "input_error",
        exit_code: CliExitCode = CliExitCode.INPUT,
        details: Any = None,
        hint: str | None = None,
    ) -> None:
        """Initialize a classified user-facing CLI failure."""
        super().__init__(message)
        self.code = code
        self.cli_exit_code = CliExitCode(exit_code)
        self.details = {} if details is None else details
        self.hint = hint


class CliArtifact(FrozenArtifactModel):
    """One path returned by a machine-readable CLI response."""

    type: str = Field(min_length=1)
    path: str = Field(min_length=1)


class CliError(FrozenArtifactModel):
    """Stable failure details in a CLI response."""

    code: str = Field(min_length=1)
    message: str
    details: Any
    hint: str | None


class CliSuccessResponse(CurrentFrozenSchemaModel):
    """Current successful CLI response envelope."""

    current_schema_version = CLI_RESPONSE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.cli_response.v1"] = (
        "sol_execbench.cli_response.v1"
    )
    ok: Literal[True]
    command: str = Field(min_length=1)
    data: Any
    artifacts: list[CliArtifact]
    warnings: list[str]


class CliFailureResponse(CurrentFrozenSchemaModel):
    """Current failed CLI response envelope."""

    current_schema_version = CLI_RESPONSE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.cli_response.v1"] = (
        "sol_execbench.cli_response.v1"
    )
    ok: Literal[False]
    command: str = Field(min_length=1)
    error: CliError


def artifact(path: Path, artifact_type: str) -> dict[str, Any]:
    """Return a serialized CLI artifact reference."""
    return CliArtifact(type=artifact_type, path=str(path)).model_dump(
        mode="json",
    )


def response_success(command: str, result: CliResult | None) -> dict[str, Any]:
    """Build a successful machine-readable CLI response."""
    result = result or CliResult()
    response = {
        "schema_version": CLI_RESPONSE_SCHEMA_VERSION,
        "ok": True,
        "command": command,
        "data": result.data,
        "artifacts": list(result.artifacts),
        "warnings": list(result.warnings),
    }
    return CliSuccessResponse.model_validate(response).model_dump(mode="json")


def response_failure(command: str, error: BaseException) -> dict[str, Any]:
    """Build a failed machine-readable CLI response."""
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
    response = {
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
    return CliFailureResponse.model_validate(response).model_dump(mode="json")


def output_format(ctx: click.Context | None = None) -> str:
    """Return the root command's requested output format."""
    ctx = ctx or click.get_current_context()
    return str(ctx.find_root().params.get("output_format", "text"))
