# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""SOL ExecBench 3.0 command-line interface."""

from __future__ import annotations

import difflib
import io
import json
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import click

from .protocol import (
    EXIT_EXECUTION,
    CliFailure,
    CliResult,
    response_failure,
    response_success,
)

VERSION = "3.0.0"


class LazyGroup(click.Group):
    """A standard Click group whose top-level domains are imported on demand."""

    _loaders = {
        "evaluate": ("sol_execbench.cli.commands.evaluate", "evaluate_cli"),
        "environment": ("sol_execbench.cli.commands.metadata", "environment_cli"),
        "contract": ("sol_execbench.cli.commands.metadata", "contract_cli"),
        "toolchain": ("sol_execbench.cli.commands.metadata", "toolchain_cli"),
        "dataset": ("sol_execbench.cli.commands.dataset", "dataset_cli"),
        "solar": ("sol_execbench.cli.commands.solar", "solar_cli"),
        "score": ("sol_execbench.cli.commands.official_score", "score_cli"),
    }

    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self._loaders)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = self._loaders.get(cmd_name)
        if target is None:
            return None
        from importlib import import_module

        module_name, attribute = target
        return getattr(import_module(module_name), attribute)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError as exc:
            if args:
                matches = difflib.get_close_matches(
                    args[0], self.list_commands(ctx), n=1
                )
                if matches:
                    raise click.UsageError(
                        f"{exc.message}\nDid you mean '{matches[0]}'?", ctx=exc.ctx
                    ) from exc
            raise


class RootGroup(LazyGroup):
    """Serialize JSON-mode results and handled failures at one boundary."""

    def main(
        self,
        args: Sequence[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        windows_expand_args: bool = True,
        **extra: Any,
    ) -> Any:
        import sys

        raw_args = list(args) if args is not None else sys.argv[1:]
        json_mode = _requested_json(raw_args)
        standalone = standalone_mode
        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        try:
            if json_mode:
                with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                    result = super().main(
                        args=raw_args,
                        prog_name=prog_name,
                        complete_var=complete_var,
                        standalone_mode=False,
                        windows_expand_args=windows_expand_args,
                        **extra,
                    )
            else:
                result = super().main(
                    args=raw_args,
                    prog_name=prog_name,
                    complete_var=complete_var,
                    standalone_mode=False,
                    windows_expand_args=windows_expand_args,
                    **extra,
                )
            cli_result = result if isinstance(result, CliResult) else CliResult()
            exit_code = cli_result.exit_code
            if json_mode:
                click.echo(
                    json.dumps(
                        response_success(_command_name(raw_args), cli_result),
                        sort_keys=True,
                    )
                )
        except click.exceptions.Exit as exc:
            exit_code = exc.exit_code
            if json_mode and exc.exit_code != 0:
                click.echo(
                    json.dumps(
                        response_failure(_command_name(raw_args), exc), sort_keys=True
                    )
                )
        except (click.ClickException, Exception) as exc:
            if isinstance(exc, click.UsageError):
                exit_code = 2
            elif isinstance(exc, CliFailure):
                exit_code = exc.cli_exit_code
            elif isinstance(exc, click.ClickException):
                exit_code = getattr(exc, "exit_code", 2)
                if type(exc) is click.ClickException:
                    exit_code = 2
            else:
                exit_code = EXIT_EXECUTION
            if json_mode:
                click.echo(
                    json.dumps(
                        response_failure(_command_name(raw_args), exc), sort_keys=True
                    )
                )
            elif isinstance(exc, click.ClickException):
                exc.show()
            else:
                click.echo(f"Error: {exc}", err=True)
        if standalone:
            raise SystemExit(exit_code)
        return exit_code if exit_code else result


def _requested_json(args: list[str]) -> bool:
    return any(
        argument == "--format=json"
        or (
            argument == "--format"
            and index + 1 < len(args)
            and args[index + 1] == "json"
        )
        for index, argument in enumerate(args)
    )


def _command_name(args: list[str]) -> str:
    values = [
        argument
        for index, argument in enumerate(args)
        if argument != "--format=json"
        and not (
            argument == "--format" or (index > 0 and args[index - 1] == "--format")
        )
    ]
    if not values or values[0].startswith("-"):
        return "sol-execbench"
    domain = values[0]
    depths = {
        "evaluate": 1,
        "environment": 2,
        "contract": 2,
        "toolchain": 2,
        "dataset": 3,
        "solar": 2,
        "score": 2,
    }
    depth = depths.get(domain, 1)
    return " ".join(values[:depth])


@click.group(
    cls=RootGroup,
    name="sol-execbench",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Response format. This root option must precede the subcommand.",
)
@click.version_option(VERSION, prog_name="sol-execbench")
@click.pass_context
def cli(ctx: click.Context, output_format: str) -> None:
    """Evaluate kernels and manage GPU benchmark evidence.

    Root options such as --format must appear before the subcommand. Run
    ``sol-execbench contract cli`` for the machine-readable command contract.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


if __name__ == "__main__":
    cli()
