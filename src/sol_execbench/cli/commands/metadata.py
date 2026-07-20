# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment, contract, and toolchain command domains."""

from __future__ import annotations

import json
from typing import Any

import click

from ...core.evaluator_contract import build_evaluator_contract
from ...core.platform.environment import build_environment_diagnostics
from ...core.platform.toolchain import (
    ToolchainArtifactType,
    ToolchainEvidenceLevel,
    ToolchainRoutingRequest,
    build_toolchain_routing_report,
    default_toolchain_registry,
)
from ..protocol import (
    CLI_CONTRACT_SCHEMA_VERSION,
    CLI_RESPONSE_SCHEMA_VERSION,
    EXIT_UNAVAILABLE,
    CliResult,
    output_format,
)


def _show(payload: Any) -> None:
    if output_format() == "text":
        click.echo(json.dumps(payload, indent=2, sort_keys=True))


@click.group("environment")
def environment_cli() -> None:
    """Inspect runtime and hardware availability."""


@environment_cli.command("doctor")
def doctor_cli() -> CliResult:
    """Diagnose the ROCm environment without running a benchmark.

    Example: ``sol-execbench --format json environment doctor``
    """
    payload = build_environment_diagnostics().model_dump(mode="json")
    _show(payload)
    unavailable = payload.get("status") == "unavailable"
    return CliResult(data=payload, exit_code=EXIT_UNAVAILABLE if unavailable else 0)


@click.group("contract")
def contract_cli() -> None:
    """Describe stable evaluator and CLI interfaces."""


@contract_cli.command("evaluator")
def evaluator_contract_cli() -> CliResult:
    """Print the current evaluator, SOLAR, corpus, and scoring boundary."""
    payload = build_evaluator_contract().model_dump(mode="json")
    _show(payload)
    return CliResult(data=payload)


@contract_cli.command("cli")
def cli_contract_cli() -> CliResult:
    """Print the generated, machine-readable CLI contract.

    Example: ``sol-execbench --format json contract cli``
    """
    from sol_execbench.cli.main import cli

    context = click.Context(cli, info_name="sol-execbench")
    payload = {
        "schema_version": CLI_CONTRACT_SCHEMA_VERSION,
        "root_options_before_subcommand": True,
        "response_schema": CLI_RESPONSE_SCHEMA_VERSION,
        "exit_codes": {
            "0": "success",
            "1": "valid evaluation or verification result did not pass",
            "2": "usage, input, or schema error",
            "3": "hardware, environment, or dependency unavailable",
            "4": "execution or internal failure",
        },
        "command_tree": _describe_command(cli, context),
        "artifacts": {
            "canonical_trace_jsonl": "Canonical evaluator Trace JSONL; schema and semantics unchanged",
            "json_file": "Command-specific JSON evidence",
            "directory": "Generated artifact directory",
        },
    }
    _show(payload)
    return CliResult(data=payload)


def _describe_command(command: click.Command, ctx: click.Context) -> dict[str, Any]:
    params = []
    for param in command.get_params(ctx):
        entry: dict[str, Any] = {
            "name": param.name,
            "kind": "argument" if isinstance(param, click.Argument) else "option",
            "type": param.type.name,
            "required": bool(param.required),
        }
        if isinstance(param, click.Option):
            entry["flags"] = list(param.opts)
            entry["default"] = (
                param.default
                if param.default is None
                or isinstance(param.default, (str, int, float, bool, list, tuple))
                else None
            )
            entry["multiple"] = param.multiple
        params.append(entry)
    result: dict[str, Any] = {
        "name": command.name,
        "help": command.help or "",
        "parameters": params,
    }
    constraints = getattr(command, "cli_constraints", None)
    if constraints:
        result["constraints"] = list(constraints)
    if isinstance(command, click.Group):
        children = []
        for name in command.list_commands(ctx):
            child = command.get_command(ctx, name)
            if child is None:
                continue
            children.append(_describe_command(child, click.Context(child, parent=ctx)))
        result["commands"] = children
    return result


@click.group("toolchain")
def toolchain_cli() -> None:
    """Inspect ROCm tool routing capabilities."""


@toolchain_cli.command("route")
@click.option(
    "--evidence-level",
    type=click.Choice([v.value for v in ToolchainEvidenceLevel]),
    default=ToolchainEvidenceLevel.PROFILING.value,
    show_default=True,
)
@click.option(
    "--artifact-type",
    type=click.Choice([v.value for v in ToolchainArtifactType]),
    default=ToolchainArtifactType.EXECUTABLE_RUN.value,
    show_default=True,
)
@click.option("--gpu-arch", "gpu_architecture")
@click.option("--hardware-generation")
@click.option("--rocm-version")
def toolchain_route_cli(
    evidence_level: str,
    artifact_type: str,
    gpu_architecture: str | None,
    hardware_generation: str | None,
    rocm_version: str | None,
) -> CliResult:
    """Choose a toolchain for a requested evidence artifact."""
    request = ToolchainRoutingRequest(
        evidence_level=ToolchainEvidenceLevel(evidence_level),
        artifact_type=ToolchainArtifactType(artifact_type),
        gpu_architecture=gpu_architecture,
        hardware_generation=hardware_generation,
        rocm_version=rocm_version,
    )
    payload = build_toolchain_routing_report(request).model_dump(mode="json")
    _show(payload)
    return CliResult(data=payload)


@toolchain_cli.command("list")
def toolchain_list_cli() -> CliResult:
    """List the registry; route-only filters are intentionally not accepted."""
    payload = [item.model_dump(mode="json") for item in default_toolchain_registry()]
    _show(payload)
    return CliResult(data=payload)
