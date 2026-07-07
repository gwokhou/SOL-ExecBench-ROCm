# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-free metadata commands for the SOL-ExecBench CLI."""

from __future__ import annotations

import json

import click

from ..core.data.contract import build_evaluator_contract
from ..core.environment import build_environment_diagnostics
from ..core.toolchain import (
    ToolchainArtifactType,
    ToolchainEvidenceLevel,
    ToolchainRoutingRequest,
    build_toolchain_routing_report,
    default_toolchain_registry,
)


@click.command("contract", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print contract JSON")
def _contract_cli(json_output: bool) -> None:
    """Print the GPU-free evaluator compatibility contract."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for contract")
    payload = build_evaluator_contract().model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.command("doctor", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print diagnostics JSON")
def _doctor_cli(json_output: bool) -> None:
    """Print ROCm environment diagnostics."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for doctor")
    payload = build_environment_diagnostics().model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))


@click.command("toolchain", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "json_output", is_flag=True, help="Print routing JSON")
@click.option(
    "--evidence-level",
    type=click.Choice([level.value for level in ToolchainEvidenceLevel]),
    default=ToolchainEvidenceLevel.PROFILING.value,
    show_default=True,
    help="Evidence level to route",
)
@click.option(
    "--artifact-type",
    type=click.Choice([artifact.value for artifact in ToolchainArtifactType]),
    default=ToolchainArtifactType.EXECUTABLE_RUN.value,
    show_default=True,
    help="Artifact type to route",
)
@click.option("--gpu-arch", "gpu_architecture", help="GPU architecture such as gfx1200")
@click.option("--hardware-generation", help="Hardware generation such as RDNA 4")
@click.option("--rocm-version", help="ROCm version such as 7.0")
@click.option(
    "--list-registry",
    is_flag=True,
    help="Print registry entries instead of a routing decision",
)
def _toolchain_cli(
    json_output: bool,
    evidence_level: str,
    artifact_type: str,
    gpu_architecture: str | None,
    hardware_generation: str | None,
    rocm_version: str | None,
    list_registry: bool,
) -> None:
    """Print ROCm toolchain routing diagnostics."""

    if not json_output:
        raise click.ClickException("Only --json output is supported for toolchain")
    if list_registry:
        payload = [
            capability.model_dump(mode="json")
            for capability in default_toolchain_registry()
        ]
        click.echo(json.dumps(payload, sort_keys=True))
        return
    request = ToolchainRoutingRequest(
        evidence_level=ToolchainEvidenceLevel(evidence_level),
        artifact_type=ToolchainArtifactType(artifact_type),
        gpu_architecture=gpu_architecture,
        hardware_generation=hardware_generation,
        rocm_version=rocm_version,
    )
    payload = build_toolchain_routing_report(request).model_dump(mode="json")
    click.echo(json.dumps(payload, sort_keys=True))
