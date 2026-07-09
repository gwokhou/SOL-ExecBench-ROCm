from __future__ import annotations

import json

from click.testing import CliRunner

from sol_execbench.cli.commands import metadata as cli_metadata
from sol_execbench.cli.main import cli
from sol_execbench.core.platform.environment import (
    EnvironmentCheckResult,
    EnvironmentDiagnostics,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
)
from sol_execbench.core.platform.toolchain import (
    ToolLifecycle,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainRoutingReport,
    ToolchainRoutingRequest,
)


def _snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        generated_at="2026-05-25T00:00:00+00:00",
        collection_status=EnvironmentEvidenceStatus.AVAILABLE,
    )


def test_contract_cli_outputs_json_without_problem_directory() -> None:
    result = CliRunner().invoke(cli, ["contract", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"].startswith("sol_execbench.evaluator_contract.")
    assert "capabilities" in payload


def test_contract_cli_rejects_non_json_mode() -> None:
    result = CliRunner().invoke(cli, ["contract"])

    assert result.exit_code != 0
    assert "Only --json output is supported for contract" in result.output


def test_doctor_cli_outputs_json_without_problem_directory(monkeypatch) -> None:
    diagnostics = EnvironmentDiagnostics(
        generated_at="2026-05-25T00:00:00+00:00",
        status=EnvironmentEvidenceStatus.AVAILABLE,
        snapshot=_snapshot(),
        checks=[
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.AVAILABLE,
                message="ok",
            )
        ],
    )
    monkeypatch.setattr(
        cli_metadata, "build_environment_diagnostics", lambda: diagnostics
    )

    result = CliRunner().invoke(cli, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == "sol_execbench.environment_diagnostics.v1"
    assert (
        payload["snapshot"]["schema_version"] == "sol_execbench.environment_snapshot.v2"
    )
    assert payload["checks"][0]["name"] == "pytorch_rocm_runtime"


def test_doctor_cli_rejects_non_json_mode() -> None:
    result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code != 0
    assert "Only --json output is supported for doctor" in result.output


def test_toolchain_cli_outputs_routing_json(monkeypatch) -> None:
    def fake_report(request: ToolchainRoutingRequest) -> ToolchainRoutingReport:
        assert request.evidence_level == ToolchainEvidenceLevel.PROFILING
        assert request.artifact_type == ToolchainArtifactType.EXECUTABLE_RUN
        assert request.gpu_architecture == "gfx1200"
        return ToolchainRoutingReport(
            generated_at="2026-05-25T00:00:00+00:00",
            request=request,
            selected_tool_id="rocprofv3",
        )

    monkeypatch.setattr(cli_metadata, "build_toolchain_routing_report", fake_report)

    result = CliRunner().invoke(
        cli,
        [
            "toolchain",
            "--json",
            "--gpu-arch",
            "gfx1200",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selected_tool_id"] == "rocprofv3"


def test_toolchain_cli_outputs_registry_json(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_metadata,
        "default_toolchain_registry",
        lambda: [
            ToolchainCapability(
                tool_id="rocprofv3",
                display_name="ROCprofiler SDK rocprofv3",
                lifecycle=ToolLifecycle.ACTIVE,
                evidence_levels=[ToolchainEvidenceLevel.PROFILING],
                artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            )
        ],
    )

    result = CliRunner().invoke(cli, ["toolchain", "--json", "--list-registry"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["tool_id"] == "rocprofv3"
    assert payload[0]["lifecycle"] == "active"


def test_toolchain_cli_rejects_non_json_mode() -> None:
    result = CliRunner().invoke(cli, ["toolchain"])

    assert result.exit_code != 0
    assert "Only --json output is supported for toolchain" in result.output
