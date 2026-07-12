from __future__ import annotations

import json
from typing import Any

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


def _json(args: list[str]) -> tuple[object, dict[str, Any]]:
    result = CliRunner().invoke(cli, ["--format", "json", *args])
    assert result.exit_code == 0, result.output
    return result, json.loads(result.output)


def test_evaluator_contract_uses_response_envelope() -> None:
    _, response = _json(["contract", "evaluator"])
    assert response["schema_version"] == "sol_execbench.cli_response.v1"
    assert response["data"]["schema_version"].startswith(
        "sol_execbench.evaluator_contract."
    )


def test_cli_contract_matches_public_tree() -> None:
    _, response = _json(["contract", "cli"])
    contract = response["data"]
    assert contract["schema_version"] == "sol_execbench.cli_contract.v1"
    assert {item["name"] for item in contract["command_tree"]["commands"]} == {
        "evaluate",
        "environment",
        "contract",
        "toolchain",
        "dataset",
        "baseline",
        "hardware",
        "score",
    }


def test_doctor_outputs_diagnostics(monkeypatch) -> None:
    diagnostics = EnvironmentDiagnostics(
        generated_at="2026-05-25T00:00:00+00:00",
        status=EnvironmentEvidenceStatus.AVAILABLE,
        snapshot=EnvironmentSnapshot(
            generated_at="2026-05-25T00:00:00+00:00",
            collection_status=EnvironmentEvidenceStatus.AVAILABLE,
        ),
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
    _, response = _json(["environment", "doctor"])
    assert response["data"]["checks"][0]["name"] == "pytorch_rocm_runtime"


def test_toolchain_route_outputs_report(monkeypatch) -> None:
    def fake_report(request: ToolchainRoutingRequest) -> ToolchainRoutingReport:
        assert request.gpu_architecture == "gfx1200"
        return ToolchainRoutingReport(
            generated_at="2026-05-25T00:00:00+00:00",
            request=request,
            selected_tool_id="rocprofv3",
        )

    monkeypatch.setattr(cli_metadata, "build_toolchain_routing_report", fake_report)
    _, response = _json(["toolchain", "route", "--gpu-arch", "gfx1200"])
    assert response["data"]["selected_tool_id"] == "rocprofv3"


def test_toolchain_list_has_no_route_filters(monkeypatch) -> None:
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
    _, response = _json(["toolchain", "list"])
    assert response["data"][0]["tool_id"] == "rocprofv3"
    rejected = CliRunner().invoke(cli, ["toolchain", "list", "--gpu-arch", "gfx1200"])
    assert rejected.exit_code == 2
