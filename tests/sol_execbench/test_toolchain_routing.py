from __future__ import annotations

import json

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.environment import ProbeCompletedProcess
from sol_execbench.core.toolchain import (
    TOOLCHAIN_ROUTING_SCHEMA_VERSION,
    ToolLifecycle,
    ToolchainArtifactType,
    ToolchainCapability,
    ToolchainEvidenceLevel,
    ToolchainRoutingRequest,
    ToolchainStatus,
    build_toolchain_routing_report,
    default_toolchain_registry,
)


def _which(binary: str) -> str | None:
    paths = {
        "rocprofv3": "/opt/rocm/bin/rocprofv3",
        "rocminfo": "/opt/rocm/bin/rocminfo",
        "llvm-objdump": "/usr/bin/llvm-objdump",
        "readelf": "/usr/bin/readelf",
    }
    return paths.get(binary)


def test_default_registry_records_lifecycle_and_static_tools():
    registry = {entry.tool_id: entry for entry in default_toolchain_registry()}

    assert registry["rocprofv3"].lifecycle == ToolLifecycle.ACTIVE
    assert registry["rocprofiler-systems"].lifecycle == ToolLifecycle.MIGRATED
    assert registry["rocprofiler-systems"].replacement_tool_id == "rocm-systems"
    assert registry["rga"].lifecycle == ToolLifecycle.PLANNED
    assert registry["llvm-objdump"].lifecycle == ToolLifecycle.ACTIVE
    assert registry["readelf"].lifecycle == ToolLifecycle.ACTIVE
    assert ToolchainEvidenceLevel.STATIC in registry["rga"].evidence_levels


def test_routing_selects_available_tool_and_preserves_authority_boundaries():
    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        assert command == ["rocprofv3", "--version"]
        assert timeout_seconds == 3.0
        return ProbeCompletedProcess(returncode=0, stdout="rocprofv3 7.0.0")

    report = build_toolchain_routing_report(
        ToolchainRoutingRequest(
            evidence_level=ToolchainEvidenceLevel.PROFILING,
            artifact_type=ToolchainArtifactType.EXECUTABLE_RUN,
            gpu_architecture="gfx1200",
        ),
        runner=runner,
        which=_which,
    )
    payload = report.model_dump(mode="json")

    assert payload["schema_version"] == TOOLCHAIN_ROUTING_SCHEMA_VERSION
    assert payload["selected_tool_id"] == "rocprofv3"
    assert payload["diagnostic_only"] is True
    assert payload["correctness_authority"] is False
    assert payload["performance_authority"] is False
    assert payload["leaderboard_authority"] is False
    selected = [decision for decision in payload["decisions"] if decision["selected"]]
    assert selected[0]["status"] == ToolchainStatus.AVAILABLE.value


def test_routing_reports_migrated_legacy_tool_and_fallback():
    report = build_toolchain_routing_report(
        ToolchainRoutingRequest(
            evidence_level=ToolchainEvidenceLevel.PROFILING,
            artifact_type=ToolchainArtifactType.EXECUTABLE_RUN,
        ),
        registry=[
            ToolchainCapability(
                tool_id="rocprofiler-systems",
                display_name="legacy",
                lifecycle=ToolLifecycle.MIGRATED,
                replacement_tool_id="rocm-systems",
                evidence_levels=[ToolchainEvidenceLevel.PROFILING],
                artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            )
        ],
    )

    decision = report.decisions[0]
    assert decision.status == ToolchainStatus.MIGRATED
    assert decision.fallback_tool_id == "rocm-systems"
    assert "migrated" in decision.reason


def test_routing_rejects_wrong_artifact_with_explicit_reason():
    report = build_toolchain_routing_report(
        ToolchainRoutingRequest(
            evidence_level=ToolchainEvidenceLevel.PROFILING,
            artifact_type=ToolchainArtifactType.ELF_OBJECT,
        ),
        registry=[
            ToolchainCapability(
                tool_id="rocprofv3",
                display_name="rocprofv3",
                lifecycle=ToolLifecycle.ACTIVE,
                evidence_levels=[ToolchainEvidenceLevel.PROFILING],
                artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
                expected_binaries=["rocprofv3"],
            )
        ],
    )

    assert report.selected_tool_id is None
    assert report.decisions[0].status == ToolchainStatus.UNSUPPORTED_ARTIFACT
    assert report.decisions[0].reason_code == "unsupported_artifact"


def test_static_tools_are_routable_and_optional_candidates_remain_nonmandatory():
    commands: list[list[str]] = []

    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        commands.append(command)
        return ProbeCompletedProcess(returncode=0, stdout=f"{command[0]} ok")

    report = build_toolchain_routing_report(
        ToolchainRoutingRequest(
            evidence_level=ToolchainEvidenceLevel.STATIC,
            artifact_type=ToolchainArtifactType.ROCM_BINARY,
        ),
        runner=runner,
        which=_which,
    )

    assert report.selected_tool_id == "readelf"
    statuses = {decision.tool_id: decision.status for decision in report.decisions}
    assert statuses["readelf"] == ToolchainStatus.AVAILABLE
    assert statuses["llvm-objdump"] == ToolchainStatus.AVAILABLE
    assert statuses["roc-objdump"] == ToolchainStatus.UNAVAILABLE
    assert statuses["rga"] == ToolchainStatus.PLANNED
    assert ["readelf", "--version"] in commands
    assert ["llvm-objdump", "--version"] in commands


def test_toolchain_cli_prints_routing_json(monkeypatch):
    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        return ProbeCompletedProcess(returncode=0, stdout="rocprofv3 7.0.0")

    monkeypatch.setattr(
        "sol_execbench.core.toolchain._run_probe",
        runner,
    )

    result = CliRunner().invoke(
        cli,
        [
            "toolchain",
            "--json",
            "--evidence-level",
            "profiling",
            "--artifact-type",
            "executable_run",
            "--gpu-arch",
            "gfx1200",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == TOOLCHAIN_ROUTING_SCHEMA_VERSION
    assert payload["request"]["gpu_architecture"] == "gfx1200"
    assert payload["correctness_authority"] is False


def test_toolchain_cli_can_list_registry():
    result = CliRunner().invoke(cli, ["toolchain", "--json", "--list-registry"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    tool_ids = {entry["tool_id"] for entry in payload}
    assert {"rocprofv3", "rocm-systems", "rga", "llvm-objdump"}.issubset(tool_ids)
