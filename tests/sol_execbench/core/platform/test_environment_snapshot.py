from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from sol_execbench.core.platform.environment import (
    ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION,
    EnvironmentCheckResult,
    EnvironmentDiagnostics,
    EnvironmentEvidenceStatus,
    EnvironmentSnapshot,
    GpuEnvironmentSummary,
    ProbeCompletedProcess,
    PytorchRocmSummary,
    ToolProbeResult,
    build_environment_diagnostics,
    collect_environment_snapshot,
    probe_tool,
)
from sol_execbench.core.platform.environment_probes import parse_probe_output
from sol_execbench.core.platform.environment_snapshot import summarize_gpus


def test_minimal_environment_snapshot_round_trips():
    snapshot = EnvironmentSnapshot(
        generated_at="2026-05-25T00:00:00+00:00",
        collection_status=EnvironmentEvidenceStatus.SKIPPED,
    )

    payload = snapshot.model_dump(mode="json")
    assert payload["schema_version"] == ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION
    assert payload["collection_status"] == "skipped"
    assert EnvironmentSnapshot.model_validate(payload) == snapshot


def test_tool_probe_result_serializes_status_and_parsed_fields():
    result = ToolProbeResult(
        tool="rocminfo",
        command=["rocminfo"],
        path="/opt/rocm/bin/rocminfo",
        status=EnvironmentEvidenceStatus.AVAILABLE,
        returncode=0,
        stdout_tail="Name: gfx1200\nMarketing Name: Radeon",
        parsed={"gfx_targets": ["gfx1200"]},
    )

    payload = result.model_dump(mode="json")
    assert payload["status"] == "available"
    assert payload["parsed"]["gfx_targets"] == ["gfx1200"]


def test_probe_tool_reports_unavailable_without_running_command():
    calls: list[list[str]] = []

    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        calls.append(command)
        return ProbeCompletedProcess(returncode=0)

    result = probe_tool(
        "rocminfo",
        ["rocminfo"],
        runner=runner,
        which=lambda _tool: None,
        timeout_seconds=1.0,
    )

    assert result.status == EnvironmentEvidenceStatus.UNAVAILABLE
    assert result.path is None
    assert calls == []


def test_probe_tool_reports_available_and_parses_gfx_target():
    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        assert command == ["/fake/rocminfo"]
        assert timeout_seconds == 2.0
        return ProbeCompletedProcess(
            returncode=0,
            stdout="Agent 1\nName: gfx942\nMarketing Name: AMD Instinct MI300X\n",
        )

    result = probe_tool(
        "rocminfo",
        ["rocminfo"],
        runner=runner,
        which=lambda _tool: "/fake/rocminfo",
        timeout_seconds=2.0,
    )

    assert result.status == EnvironmentEvidenceStatus.AVAILABLE
    assert result.returncode == 0
    assert result.parsed["gfx_targets"] == ["gfx942"]


def test_parse_probe_output_ignores_rocm_generic_isa_labels() -> None:
    parsed = parse_probe_output("ISA: gfx12\nName: gfx1200\n")

    assert parsed["gfx_targets"] == ["gfx1200"]


def test_summarize_gpus_deduplicates_single_pytorch_device_from_tools():
    tools = {
        "rocminfo": ToolProbeResult(
            tool="rocminfo",
            status=EnvironmentEvidenceStatus.AVAILABLE,
            parsed={"gfx_targets": ["gfx1200"]},
        )
    }
    pytorch = PytorchRocmSummary(
        available=True,
        device_count=1,
        device_name="AMD Radeon RX 9060 XT",
        gfx_target="gfx1200",
    )

    assert summarize_gpus(tools, pytorch) == [
        GpuEnvironmentSummary(
            source="pytorch",
            index=0,
            name="AMD Radeon RX 9060 XT",
            gfx_target="gfx1200",
        )
    ]


def test_probe_tool_reports_nonzero_exit_as_failed():
    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        return ProbeCompletedProcess(returncode=1, stderr="permission denied")

    result = probe_tool(
        "amd-smi",
        ["amd-smi", "static", "-a"],
        runner=runner,
        which=lambda _tool: "/opt/rocm/bin/amd-smi",
    )

    assert result.status == EnvironmentEvidenceStatus.FAILED
    assert result.returncode == 1
    assert "permission denied" in result.stderr_tail


def test_probe_tool_reports_timeout():
    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        raise subprocess.TimeoutExpired(command, timeout_seconds, output="partial")

    result = probe_tool(
        "rocminfo",
        ["rocminfo"],
        runner=runner,
        which=lambda _tool: "/opt/rocm/bin/rocminfo",
        timeout_seconds=0.5,
    )

    assert result.status == EnvironmentEvidenceStatus.TIMEOUT
    assert result.stdout_tail == "partial"


def test_collect_environment_snapshot_uses_injected_runner_and_is_gpu_free():
    commands: list[list[str]] = []

    def runner(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
        commands.append(command)
        if command == ["/fake/rocm_agent_enumerator"]:
            return ProbeCompletedProcess(returncode=0, stdout="gfx1200\n")
        return ProbeCompletedProcess(returncode=0, stdout="")

    snapshot = collect_environment_snapshot(
        runner=runner,
        which=lambda tool: f"/fake/{tool}",
        collect_pytorch=False,
        now=lambda: datetime(2026, 5, 25, tzinfo=UTC),
    )

    assert snapshot.schema_version == ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION
    assert snapshot.collection_status == EnvironmentEvidenceStatus.AVAILABLE
    assert [gpu.gfx_target for gpu in snapshot.gpus] == ["gfx1200"]
    assert ["/fake/amd-smi", "static", "-a"] in commands
    assert ["/fake/rocminfo"] in commands
    assert ["/fake/rocm_agent_enumerator"] in commands


def test_collect_environment_snapshot_handles_missing_tools():
    snapshot = collect_environment_snapshot(
        runner=lambda command, timeout_seconds: ProbeCompletedProcess(returncode=0),
        which=lambda _tool: None,
        collect_pytorch=False,
        now=lambda: datetime(2026, 5, 25, tzinfo=UTC),
    )

    assert snapshot.collection_status == EnvironmentEvidenceStatus.UNAVAILABLE
    assert set(snapshot.tools) == {"amd-smi", "rocminfo", "rocm_agent_enumerator"}
    assert all(
        result.status == EnvironmentEvidenceStatus.UNAVAILABLE
        for result in snapshot.tools.values()
    )


def test_build_environment_diagnostics_combines_snapshot_and_checks():
    snapshot = EnvironmentSnapshot(
        generated_at="2026-05-25T00:00:00+00:00",
        collection_status=EnvironmentEvidenceStatus.UNAVAILABLE,
        tools={
            "rocminfo": ToolProbeResult(
                tool="rocminfo",
                command=["rocminfo"],
                status=EnvironmentEvidenceStatus.UNAVAILABLE,
            )
        },
    )
    diagnostics = build_environment_diagnostics(
        snapshot_collector=lambda: snapshot,
        smoke_checker=lambda: [
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.SKIPPED,
                message="skipped",
            )
        ],
        now=lambda: datetime(2026, 5, 25, tzinfo=UTC),
    )

    assert isinstance(diagnostics, EnvironmentDiagnostics)
    assert diagnostics.status == EnvironmentEvidenceStatus.UNAVAILABLE
    assert [check.name for check in diagnostics.checks] == [
        "tool:rocminfo",
        "pytorch_rocm_runtime",
    ]
