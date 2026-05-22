from __future__ import annotations

import pytest

from sol_execbench.core.diagnostics import (
    DiagnosticStage,
    ProfilerBackend,
    StageDiagnostic,
    SolExecBenchError,
    cdna3_validation_readiness,
    classify_gfx,
    local_gfx_target,
    rocm_tool_diagnostics,
    select_profiler_backend,
)


def test_gfx_classification_covers_supported_architectures():
    assert classify_gfx("gfx1200") == "rdna4"
    assert classify_gfx("gfx942") == "cdna3"
    assert classify_gfx(None) == "unknown"


def test_cdna3_full_profile_falls_back_without_cdna_tools():
    readiness = select_profiler_backend(
        "full",
        "gfx942",
        rocprofiler_compute=False,
        omniperf=False,
        rocprofv3=True,
    )
    assert readiness.backend == ProfilerBackend.ROCPROFV3
    assert readiness.fallback_applied is True
    assert readiness.effective_level == "basic"
    assert "CDNA 3" in readiness.reason


def test_cdna3_full_profile_prefers_rocprofiler_compute():
    readiness = select_profiler_backend(
        "full",
        "gfx940",
        rocprofiler_compute=True,
        omniperf=True,
        rocprofv3=True,
    )
    assert readiness.backend == ProfilerBackend.ROCPROFILER_COMPUTE
    assert readiness.fallback_applied is False


def test_invalid_profile_level_has_actionable_stage_error():
    with pytest.raises(SolExecBenchError) as exc_info:
        select_profiler_backend("deep", "gfx1200")

    assert exc_info.value.stage == DiagnosticStage.ENVIRONMENT
    assert "Fix:" in str(exc_info.value)


def test_rocm_tool_diagnostics_reports_missing_tools():
    diagnostics = rocm_tool_diagnostics(which=lambda _: None)
    assert {diag.message for diag in diagnostics} == {
        "hipcc not found",
        "rocminfo not found",
        "rocm-smi not found",
        "rocprofv3 not found",
    }
    assert all(diag.hint for diag in diagnostics)


def test_local_gfx_target_uses_first_available_rocm_output():
    def fake_check_output(cmd, **_kwargs):
        if cmd[0] == "rocm_agent_enumerator":
            return "gfx000\n"
        return "Name: gfx1200\n"

    assert local_gfx_target(check_output=fake_check_output) == "gfx1200"


def test_cdna3_validation_readiness_for_gfx94_target():
    readiness = cdna3_validation_readiness("gfx942")
    payload = readiness.to_dict()

    assert readiness.target_family == "cdna3"
    assert readiness.ready is True
    assert readiness.claim == "cdna3_readiness_implemented"
    assert "uv run --no-sync pytest tests/" in readiness.commands
    assert any("gfx94" in item for item in readiness.acceptance_criteria)
    assert payload["commands"][0] == "uv run --no-sync pytest tests/"


def test_cdna3_validation_readiness_blocks_rdna4_target():
    readiness = cdna3_validation_readiness("gfx1200")

    assert readiness.target_family == "rdna4"
    assert readiness.ready is False
    assert readiness.claim == "cdna3_hardware_validation_deferred"
    assert readiness.blockers == (
        "Detected RDNA 4 target; CDNA 3 validation requires gfx94* hardware.",
    )


def test_cdna3_validation_readiness_reports_missing_tools():
    diagnostics = [
        StageDiagnostic(
            stage=DiagnosticStage.ENVIRONMENT,
            status="missing",
            message="rocminfo not found",
            hint="Install ROCm runtime tools.",
        )
    ]
    readiness = cdna3_validation_readiness("gfx942", tool_diagnostics=diagnostics)

    assert readiness.ready is False
    assert readiness.claim == "cdna3_hardware_validation_deferred"
    assert readiness.blockers == ("Missing ROCm validation tools: rocminfo",)
