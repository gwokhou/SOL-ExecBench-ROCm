from __future__ import annotations

import pytest

from sol_execbench.core.diagnostics import (
    DiagnosticStage,
    ProfilerBackend,
    SolExecBenchError,
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
