from __future__ import annotations

import pytest

from sol_execbench.core.platform.diagnostics import (
    DiagnosticStage,
    MI300X_FP8_READINESS,
    MI300X_REQUIRED_ARTIFACTS,
    MI300X_VALIDATION_RESULT_CATEGORIES,
    RDNA4_REQUIRED_ARTIFACTS,
    RDNA4_VALIDATION_RESULT_CATEGORIES,
    ROCM_LIBRARY_SPECS,
    ProfilerBackend,
    StageDiagnostic,
    SolExecBenchError,
    can_mark_rdna4_validation_upgraded,
    can_mark_mi300x_hardware_validated,
    cdna3_validation_readiness,
    classify_gfx,
    local_gfx_target,
    mi300x_validation_claim_blockers,
    rdna4_validation_claim_blockers,
    rocm_library_diagnostics,
    rocm_library_readiness,
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
    diagnostics = rocm_tool_diagnostics(
        which=lambda _: None,
        tool_resolver=lambda _tool, **_kwargs: None,
    )
    assert {diag.message for diag in diagnostics} == {
        "hipcc not found",
        "rocminfo not found",
        "rocm-smi not found",
        "rocprofv3 not found",
    }
    assert all(diag.hint for diag in diagnostics)


def test_rocm_library_specs_cover_supported_native_categories():
    assert set(ROCM_LIBRARY_SPECS) == {"hipblas", "miopen", "ck", "rocwmma"}
    assert ROCM_LIBRARY_SPECS["miopen"].headers == ("miopen/miopen.h",)
    assert "MIOpen" in ROCM_LIBRARY_SPECS["miopen"].libraries
    assert ROCM_LIBRARY_SPECS["ck"].libraries == ()
    assert ROCM_LIBRARY_SPECS["rocwmma"].headers == ("rocwmma/rocwmma.hpp",)


def test_rocm_library_readiness_reports_missing_headers_and_libraries(tmp_path):
    readiness = rocm_library_readiness(
        "miopen",
        roots=(tmp_path,),
        find_library=lambda _name: None,
        exists=lambda _path: False,
    )

    assert readiness.ready is False
    assert readiness.status == "missing"
    assert readiness.missing_headers == ("miopen/miopen.h",)
    assert "MIOpen" in readiness.missing_libraries
    diagnostic = readiness.to_diagnostic()
    assert diagnostic.stage == DiagnosticStage.ENVIRONMENT
    assert diagnostic.status == "missing"
    assert "headers: miopen/miopen.h" in diagnostic.message
    assert diagnostic.hint


def test_rocm_library_readiness_accepts_header_only_libraries(tmp_path):
    header = tmp_path / "include" / "rocwmma" / "rocwmma.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("// test header")

    readiness = rocm_library_readiness(
        "rocwmma",
        roots=(tmp_path,),
        find_library=lambda _name: None,
    )

    assert readiness.ready is True
    assert readiness.header_paths == (str(header),)
    assert readiness.library_paths == ()
    assert readiness.to_diagnostic().status == "available"


def test_rocm_library_diagnostics_reports_each_requested_library(tmp_path):
    diagnostics = rocm_library_diagnostics(
        ("miopen", "ck"),
        roots=(tmp_path,),
        find_library=lambda _name: None,
        exists=lambda _path: False,
    )

    assert [diagnostic.status for diagnostic in diagnostics] == ["missing", "missing"]
    assert "MIOpen" in diagnostics[0].message
    assert "Composable Kernel" in diagnostics[1].message


def test_local_gfx_target_uses_first_available_rocm_output():
    def fake_check_output(cmd, **_kwargs):
        if cmd[0] == "rocm_agent_enumerator":
            return "gfx000\n"
        return "Name: gfx1200\n"

    assert (
        local_gfx_target(
            check_output=fake_check_output,
            tool_resolver=lambda _tool: None,
        )
        == "gfx1200"
    )


def test_cdna3_validation_readiness_for_gfx94_target():
    readiness = cdna3_validation_readiness("gfx942")
    payload = readiness.to_dict()

    assert readiness.target_family == "cdna3"
    assert readiness.ready is True
    assert readiness.claim == "cdna3_readiness_implemented"
    assert "uv run pytest tests/" in readiness.commands
    assert any("gfx94" in item for item in readiness.acceptance_criteria)
    assert payload["commands"][0] == "uv run pytest tests/"


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


def test_mi300x_validation_claim_requires_complete_evidence():
    blockers = mi300x_validation_claim_blockers(
        {
            "gpu_name": "AMD Radeon RX 9070 XT",
            "gfx": "gfx1200",
            "rocm_version": "7.0.0",
            "clocks_locked": False,
            "full_suite_passed": False,
            "artifacts": [],
            "result_categories": [],
            "fp8_validation": "missing",
            "nvfp4_mxfp4_validation": "passed",
        }
    )

    assert can_mark_mi300x_hardware_validated({}) is False
    assert "gpu_name must identify AMD Instinct MI300X" in blockers
    assert "gfx must be a CDNA 3 gfx94* target" in blockers
    assert "clock-lock evidence must record clocks_locked=True" in blockers
    assert any("missing validation artifacts" in blocker for blocker in blockers)
    assert any(
        "missing validation result categories" in blocker for blocker in blockers
    )


def test_mi300x_validation_claim_allows_complete_evidence():
    evidence = {
        "gpu_name": "AMD Instinct MI300X",
        "gfx": "gfx942",
        "rocm_version": "7.0.0",
        "clocks_locked": True,
        "full_suite_passed": True,
        "artifacts": list(MI300X_REQUIRED_ARTIFACTS),
        "result_categories": list(MI300X_VALIDATION_RESULT_CATEGORIES),
        "fp8_validation": "passed",
        "nvfp4_mxfp4_validation": "deferred_no_amd_path",
    }

    assert mi300x_validation_claim_blockers(evidence) == ()
    assert can_mark_mi300x_hardware_validated(evidence) is True
    assert "per_problem_traces" in MI300X_REQUIRED_ARTIFACTS
    assert "rocm_timing_evidence" in MI300X_REQUIRED_ARTIFACTS
    assert "amd_native_score_report" in MI300X_REQUIRED_ARTIFACTS
    assert "missing_tools" in MI300X_VALIDATION_RESULT_CATEGORIES
    assert "timing_instability" in MI300X_VALIDATION_RESULT_CATEGORIES
    assert "MI300X, as a CDNA 3 GPU, can validate FP8" in MI300X_FP8_READINESS[0]
    assert "NVFP4/MXFP4 validation is deferred" in MI300X_FP8_READINESS[1]


def test_rdna4_validation_claim_blocks_current_bounded_v1_31_shape():
    evidence = {
        "gpu_name": "AMD Radeon RX 9060 XT",
        "gfx": "gfx1200",
        "rocm_version": "7.1.1",
        "clocks_locked": True,
        "profiler_backed_timing": False,
        "full_paper_problem_count": 121,
        "failed_workload_count": 146,
        "derived_sidecar_blocker_count": 7,
        "artifacts": list(RDNA4_REQUIRED_ARTIFACTS),
        "result_categories": list(RDNA4_VALIDATION_RESULT_CATEGORIES),
    }

    blockers = rdna4_validation_claim_blockers(evidence)

    assert can_mark_rdna4_validation_upgraded(evidence) is False
    assert (
        "profiler-backed rocprofv3 kernel activity timing must be recorded" in blockers
    )
    assert "full paper denominator must account for 235 problems" in blockers
    assert "failed workload count must be zero for claim upgrade" in blockers
    assert "derived sidecar blocker count must be zero for claim upgrade" in blockers
    assert not any("missing validation artifacts" in blocker for blocker in blockers)
    assert not any(
        "missing validation result categories" in blocker for blocker in blockers
    )


def test_rdna4_validation_claim_requires_rdna4_target_and_artifacts():
    blockers = rdna4_validation_claim_blockers(
        {
            "gpu_name": "",
            "gfx": "gfx942",
            "rocm_version": "",
            "clocks_locked": False,
            "profiler_backed_timing": False,
            "artifacts": ["environment_sidecar"],
            "result_categories": ["attempted_passed"],
        }
    )

    assert "gpu_name must be recorded" in blockers
    assert "gfx must be an RDNA 4 gfx12* target" in blockers
    assert "rocm_version must be recorded" in blockers
    assert "failed_workload_count must be recorded" in blockers
    assert "derived_sidecar_blocker_count must be recorded" in blockers
    assert any("missing validation artifacts" in blocker for blocker in blockers)
    assert any(
        "missing validation result categories" in blocker for blocker in blockers
    )


def test_rdna4_validation_claim_allows_complete_hypothetical_evidence():
    evidence = {
        "gpu_name": "AMD Radeon RX 9060 XT",
        "gfx": "gfx1200",
        "rocm_version": "7.1.1",
        "clocks_locked": True,
        "profiler_backed_timing": True,
        "full_paper_problem_count": 235,
        "failed_workload_count": 0,
        "derived_sidecar_blocker_count": 0,
        "artifacts": list(RDNA4_REQUIRED_ARTIFACTS),
        "result_categories": list(RDNA4_VALIDATION_RESULT_CATEGORIES),
    }

    assert rdna4_validation_claim_blockers(evidence) == ()
    assert can_mark_rdna4_validation_upgraded(evidence) is True
    assert "profiler_backed_timing" in RDNA4_REQUIRED_ARTIFACTS
    assert "derived_sidecar_memory_blockers" in RDNA4_VALIDATION_RESULT_CATEGORIES
