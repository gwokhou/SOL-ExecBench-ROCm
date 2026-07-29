from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceCase,
    DiagnosticAcceptanceManifest,
    evaluate_diagnostic_acceptance,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    WorkloadKind,
)
from sol_execbench.core.integrity import stable_json_checksum

_ATTRIBUTIONS = {
    WorkloadKind.ELEMENTWISE: "launch_bound",
    WorkloadKind.TRANSPOSE: "memory_access_efficiency",
    WorkloadKind.REDUCTION: "lds_barrier_pressure",
    WorkloadKind.MATMUL: "matrix_path_missing",
}


def _identity() -> CalibrationIdentity:
    return CalibrationIdentity(
        gpu_architecture="gfx1200",
        gpu_id="gpu-0",
        gpu_bdf="0000:03:00.0",
        rocm_version="7.2",
        compiler_version="hipcc-7.2",
        clock_mode="locked",
        power_profile="stable_peak",
    )


def _case(kind: WorkloadKind, index: int) -> DiagnosticAcceptanceCase:
    identity = f"{kind}:{index}"
    return DiagnosticAcceptanceCase(
        workload_uuid=identity,
        workload_kind=kind,
        candidate_sha256=stable_json_checksum([identity, "candidate"]),
        evidence_manifest_sha256=stable_json_checksum([identity, "evidence"]),
        performance_diagnostic_sha256=stable_json_checksum(
            [identity, "diagnostic"]
        ),
        predicted_ms=1.05,
        measured_ms=1.0,
        primary_attribution_code=_ATTRIBUTIONS[kind],
    )


def _manifest() -> DiagnosticAcceptanceManifest:
    return DiagnosticAcceptanceManifest(
        calibration_profile_sha256="a" * 64,
        calibration_identity=_identity(),
        frozen_configuration_sha256="b" * 64,
        tuning_evidence_sha256=["c" * 64],
        cases=[
            _case(kind, index) for kind in _ATTRIBUTIONS for index in range(10)
        ],
    )


def test_frozen_acceptance_requires_ten_cases_per_family() -> None:
    result = evaluate_diagnostic_acceptance(_manifest())

    assert result.accepted is True
    assert result.case_count == 40
    assert set(result.family_case_counts.values()) == {10}
    assert result.median_absolute_percentage_error <= 15.0
    assert result.p90_absolute_percentage_error <= 30.0


def test_tuning_samples_are_rejected_by_contract() -> None:
    with pytest.raises(ValidationError):
        _case(WorkloadKind.ELEMENTWISE, 0).model_copy(
            update={"tuning_sample": True}
        ).model_validate(
            {
                **_case(
                    WorkloadKind.ELEMENTWISE,
                    0,
                ).model_dump(mode="json"),
                "tuning_sample": True,
            }
        )


def test_wrong_attribution_fails_acceptance() -> None:
    manifest = _manifest()
    cases = list(manifest.cases)
    cases[0] = cases[0].model_copy(update={"primary_attribution_code": "wrong"})

    result = evaluate_diagnostic_acceptance(
        manifest.model_copy(update={"cases": cases})
    )

    assert result.accepted is False
    assert "primary_attribution_mismatch" in result.reason_codes
