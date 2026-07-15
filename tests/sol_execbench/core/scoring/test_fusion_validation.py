# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast

import pytest

from sol_execbench.core.scoring.fusion_validation import (
    FusionCapacityStatus,
    FusionPerformanceStatus,
    FusionSignature,
    FusionValidationArtifact,
    FusionValidationCase,
    KernelResourceEvidence,
    PerformanceEvidence,
    fusion_validation_from_dict,
    performance_from_rounds,
)


def _artifact() -> FusionValidationArtifact:
    kernel = KernelResourceEvidence(
        kernel_name="fused_reduction",
        binary_sha256="a" * 64,
        source_sha256="b" * 64,
        compile_command=("hipcc", "--offload-arch=gfx1200", "probe.hip"),
        architecture="gfx1200",
        vgpr_count=32,
        sgpr_count=24,
        vgpr_spill_count=0,
        sgpr_spill_count=0,
        private_segment_bytes=0,
        static_lds_bytes=4096,
        dynamic_lds_bytes=0,
        lds_limit_bytes=65536,
        active_blocks_per_multiprocessor=1,
        launch_passed=True,
        correctness_passed=True,
    )
    signature = FusionSignature(
        "reduction_epilogue.v1",
        1,
        ("sum", "mul"),
        "float32",
        ((2, 4096),),
        ((2,),),
        {"required_lds_bytes": 4096},
    )
    return FusionValidationArtifact(
        architecture="gfx1200",
        gpu_uuid="GPU-1",
        rocm_version="7.1",
        hipcc_version="7.1",
        clocks_locked=True,
        suite_manifest_sha256="c" * 64,
        benchmark_root_sha256="d" * 64,
        generated_at="2026-07-11T00:00:00Z",
        cases=(
            FusionValidationCase(
                "rmsnorm-b2",
                "workload-rmsnorm-b2",
                "rmsnorm_fp32_h4096",
                signature,
                kernel,
                (kernel,),
                FusionCapacityStatus.PASSED,
                PerformanceEvidence(
                    FusionPerformanceStatus.NOT_MEASURED, (), (), None, None
                ),
            ),
        ),
    )


def test_fusion_validation_round_trip_is_strict() -> None:
    artifact = _artifact()
    assert fusion_validation_from_dict(artifact.to_dict()) == artifact
    invalid = copy.deepcopy(artifact.to_dict())
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        fusion_validation_from_dict(invalid)


def test_fusion_statuses_are_typed_and_invalid_in_memory_values_are_rejected() -> None:
    artifact = _artifact()
    case = artifact.cases[0]

    assert case.capacity_status is FusionCapacityStatus.PASSED
    assert case.performance.status is FusionPerformanceStatus.NOT_MEASURED
    with pytest.raises(ValueError, match="not_a_status"):
        PerformanceEvidence(
            cast(FusionPerformanceStatus, "not_a_status"), (), (), None, None
        )
    with pytest.raises(ValueError, match="not_a_status"):
        replace(case, capacity_status=cast(FusionCapacityStatus, "not_a_status"))


def test_fusion_validation_accepts_exact_scalar_shapes() -> None:
    artifact = _artifact()
    scalar_signature = replace(
        artifact.cases[0].signature,
        input_shapes=((2, 4096), ()),
        output_shapes=((),),
    )
    scalar_case = replace(artifact.cases[0], signature=scalar_signature)
    scalar_artifact = replace(artifact, cases=(scalar_case,))

    assert fusion_validation_from_dict(scalar_artifact.to_dict()) == scalar_artifact


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vgpr_spill_count", 1),
        ("private_segment_bytes", 4),
        ("active_blocks_per_multiprocessor", 0),
        ("correctness_passed", False),
    ],
)
def test_capacity_status_cannot_contradict_resources(field: str, value: object) -> None:
    invalid = cast(dict[str, Any], _artifact().to_dict())
    invalid["cases"][0]["fused"][field] = value
    with pytest.raises(ValueError, match="capacity_status contradicts"):
        fusion_validation_from_dict(invalid)


def test_performance_policy_separates_ratio_and_stability() -> None:
    assert performance_from_rounds((1.0, 1.01, 1.0), (1.0, 1.0, 1.0)).status == "passed"
    assert (
        performance_from_rounds((1.03, 1.03, 1.03), (1.0, 1.0, 1.0)).status == "failed"
    )
    assert (
        performance_from_rounds((1.0, 1.0, 1.2), (1.0, 1.0, 1.0)).status == "unstable"
    )


def test_duplicate_exact_signature_is_rejected() -> None:
    invalid = cast(dict[str, Any], _artifact().to_dict())
    duplicate = copy.deepcopy(invalid["cases"][0])
    duplicate["evidence_id"] = "duplicate"
    invalid["cases"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate canonical signature"):
        fusion_validation_from_dict(invalid)
