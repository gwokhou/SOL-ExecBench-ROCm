# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest
from dataclasses import replace

from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.enums import OpFamily
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.hardware_profile_requirements import (
    hardware_profile_requirements_from_dict,
    requirements_from_estimates,
)


def _estimate() -> OperatorWorkEstimate:
    return OperatorWorkEstimate(
        node_id="node",
        op_family=OpFamily.GEMM,
        op_name="matmul",
        formula_kind="gemm",
        formula="2mnk",
        formula_inputs={},
        flops=1.0,
        read_bytes=1.0,
        write_bytes=1.0,
        intermediate_bytes=0.0,
        movement_bytes=0.0,
        total_bytes=2.0,
        confidence=EstimateConfidence.SUPPORTED,
        rationale="test",
        compute_operation="matrix",
        input_dtype="bf16",
        output_dtype="bf16",
        compute_path="wmma",
        memory_access="stream_copy",
        memory_path="gfx12",
    )


def test_requirements_bind_exact_compute_and_memory_profiles() -> None:
    requirements = requirements_from_estimates(
        architecture="gfx1200", estimates=[_estimate()], scope="test-suite"
    )

    assert requirements.required_profile_keys == (
        "compute.matrix.bf16.bf16.wmma",
        "memory.stream_copy.bf16.bf16.gfx12",
    )
    assert (
        hardware_profile_requirements_from_dict(requirements.to_dict()) == requirements
    )


def test_requirements_reject_missing_exact_profile_fields() -> None:
    estimate = _estimate()
    estimate = replace(estimate, compute_path=None)

    with pytest.raises(ValueError, match="exact compute profile"):
        requirements_from_estimates(
            architecture="gfx1200", estimates=[estimate], scope="test-suite"
        )
