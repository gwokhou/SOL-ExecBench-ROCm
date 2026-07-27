# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-workload AKA tolerance calibration."""

from __future__ import annotations

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import ToleranceSpec, Workload
from sol_execbench.core.dataset.aka_tolerance import (
    calibrate_tolerance,
    dtype_default_tolerance,
    workload_contract_sha256,
)


def _definition() -> Definition:
    return Definition.model_validate(
        {
            "name": "two_outputs",
            "op_type": "test",
            "axes": {"N": {"type": "var", "description": "elements"}},
            "inputs": {
                "x": {
                    "shape": ["N"],
                    "dtype": "float32",
                    "description": "input",
                },
            },
            "outputs": {
                "fp32": {
                    "shape": ["N"],
                    "dtype": "float32",
                    "description": "fp32 output",
                },
                "bf16": {
                    "shape": ["N"],
                    "dtype": "bfloat16",
                    "description": "bf16 output",
                },
            },
            "reference": "def run(x):\n    return x, x.to(torch.bfloat16)\n",
        },
    )


def _workload(tolerance: ToleranceSpec) -> Workload:
    return Workload.model_validate(
        {
            "axes": {"N": 8},
            "inputs": {"x": {"type": "random"}},
            "uuid": "calibration-test",
            "tolerance": tolerance.model_dump(mode="json"),
        },
    )


def test_runtime_observation_can_only_widen_dtype_floors():
    calibrated = calibrate_tolerance(
        ["float32", "bfloat16"],
        observed_max_atol=0.02,
        observed_max_rtol=0.03,
    )

    assert calibrated.max_atol == 0.025
    assert calibrated.max_rtol == 0.0375
    assert calibrated.max_atol >= dtype_default_tolerance("bfloat16").max_atol


def test_workload_contract_hash_excludes_only_tolerance():
    definition = _definition()
    first = _workload(ToleranceSpec(max_atol=1e-5, max_rtol=1e-5))
    second = _workload(ToleranceSpec(max_atol=1e-1, max_rtol=1e-1))

    assert workload_contract_sha256(
        definition,
        first,
    ) == workload_contract_sha256(
        definition,
        second,
    )
