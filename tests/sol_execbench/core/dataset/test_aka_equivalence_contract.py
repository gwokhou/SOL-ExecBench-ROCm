# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for all-output AKA equivalence validation."""

from __future__ import annotations

import pytest
import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_equivalence import normalize_outputs


def _contract() -> tuple[Definition, Workload]:
    definition = Definition.model_validate(
        {
            "name": "multi_output",
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
                "first": {
                    "shape": ["N"],
                    "dtype": "float32",
                    "description": "first output",
                },
                "second": {
                    "shape": ["N"],
                    "dtype": "float16",
                    "description": "second output",
                },
            },
            "reference": "def run(x):\n    return x, x.half()\n",
        },
    )
    workload = Workload.model_validate(
        {
            "axes": {"N": 4},
            "inputs": {"x": {"type": "random"}},
            "uuid": "all-output-check",
        },
    )
    return definition, workload


def test_normalize_outputs_checks_every_declared_output():
    definition, workload = _contract()

    outputs = normalize_outputs(
        {
            "second": torch.ones(4, dtype=torch.float16),
            "first": torch.ones(4, dtype=torch.float32),
        },
        definition,
        workload,
        source="test",
    )

    assert [output.dtype for output in outputs] == [
        torch.float32,
        torch.float16,
    ]


def test_normalize_outputs_rejects_dtype_mismatch_in_later_output():
    definition, workload = _contract()

    with pytest.raises(ValueError, match=r"test\.second dtype"):
        normalize_outputs(
            (
                torch.ones(4, dtype=torch.float32),
                torch.ones(4, dtype=torch.float32),
            ),
            definition,
            workload,
            source="test",
        )


def test_normalize_outputs_rejects_missing_mapping_key():
    definition, workload = _contract()

    with pytest.raises(ValueError, match="output keys"):
        normalize_outputs(
            {"first": torch.ones(4)},
            definition,
            workload,
            source="test",
        )
