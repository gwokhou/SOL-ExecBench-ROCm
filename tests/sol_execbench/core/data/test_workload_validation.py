# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Definition–Workload contract validation."""

from __future__ import annotations

import pytest
from sol_execbench_type_helpers import make_definition, make_workload

from sol_execbench.core.data.workload_validation import (
    WorkloadContractError,
    validate_workload_contract,
)


def _definition():
    return make_definition(
        name="contract",
        op_type="test",
        axes={"N": {"type": "var"}},
        inputs={
            "x": {"shape": ["N"], "dtype": "float32"},
            "index": {"shape": ["N"], "dtype": "int32"},
        },
        outputs={
            "value": {"shape": ["N"], "dtype": "float32"},
            "ids": {"shape": ["N"], "dtype": "int32"},
        },
        custom_inputs_entrypoint="gen",
        reference=(
            "def run(x, index): return x, index\n"
            "def gen(values, device): return {'index': None}\n"
        ),
    )


def test_rejects_missing_and_extra_input_names() -> None:
    workload = make_workload(
        uuid="invalid",
        axes={"N": 4},
        inputs={"x": {"type": "random"}, "extra": {"type": "random"}},
        checks=[
            {"type": "numeric", "output": "value"},
            {"type": "exact", "output": "ids"},
        ],
    )

    with pytest.raises(WorkloadContractError, match=r"missing=.*index.*extra"):
        validate_workload_contract(_definition(), workload)


def test_rejects_duplicate_or_uncovered_output_checks() -> None:
    workload = make_workload(
        uuid="invalid",
        axes={"N": 4},
        inputs={"x": {"type": "random"}, "index": {"type": "custom"}},
        checks=[
            {"type": "numeric", "output": "value"},
            {"type": "numeric", "output": "value"},
        ],
    )

    with pytest.raises(WorkloadContractError, match=r"duplicates=.*value"):
        validate_workload_contract(_definition(), workload)


def test_accepts_partial_custom_and_exact_integer_output() -> None:
    workload = make_workload(
        uuid="valid",
        axes={"N": 4},
        inputs={"x": {"type": "random"}, "index": {"type": "custom"}},
        checks=[
            {"type": "numeric", "output": "value"},
            {"type": "exact", "output": "ids"},
        ],
    )

    validate_workload_contract(_definition(), workload)
