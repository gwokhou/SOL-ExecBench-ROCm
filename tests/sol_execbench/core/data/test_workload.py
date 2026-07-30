# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Tests for sol_execbench.core.data.workload.Workload."""

import pytest
from pydantic import ValidationError

from sol_execbench.core.data.workload import (
    GeneratedInput,
    NumericCheck,
    RandomInput,
    ScalarInput,
    Workload,
)
from sol_execbench.core.integrity.schema_versions import WORKLOAD_SCHEMA_VERSION


def _wkl(**inputs):
    return Workload(
        uuid="test-uuid",
        axes={},
        inputs=inputs,
        checks=[NumericCheck(output="output")],
    )


class TestGetScalarInputs:
    def test_no_scalars_returns_empty(self):
        wkl = _wkl(a={"type": "random"})
        assert wkl.get_scalar_inputs() == {}

    def test_single_int_scalar(self):
        wkl = _wkl(x={"type": "scalar", "value": 42})
        assert wkl.get_scalar_inputs() == {"x": 42}

    def test_single_float_scalar(self):
        wkl = _wkl(scale={"type": "scalar", "value": 0.5})
        assert wkl.get_scalar_inputs() == {"scale": 0.5}

    def test_multiple_scalars(self):
        wkl = _wkl(
            a={"type": "scalar", "value": 1},
            b={"type": "scalar", "value": 2},
        )
        assert wkl.get_scalar_inputs() == {"a": 1, "b": 2}

    def test_mixed_inputs_returns_only_scalars(self):
        wkl = _wkl(
            a={"type": "random"},
            b={"type": "scalar", "value": 7},
            c={
                "type": "safetensors",
                "path": "f.safetensors",
                "tensor_key": "k",
            },
        )
        assert wkl.get_scalar_inputs() == {"b": 7}

    def test_all_custom_returns_empty(self):
        wkl = _wkl(a={"type": "custom"}, b={"type": "custom"})
        assert wkl.get_scalar_inputs() == {}

    def test_input_type_discriminator_selects_the_declared_model(self):
        wkl = _wkl(value={"type": "scalar", "value": 42})

        assert isinstance(wkl.inputs["value"], ScalarInput)

    def test_input_type_discriminator_rejects_unknown_types(self):
        with pytest.raises(ValidationError, match="union_tag_invalid"):
            _wkl(value={"type": "mystery", "value": 42})

    def test_generated_input_uses_a_typed_generator(self):
        wkl = _wkl(
            value={
                "type": "generated",
                "generator": {"type": "integer", "low": 0, "high": "C"},
            },
        )

        assert isinstance(wkl.inputs["value"], GeneratedInput)


class TestOutputChecks:
    def test_rejects_empty_checks(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            Workload(uuid="test-uuid", axes={}, inputs={}, checks=[])

    def test_rejects_removed_tolerance_wire_field(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            Workload.model_validate(
                {
                    "schema_version": WORKLOAD_SCHEMA_VERSION,
                    "uuid": "test-uuid",
                    "axes": {},
                    "inputs": {"x": {"type": "random"}},
                    "checks": [{"type": "numeric", "output": "output"}],
                    "tolerance": {"max_atol": 0.1},
                },
            )

    def test_allows_partial_custom_inputs(self):
        workload = Workload(
            uuid="test-uuid",
            axes={},
            inputs={"x": RandomInput(), "offsets": {"type": "custom"}},
            checks=[NumericCheck(output="output", required_matched_ratio=0.97)],
        )

        check = workload.checks[0]
        assert isinstance(check, NumericCheck)
        assert check.required_matched_ratio == 0.97
