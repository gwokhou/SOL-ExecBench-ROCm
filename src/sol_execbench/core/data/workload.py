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

"""Specification for workload inputs and correctness checks."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from sol_execbench.core.data.base_model import (
    BaseModelWithDocstrings,
    CurrentSchemaModel,
    NonEmptyString,
    NonNegativeInt,
)
from sol_execbench.core.integrity.schema_versions import WORKLOAD_SCHEMA_VERSION


class RandomInput(BaseModelWithDocstrings):
    """Random input generation descriptor.

    Represents a specification for generating random tensor input data
    during workload execution and benchmarking.
    """

    type: Literal["random"] = "random"
    """The input type identifier for random data generation."""


class NormalGenerator(BaseModelWithDocstrings):
    """Normally distributed tensor values."""

    type: Literal["normal"] = "normal"
    mean: float = 0.0
    std: float = Field(default=1.0, gt=0.0)

    @field_validator("mean", "std")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("generator parameters must be finite")
        return value


class UniformGenerator(BaseModelWithDocstrings):
    """Uniformly distributed tensor values in ``[low, high)``."""

    type: Literal["uniform"] = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _valid_range(self) -> UniformGenerator:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("uniform bounds must be finite")
        if self.low >= self.high:
            raise ValueError("uniform low must be smaller than high")
        return self


class IntegerGenerator(BaseModelWithDocstrings):
    """Integer values in the half-open interval ``[low, high)``."""

    type: Literal["integer"] = "integer"
    low: int | NonEmptyString
    high: int | NonEmptyString


class BernoulliGenerator(BaseModelWithDocstrings):
    """Bernoulli values cast to the declared tensor dtype."""

    type: Literal["bernoulli"] = "bernoulli"
    probability: float = Field(default=0.5, ge=0.0, le=1.0)


class ConstantGenerator(BaseModelWithDocstrings):
    """A tensor filled with one scalar value."""

    type: Literal["constant"] = "constant"
    value: int | float | bool

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float | bool) -> int | float | bool:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("constant value must be finite")
        return value


class SimplexGenerator(BaseModelWithDocstrings):
    """A probability simplex produced by a float32 softmax."""

    type: Literal["simplex"] = "simplex"
    axis: int = -1
    temperature: float = Field(default=1.0, gt=0.0)

    @field_validator("temperature")
    @classmethod
    def _finite_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("simplex temperature must be finite")
        return value


GeneratorSpec = Annotated[
    NormalGenerator
    | UniformGenerator
    | IntegerGenerator
    | BernoulliGenerator
    | ConstantGenerator
    | SimplexGenerator,
    Field(discriminator="type"),
]


class GeneratedInput(BaseModelWithDocstrings):
    """Explicit, auditable tensor generation descriptor."""

    type: Literal["generated"] = "generated"
    generator: GeneratorSpec


class ScalarInput(BaseModelWithDocstrings):
    """Scalar literal input specification.

    Represents a scalar value (integer, float, or boolean) that will be
    used as a direct input parameter to the computational workload.
    """

    type: Literal["scalar"] = "scalar"
    """The input type identifier for scalar values."""
    value: int | float | bool
    """The scalar value to be used as input. Must be int, float, or bool."""


class SafetensorsInput(BaseModelWithDocstrings):
    """Input specification for data loaded from safetensors files.

    Represents tensor data that will be loaded from a safetensors file
    using a specific tensor key within that file.
    """

    type: Literal["safetensors"] = "safetensors"
    """The input type identifier for safetensors data."""
    path: NonEmptyString
    """Path to the safetensors file containing the tensor data. The path is relative to the root
    path of the TraceSet."""
    tensor_key: NonEmptyString
    """Key identifier for the specific tensor within the safetensors file."""


class CustomInput(BaseModelWithDocstrings):
    """Custom input specification.

    Only fields marked as custom are returned by the trusted generator. Custom
    fields may be mixed with ordinary workload inputs.
    """

    type: Literal["custom"] = "custom"
    """The input type identifier for inputs generated by the reference code."""


InputSpec = Annotated[
    RandomInput | GeneratedInput | SafetensorsInput | ScalarInput | CustomInput,
    Field(discriminator="type"),
]
"""Discriminated union representing all possible input specification types."""


class ToleranceSpec(BaseModelWithDocstrings):
    """Numerical tolerance fields shared by numeric checks.

    This model is retained as an internal calculation primitive; workload wire
    contracts use :class:`NumericCheck`.
    """

    max_atol: float = Field(default=1e-2)
    """The maximum absolute error allowed for the problem."""
    max_rtol: float = Field(default=1e-2)
    """The maximum relative error allowed for the problem."""
    required_matched_ratio: float = Field(default=0.99)
    """The ratio of elements that must pass the correctness bounds to be considered correct."""
    max_error_cap: float | None = Field(default=None)
    """Hard ceiling on maximum absolute error. If set, correctness fails when any
    element's absolute error exceeds this cap, regardless of matched ratio."""
    allow_negative_inf: bool = Field(default=False)
    """When True, matching -inf values in both output and reference are treated as
    correct and excluded from error computation. Positions where only one tensor
    has -inf still fail. +inf and NaN are unaffected by this flag."""


class NumericCheckMode(StrEnum):
    """Supported numerical comparison metrics."""

    ELEMENTWISE = "elementwise"
    NORMALIZED_MAX = "normalized_max"


class NumericCheck(ToleranceSpec):
    """Numerically compare one named output."""

    type: Literal["numeric"] = "numeric"
    output: NonEmptyString
    mode: NumericCheckMode = NumericCheckMode.ELEMENTWISE
    denominator_epsilon: float = Field(default=1e-12, gt=0.0)


class ExactCheck(BaseModelWithDocstrings):
    """Require exact elementwise equality for one output."""

    type: Literal["exact"] = "exact"
    output: NonEmptyString


class CodeDistanceMode(StrEnum):
    """Interpretation used when comparing quantized codes."""

    VALUE = "value"
    RAW_BITS = "raw_bits"


class CodeDistanceCheck(BaseModelWithDocstrings):
    """Bound integer or raw-storage code distance for one output."""

    type: Literal["code_distance"] = "code_distance"
    output: NonEmptyString
    mode: CodeDistanceMode = CodeDistanceMode.VALUE
    max_distance: NonNegativeInt
    required_matched_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class TopKRoutingCheck(BaseModelWithDocstrings):
    """Joint, tie-aware comparison of MoE routing IDs and weights."""

    type: Literal["topk_routing"] = "topk_routing"
    ids_output: NonEmptyString
    weights_output: NonEmptyString
    gating_input: NonEmptyString
    bias_input: NonEmptyString | None = None
    topk: int = Field(gt=0)
    tie_atol: float = Field(default=1e-4, ge=0.0)
    weight_atol: float = Field(default=1e-2, ge=0.0)
    max_mismatch_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


OutputCheck = Annotated[
    NumericCheck | ExactCheck | CodeDistanceCheck | TopKRoutingCheck,
    Field(discriminator="type"),
]


def conservative_numeric_tolerance(
    checks: list[OutputCheck],
) -> ToleranceSpec:
    """Collapse typed numeric checks into one conservative replay policy."""
    numeric = [check for check in checks if isinstance(check, NumericCheck)]
    caps = [
        check.max_error_cap
        for check in numeric
        if check.max_error_cap is not None
    ]
    return ToleranceSpec(
        max_atol=min((check.max_atol for check in numeric), default=0.0),
        max_rtol=min((check.max_rtol for check in numeric), default=0.0),
        required_matched_ratio=max(
            (check.required_matched_ratio for check in numeric),
            default=1.0,
        ),
        max_error_cap=min(caps) if caps else None,
        allow_negative_inf=all(check.allow_negative_inf for check in numeric),
    )


class Workload(CurrentSchemaModel):
    """Concrete workload configuration for benchmarking.

    Defines a specific instance of a computational workload with concrete
    values for all variable axes and specifications for all input data.
    This represents an executable configuration that can be benchmarked.
    """

    current_schema_version = WORKLOAD_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.workload.v1"] = (
        WORKLOAD_SCHEMA_VERSION
    )
    axes: dict[str, NonNegativeInt]
    """Dictionary mapping axis names to their concrete integer values. All values must be
    positive."""
    inputs: dict[str, InputSpec]
    """Dictionary mapping input names to their data specifications."""
    uuid: NonEmptyString
    """Unique identifier for this specific workload configuration."""
    checks: list[OutputCheck]
    """Closed correctness checks covering every Definition output exactly once."""

    @field_validator("checks")
    @classmethod
    def _non_empty_checks(cls, checks: list[OutputCheck]) -> list[OutputCheck]:
        if not checks:
            raise ValueError("workload checks must not be empty")
        return checks

    def get_scalar_inputs(self) -> dict[str, int | float | bool]:
        """Return scalar input values keyed by input name."""
        return {
            name: input_spec.value
            for name, input_spec in self.inputs.items()
            if isinstance(input_spec, ScalarInput)
        }
