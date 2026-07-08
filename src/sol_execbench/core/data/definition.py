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

"""The definition of kernels in the FlashInfer Trace schema."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Iterable, Optional

from pydantic import Field, field_validator, model_validator

from .base_model import BaseModelWithDocstrings, NonEmptyString
from .definition_axes import (
    const_axes as _const_axes,
    expr_axes as _expr_axes,
    get_axes_values as _get_axes_values,
    get_axes_values_from_inputs as _get_axes_values_from_inputs,
    get_input_shapes as _get_input_shapes,
    get_output_shapes as _get_output_shapes,
    get_resolved_axes_values as _get_resolved_axes_values,
    get_shapes as _get_shapes,
    torch_input_dtypes as _torch_input_dtypes,
    torch_output_dtypes as _torch_output_dtypes,
    var_axes as _var_axes,
)
from .definition_models import (
    AxisConst,
    AxisExpr,
    AxisSpec,
    AxisVar,
    DType,
    TensorSpec,
)
from .definition_reference import (
    validate_reference_code,
    validate_reference_inputs_match,
    verify_custom_inputs_entrypoint,
)

if TYPE_CHECKING:
    import torch


class Definition(BaseModelWithDocstrings):
    """Complete definition of a computational workload."""

    name: NonEmptyString
    """A unique, human-readable name for the kernel definition."""
    op_type: Optional[NonEmptyString] = Field(default=None)
    """The general compute category."""
    axes: dict[NonEmptyString, AxisSpec]
    """Dictionary of symbolic dimensions used in tensor shapes."""
    custom_inputs_entrypoint: Optional[NonEmptyString] = Field(default=None)
    """The entrypoint function to generate the inputs."""
    inputs: dict[NonEmptyString, TensorSpec]
    """Named input tensors required by this kernel."""
    outputs: dict[NonEmptyString, TensorSpec]
    """Named output tensors produced by this kernel."""
    reference: NonEmptyString
    """Reference implementation code with a top-level run function."""
    description: Optional[str] = Field(default=None)
    """Optional human-readable description of the kernel's purpose."""
    hf_id: Optional[NonEmptyString] = Field(default=None)
    """Optional HuggingFace model ID that the definition was sourced from."""

    @field_validator("hf_id", mode="before")
    @classmethod
    def _normalize_empty_hf_id(cls, value: Any) -> Any:
        """Treat empty dataset IDs as absent optional metadata."""
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def _validate_reference_code(self) -> Definition:
        return validate_reference_code(self)

    @model_validator(mode="after")
    def _validate_reference_inputs_match(self) -> Definition:
        return validate_reference_inputs_match(self)

    @model_validator(mode="after")
    def _verify_custom_inputs_entrypoint(self) -> Definition:
        return verify_custom_inputs_entrypoint(self)

    @model_validator(mode="after")
    def _validate_input_names_are_not_axes(self) -> Definition:
        """Validate that input names are not axes."""
        for name in self.inputs.keys():
            if name in self.axes:
                raise ValueError(f"Input name '{name}' is not allowed to be an axis.")
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> Definition:
        """Validate that input and output names are unique and do not overlap."""
        if set(self.inputs.keys()) & set(self.outputs.keys()):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> Definition:
        """Validate that tensor shapes reference defined axes."""
        all_tensors = {**self.inputs, **self.outputs}

        for tensor_name, tensor_spec in all_tensors.items():
            if tensor_spec.shape is None:
                continue
            for axis_name in tensor_spec.shape:
                if axis_name.isdigit():
                    continue
                if axis_name not in self.axes:
                    tensor_type = "input" if tensor_name in self.inputs else "output"
                    raise ValueError(
                        f'{tensor_type.capitalize()} "{tensor_name}" references undefined '
                        f'axis "{axis_name}".'
                    )
        return self

    @cached_property
    def const_axes(self) -> dict[str, int]:
        """Get all constant axes and their values."""
        return _const_axes(self)

    @cached_property
    def var_axes(self) -> list[str]:
        """Get all variable axis names."""
        return _var_axes(self)

    @cached_property
    def expr_axes(self) -> dict[str, AxisExpr]:
        """Get all expression axis specs keyed by name."""
        return _expr_axes(self)

    def get_axes_values(
        self, input_shapes: Iterable[Optional[tuple[int, ...]]]
    ) -> dict[str, int]:
        """Get concrete variable axis values from input shapes."""
        return _get_axes_values(self, input_shapes)

    def get_axes_values_from_inputs(self, inputs: Iterable[Any]) -> dict[str, int]:
        """Get concrete variable axis values directly from input values."""
        return _get_axes_values_from_inputs(self, inputs)

    def get_resolved_axes_values(self, var_axes_values: dict[str, int]) -> dict[str, int]:
        """Get concrete axis values from variable axis values."""
        return _get_resolved_axes_values(self, var_axes_values)

    def _get_shapes(
        self,
        tensors: Iterable[TensorSpec],
        var_axes_values: Optional[dict[str, int]] = None,
    ) -> list[Optional[tuple[int, ...]]]:
        """Get concrete tensor shapes given variable axis values."""
        return _get_shapes(self, tensors, var_axes_values)

    def get_input_shapes(
        self, var_axes_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        """Get concrete input shapes given variable axis values."""
        return _get_input_shapes(self, var_axes_values)

    def get_output_shapes(
        self, var_values: Optional[dict[str, int]] = None
    ) -> dict[str, Optional[tuple[int, ...]]]:
        """Get concrete output shapes given variable axis values."""
        return _get_output_shapes(self, var_values)

    @cached_property
    def torch_input_dtypes(self) -> list[torch.dtype]:
        """Get the torch data types of the input tensors."""
        return _torch_input_dtypes(self)

    @cached_property
    def torch_output_dtypes(self) -> list[torch.dtype]:
        """Get the torch data types of the output tensors."""
        return _torch_output_dtypes(self)
