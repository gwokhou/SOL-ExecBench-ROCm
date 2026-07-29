# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Cross-model validation for one Definition and its workloads."""

from __future__ import annotations

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.shapes import resolve_shape_expression
from sol_execbench.core.data.workload import (
    CodeDistanceCheck,
    CodeDistanceMode,
    CustomInput,
    ExactCheck,
    GeneratedInput,
    IntegerGenerator,
    NumericCheck,
    SimplexGenerator,
    TopKRoutingCheck,
    Workload,
)


class WorkloadContractError(ValueError):
    """Stable Definition–Workload contract failure."""

    code = "invalid_workload_contract"


def _check_outputs(check: object) -> tuple[str, ...]:
    if isinstance(check, (NumericCheck, ExactCheck, CodeDistanceCheck)):
        return (check.output,)
    if isinstance(check, TopKRoutingCheck):
        return (check.ids_output, check.weights_output)
    raise TypeError(f"unsupported output check {type(check).__name__}")


def _validate_input_inventory(
    definition: Definition,
    workload: Workload,
) -> None:
    expected = set(definition.inputs)
    observed = set(workload.inputs)
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise WorkloadContractError(
            f"workload {workload.uuid} input mismatch: missing={missing}, extra={extra}",
        )
    has_custom = any(
        isinstance(item, CustomInput) for item in workload.inputs.values()
    )
    if has_custom and not definition.custom_inputs_entrypoint:
        raise WorkloadContractError(
            f"workload {workload.uuid} uses custom inputs without a trusted entrypoint",
        )


def _validate_generated_input(
    definition: Definition,
    workload: Workload,
    name: str,
    input_spec: GeneratedInput,
) -> None:
    tensor_spec = definition.inputs[name]
    dtype = dtype_str_to_torch_dtype(tensor_spec.dtype)
    shape = definition.get_input_shapes(workload.axes)[name]
    generator = input_spec.generator
    if isinstance(generator, IntegerGenerator):
        if dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise WorkloadContractError(
                f"integer generator for {name} requires integer dtype"
            )
        axes = definition.get_resolved_axes_values(workload.axes)
        low = (
            generator.low
            if isinstance(generator.low, int)
            else resolve_shape_expression(generator.low, axes)
        )
        high = (
            generator.high
            if isinstance(generator.high, int)
            else resolve_shape_expression(generator.high, axes)
        )
        limits = torch.iinfo(dtype)
        if low >= high or low < limits.min or high - 1 > limits.max:
            raise WorkloadContractError(
                f"integer generator bounds for {name} exceed {dtype}"
            )
    if isinstance(generator, SimplexGenerator):
        rank = 0 if shape is None else len(shape)
        axis = generator.axis if generator.axis >= 0 else rank + generator.axis
        if not dtype.is_floating_point or rank == 0 or axis not in range(rank):
            raise WorkloadContractError(f"invalid simplex generator for {name}")


def _validate_output_inventory(
    definition: Definition,
    workload: Workload,
) -> None:
    covered: list[str] = []
    for check in workload.checks:
        covered.extend(_check_outputs(check))
    expected = set(definition.outputs)
    observed = set(covered)
    duplicates = sorted({name for name in covered if covered.count(name) > 1})
    if expected != observed or duplicates:
        raise WorkloadContractError(
            f"workload {workload.uuid} output checks mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}, "
            f"duplicates={duplicates}",
        )


def _validate_check_dtypes(definition: Definition, workload: Workload) -> None:
    for check in workload.checks:
        if isinstance(check, ExactCheck):
            dtype = dtype_str_to_torch_dtype(
                definition.outputs[check.output].dtype
            )
            if dtype.is_floating_point:
                raise WorkloadContractError(
                    "exact checks require integer or boolean output"
                )
        elif isinstance(check, CodeDistanceCheck):
            dtype = dtype_str_to_torch_dtype(
                definition.outputs[check.output].dtype
            )
            if check.mode is CodeDistanceMode.VALUE and dtype.is_floating_point:
                raise WorkloadContractError(
                    "value code distance requires integer output"
                )
            if check.mode is CodeDistanceMode.RAW_BITS and dtype.itemsize != 1:
                raise WorkloadContractError(
                    "raw-bit code distance requires one-byte output"
                )
        elif isinstance(check, TopKRoutingCheck):
            _validate_topk_check(definition, workload, check)


def _validate_topk_check(
    definition: Definition,
    workload: Workload,
    check: TopKRoutingCheck,
) -> None:
    inputs = definition.inputs
    if check.gating_input not in inputs or (
        check.bias_input is not None and check.bias_input not in inputs
    ):
        raise WorkloadContractError(
            "top-k routing check references an unknown input"
        )
    output_shapes = definition.get_output_shapes(workload.axes)
    ids_shape = output_shapes[check.ids_output]
    weights_shape = output_shapes[check.weights_output]
    if (
        ids_shape != weights_shape
        or not ids_shape
        or ids_shape[-1] != check.topk
    ):
        raise WorkloadContractError(
            "top-k routing outputs must share a [..., topk] shape"
        )
    ids_dtype = dtype_str_to_torch_dtype(
        definition.outputs[check.ids_output].dtype
    )
    weights_dtype = dtype_str_to_torch_dtype(
        definition.outputs[check.weights_output].dtype,
    )
    if ids_dtype.is_floating_point or not weights_dtype.is_floating_point:
        raise WorkloadContractError(
            "top-k routing requires integer IDs and float weights"
        )


def validate_workload_contract(
    definition: Definition,
    workload: Workload,
) -> None:
    """Validate one workload against its Definition or fail closed."""
    definition.get_resolved_axes_values(workload.axes)
    _validate_input_inventory(definition, workload)
    for name, input_spec in workload.inputs.items():
        if isinstance(input_spec, GeneratedInput):
            _validate_generated_input(definition, workload, name, input_spec)
    _validate_output_inventory(definition, workload)
    _validate_check_dtypes(definition, workload)


def validate_problem_contract(
    definition: Definition,
    workloads: list[Workload],
) -> None:
    """Validate every workload in a problem and reject duplicate UUIDs."""
    uuids = [workload.uuid for workload in workloads]
    if len(set(uuids)) != len(uuids):
        raise WorkloadContractError(
            "workload UUIDs must be unique within a problem"
        )
    for workload in workloads:
        validate_workload_contract(definition, workload)


__all__ = [
    "WorkloadContractError",
    "validate_problem_contract",
    "validate_workload_contract",
]
