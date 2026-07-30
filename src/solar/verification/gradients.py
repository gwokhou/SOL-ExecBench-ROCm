# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""First-order gradient equivalence checks for IR replay."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from solar.errors import GradientVerificationError
from solar.types import GraphValue
from solar.verification.contracts import VerificationPolicy
from solar.verification.errors import VerificationError
from solar.verification.executor import IRGraphExecutor
from solar.verification.numerics import assert_close, clone


def capture_rng_state() -> tuple[GraphValue, list[GraphValue] | None]:
    """Capture CPU and available accelerator RNG state."""
    import torch

    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu, cuda


def restore_rng_state(
    state: tuple[GraphValue, list[GraphValue] | None],
) -> None:
    """Restore CPU and available accelerator RNG state."""
    import torch

    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _gradient_indices(
    inputs: Sequence[GraphValue],
    graph: Mapping[str, GraphValue],
    policy: VerificationPolicy,
) -> tuple[int, ...]:
    import torch

    source = graph.get("source_input_indices")
    available = (
        tuple(int(index) for index in source)
        if isinstance(source, list)
        else tuple(
            index
            for index, value in enumerate(inputs)
            if isinstance(value, torch.Tensor)
        )
    )
    selected = (
        tuple(int(index) for index in policy.gradient_input_indices)
        if policy.gradient_input_indices is not None
        else tuple(
            index
            for index in available
            if isinstance(inputs[index], torch.Tensor)
            and (
                inputs[index].is_floating_point() or inputs[index].is_complex()
            )
        )
    )
    if any(index not in available for index in selected):
        raise VerificationError(
            "gradient_input_indices select unavailable graph inputs"
        )
    return selected


def _gradient_inputs(
    inputs: Sequence[GraphValue],
    indices: Sequence[int],
) -> tuple[GraphValue, ...]:
    import torch

    result = list(clone(tuple(inputs)))
    selected = set(indices)
    for index, item in enumerate(result):
        if isinstance(item, torch.Tensor):
            value = item.detach()
            if index in selected:
                value.requires_grad_(True)
            result[index] = value
    return tuple(result)


def _differentiable_scalar(outputs: GraphValue) -> GraphValue | None:
    import torch

    leaves: list[GraphValue] = []

    def collect(value: GraphValue) -> None:
        if isinstance(value, torch.Tensor):
            if value.requires_grad and (
                value.is_floating_point() or value.is_complex()
            ):
                leaves.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)

    collect(outputs)
    if not leaves:
        return None
    scalars = [
        value.real.sum() + value.imag.sum()
        if value.is_complex()
        else value.sum()
        for value in leaves
    ]
    return sum(scalars)


def _run_gradients(
    callable_: Callable[..., GraphValue],
    inputs: tuple[GraphValue, ...],
    selected: Sequence[int],
) -> tuple[tuple[GraphValue | None, ...], BaseException | None]:
    import torch

    watched = tuple(inputs[index] for index in selected)
    try:
        with torch.enable_grad():
            outputs = callable_(*inputs)
            scalar = _differentiable_scalar(outputs)
            if scalar is None:
                return tuple(None for _ in watched), None
            gradients = torch.autograd.grad(
                scalar,
                watched,
                allow_unused=True,
            )
            return tuple(gradients), None
    except (RuntimeError, ValueError, TypeError) as exc:
        return (), exc


def verify_gradients(
    reference: Callable[..., GraphValue],
    executor: IRGraphExecutor,
    graph: Mapping[str, GraphValue],
    inputs: Sequence[GraphValue],
    policy: VerificationPolicy,
) -> dict[str, float]:
    """Compare first-order input gradients and public error semantics."""
    indices = _gradient_indices(inputs, graph, policy)
    if not indices:
        return {"gradient_inputs_verified": 0.0}
    reference_inputs = _gradient_inputs(inputs, indices)
    source = graph.get("source_input_indices")
    source_indices = (
        tuple(int(index) for index in source)
        if isinstance(source, list)
        else tuple(
            index
            for index, value in enumerate(reference_inputs)
            if hasattr(value, "shape")
        )
    )
    tensor_inputs = tuple(reference_inputs[index] for index in source_indices)
    executor_selected = tuple(
        slot for slot, index in enumerate(source_indices) if index in indices
    )
    executor_inputs = _gradient_inputs(tensor_inputs, executor_selected)
    state = capture_rng_state()
    expected, reference_error = _run_gradients(
        reference, reference_inputs, indices
    )
    restore_rng_state(state)
    actual, executor_error = _run_gradients(
        executor, executor_inputs, executor_selected
    )
    if reference_error is not None or executor_error is not None:
        if type(reference_error) is type(executor_error):
            return {"gradient_error_semantics_verified": 1.0}
        raise GradientVerificationError("gradient error semantics differ")
    if len(expected) != len(actual):
        raise GradientVerificationError("gradient arity differs")
    atol = (
        policy.gradient_atol
        if policy.gradient_atol is not None
        else policy.atol
    )
    rtol = (
        policy.gradient_rtol
        if policy.gradient_rtol is not None
        else policy.rtol
    )
    for expected_item, actual_item in zip(expected, actual, strict=True):
        if (expected_item is None) != (actual_item is None):
            raise GradientVerificationError("None gradient pattern differs")
        if expected_item is not None:
            try:
                assert_close(actual_item, expected_item, atol, rtol)
            except VerificationError as exc:
                raise GradientVerificationError(str(exc)) from exc
    return {"gradient_inputs_verified": float(len(indices))}


__all__ = ["capture_rng_state", "restore_rng_state", "verify_gradients"]
