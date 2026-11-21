# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""First-order gradient equivalence checks for IR replay."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from solar.errors import GradientVerificationError, SolarError
from solar.types import GraphValue
from solar.verification.contracts import VerificationPolicy
from solar.verification.errors import VerificationError
from solar.verification.executor import IRGraphExecutor
from solar.verification.numerics import assert_close, clone, tensor_leaves

_COTANGENT_BASE_SEED = 0x501A_2026
_COTANGENT_STRIDE = 1_000_003
_AUTOGRAD_ERROR_MARKERS: tuple[tuple[str, str], ...] = (
    (
        "leaf variable that requires grad is being used in an in-place operation",
        "inplace_on_grad_leaf",
    ),
    (
        "view of a leaf variable that requires grad is being used in an in-place operation",
        "inplace_on_leaf_view",
    ),
    (
        "one of the variables needed for gradient computation has been modified by an inplace operation",
        "saved_tensor_mutated_inplace",
    ),
    ("derivative for", "missing_derivative"),
    (
        "element 0 of tensors does not require grad and does not have a grad_fn",
        "non_differentiable_backward",
    ),
)


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


def _cotangent_for(
    tensor: GraphValue,
    *,
    probe_index: int,
) -> GraphValue:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("cotangent target must be a tensor")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        _COTANGENT_BASE_SEED + probe_index * _COTANGENT_STRIDE
    )
    numel = max(1, tensor.numel())
    magnitudes = torch.empty(numel, dtype=torch.float64).uniform_(
        0.5,
        1.5,
        generator=generator,
    )
    signs = torch.where(
        torch.randint(0, 2, (numel,), generator=generator) == 0,
        1.0,
        -1.0,
    )
    values = (magnitudes * signs).reshape(tensor.shape)
    return values.to(device=tensor.device, dtype=tensor.dtype)


def _autograd_error_kind(exc: BaseException | None) -> str | None:
    if (
        exc is None
        or isinstance(exc, SolarError)
        or type(exc) is not RuntimeError
    ):
        return None
    message = str(exc).lower()
    for marker, kind in _AUTOGRAD_ERROR_MARKERS:
        if marker in message:
            if (
                kind == "missing_derivative"
                and "not implemented" not in message
            ):
                continue
            return kind
    return None


def _probe_leaf_gradients(
    value: GraphValue,
    watched: tuple[GraphValue, ...],
    *,
    probe_offset: int,
    retain_after: bool,
) -> tuple[bool, list[GraphValue | None], int]:
    import torch

    if not isinstance(value, torch.Tensor) or not value.requires_grad:
        return False, [], probe_offset
    targets = (value.real, value.imag) if value.is_complex() else (value,)
    collected: list[GraphValue | None] = []
    for idx, target in enumerate(targets):
        cotangent = _cotangent_for(target, probe_index=probe_offset + idx)
        scalar = (target * cotangent).sum()
        keep_graph = retain_after or (idx + 1 < len(targets))
        grads = torch.autograd.grad(
            scalar,
            watched,
            allow_unused=True,
            retain_graph=keep_graph,
        )
        collected.extend(grads)
    connected = any(grad is not None for grad in collected)
    return (
        connected,
        (collected if connected else []),
        probe_offset + len(targets),
    )


def _run_gradients(
    callable_: Callable[..., GraphValue],
    inputs: tuple[GraphValue, ...],
    selected: Sequence[int],
) -> tuple[
    tuple[bool, ...], tuple[GraphValue | None, ...], BaseException | None
]:
    import torch

    watched = tuple(inputs[index] for index in selected)
    try:
        with torch.enable_grad():
            leaves = tensor_leaves(callable_(*inputs))
            last_grad_idx = max(
                (
                    idx
                    for idx, leaf in enumerate(leaves)
                    if bool(leaf.requires_grad)
                ),
                default=-1,
            )
            differentiable: list[bool] = []
            gradients: list[GraphValue | None] = []
            probe_offset = 0
            for idx, value in enumerate(leaves):
                connected, leaf_grads, probe_offset = _probe_leaf_gradients(
                    value,
                    watched,
                    probe_offset=probe_offset,
                    retain_after=idx < last_grad_idx,
                )
                differentiable.append(connected)
                gradients.extend(leaf_grads)
            return tuple(differentiable), tuple(gradients), None
    except SolarError as exc:
        return (), (), exc
    except (RuntimeError, ValueError, TypeError) as exc:
        return (), (), exc


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
    executor_selected = tuple(source_indices.index(index) for index in indices)
    executor_inputs = _gradient_inputs(tensor_inputs, executor_selected)
    state = capture_rng_state()
    expected_pattern, expected, reference_error = _run_gradients(
        reference, reference_inputs, indices
    )
    restore_rng_state(state)
    actual_pattern, actual, executor_error = _run_gradients(
        executor, executor_inputs, executor_selected
    )
    if reference_error is not None or executor_error is not None:
        ref_kind = _autograd_error_kind(reference_error)
        exec_kind = _autograd_error_kind(executor_error)
        if ref_kind is not None and ref_kind == exec_kind:
            return {"gradient_error_semantics_verified": 1.0}
        cause = (
            executor_error if executor_error is not None else reference_error
        )
        raise GradientVerificationError(
            "gradient error semantics differ"
        ) from cause
    if expected_pattern != actual_pattern:
        raise GradientVerificationError("output differentiability differs")
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
                assert_close(
                    actual_item,
                    expected_item,
                    atol,
                    rtol,
                    allow_matching_nan=True,
                )
            except VerificationError as exc:
                raise GradientVerificationError(str(exc)) from exc
    return {"gradient_inputs_verified": float(len(indices))}


__all__ = ["capture_rng_state", "restore_rng_state", "verify_gradients"]
