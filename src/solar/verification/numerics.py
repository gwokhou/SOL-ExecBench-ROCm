# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded numerical, alias, and input-pattern verification primitives."""

from __future__ import annotations

from collections.abc import Iterable

from solar.common.types import DynamicValue
from solar.verification.errors import VerificationError


def clone(value: DynamicValue) -> DynamicValue:
    """Clone nested tensor inputs without changing scalar arguments."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if isinstance(value, list):
        return [clone(item) for item in value]
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return value


def _tensor_leaves(value: DynamicValue) -> list[DynamicValue]:
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _tensor_leaves(item)]
    if isinstance(value, dict):
        return [leaf for key in value for leaf in _tensor_leaves(value[key])]
    return []


def _same_storage(left: DynamicValue, right: DynamicValue) -> bool:
    import torch

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left is right:
        return True
    try:
        return left.untyped_storage()._cdata == right.untyped_storage()._cdata
    except RuntimeError:
        return False


def alias_relation(
    outputs: DynamicValue, inputs: DynamicValue
) -> tuple[tuple[bool, ...], ...]:
    """Return the complete input/output tensor-storage alias relation."""
    leaves = [*_tensor_leaves(inputs), *_tensor_leaves(outputs)]
    return tuple(
        tuple(_same_storage(left, right) for right in leaves) for left in leaves
    )


def assert_close(
    actual: DynamicValue,
    expected: DynamicValue,
    atol: float,
    rtol: float,
    *,
    required_matched_ratio: float = 1.0,
    max_error_cap: float | None = None,
    allow_negative_inf: bool = False,
    allow_matching_nan: bool = False,
) -> dict[str, float]:
    """Compare nested outputs with bounded temporary GPU memory."""
    import torch

    policy = (
        atol,
        rtol,
        required_matched_ratio,
        max_error_cap,
        allow_negative_inf,
        allow_matching_nan,
    )
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        return _tensor_close_stats(actual, expected, *policy)
    if isinstance(actual, (tuple, list)) and isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise VerificationError("output arity mismatch")
        return _nested_close_stats(zip(actual, expected), policy)
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise VerificationError("output mapping keys differ")
        return _nested_close_stats(
            ((actual[key], expected[key]) for key in actual), policy
        )
    if actual != expected:
        raise VerificationError(
            f"non-tensor output mismatch: {actual!r} != {expected!r}"
        )
    return {"max_abs_error": 0.0, "matched_ratio": 1.0}


def _nested_close_stats(
    pairs: Iterable[tuple[DynamicValue, DynamicValue]],
    policy: tuple[float, float, float, float | None, bool, bool],
) -> dict[str, float]:
    stats = [
        assert_close(
            actual,
            expected,
            policy[0],
            policy[1],
            required_matched_ratio=policy[2],
            max_error_cap=policy[3],
            allow_negative_inf=policy[4],
            allow_matching_nan=policy[5],
        )
        for actual, expected in pairs
    ]
    return {
        "max_abs_error": max((item["max_abs_error"] for item in stats), default=0.0),
        "matched_ratio": min(
            (item.get("matched_ratio", 1.0) for item in stats), default=1.0
        ),
    }


def _tensor_close_stats(
    actual: DynamicValue,
    expected: DynamicValue,
    atol: float,
    rtol: float,
    required_matched_ratio: float,
    max_error_cap: float | None,
    allow_negative_inf: bool,
    allow_matching_nan: bool,
) -> dict[str, float]:
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("tensor comparison requires tensor operands")
    if actual.shape != expected.shape:
        raise VerificationError(
            f"output shape mismatch: einsum={tuple(actual.shape)}, "
            f"reference={tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        raise VerificationError(
            f"output dtype mismatch: einsum={actual.dtype}, reference={expected.dtype}"
        )
    if not (actual.is_floating_point() or actual.is_complex()):
        if not torch.equal(actual, expected):
            raise VerificationError("integer/bool tensor values differ")
        return {"max_abs_error": 0.0}
    return _floating_close_stats(
        actual,
        expected,
        atol,
        rtol,
        required_matched_ratio=required_matched_ratio,
        max_error_cap=max_error_cap,
        allow_negative_inf=allow_negative_inf,
        allow_matching_nan=allow_matching_nan,
    )


def _floating_close_stats(
    actual: DynamicValue,
    expected: DynamicValue,
    atol: float,
    rtol: float,
    *,
    required_matched_ratio: float,
    max_error_cap: float | None,
    allow_negative_inf: bool,
    allow_matching_nan: bool,
) -> dict[str, float]:
    """Compare large tensors in bounded chunks to cap verifier peak memory."""
    import torch

    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("floating comparison requires tensor operands")
    dtype = torch.complex64 if actual.is_complex() else torch.float32
    output, reference = actual.reshape(-1), expected.reshape(-1)
    matched_count = finite_count = matching_nan_count = 0
    max_abs = 0.0
    output_nonzero = reference_nonzero = False
    for start in range(0, output.numel(), 1_048_576):
        out = output[start : start + 1_048_576].to(dtype)
        ref = reference[start : start + 1_048_576].to(dtype)
        reference_nonzero |= bool((ref != 0).any())
        output_nonzero |= bool((out != 0).any())
        matching_negative_inf = torch.zeros_like(out, dtype=torch.bool)
        matching_nan = torch.zeros_like(out, dtype=torch.bool)
        if allow_negative_inf:
            matching_negative_inf = torch.isneginf(out) & torch.isneginf(ref)
        if allow_matching_nan:
            matching_nan = torch.isnan(out) & torch.isnan(ref)
        matching_nonfinite = matching_negative_inf | matching_nan
        if bool(
            ((~torch.isfinite(out)) & ~matching_nonfinite).any()
            or ((~torch.isfinite(ref)) & ~matching_nonfinite).any()
        ):
            raise VerificationError("non-finite tensor values are not allowed")
        if bool(matching_nonfinite.any()):
            finite = ~matching_nonfinite
            out, ref = out[finite], ref[finite]
        difference = (out - ref).abs()
        matched_count += int((difference <= atol + rtol * ref.abs()).sum().item())
        finite_count += difference.numel()
        if difference.numel():
            max_abs = max(max_abs, float(difference.max().item()))
        matching_nan_count += int(matching_nan.sum().item())
    if reference_nonzero and not output_nonzero:
        raise VerificationError("all-zero output disagrees with reference")
    matched_ratio = matched_count / finite_count if finite_count else 1.0
    if matched_ratio < required_matched_ratio:
        raise VerificationError(
            f"numerical mismatch: matched_ratio={matched_ratio:.6g}, "
            f"required={required_matched_ratio:.6g}, max_abs={max_abs:.6g}"
        )
    if max_error_cap is not None and max_abs > max_error_cap:
        raise VerificationError(
            f"maximum error {max_abs:.6g} exceeds cap {max_error_cap:.6g}"
        )
    stats = {"max_abs_error": max_abs, "matched_ratio": matched_ratio}
    if allow_matching_nan and matching_nan_count:
        stats["matching_nan_count"] = float(matching_nan_count)
    return stats


def pattern_inputs(
    inputs: tuple[DynamicValue, ...], pattern: str
) -> tuple[DynamicValue, ...]:
    """Apply a deterministic verification pattern to tensor inputs."""
    import torch

    def transform(value: DynamicValue) -> DynamicValue:
        if not isinstance(value, torch.Tensor):
            return value
        if pattern == "random":
            return value
        if pattern == "zeros":
            return torch.zeros_like(value)
        if pattern == "boundary":
            if value.is_floating_point():
                flat = torch.arange(value.numel(), device=value.device).reshape(
                    value.shape
                )
                return ((flat % 3) - 1).to(value.dtype)
            if value.dtype == torch.bool:
                flat = torch.arange(value.numel(), device=value.device).reshape(
                    value.shape
                )
                return (flat % 2).bool()
            return torch.zeros_like(value)
        raise VerificationError(f"unknown verification input pattern: {pattern}")

    return tuple(transform(value) for value in inputs)


__all__ = ["alias_relation", "assert_close", "clone", "pattern_inputs"]
