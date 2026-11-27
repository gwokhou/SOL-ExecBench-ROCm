# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared setup-time pre-packing helpers for Tier 2 vLLM / Tier 3 AITER adapters."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch


class DynamicCallable(Protocol):
    """Typed protocol for dynamically resolved ROCm vendor callables."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Invoke the dynamically resolved vendor callable."""
        ...


def _as_callable(value: object) -> DynamicCallable | None:
    if callable(value):
        return cast(DynamicCallable, value)
    return None


_FP8_TORCH_DTYPES = (
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2fnuz,
    torch.float8_e5m2,
)


def _as_tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(value).__name__}")
    return value


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected int, got {type(value).__name__}")
    return int(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedRunner:
    """Resolved operator callable with setup-time pre-packed candidate args."""

    fn: Callable[..., object] | None
    implementation_name: str
    candidate_args: list[object]
    unavailable_reason: str | None = None


def materialize_comparable_output(out: object) -> torch.Tensor | None:
    """Convert Tensor or ``(fp8, fp32_scale)`` pair to comparable Tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if (
        isinstance(out, tuple)
        and len(out) == 2
        and isinstance(out[0], torch.Tensor)
        and isinstance(out[1], torch.Tensor)
    ):
        return (out[0].float() * out[1].float()).to(torch.bfloat16)
    return None


def _unavailable(prototype_args: list[object], reason: str) -> PreparedRunner:
    return PreparedRunner(
        fn=None,
        implementation_name=f"unavailable:{reason}",
        candidate_args=prototype_args,
        unavailable_reason=reason,
    )


def _is_rocm_torch() -> bool:
    return getattr(getattr(torch, "version", None), "hip", None) is not None


def _clone_call_args(args: list[object]) -> list[object]:
    return [
        arg.detach().clone() if isinstance(arg, torch.Tensor) else arg
        for arg in args
    ]


def _self_validate(
    cand_fn: Callable[..., object],
    cand_args: list[object],
    ref_fn: Callable[..., object],
    ref_args: list[object],
    max_atol: float,
) -> bool:
    """Dry-run vendor adapter on cloned pre-packed args and verify finite error."""
    try:
        with torch.no_grad():
            ref_out = materialize_comparable_output(
                ref_fn(*_clone_call_args(ref_args))
            )
            cand_out = materialize_comparable_output(
                cand_fn(*_clone_call_args(cand_args))
            )
        if cand_out is None or ref_out is None:
            return False
        if cand_out.shape != ref_out.shape:
            return False
        if not bool(torch.isfinite(cand_out).all().item()):
            return False
        err = float((cand_out.float() - ref_out.float()).abs().max().item())
        return err <= max_atol
    except Exception:  # noqa: BLE001 -- degrade cleanly on API/arch mismatch
        return False


def _import_vllm_custom_ops() -> tuple[object, object] | None:
    try:
        vllm_mod = importlib.import_module("vllm")
    except Exception:  # noqa: BLE001 -- vllm optional on host
        return None
    ops_mod = getattr(vllm_mod, "_custom_ops", None)
    if ops_mod is None:
        try:
            ops_mod = importlib.import_module("vllm._custom_ops")
        except Exception:  # noqa: BLE001 -- optional _custom_ops
            ops_mod = None
    return vllm_mod, ops_mod


def _ensure_aiter_hipblas(aiter_mod: object) -> None:
    """Initialize AITER hipBLASLt extension lazily before hipb_mm probes."""
    init_fn = getattr(aiter_mod, "hipb_create_extension", None)
    if callable(init_fn):
        with contextlib.suppress(
            RuntimeError, TypeError, ValueError, AttributeError
        ):
            init_fn()


def _try_aiter_preshuffle_helper() -> Callable[..., torch.Tensor] | None:
    try:
        shuffle_mod = importlib.import_module("aiter.ops.shuffle")
        return getattr(shuffle_mod, "shuffle_weight", None)
    except Exception:  # noqa: BLE001 -- optional shuffle helper
        return None
