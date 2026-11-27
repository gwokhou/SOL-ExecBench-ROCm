# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM and Tier 3 AITER adapters for SwiGLU + FP8 quantization."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable

import torch

import sol_execbench.vertical.adapters.common as _common
from sol_execbench.vertical.adapters.common import (
    PreparedRunner,
    _as_callable,
    _as_tensor,
    _self_validate,
    _unavailable,
)


def _build_swiglu_quant_runners(
    fused_fn: object,
    silu_fn: object,
    quant_fn: object,
    *,
    scaled_quant_fn: object,
    m: int,
    inter: int,
    fp8_dtype: torch.dtype,
    fused_name: str,
    split_name: str,
) -> list[tuple[str, Callable[[torch.Tensor], object]]]:
    pairs: list[tuple[str, Callable[[torch.Tensor], object]]] = []
    fused_call = _as_callable(fused_fn)
    silu_call = _as_callable(silu_fn)
    quant_call = _as_callable(quant_fn)
    scaled_call = _as_callable(scaled_quant_fn)
    if fused_call is not None:

        def _run_fused_out(
            gu_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            y_fp8 = torch.empty(
                (m, inter), dtype=fp8_dtype, device=gu_in.device
            )
            y_scale = torch.empty(
                (m, 1), dtype=torch.float32, device=gu_in.device
            )
            fused_call(y_fp8, y_scale, gu_in)
            return y_fp8, y_scale

        pairs.extend(((fused_name, fused_call), (fused_name, _run_fused_out)))

    if silu_call is not None and quant_call is not None:

        def _run_two_step(
            gu_in: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            act_out = torch.empty(
                (m, inter), dtype=gu_in.dtype, device=gu_in.device
            )
            silu_call(act_out, gu_in)
            y_fp8 = torch.empty(
                (m, inter), dtype=fp8_dtype, device=gu_in.device
            )
            y_scale = torch.empty(
                (m, 1), dtype=torch.float32, device=gu_in.device
            )
            quant_call(y_fp8, act_out, y_scale)
            return y_fp8, y_scale

        pairs.append((split_name, _run_two_step))

    if silu_call is not None and scaled_call is not None:

        def _run_scaled_two_step(gu_in: torch.Tensor) -> object:
            act_out = torch.empty(
                (m, inter), dtype=gu_in.dtype, device=gu_in.device
            )
            silu_call(act_out, gu_in)
            return scaled_call(act_out, use_per_token_if_dynamic=True)

        pairs.append((split_name, _run_scaled_two_step))
    return pairs


def _probe_swiglu_quant_pair(
    fused_fn: object,
    silu_fn: object,
    quant_fn: object,
    cand_args: list[object],
    *,
    scaled_quant_fn: object = None,
    fused_name: str,
    split_name: str,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., object],
    max_atol: float,
) -> PreparedRunner | None:
    gate_up = _as_tensor(cand_args[0])
    m, i2 = int(gate_up.shape[0]), int(gate_up.shape[1])
    for impl_name, runner in _build_swiglu_quant_runners(
        fused_fn,
        silu_fn,
        quant_fn,
        scaled_quant_fn=scaled_quant_fn,
        m=m,
        inter=i2 // 2,
        fp8_dtype=fp8_dtype,
        fused_name=fused_name,
        split_name=split_name,
    ):
        if _self_validate(runner, cand_args, ref_fn, cand_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name=impl_name,
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_swiglu_quant_runner(
    prototype_args: list[object],
    *,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a Tier-2 vLLM SwiGLU + Dynamic Per-Token FP8 Quant runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _common._import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    silu_fn = getattr(ops_mod, "silu_and_mul", None)
    if not callable(silu_fn):
        c_ns = getattr(torch.ops, "_C", None)
        silu_fn = getattr(c_ns, "silu_and_mul", None) if c_ns else None
    matched = _probe_swiglu_quant_pair(
        getattr(ops_mod, "silu_and_mul_quant", None),
        silu_fn,
        getattr(ops_mod, "dynamic_per_token_scaled_fp8_quant", None),
        prototype_args,
        scaled_quant_fn=getattr(ops_mod, "scaled_fp8_quant", None),
        fused_name="vllm_silu_and_mul_fp8_quant",
        split_name="vllm_silu_mul_then_fp8_quant",
        fp8_dtype=fp8_dtype,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if matched is not None:
        return matched
    return _unavailable(prototype_args, "vllm_swiglu_quant_unsupported")


def try_make_aiter_swiglu_quant_runner(
    prototype_args: list[object],
    *,
    fp8_dtype: torch.dtype,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a Tier-3 AITER SwiGLU + Dynamic Per-Token FP8 Quant runner."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    fused_fn = getattr(aiter_mod, "silu_and_mul_scaled_quant", None)
    if fused_fn is None:
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            act_mod = importlib.import_module("aiter.ops.activation")
            fused_fn = getattr(act_mod, "silu_and_mul_quant", None)
    matched = _probe_swiglu_quant_pair(
        fused_fn,
        getattr(aiter_mod, "silu_and_mul", None),
        getattr(aiter_mod, "dynamic_per_token_scaled_quant", None),
        prototype_args,
        fused_name="aiter_silu_and_mul_fp8_quant",
        split_name="aiter_silu_mul_then_fp8_quant",
        fp8_dtype=fp8_dtype,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if matched is not None:
        return matched
    return _unavailable(prototype_args, "aiter_swiglu_quant_unsupported")
