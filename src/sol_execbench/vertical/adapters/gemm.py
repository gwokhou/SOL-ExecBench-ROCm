# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tier 2 vLLM and Tier 3 AITER adapters for skinny BF16/FP8 GEMM."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from functools import partial

import torch

import sol_execbench.vertical.adapters.common as _common
from sol_execbench.vertical.adapters.common import (
    PreparedRunner,
    _as_tensor,
    _ensure_aiter_hipblas,
    _self_validate,
    _try_aiter_preshuffle_helper,
    _unavailable,
)


def _try_vllm_bf16_gemm(
    prototype_args: list[object],
    ops_mod: object,
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    if ops_mod is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    x0, w_t0 = _as_tensor(prototype_args[0]), _as_tensor(prototype_args[1])
    w_nk = w_t0.t().contiguous()
    cand_args: list[object] = [x0, w_nk]
    wv_fn = getattr(ops_mod, "wvSplitK", None)
    if wv_fn is not None and int(x0.shape[0]) <= 4:
        cu_count = (
            int(
                torch.cuda.get_device_properties(
                    x0.device
                ).multi_processor_count
            )
            if x0.is_cuda
            else 304
        )

        def _run_wv(x_in: torch.Tensor, w_nk_in: torch.Tensor) -> torch.Tensor:
            return wv_fn(w_nk_in, x_in, cu_count)

        if _self_validate(_run_wv, cand_args, ref_fn, prototype_args, max_atol):
            return PreparedRunner(
                fn=_run_wv,
                implementation_name="vllm_wvSplitK",
                candidate_args=cand_args,
            )

    def _run_linear(x_in: torch.Tensor, w_nk_in: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x_in, w_nk_in)

    if _self_validate(_run_linear, cand_args, ref_fn, prototype_args, max_atol):
        return PreparedRunner(
            fn=_run_linear,
            implementation_name="vllm_torch_linear_fallback",
            candidate_args=cand_args,
        )
    return _unavailable(prototype_args, "vllm_bf16_gemm_unsupported")


def _build_fp8_ref_args(
    x0: torch.Tensor,
    w_t0: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    ref_fn: Callable[..., object],
) -> list[object]:
    if isinstance(ref_fn, partial) and (
        "scale_a" in ref_fn.keywords or "scale_b" in ref_fn.keywords
    ):
        return [x0.float(), w_t0.float()]
    sa_f = scale_a.float()
    sb_f = scale_b.float()
    if (
        sb_f.ndim == 2
        and sb_f.shape == (w_t0.shape[1], 1)
        and w_t0.shape[0] != w_t0.shape[1]
    ):
        sb_f = sb_f.t()
    return [x0.float() * sa_f, w_t0.float() * sb_f]


def _try_vllm_fp8_custom_op(
    op_fn: Callable[..., torch.Tensor],
    cand_args: list[object],
    ref_args: list[object],
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    sa_2d = scale_a.expand(_as_tensor(cand_args[0]).shape[0], 1).contiguous()
    sb_2d = scale_b.expand(_as_tensor(cand_args[1]).shape[1], 1).contiguous()
    for s_a, s_b in ((sa_2d, sb_2d), (scale_a, scale_b)):

        def _make_runner(
            sa: torch.Tensor, sb: torch.Tensor
        ) -> Callable[..., torch.Tensor]:
            def _run_op(
                x_in: torch.Tensor, w_col_in: torch.Tensor
            ) -> torch.Tensor:
                return op_fn(
                    x_in,
                    w_col_in,
                    scale_a=sa,
                    scale_b=sb,
                    out_dtype=torch.bfloat16,
                )

            return _run_op

        runner = _make_runner(s_a, s_b)
        if _self_validate(runner, cand_args, ref_fn, ref_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name="vllm_fp8_scaled_mm",
                candidate_args=cand_args,
            )
    return None


def _try_vllm_fp8_pad16_fallback(
    x0: torch.Tensor,
    cand_args: list[object],
    ref_args: list[object],
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    m, k = int(x0.shape[0]), int(x0.shape[1])
    if m % 16 == 0:
        return None
    m_pad = math.ceil(m / 16) * 16
    pad_x = torch.zeros((m_pad, k), dtype=x0.dtype, device=x0.device)

    def _run_pad16(x_in: torch.Tensor, w_c: torch.Tensor) -> torch.Tensor:
        rows = x_in.shape[0]
        pad_x[:rows].copy_(x_in)
        return torch._scaled_mm(
            pad_x,
            w_c,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )[:rows]

    if _self_validate(_run_pad16, cand_args, ref_fn, ref_args, max_atol):
        return PreparedRunner(
            fn=_run_pad16,
            implementation_name="vllm_torch_scaled_mm_pad16_fallback",
            candidate_args=cand_args,
        )
    return None


def _try_vllm_fp8_gemm(
    prototype_args: list[object],
    ops_mod: object,
    *,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    if ops_mod is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    if scale_a is None or scale_b is None:
        return _unavailable(prototype_args, "fp8_scales_required")
    x0, w_t0 = _as_tensor(prototype_args[0]), _as_tensor(prototype_args[1])
    w_col = w_t0.t().contiguous().t()
    cand_args: list[object] = [x0, w_col]
    ref_args = _build_fp8_ref_args(x0, w_t0, scale_a, scale_b, ref_fn)

    for attr in ("cutlass_scaled_mm", "scaled_mm"):
        op_fn = getattr(ops_mod, attr, None)
        if op_fn is None:
            continue
        matched = _try_vllm_fp8_custom_op(
            op_fn,
            cand_args,
            ref_args,
            scale_a=scale_a,
            scale_b=scale_b,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched

    def _run_scaled(x_in: torch.Tensor, w_c: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(
            x_in,
            w_c,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    if _self_validate(_run_scaled, cand_args, ref_fn, ref_args, max_atol):
        return PreparedRunner(
            fn=_run_scaled,
            implementation_name="vllm_torch_scaled_mm_fallback",
            candidate_args=cand_args,
        )

    padded = _try_vllm_fp8_pad16_fallback(
        x0,
        cand_args,
        ref_args,
        scale_a=scale_a,
        scale_b=scale_b,
        ref_fn=ref_fn,
        max_atol=max_atol,
    )
    if padded is not None:
        return padded
    return _unavailable(prototype_args, "vllm_fp8_gemm_unsupported")


def try_make_vllm_gemm_runner(
    prototype_args: list[object],
    *,
    fp8: bool,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM GEMM runner (BF16 or FP8)."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _common._import_vllm_custom_ops()
    if loaded is None:
        return _unavailable(prototype_args, "vllm_not_installed")
    _vllm_mod, ops_mod = loaded
    if fp8:
        return _try_vllm_fp8_gemm(
            prototype_args,
            ops_mod,
            scale_a=scale_a,
            scale_b=scale_b,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
    return _try_vllm_bf16_gemm(
        prototype_args, ops_mod, ref_fn=ref_fn, max_atol=max_atol
    )


def _try_aiter_bf16_gemm(
    prototype_args: list[object],
    aiter_mod: object,
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    x0, w_t0 = _as_tensor(prototype_args[0]), _as_tensor(prototype_args[1])
    w_nk = w_t0.t().contiguous()
    w_col = w_nk.t()
    args_nk: list[object] = [x0, w_nk]
    args_col: list[object] = [x0, w_col]

    a16_fn = getattr(aiter_mod, "gemm_a16w16", None)
    if a16_fn is not None:
        candidates_a16: tuple[Callable[..., torch.Tensor], ...] = (
            lambda x_in, w_in: a16_fn(x_in, w_in),
            lambda x_in, w_in: a16_fn(x_in, w_in, None, torch.bfloat16),
        )
        for runner in candidates_a16:
            if _self_validate(
                runner, args_nk, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="aiter_gemm_a16w16",
                    candidate_args=args_nk,
                )

    hipb_fn = getattr(aiter_mod, "hipb_mm", None)
    if hipb_fn is not None:
        candidates_hipb: tuple[Callable[..., torch.Tensor], ...] = (
            lambda x_in, w_in: hipb_fn(x_in, w_in),
            lambda x_in, w_in: hipb_fn(x_in, w_in, -1, None, torch.bfloat16),
            lambda x_in, w_in: hipb_fn(
                x_in, w_in, None, out_dtype=torch.bfloat16
            ),
        )
        for runner in candidates_hipb:
            if _self_validate(
                runner, args_col, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="aiter_hipb_mm",
                    candidate_args=args_col,
                )
    return _unavailable(prototype_args, "aiter_bf16_gemm_unsupported")


def _probe_aiter_fp8_fn(
    gemm_fn: Callable[..., torch.Tensor],
    cand_args: list[object],
    ref_args: list[object],
    *,
    impl_name: str,
    scale_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    for sa, sb in scale_pairs:

        def _make_runner(
            cur_sa: torch.Tensor, cur_sb: torch.Tensor
        ) -> Callable[..., torch.Tensor]:
            def _run_aiter(
                x_in: torch.Tensor, w_in: torch.Tensor
            ) -> torch.Tensor:
                return gemm_fn(x_in, w_in, cur_sa, cur_sb, None, torch.bfloat16)

            return _run_aiter

        runner = _make_runner(sa, sb)
        if _self_validate(runner, cand_args, ref_fn, ref_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name=impl_name,
                candidate_args=cand_args,
            )
    return None


def _try_aiter_fp8_gemm(
    prototype_args: list[object],
    aiter_mod: object,
    *,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    if scale_a is None or scale_b is None:
        return _unavailable(prototype_args, "fp8_scales_required")
    x0, w_t0 = _as_tensor(prototype_args[0]), _as_tensor(prototype_args[1])
    w_nk = w_t0.t().contiguous()
    ref_args = _build_fp8_ref_args(x0, w_t0, scale_a, scale_b, ref_fn)
    sa_2d = scale_a.expand(x0.shape[0], 1).contiguous()
    sb_2d = scale_b.expand(w_nk.shape[0], 1).contiguous()
    scale_pairs = ((sa_2d, sb_2d), (scale_a, scale_b))

    preshuffle_fn = getattr(aiter_mod, "gemm_a8w8_bpreshuffle", None)
    if preshuffle_fn is not None:
        shuffle_helper = _try_aiter_preshuffle_helper()
        try:
            w_shuf = (
                shuffle_helper(w_nk, layout=(16, 16))
                if shuffle_helper is not None
                else w_nk
            )
        except Exception:  # noqa: BLE001 -- fallback to unshuffled layout
            w_shuf = w_nk
        matched = _probe_aiter_fp8_fn(
            preshuffle_fn,
            [x0, w_shuf],
            ref_args,
            impl_name="aiter_gemm_a8w8_bpreshuffle",
            scale_pairs=scale_pairs,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched

    gemm_fn = getattr(aiter_mod, "gemm_a8w8", None)
    if gemm_fn is not None:
        for w_cand in (w_nk, w_t0):
            matched = _probe_aiter_fp8_fn(
                gemm_fn,
                [x0, w_cand],
                ref_args,
                impl_name="aiter_gemm_a8w8",
                scale_pairs=scale_pairs,
                ref_fn=ref_fn,
                max_atol=max_atol,
            )
            if matched is not None:
                return matched
    return _unavailable(prototype_args, "aiter_fp8_gemm_unsupported")


def try_make_aiter_gemm_runner(
    prototype_args: list[object],
    *,
    fp8: bool,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER GEMM runner (BF16 or FP8)."""
    if not _common._is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    _ensure_aiter_hipblas(aiter_mod)
    if fp8:
        return _try_aiter_fp8_gemm(
            prototype_args,
            aiter_mod,
            scale_a=scale_a,
            scale_b=scale_b,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
    return _try_aiter_bf16_gemm(
        prototype_args, aiter_mod, ref_fn=ref_fn, max_atol=max_atol
    )
