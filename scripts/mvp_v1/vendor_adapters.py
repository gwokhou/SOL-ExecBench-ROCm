# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Setup-time pre-packing ROCm vendor adapters (Tier 2 vLLM / Tier 3 AITER)."""

from __future__ import annotations

import contextlib
import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedRunner:
    """Resolved operator callable with setup-time pre-packed candidate args."""

    fn: Callable[..., Any] | None
    implementation_name: str
    candidate_args: list[Any]
    unavailable_reason: str | None = None


def materialize_comparable_output(out: Any) -> torch.Tensor | None:
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


def _unavailable(prototype_args: list[Any], reason: str) -> PreparedRunner:
    return PreparedRunner(
        fn=None,
        implementation_name=f"unavailable:{reason}",
        candidate_args=prototype_args,
        unavailable_reason=reason,
    )


def _is_rocm_torch() -> bool:
    return getattr(getattr(torch, "version", None), "hip", None) is not None


def _clone_call_args(args: list[Any]) -> list[Any]:
    return [
        arg.detach().clone() if isinstance(arg, torch.Tensor) else arg
        for arg in args
    ]


def _self_validate(
    cand_fn: Callable[..., Any],
    cand_args: list[Any],
    ref_fn: Callable[..., Any],
    ref_args: list[Any],
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


def _import_vllm_custom_ops() -> tuple[Any, Any] | None:
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


def _try_vllm_bf16_gemm(
    prototype_args: list[Any],
    ops_mod: Any,
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    if ops_mod is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    x0, w_t0 = prototype_args
    w_nk = w_t0.t().contiguous()
    cand_args: list[Any] = [x0, w_nk]
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
    ref_fn: Callable[..., Any],
) -> list[Any]:
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
    cand_args: list[Any],
    ref_args: list[Any],
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    sa_2d = scale_a.expand(cand_args[0].shape[0], 1).contiguous()
    sb_2d = scale_b.expand(cand_args[1].shape[1], 1).contiguous()
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
    cand_args: list[Any],
    ref_args: list[Any],
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
    prototype_args: list[Any],
    ops_mod: Any,
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
    x0, w_t0 = prototype_args
    w_col = w_t0.t().contiguous().t()
    cand_args: list[Any] = [x0, w_col]
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
    prototype_args: list[Any],
    *,
    fp8: bool,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM GEMM runner (BF16 or FP8)."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _import_vllm_custom_ops()
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


def _ensure_aiter_hipblas(aiter_mod: Any) -> None:
    """Initialize AITER hipBLASLt extension lazily before hipb_mm probes."""
    init_fn = getattr(aiter_mod, "hipb_create_extension", None)
    if callable(init_fn):
        with contextlib.suppress(
            RuntimeError, TypeError, ValueError, AttributeError
        ):
            init_fn()


def _try_aiter_bf16_gemm(
    prototype_args: list[Any],
    aiter_mod: Any,
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    x0, w_t0 = prototype_args
    w_nk = w_t0.t().contiguous()
    w_col = w_nk.t()
    args_nk: list[Any] = [x0, w_nk]
    args_col: list[Any] = [x0, w_col]

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


def _try_aiter_preshuffle_helper() -> Callable[..., torch.Tensor] | None:
    try:
        shuffle_mod = importlib.import_module("aiter.ops.shuffle")
        return getattr(shuffle_mod, "shuffle_weight", None)
    except Exception:  # noqa: BLE001 -- optional shuffle helper
        return None


def _probe_aiter_fp8_fn(
    gemm_fn: Callable[..., torch.Tensor],
    cand_args: list[Any],
    ref_args: list[Any],
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
    prototype_args: list[Any],
    aiter_mod: Any,
    *,
    scale_a: torch.Tensor | None,
    scale_b: torch.Tensor | None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    if scale_a is None or scale_b is None:
        return _unavailable(prototype_args, "fp8_scales_required")
    x0, w_t0 = prototype_args
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
    prototype_args: list[Any],
    *,
    fp8: bool,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER GEMM runner (BF16 or FP8)."""
    if not _is_rocm_torch():
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


def _pack_moe_weights_once(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1_packed = torch.cat([gate_w, up_w], dim=-1).transpose(1, 2).contiguous()
    w2_packed = down_w.transpose(1, 2).contiguous()
    return w1_packed, w2_packed


def _resolve_vllm_moe_callables() -> tuple[Any | None, Any | None] | None:
    try:
        moe_mod = importlib.import_module(
            "vllm.model_executor.layers.fused_moe"
        )
    except Exception:  # noqa: BLE001 -- vllm fused_moe optional
        return None
    fused_moe_fn = getattr(moe_mod, "fused_moe", None)
    if fused_moe_fn is not None and not callable(fused_moe_fn):
        fused_moe_fn = getattr(fused_moe_fn, "fused_moe", None)
    fused_experts_fn = getattr(moe_mod, "fused_experts", None)
    if fused_moe_fn is None and fused_experts_fn is None:
        sub_mod = getattr(moe_mod, "fused_moe", None)
        if sub_mod is not None:
            fused_moe_fn = getattr(sub_mod, "fused_moe", None)
            fused_experts_fn = getattr(sub_mod, "fused_experts", None)
    if fused_experts_fn is None:
        with contextlib.suppress(ImportError, RuntimeError, AttributeError):
            sub_mod = importlib.import_module(
                "vllm.model_executor.layers.fused_moe.fused_moe"
            )
            fused_experts_fn = getattr(sub_mod, "fused_experts", None)
    return fused_moe_fn, fused_experts_fn


_FP8_TORCH_DTYPES = (torch.float8_e4m3fnuz, torch.float8_e4m3fn)


def _probe_vllm_bf16_experts(
    fused_experts_fn: Callable[..., torch.Tensor],
    cand_args: list[Any],
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    for use_inplace in (False, True):

        def _make_experts_runner(
            *, pass_inplace: bool
        ) -> Callable[..., torch.Tensor]:
            def _run_experts(
                t: torch.Tensor,
                w1: torch.Tensor,
                w2: torch.Tensor,
                lg: torch.Tensor,
                k: int,
            ) -> torch.Tensor:
                probs = torch.softmax(lg.float(), dim=-1)
                weights, ids = torch.topk(probs, k=int(k), dim=-1)
                w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(
                    t.dtype
                )
                if pass_inplace:
                    return fused_experts_fn(
                        t,
                        w1,
                        w2,
                        w_norm,
                        ids.to(torch.int32),
                        inplace=False,
                    )
                return fused_experts_fn(t, w1, w2, w_norm, ids.to(torch.int32))

            return _run_experts

        experts_runner = _make_experts_runner(pass_inplace=use_inplace)
        if _self_validate(
            experts_runner, cand_args, ref_fn, prototype_args, max_atol
        ):
            return PreparedRunner(
                fn=experts_runner,
                implementation_name="vllm_fused_moe",
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_moe_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM Fused MoE runner (BF16 or FP8)."""
    if prototype_args[0].dtype in _FP8_TORCH_DTYPES:
        from scripts.mvp_v1.vendor_adapters_ext import (
            try_make_vllm_fp8_moe_runner,
        )

        return try_make_vllm_fp8_moe_runner(
            prototype_args, ref_fn=ref_fn, max_atol=max_atol
        )
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    resolved = _resolve_vllm_moe_callables()
    if resolved is None:
        return _unavailable(prototype_args, "vllm_moe_not_installed")
    fused_moe_fn, fused_experts_fn = resolved
    tokens, gate_w, up_w, down_w, logits, top_k = prototype_args
    w1_packed, w2_packed = _pack_moe_weights_once(gate_w, up_w, down_w)
    cand_args: list[Any] = [tokens, w1_packed, w2_packed, logits, int(top_k)]

    if callable(fused_moe_fn):
        for runner in (
            lambda t, w1, w2, lg, k: fused_moe_fn(
                t, w1, w2, lg, k, renormalize=True
            ),
            lambda t, w1, w2, lg, k: fused_moe_fn(t, w1, w2, lg, k),
        ):
            if _self_validate(
                runner, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=runner,
                    implementation_name="vllm_fused_moe",
                    candidate_args=cand_args,
                )

    if callable(fused_experts_fn):
        matched = _probe_vllm_bf16_experts(
            fused_experts_fn,
            cand_args,
            prototype_args,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "vllm_moe_unsupported")


def _build_aiter_moe_weight_pairs(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    w1_packed, w2_packed = _pack_moe_weights_once(gate_w, up_w, down_w)
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    shuffle_helper = _try_aiter_preshuffle_helper()
    if shuffle_helper is not None:
        with contextlib.suppress(
            RuntimeError, TypeError, ValueError, AttributeError
        ):
            pairs.append(
                (
                    shuffle_helper(w1_packed, layout=(16, 16)),
                    shuffle_helper(w2_packed, layout=(16, 16)),
                )
            )
    pairs.append((w1_packed, w2_packed))
    return pairs


def try_make_aiter_moe_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER Fused MoE runner (BF16 or FP8)."""
    if prototype_args[0].dtype in _FP8_TORCH_DTYPES:
        from scripts.mvp_v1.vendor_adapters_ext import (
            try_make_aiter_fp8_moe_runner,
        )

        return try_make_aiter_fp8_moe_runner(
            prototype_args, ref_fn=ref_fn, max_atol=max_atol
        )
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")

    tokens, gate_w, up_w, down_w, logits, top_k = prototype_args
    weight_pairs = _build_aiter_moe_weight_pairs(gate_w, up_w, down_w)
    candidates: list[tuple[str, Callable[..., Any]]] = []
    with contextlib.suppress(ImportError, RuntimeError, AttributeError):
        fm_mod = importlib.import_module("aiter.fused_moe")
        fm_fn = getattr(fm_mod, "fused_moe", None)
        if callable(fm_fn):
            candidates.append(("aiter_fused_moe", fm_fn))
    for attr, impl_name in (
        ("fused_moe", "aiter_fused_moe"),
        ("ck_moe", "aiter_ck_moe"),
    ):
        target = getattr(aiter_mod, attr, None)
        if target is not None and not callable(target):
            target = getattr(target, attr, None)
        if callable(target):
            candidates.append((impl_name, target))
    if not candidates:
        return _unavailable(prototype_args, "aiter_moe_unavailable")

    for w1_cand, w2_cand in weight_pairs:
        cand_args: list[Any] = [tokens, w1_cand, w2_cand, logits, int(top_k)]
        for impl_name, moe_fn in candidates:

            def _make_runner(
                fn: Callable[..., torch.Tensor], wt_dtype: torch.dtype | None
            ) -> Callable[..., torch.Tensor]:
                def _run(
                    t: torch.Tensor,
                    w1: torch.Tensor,
                    w2: torch.Tensor,
                    lg: torch.Tensor,
                    k: int,
                ) -> torch.Tensor:
                    probs = torch.softmax(lg.float(), dim=-1)
                    weights, ids = torch.topk(probs, k=int(k), dim=-1)
                    w_norm = weights / weights.sum(dim=-1, keepdim=True)
                    if wt_dtype is not None:
                        w_norm = w_norm.to(wt_dtype)
                    return fn(t, w1, w2, w_norm, ids.to(torch.int32))

                return _run

            for wt_dtype in (torch.float32, tokens.dtype):
                runner = _make_runner(moe_fn, wt_dtype)
                if _self_validate(
                    runner, cand_args, ref_fn, prototype_args, max_atol
                ):
                    return PreparedRunner(
                        fn=runner,
                        implementation_name=impl_name,
                        candidate_args=cand_args,
                    )
    return _unavailable(prototype_args, "aiter_moe_unsupported")


@dataclass(frozen=True, slots=True, kw_only=True)
class _PackedROCmKV:
    k_paged: torch.Tensor
    v_linear: torch.Tensor
    v_shuffled: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    page_size: int
    q_heads: int
    kv_heads: int
    head_dim: int


def build_paged_tables_and_seqlens(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_len: torch.Tensor,
    *,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build dense ``block_tables`` and ``seq_lens`` once from ragged CSR KV."""
    starts = [int(v) for v in kv_indptr.tolist()]
    batch = max(0, len(starts) - 1)
    counts = [starts[i + 1] - starts[i] for i in range(batch)]
    max_blocks = max(counts, default=1)
    block_tables = torch.zeros(
        (batch, max_blocks), dtype=torch.int32, device=device
    )
    seq_lens_list: list[int] = []
    last_lens = [int(v) for v in kv_last_len.tolist()]
    indices_i32 = kv_indices.to(dtype=torch.int32)
    for b_idx in range(batch):
        p0, p1 = starts[b_idx], starts[b_idx + 1]
        cnt = p1 - p0
        if cnt > 0:
            block_tables[b_idx, :cnt] = indices_i32[p0:p1]
        slen = (
            max(0, (cnt - 1) * page_size + last_lens[b_idx]) if cnt > 0 else 0
        )
        seq_lens_list.append(slen)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    return block_tables, seq_lens, max(seq_lens_list, default=page_size)


def _pack_rocm_paged_kv_once(
    prototype_args: list[Any],
) -> _PackedROCmKV | None:
    q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_len = prototype_args
    n_pages, ps, hk, d = k_cache.shape
    _, q_heads, _ = q.shape
    x_pack = 16 // max(int(k_cache.element_size()), 1)
    if x_pack <= 0 or d % x_pack != 0 or ps % x_pack != 0:
        return None
    k_paged = (
        k_cache.view(n_pages, ps, hk, d // x_pack, x_pack)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_linear = v_cache.permute(0, 2, 3, 1).contiguous()
    v_shuffled = (
        v_cache.view(n_pages, ps // x_pack, x_pack, hk, d)
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    block_tables, seq_lens, max_seq_len = build_paged_tables_and_seqlens(
        kv_indptr,
        kv_indices,
        kv_last_len,
        page_size=ps,
        device=q.device,
    )
    return _PackedROCmKV(
        k_paged=k_paged,
        v_linear=v_linear,
        v_shuffled=v_shuffled,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        page_size=ps,
        q_heads=q_heads,
        kv_heads=hk,
        head_dim=d,
    )


def _make_vllm_pa_v2_runner(
    pa_fn: Callable[..., Any],
    packed: _PackedROCmKV,
    q: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    partition_size: int = 256,
    *,
    with_query_start_loc: bool = False,
) -> Callable[..., torch.Tensor]:
    num_parts = max(1, math.ceil(packed.max_seq_len / partition_size))
    scratch_shape = (q.shape[0], packed.q_heads, num_parts)
    exp_sums = torch.empty(scratch_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty(
        scratch_shape, dtype=torch.float32, device=q.device
    )
    tmp_out = torch.empty(
        (*scratch_shape, packed.head_dim), dtype=q.dtype, device=q.device
    )

    def _run_rocm_v2(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        bt_in: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q_in)
        if with_query_start_loc:
            pa_fn(
                out,
                exp_sums,
                max_logits,
                tmp_out,
                q_in,
                k_in,
                v_in,
                packed.kv_heads,
                scale,
                bt_in,
                sl_in,
                None,
                packed.page_size,
                packed.max_seq_len,
                None,
                "auto",
                k_scale,
                v_scale,
                None,
            )
            return out
        pa_fn(
            out,
            exp_sums,
            max_logits,
            tmp_out,
            q_in,
            k_in,
            v_in,
            packed.kv_heads,
            scale,
            bt_in,
            sl_in,
            packed.page_size,
            packed.max_seq_len,
            None,
            "auto",
            k_scale,
            v_scale,
            None,
        )
        return out

    return _run_rocm_v2


def _make_vllm_pa_v1_runner(
    pa_fn: Callable[..., Any],
    packed: _PackedROCmKV,
    scale: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> Callable[..., torch.Tensor]:
    def _run_v1(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        bt_in: torch.Tensor,
        sl_in: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(q_in)
        pa_fn(
            out,
            q_in,
            k_in,
            v_in,
            packed.kv_heads,
            scale,
            bt_in,
            sl_in,
            packed.page_size,
            packed.max_seq_len,
            None,
            "auto",
            k_scale,
            v_scale,
            0,
            0,
            0,
            1,
            0,
        )
        return out

    return _run_v1


def _probe_vllm_pa_op(
    pa_fn: Callable[..., Any],
    packed: _PackedROCmKV,
    cand_args: list[Any],
    prototype_args: list[Any],
    *,
    impl_name: str,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner | None:
    q = cand_args[0]
    scale = 1.0 / math.sqrt(packed.head_dim)
    k_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    v_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    candidates = (
        _make_vllm_pa_v2_runner(
            pa_fn,
            packed,
            q,
            scale,
            k_scale,
            v_scale,
            partition_size=256,
            with_query_start_loc=True,
        ),
        _make_vllm_pa_v2_runner(
            pa_fn,
            packed,
            q,
            scale,
            k_scale,
            v_scale,
            partition_size=256,
            with_query_start_loc=False,
        ),
        _make_vllm_pa_v2_runner(
            pa_fn,
            packed,
            q,
            scale,
            k_scale,
            v_scale,
            partition_size=512,
            with_query_start_loc=False,
        ),
        _make_vllm_pa_v1_runner(pa_fn, packed, scale, k_scale, v_scale),
    )

    for runner in candidates:
        if _self_validate(runner, cand_args, ref_fn, prototype_args, max_atol):
            return PreparedRunner(
                fn=runner,
                implementation_name=impl_name,
                candidate_args=cand_args,
            )
    return None


def try_make_vllm_gqa_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-2 vLLM PagedAttention GQA runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    loaded = _import_vllm_custom_ops()
    if loaded is None or loaded[1] is None:
        return _unavailable(prototype_args, "vllm_ops_not_installed")
    _vllm_mod, ops_mod = loaded
    packed = _pack_rocm_paged_kv_once(prototype_args)
    if packed is None:
        return _unavailable(prototype_args, "rocm_kv_shape_not_divisible")

    q = prototype_args[0]
    cand_args: list[Any] = [
        q,
        packed.k_paged,
        packed.v_linear,
        packed.block_tables,
        packed.seq_lens,
    ]
    for attr, impl_name in (
        ("paged_attention_rocm", "vllm_paged_attention_rocm"),
        ("paged_attention_v1", "vllm_paged_attention_v1"),
    ):
        pa_fn = getattr(ops_mod, attr, None)
        if pa_fn is None:
            continue
        matched = _probe_vllm_pa_op(
            pa_fn,
            packed,
            cand_args,
            prototype_args,
            impl_name=impl_name,
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "vllm_gqa_unsupported")


def try_make_aiter_gqa_runner(
    prototype_args: list[Any],
    *,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> PreparedRunner:
    """Build a setup-packed Tier-3 AITER PagedAttention GQA runner."""
    if not _is_rocm_torch():
        return _unavailable(prototype_args, "rocm_torch_required")
    try:
        aiter_mod = importlib.import_module("aiter")
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return _unavailable(prototype_args, "aiter_not_installed")
    packed = _pack_rocm_paged_kv_once(prototype_args)
    if packed is None:
        return _unavailable(prototype_args, "rocm_kv_shape_not_divisible")

    q = prototype_args[0]
    max_blocks = int(packed.block_tables.shape[1])
    pa_asm_fn = getattr(aiter_mod, "pa_fwd_asm", None)
    if callable(pa_asm_fn) and packed.head_dim == 128:
        for v_cand in (packed.v_shuffled, packed.v_linear):
            cand_args: list[Any] = [
                q,
                packed.k_paged,
                v_cand,
                packed.block_tables,
                packed.seq_lens,
            ]

            def _run_asm(
                q_in: torch.Tensor,
                k_in: torch.Tensor,
                v_in: torch.Tensor,
                bt_in: torch.Tensor,
                sl_in: torch.Tensor,
            ) -> torch.Tensor:
                return pa_asm_fn(q_in, k_in, v_in, bt_in, sl_in, max_blocks)

            if _self_validate(
                _run_asm, cand_args, ref_fn, prototype_args, max_atol
            ):
                return PreparedRunner(
                    fn=_run_asm,
                    implementation_name="aiter_pa_fwd_asm",
                    candidate_args=cand_args,
                )

    pa_rocm_fn = getattr(aiter_mod, "paged_attention_rocm", None)
    if callable(pa_rocm_fn):
        cand_args_lin: list[Any] = [
            q,
            packed.k_paged,
            packed.v_linear,
            packed.block_tables,
            packed.seq_lens,
        ]
        matched = _probe_vllm_pa_op(
            pa_rocm_fn,
            packed,
            cand_args_lin,
            prototype_args,
            impl_name="aiter_paged_attention_rocm",
            ref_fn=ref_fn,
            max_atol=max_atol,
        )
        if matched is not None:
            return matched
    return _unavailable(prototype_args, "aiter_gqa_unsupported")
