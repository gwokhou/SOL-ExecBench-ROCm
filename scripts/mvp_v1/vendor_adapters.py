# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional self-validating vendor-library adapters (FlashInfer / AITER)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch

_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024


def _is_rocm_torch() -> bool:
    return getattr(getattr(torch, "version", None), "hip", None) is not None


def _self_validate(
    cand_fn: Callable[..., torch.Tensor],
    cand_args: list[Any],
    ref_fn: Callable[..., torch.Tensor],
    ref_args: list[Any],
    max_atol: float,
) -> bool:
    """Dry-run vendor adapter on prototype args and verify finite error."""
    try:
        with torch.no_grad():
            cand_out = cand_fn(*cand_args)
            ref_out = ref_fn(*ref_args)
        if not isinstance(cand_out, torch.Tensor):
            return False
        if cand_out.shape != ref_out.shape:
            return False
        if not bool(torch.isfinite(cand_out).all().item()):
            return False
        err = float((cand_out.float() - ref_out.float()).abs().max().item())
        return err <= max_atol
    except Exception:  # noqa: BLE001 -- degrade cleanly on API/arch mismatch
        return False


def try_make_flashinfer_gqa_runner(
    prototype_args: list[Any],
    *,
    device: str,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> tuple[Callable[..., torch.Tensor], str] | None:
    """Build and validate a FlashInfer paged-decode runner on CUDA if available."""
    if device == "cpu" or _is_rocm_torch():
        return None
    try:
        flashinfer = importlib.import_module("flashinfer")
        wrapper_cls = getattr(
            flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None
        )
        if wrapper_cls is None:
            return None
        q, k_cache, _v, kv_indptr, kv_indices, kv_last_len = prototype_args
        workspace = torch.empty(
            _FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=device
        )
        wrapper = wrapper_cls(workspace, "NHD", use_tensor_cores=True)
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_len,
            num_qo_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim=q.shape[2],
            page_size=k_cache.shape[1],
            q_data_type=q.dtype,
            kv_data_type=k_cache.dtype,
        )
    except Exception:  # noqa: BLE001 -- flashinfer optional on cloud CUDA
        return None

    def _run_flashinfer(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        _indptr: torch.Tensor,
        _indices: torch.Tensor,
        _last_len: torch.Tensor,
    ) -> torch.Tensor:
        return wrapper.run(q_in, (k_in, v_in))

    if _self_validate(
        _run_flashinfer, prototype_args, ref_fn, prototype_args, max_atol
    ):
        return _run_flashinfer, "flashinfer_paged_gqa"
    return None


def try_make_aiter_fp8_gemm_runner(
    prototype_args: list[Any],
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    device: str,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> tuple[Callable[..., torch.Tensor], str] | None:
    """Build and validate an AMD AITER FP8 GEMM runner on ROCm if installed."""
    if device == "cpu" or not _is_rocm_torch():
        return None
    try:
        aiter_mod = importlib.import_module("aiter")
        gemm_fn = getattr(aiter_mod, "gemm_a8w8", None)
        if gemm_fn is None:
            return None
    except Exception:  # noqa: BLE001 -- aiter optional on ROCm hosts
        return None

    x0, w_t0 = prototype_args
    ref_args = [x0.float(), w_t0.float()]
    for w_transpose in (True, False):

        def _make_runner(*, transpose_w: bool) -> Callable[..., torch.Tensor]:
            def _run_aiter(
                x_in: torch.Tensor,
                w_t_in: torch.Tensor,
            ) -> torch.Tensor:
                w_arg = w_t_in.t().contiguous() if transpose_w else w_t_in
                return gemm_fn(
                    x_in,
                    w_arg,
                    scale_a,
                    scale_b,
                    None,
                    torch.bfloat16,
                )

            return _run_aiter

        runner = _make_runner(transpose_w=w_transpose)
        if _self_validate(runner, prototype_args, ref_fn, ref_args, max_atol):
            return runner, "aiter_gemm_a8w8"
    return None


def try_make_vendor_moe_runner(
    prototype_args: list[Any],
    *,
    device: str,
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> tuple[Callable[..., torch.Tensor], str] | None:
    """Probe FlashInfer / AITER Fused MoE adapters when installed and valid."""
    if device == "cpu":
        return None
    if not _is_rocm_torch():
        return _try_flashinfer_moe(prototype_args, ref_fn, max_atol)
    return _try_aiter_moe(prototype_args, ref_fn, max_atol)


def _try_flashinfer_moe(
    prototype_args: list[Any],
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> tuple[Callable[..., torch.Tensor], str] | None:
    try:
        mod = importlib.import_module("flashinfer.fused_moe")
        fn = getattr(mod, "cutlass_fused_moe", None)
        if fn is None:
            return None
    except Exception:  # noqa: BLE001 -- optional CUDA MoE extension
        return None

    def _run_fi_moe(
        tokens: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, k=int(top_k), dim=-1)
        w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(tokens.dtype)
        fc1 = torch.cat([gate_w, up_w], dim=-1).transpose(1, 2).contiguous()
        fc2 = down_w.transpose(1, 2).contiguous()
        return fn(tokens, ids.to(torch.int32), w_norm, fc1, fc2)

    if _self_validate(
        _run_fi_moe, prototype_args, ref_fn, prototype_args, max_atol
    ):
        return _run_fi_moe, "flashinfer_fused_moe"
    return None


def _try_aiter_moe(
    prototype_args: list[Any],
    ref_fn: Callable[..., torch.Tensor],
    max_atol: float,
) -> tuple[Callable[..., torch.Tensor], str] | None:
    try:
        aiter_mod = importlib.import_module("aiter")
        fn = getattr(aiter_mod, "fused_moe", None) or getattr(
            aiter_mod, "ck_moe", None
        )
        if fn is None:
            return None
    except Exception:  # noqa: BLE001 -- optional ROCm MoE extension
        return None

    def _run_aiter_moe(
        tokens: torch.Tensor,
        gate_w: torch.Tensor,
        up_w: torch.Tensor,
        down_w: torch.Tensor,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probs, k=int(top_k), dim=-1)
        w_norm = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.float32)
        w1 = torch.cat([gate_w, up_w], dim=-1).transpose(1, 2).contiguous()
        w2 = down_w.transpose(1, 2).contiguous()
        return fn(tokens, w1, w2, w_norm, ids.to(torch.int32))

    if _self_validate(
        _run_aiter_moe, prototype_args, ref_fn, prototype_args, max_atol
    ):
        return _run_aiter_moe, "aiter_fused_moe"
    return None
