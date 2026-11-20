# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Cross-vendor kernels and PyTorch references for the 3-operator MVP."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import StrEnum
from functools import cache
from typing import Any

import torch

from scripts.mvp_v1.triton_kernels import (
    allocate_fused_moe_workspace,
    allocate_splitk_gqa_workspace,
    run_fused_moe_triton,
    run_gqa_paged_decode_splitk,
    select_gqa_num_splits,
    supports_splitk_gqa,
)
from scripts.mvp_v1.vendor_adapters import (
    try_make_aiter_fp8_gemm_runner,
    try_make_flashinfer_gqa_runner,
    try_make_vendor_moe_runner,
)


class KernelBackend(StrEnum):
    """Execution backend preference for GPU runs."""

    AUTO = "auto"
    VENDOR = "vendor"
    TRITON = "triton"
    PYTORCH = "pytorch"


def ref_skinny_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference BF16 output for Skinny GEMM (also used for FP8 dequant ref)."""
    return (x.float() @ weight.float()).to(torch.bfloat16)


def make_skinny_gemm_runner(
    *,
    prototype_args: list[Any],
    fp8: bool,
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    max_atol: float = 5e-1,
) -> tuple[Callable[..., torch.Tensor], str]:
    """Build candidate runner for BF16 matmul, AITER FP8, or FP8 scaled_mm."""
    if not fp8:
        return torch.matmul, "torch.matmul"
    if device == "cpu" or scale_a is None or scale_b is None:
        return (
            lambda x, w_t: (x.float() @ w_t.float()).to(torch.bfloat16),
            "cpu_fp8_ref",
        )
    if backend in (KernelBackend.AUTO, KernelBackend.VENDOR):
        aiter_match = try_make_aiter_fp8_gemm_runner(
            prototype_args,
            scale_a=scale_a,
            scale_b=scale_b,
            device=device,
            ref_fn=ref_skinny_gemm,
            max_atol=max_atol,
        )
        if aiter_match is not None:
            return aiter_match

    def _run_fp8(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(
            x,
            weight_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    return _run_fp8, "torch._scaled_mm"


def ref_fused_moe(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Reference FP32-accumulated SwiGLU top-k MoE."""
    probs = torch.softmax(gate_logits.float(), dim=-1)
    weights, expert_ids = torch.topk(probs, k=int(top_k), dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(tokens, dtype=torch.float32)
    tf = tokens.float()
    for token_idx in range(tokens.shape[0]):
        for slot in range(int(top_k)):
            eid = int(expert_ids[token_idx, slot].item())
            w = weights[token_idx, slot]
            x = tf[token_idx : token_idx + 1]
            g = torch.nn.functional.silu(x @ gate_weights[eid].float())
            u = x @ up_weights[eid].float()
            out[token_idx] += w * ((g * u) @ down_weights[eid].float())[0]
    return out.to(torch.bfloat16)


def run_fused_moe_grouped(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Active-expert grouped SwiGLU MoE avoiding inactive expert loads."""
    probs = torch.softmax(gate_logits.float(), dim=-1)
    weights, expert_ids = torch.topk(probs, k=int(top_k), dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(tokens.dtype)
    flat_experts = expert_ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    token_repeat = torch.arange(
        tokens.shape[0],
        device=tokens.device,
    ).repeat_interleave(int(top_k))
    order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[order]
    sorted_tokens = token_repeat[order]
    sorted_weights = flat_weights[order]
    unique_e, counts = torch.unique_consecutive(
        sorted_experts, return_counts=True
    )
    out = torch.zeros_like(tokens, dtype=torch.float32)
    offset = 0
    for eid_t, count_t in zip(unique_e.tolist(), counts.tolist(), strict=True):
        sl = slice(offset, offset + count_t)
        t_idx = sorted_tokens[sl]
        w_e = sorted_weights[sl].unsqueeze(-1)
        x_e = tokens[t_idx]
        hidden = torch.nn.functional.silu(x_e @ gate_weights[eid_t]) * (
            x_e @ up_weights[eid_t]
        )
        contrib = (hidden @ down_weights[eid_t]).float() * w_e.float()
        out.index_add_(0, t_idx, contrib)
        offset += count_t
    return out.to(torch.bfloat16)


def make_fused_moe_runner(
    *,
    prototype_args: list[Any],
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 5e-2,
) -> tuple[Callable[..., torch.Tensor], str]:
    """Select vendor Fused MoE, fused Triton MoE, or grouped PyTorch fallback."""
    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return run_fused_moe_grouped, "grouped_swiglu_moe"
    if backend in (KernelBackend.AUTO, KernelBackend.VENDOR):
        vendor = try_make_vendor_moe_runner(
            prototype_args,
            device=device,
            ref_fn=ref_fused_moe,
            max_atol=max_atol,
        )
        if vendor is not None:
            return vendor
    tokens, gate_w, _up, _down, _logits, top_k = prototype_args
    workspace = allocate_fused_moe_workspace(
        tokens, gate_w.shape[2], int(top_k)
    )

    def _run_triton_moe(*args: Any) -> torch.Tensor:
        return run_fused_moe_triton(*args, workspace=workspace)

    try:
        _run_triton_moe(*prototype_args)
        return _run_triton_moe, "triton_fused_swiglu_moe"
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton fails
        return run_fused_moe_grouped, "grouped_swiglu_moe"


def ref_gqa_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> torch.Tensor:
    """Reference NHD paged GQA decode implementation in FP32 -> BF16."""
    batch, q_heads, head_dim = q.shape
    page_size = k_cache.shape[1]
    kv_heads = k_cache.shape[2]
    group_size = q_heads // kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    for b in range(batch):
        p0 = int(kv_indptr[b].item())
        p1 = int(kv_indptr[b + 1].item())
        pages = kv_indices[p0:p1]
        seq_len = (len(pages) - 1) * page_size + int(kv_last_page_len[b].item())
        k_seq = k_cache[pages].reshape(-1, kv_heads, head_dim)[:seq_len]
        v_seq = v_cache[pages].reshape(-1, kv_heads, head_dim)[:seq_len]
        k_exp = k_seq.transpose(0, 1).repeat_interleave(group_size, dim=0)
        v_exp = v_seq.transpose(0, 1).repeat_interleave(group_size, dim=0)
        q_b = q[b].float().unsqueeze(1)
        scores = torch.bmm(q_b, k_exp.float().transpose(1, 2)) * scale
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(probs, v_exp.float()).squeeze(1)
        out[b] = ctx.to(q.dtype)
    return out


@cache
def _load_triton_paged_decode_kernel() -> Any:
    """Compile and cache the portable Triton NHD GQA Paged Decode kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_gqa_kernel(
        q_ptr,  # noqa: ANN001 -- Triton JIT pointer
        k_ptr,  # noqa: ANN001 -- Triton JIT pointer
        v_ptr,  # noqa: ANN001 -- Triton JIT pointer
        indptr_ptr,  # noqa: ANN001 -- Triton JIT pointer
        indices_ptr,  # noqa: ANN001 -- Triton JIT pointer
        last_len_ptr,  # noqa: ANN001 -- Triton JIT pointer
        out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)
        hq = tl.program_id(1)
        q_heads = tl.num_programs(1)
        hk = hq // (q_heads // kv_heads)
        d_offs = tl.arange(0, head_dim)
        p_offs = tl.arange(0, page_size)
        sm_scale = 1.0 / tl.sqrt(float(head_dim))

        q_off = (b * q_heads + hq) * head_dim + d_offs
        q_vec = tl.load(q_ptr + q_off).to(tl.float32)
        start_page = tl.load(indptr_ptr + b)
        end_page = tl.load(indptr_ptr + b + 1)
        last_len = tl.load(last_len_ptr + b)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([head_dim], dtype=tl.float32)
        stride_kp = page_size * kv_heads * head_dim
        stride_kn = kv_heads * head_dim
        stride_kh = head_dim

        for p_iter in range(start_page, end_page):
            phys_page = tl.load(indices_ptr + p_iter)
            valid_toks = tl.where(p_iter + 1 == end_page, last_len, page_size)
            mask_n = p_offs < valid_toks
            kv_base = phys_page * stride_kp + hk * stride_kh
            offs_kv = kv_base + p_offs[:, None] * stride_kn + d_offs[None, :]
            k_tile = tl.load(k_ptr + offs_kv, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptr + offs_kv, mask=mask_n[:, None], other=0.0)
            qk = tl.sum(k_tile.to(tl.float32) * q_vec[None, :], axis=1)
            qk = tl.where(mask_n, qk * sm_scale, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp(qk - m_ij)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(
                p[:, None] * v_tile.to(tl.float32), axis=0
            )
            m_i = m_ij

        tl.store(out_ptr + q_off, (acc / l_i).to(out_ptr.dtype.element_ty))

    return _paged_gqa_kernel


def run_gqa_paged_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> torch.Tensor:
    """Launch the portable Triton NHD GQA Paged Decode kernel."""
    q_c = q.contiguous()
    k_c = k_cache.contiguous()
    v_c = v_cache.contiguous()
    batch, q_heads, head_dim = q_c.shape
    page_size = k_c.shape[1]
    kv_heads = k_c.shape[2]
    out = torch.empty_like(q_c)
    kernel = _load_triton_paged_decode_kernel()
    kernel[(batch, q_heads)](
        q_c,
        k_c,
        v_c,
        kv_indptr.contiguous(),
        kv_indices.contiguous(),
        kv_last_page_len.contiguous(),
        out,
        kv_heads=kv_heads,
        page_size=page_size,
        head_dim=head_dim,
    )
    return out


def _try_splitk_gqa(
    prototype_args: list[Any],
) -> tuple[Callable[..., torch.Tensor], str] | None:
    q, k_cache, _v, _indptr, kv_indices, _last_len = prototype_args
    if not supports_splitk_gqa(q, k_cache):
        return None
    num_cus = (
        int(torch.cuda.get_device_properties(q.device).multi_processor_count)
        if q.is_cuda
        else None
    )
    num_splits = select_gqa_num_splits(
        batch=q.shape[0],
        kv_heads=k_cache.shape[2],
        total_pages=kv_indices.numel(),
        num_cus=num_cus,
    )
    workspace = allocate_splitk_gqa_workspace(q, num_splits)

    def _run_splitk(*args: Any) -> torch.Tensor:
        return run_gqa_paged_decode_splitk(*args, workspace=workspace)

    try:
        _run_splitk(*prototype_args)
        return _run_splitk, "triton_splitk_paged_gqa"
    except Exception:  # noqa: BLE001 -- degrade to single-split kernel
        return None


def make_gqa_paged_decode_runner(
    *,
    prototype_args: list[Any],
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 2e-2,
) -> tuple[Callable[..., torch.Tensor], str]:
    """Select FlashInfer, Split-KV Triton, single-split Triton, or CPU ref."""
    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return ref_gqa_paged_decode, "pytorch_ref"
    if backend in (KernelBackend.AUTO, KernelBackend.VENDOR):
        vendor = try_make_flashinfer_gqa_runner(
            prototype_args,
            device=device,
            ref_fn=ref_gqa_paged_decode,
            max_atol=max_atol,
        )
        if vendor is not None:
            return vendor
    splitk = _try_splitk_gqa(prototype_args)
    if splitk is not None:
        return splitk
    try:
        run_gqa_paged_decode_triton(*prototype_args)
        return run_gqa_paged_decode_triton, "triton_paged_gqa"
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton unavailable
        return ref_gqa_paged_decode, "pytorch_ref_fallback"
