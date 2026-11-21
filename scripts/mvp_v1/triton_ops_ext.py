# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Portable Tier-1 Triton kernels for MLA decode, Add-RMSNorm+RoPE, and SwiGLU-FP8."""

from __future__ import annotations

from functools import cache
from typing import Any

import torch

from scripts.mvp_v1.triton_kernels import (
    _load_splitk_gqa_reduce,
    select_gqa_num_splits,
)


@cache
def _load_mla_splitk_stage1() -> Any:
    """Compile and cache Split-KV Absorbed MLA Paged Decode Stage 1 kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _mla_splitk_stage1(
        q_ptr,  # noqa: ANN001 -- Triton JIT pointer
        kv_ptr,  # noqa: ANN001 -- Triton JIT pointer
        indptr_ptr,  # noqa: ANN001 -- Triton JIT pointer
        indices_ptr,  # noqa: ANN001 -- Triton JIT pointer
        last_len_ptr,  # noqa: ANN001 -- Triton JIT pointer
        part_out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        part_lse_ptr,  # noqa: ANN001 -- Triton JIT pointer
        q_heads: tl.constexpr,
        d_ckv: tl.constexpr,
        d_kpe: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)
        split_id = tl.program_id(1)
        num_splits = tl.num_programs(1)
        d_total = d_ckv + d_kpe
        h_offs = tl.arange(0, q_heads)
        ckv_offs = tl.arange(0, d_ckv)
        kpe_offs = tl.arange(0, d_kpe)
        p_offs = tl.arange(0, 16)
        start_page = tl.load(indptr_ptr + b)
        end_page = tl.load(indptr_ptr + b + 1)
        last_len = tl.load(last_len_ptr + b)
        pps = tl.cdiv(end_page - start_page, num_splits)
        s_start = start_page + split_id * pps
        s_end = tl.minimum(end_page, s_start + pps)
        lse_off = (b * q_heads + h_offs) * num_splits + split_id
        if s_start >= s_end:
            neg = tl.full([q_heads], -float("inf"), dtype=tl.float32)
            tl.store(part_lse_ptr + lse_off, neg)
            return
        q_base = (b * q_heads + h_offs[:, None]) * d_total
        q_ckv = tl.load(q_ptr + q_base + ckv_offs[None, :]).to(tl.bfloat16)
        q_kpe = tl.load(q_ptr + q_base + d_ckv + kpe_offs[None, :]).to(
            tl.bfloat16
        )
        m_i = tl.full([q_heads], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([q_heads], dtype=tl.float32)
        acc = tl.zeros([q_heads, d_ckv], dtype=tl.float32)
        sm_scale = 1.0 / tl.sqrt(d_total * 1.0)
        for p_iter in range(s_start, s_end):
            phys_page = tl.load(indices_ptr + p_iter)
            valid = tl.where(p_iter + 1 == end_page, last_len, 16)
            mask_n = p_offs < valid
            mn = mask_n[:, None]
            kv_base = phys_page * (16 * d_total) + p_offs[:, None] * d_total
            c_kv = tl.load(
                kv_ptr + kv_base + ckv_offs[None, :], mask=mn, other=0.0
            ).to(tl.bfloat16)
            k_pe = tl.load(
                kv_ptr + kv_base + d_ckv + kpe_offs[None, :], mask=mn, other=0.0
            ).to(tl.bfloat16)
            scores = (
                tl.dot(q_ckv, tl.trans(c_kv)) + tl.dot(q_kpe, tl.trans(k_pe))
            ) * sm_scale
            qk = tl.where(mask_n[None, :], scores, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), c_kv)
            m_i = m_ij
        o_off = ((b * q_heads + h_offs[:, None]) * num_splits + split_id) * (
            d_ckv
        ) + ckv_offs[None, :]
        tl.store(part_out_ptr + o_off, acc / l_i[:, None])
        tl.store(part_lse_ptr + lse_off, m_i + tl.log(l_i))

    return _mla_splitk_stage1


def supports_splitk_mla(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    d_ckv: int,
) -> bool:
    """Return True when Split-KV Triton MLA supports the latent KV layout."""
    _, q_heads, d_total = q.shape
    page_size = kv_cache.shape[1]
    d_kpe = d_total - d_ckv
    return (
        page_size == 16
        and kv_cache.shape[2] == d_total
        and q_heads in {16, 32, 64}
        and d_ckv in {64, 128, 256, 512}
        and d_kpe in {16, 32, 64, 128}
    )


def allocate_splitk_mla_workspace(
    q: torch.Tensor,
    d_ckv: int,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate Split-KV partial output and log-sum-exp buffers for MLA."""
    batch, q_heads, _ = q.shape
    part_out = torch.empty(
        (batch, q_heads, num_splits, d_ckv),
        dtype=torch.float32,
        device=q.device,
    )
    part_lse = torch.empty(
        (batch, q_heads, num_splits),
        dtype=torch.float32,
        device=q.device,
    )
    return part_out, part_lse


def run_mla_paged_decode_splitk(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    d_ckv: int,
    *,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute absorbed Split-KV MLA Paged Decode in two Triton passes."""
    if not supports_splitk_mla(q, kv_cache, d_ckv):
        raise ValueError("Shape unsupported by Split-KV Triton MLA kernel")
    q_c = q.contiguous()
    kv_c = kv_cache.contiguous()
    batch, q_heads, d_total = q_c.shape
    d_kpe = d_total - d_ckv
    num_cus = (
        int(torch.cuda.get_device_properties(q_c.device).multi_processor_count)
        if q_c.is_cuda
        else None
    )
    num_splits = select_gqa_num_splits(
        batch=batch,
        kv_heads=1,
        total_pages=kv_indices.numel(),
        num_cus=num_cus,
    )
    expected_shape = (batch, q_heads, num_splits, d_ckv)
    if (
        workspace is not None
        and workspace[0].shape == expected_shape
        and workspace[0].device == q_c.device
    ):
        part_out, part_lse = workspace
    else:
        part_out, part_lse = allocate_splitk_mla_workspace(
            q_c, d_ckv, num_splits
        )

    stage1 = _load_mla_splitk_stage1()
    stage2 = _load_splitk_gqa_reduce()
    stage1[(batch, num_splits)](
        q_c,
        kv_c,
        kv_indptr.contiguous(),
        kv_indices.contiguous(),
        kv_last_page_len.contiguous(),
        part_out,
        part_lse,
        q_heads=q_heads,
        d_ckv=d_ckv,
        d_kpe=d_kpe,
    )
    out = torch.empty(
        (batch, q_heads, d_ckv), dtype=q_c.dtype, device=q_c.device
    )
    split_pad = 1 << (num_splits - 1).bit_length()
    stage2[(batch, q_heads)](
        part_out,
        part_lse,
        out,
        num_splits=num_splits,
        split_pad=split_pad,
        head_dim=d_ckv,
    )
    return out


@cache
def _load_rmsnorm_rope_kernel() -> Any:
    """Compile and cache single-pass Fused Add-RMSNorm + NeoX RoPE kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_rmsnorm_rope_kernel(
        x_ptr,  # noqa: ANN001 -- Triton JIT pointer
        res_ptr,  # noqa: ANN001 -- Triton JIT pointer
        w_ptr,  # noqa: ANN001 -- Triton JIT pointer
        cs_ptr,  # noqa: ANN001 -- Triton JIT pointer
        out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        stride_m: tl.constexpr,
        eps,  # noqa: ANN001 -- scalar float
        d2: tl.constexpr,
    ) -> None:
        m = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, d2)
        d_full = 2 * d2
        tok_off = m * stride_m + h * d_full
        x0 = tl.load(x_ptr + tok_off + offs).to(tl.float32)
        x1 = tl.load(x_ptr + tok_off + d2 + offs).to(tl.float32)
        r0 = tl.load(res_ptr + tok_off + offs).to(tl.float32)
        r1 = tl.load(res_ptr + tok_off + d2 + offs).to(tl.float32)
        h0 = x0 + r0
        h1 = x1 + r1
        mean_sq = tl.sum(h0 * h0 + h1 * h1, axis=0) / d_full
        rms = tl.rsqrt(mean_sq + eps)
        w0 = tl.load(w_ptr + offs).to(tl.float32)
        w1 = tl.load(w_ptr + d2 + offs).to(tl.float32)
        u = h0 * rms * w0
        v = h1 * rms * w1
        cos_v = tl.load(cs_ptr + m * d_full + offs).to(tl.float32)
        sin_v = tl.load(cs_ptr + m * d_full + d2 + offs).to(tl.float32)
        out_dt = out_ptr.dtype.element_ty
        tl.store(out_ptr + tok_off + offs, (u * cos_v - v * sin_v).to(out_dt))
        tl.store(
            out_ptr + tok_off + d2 + offs, (v * cos_v + u * sin_v).to(out_dt)
        )

    return _fused_rmsnorm_rope_kernel


def run_fused_rmsnorm_rope_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Execute single-pass Add + RMSNorm + NeoX RoPE in Triton."""
    m, h, d = x.shape
    if d % 2 != 0 or (d // 2) not in {16, 32, 64, 128}:
        raise ValueError(f"Unsupported head_dim {d} for Triton RMSNorm+RoPE")
    x_c = x.contiguous()
    res_c = residual.contiguous()
    w_c = weight.contiguous()
    cs_c = cos_sin.contiguous()
    out = torch.empty_like(x_c)
    kernel = _load_rmsnorm_rope_kernel()
    kernel[(m, h)](
        x_c, res_c, w_c, cs_c, out, stride_m=h * d, eps=eps, d2=d // 2
    )
    return out


@cache
def _load_swiglu_fp8_quant_kernel() -> Any:
    """Compile and cache row-wise Fused SwiGLU + Dynamic FP8 Quant kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fp8_quant_kernel(
        gu_ptr,  # noqa: ANN001 -- Triton JIT pointer
        y_ptr,  # noqa: ANN001 -- Triton JIT pointer
        s_ptr,  # noqa: ANN001 -- Triton JIT pointer
        inter: tl.constexpr,
        fp8_max,  # noqa: ANN001 -- scalar float
        block_i: tl.constexpr,
    ) -> None:
        m = tl.program_id(0)
        row_in = m * (2 * inter)
        row_out = m * inter
        amax = 0.0
        for start in tl.range(0, inter, block_i):
            offs = start + tl.arange(0, block_i)
            mask = offs < inter
            g = tl.load(gu_ptr + row_in + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            u = tl.load(
                gu_ptr + row_in + inter + offs, mask=mask, other=0.0
            ).to(tl.float32)
            act = (g * tl.sigmoid(g)) * u
            amax = tl.maximum(amax, tl.max(tl.abs(act), axis=0))
        scale = tl.maximum(amax, 1e-12) / fp8_max
        tl.store(s_ptr + m, scale)
        inv_scale = 1.0 / scale
        for start in tl.range(0, inter, block_i):
            offs = start + tl.arange(0, block_i)
            mask = offs < inter
            g = tl.load(gu_ptr + row_in + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            u = tl.load(
                gu_ptr + row_in + inter + offs, mask=mask, other=0.0
            ).to(tl.float32)
            q = (g * tl.sigmoid(g)) * u * inv_scale
            q_clamped = tl.maximum(tl.minimum(q, fp8_max), -fp8_max)
            tl.store(
                y_ptr + row_out + offs,
                q_clamped.to(y_ptr.dtype.element_ty),
                mask=mask,
            )

    return _swiglu_fp8_quant_kernel


def run_swiglu_fp8_quant_triton(
    gate_up: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute fused SwiGLU + dynamic per-token FP8 quantization in Triton."""
    gu_c = gate_up.contiguous()
    m, i2 = gu_c.shape
    inter = i2 // 2
    fp8_max = float(torch.finfo(fp8_dtype).max)
    y_fp8 = torch.empty((m, inter), dtype=fp8_dtype, device=gu_c.device)
    scales = torch.empty((m, 1), dtype=torch.float32, device=gu_c.device)
    pow2 = 1 << max(0, inter - 1).bit_length()
    block_i = min(2048, max(64, pow2))
    kernel = _load_swiglu_fp8_quant_kernel()
    kernel[(m,)](
        gu_c,
        y_fp8,
        scales,
        inter=inter,
        fp8_max=fp8_max,
        block_i=block_i,
    )
    return y_fp8, scales
