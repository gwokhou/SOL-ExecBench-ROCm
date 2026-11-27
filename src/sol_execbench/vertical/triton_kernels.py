# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""High-occupancy portable Triton kernels for Split-KV GQA decode and Fused MoE."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Protocol, cast

import torch


class TritonKernel(Protocol):
    """Grid-indexable Triton JIT kernel launcher protocol."""

    def __getitem__(self, grid: tuple[int, ...]) -> Callable[..., object]:
        """Bind a Triton launch grid and return the kernel launcher."""
        ...


def select_gqa_num_splits(
    *,
    batch: int,
    kv_heads: int,
    total_pages: int,
    num_cus: int | None = None,
) -> int:
    """Choose a power-of-2 Split-KV factor saturating datacenter CUs/SMs."""
    avg_pages = max(1, total_pages // max(1, batch))
    max_by_len = max(1, avg_pages // 2)
    target_ctas = (
        256
        if (num_cus is None or num_cus <= 0)
        else min(256, max(64, num_cus * 2))
    )
    target = max(1, min(64, target_ctas // max(1, batch * kv_heads)))
    raw = min(max_by_len, target)
    return 1 << (raw.bit_length() - 1)


@cache
def _load_splitk_gqa_stage1() -> TritonKernel:
    """Compile and cache grouped Split-KV GQA Stage 1 kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_gqa_splitk_stage1(
        q_ptr,  # noqa: ANN001 -- Triton JIT pointer
        k_ptr,  # noqa: ANN001 -- Triton JIT pointer
        v_ptr,  # noqa: ANN001 -- Triton JIT pointer
        indptr_ptr,  # noqa: ANN001 -- Triton JIT pointer
        indices_ptr,  # noqa: ANN001 -- Triton JIT pointer
        last_len_ptr,  # noqa: ANN001 -- Triton JIT pointer
        part_out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        part_lse_ptr,  # noqa: ANN001 -- Triton JIT pointer
        group_size: tl.constexpr,
        head_dim: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)
        hk = tl.program_id(1)
        kv_heads = tl.num_programs(1)
        split_id = tl.program_id(2)
        num_splits = tl.num_programs(2)
        q_heads = kv_heads * group_size
        g_offs = tl.arange(0, 16)
        mask_g = g_offs < group_size
        mg = mask_g[:, None]
        hq_offs = hk * group_size + g_offs
        d_offs = tl.arange(0, head_dim)
        p_offs = tl.arange(0, 16)
        start_page = tl.load(indptr_ptr + b)
        end_page = tl.load(indptr_ptr + b + 1)
        last_len = tl.load(last_len_ptr + b)
        pps = tl.cdiv(end_page - start_page, num_splits)
        s_start = start_page + split_id * pps
        s_end = tl.minimum(end_page, s_start + pps)
        lse_off = (b * q_heads + hq_offs) * num_splits + split_id
        if s_start >= s_end:
            neg = tl.full([16], -float("inf"), dtype=tl.float32)
            tl.store(part_lse_ptr + lse_off, neg, mask=mask_g)
            return
        q_off = (b * q_heads + hq_offs[:, None]) * head_dim + d_offs[None, :]
        q_bf = tl.load(q_ptr + q_off, mask=mg, other=0.0).to(tl.bfloat16)
        m_i = tl.full([16], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([16], dtype=tl.float32)
        acc = tl.zeros([16, head_dim], dtype=tl.float32)
        sm_scale = 1.0 / tl.sqrt(float(head_dim))
        s_row = kv_heads * head_dim
        s_page = 16 * s_row
        for p_iter in range(s_start, s_end):
            phys_page = tl.load(indices_ptr + p_iter)
            valid = tl.where(p_iter + 1 == end_page, last_len, 16)
            mask_n = p_offs < valid
            mn = mask_n[:, None]
            base = phys_page * s_page + hk * head_dim
            kv = base + p_offs[:, None] * s_row + d_offs[None, :]
            k_bf = tl.load(k_ptr + kv, mask=mn, other=0.0).to(tl.bfloat16)
            v_bf = tl.load(v_ptr + kv, mask=mn, other=0.0).to(tl.bfloat16)
            scores = tl.dot(q_bf, tl.trans(k_bf)) * sm_scale
            qk = tl.where(mask_n[None, :], scores, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_bf)
            m_i = m_ij
        o_base = (b * q_heads + hq_offs[:, None]) * num_splits + split_id
        o_off = o_base * head_dim + d_offs[None, :]
        tl.store(part_out_ptr + o_off, acc / l_i[:, None], mask=mg)
        tl.store(part_lse_ptr + lse_off, m_i + tl.log(l_i), mask=mask_g)

    return _paged_gqa_splitk_stage1


@cache
def _load_splitk_gqa_reduce() -> TritonKernel:
    """Compile and cache Split-KV GQA Stage 2 LSE reduction kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_gqa_splitk_reduce(
        part_out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        part_lse_ptr,  # noqa: ANN001 -- Triton JIT pointer
        out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        num_splits: tl.constexpr,
        split_pad: tl.constexpr,
        head_dim: tl.constexpr,
    ) -> None:
        b = tl.program_id(0)
        hq = tl.program_id(1)
        q_heads = tl.num_programs(1)
        s_offs = tl.arange(0, split_pad)
        d_offs = tl.arange(0, head_dim)
        mask_s = s_offs < num_splits
        lse_base = (b * q_heads + hq) * num_splits
        lse = tl.load(
            part_lse_ptr + lse_base + s_offs,
            mask=mask_s,
            other=-float("inf"),
        )
        valid_s = lse > -float("inf")
        m_max = tl.max(lse, axis=0)
        w = tl.where(valid_s, tl.exp(lse - m_max), 0.0)
        denom = tl.sum(w, axis=0)
        po_off = (lse_base + s_offs[:, None]) * head_dim + d_offs[None, :]
        po = tl.where(
            valid_s[:, None],
            tl.load(part_out_ptr + po_off, mask=mask_s[:, None], other=0.0),
            0.0,
        )
        acc = tl.sum(w[:, None] * po, axis=0) / denom
        out_off = (b * q_heads + hq) * head_dim + d_offs
        tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty))

    return _paged_gqa_splitk_reduce


def allocate_splitk_gqa_workspace(
    q: torch.Tensor,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate Split-KV partial output and log-sum-exp buffers."""
    batch, q_heads, head_dim = q.shape
    part_out = torch.empty(
        (batch, q_heads, num_splits, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    part_lse = torch.empty(
        (batch, q_heads, num_splits),
        dtype=torch.float32,
        device=q.device,
    )
    return part_out, part_lse


def supports_splitk_gqa(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Return True when grouped Split-KV Triton decode handles the tensor layout."""
    _, q_heads, head_dim = q.shape
    page_size = k_cache.shape[1]
    kv_heads = k_cache.shape[2]
    if page_size != 16 or kv_heads <= 0 or q_heads % kv_heads != 0:
        return False
    group_size = q_heads // kv_heads
    return 1 <= group_size <= 16 and head_dim in {16, 32, 64, 128, 256}


def run_gqa_paged_decode_splitk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    *,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute grouped Split-KV GQA Paged Decode in two Triton passes."""
    if not supports_splitk_gqa(q, k_cache):
        raise ValueError("Shape unsupported by grouped Split-KV Triton kernel")
    q_c = q.contiguous()
    k_c = k_cache.contiguous()
    v_c = v_cache.contiguous()
    batch, q_heads, head_dim = q_c.shape
    kv_heads = k_c.shape[2]
    group_size = q_heads // kv_heads
    num_cus = (
        int(torch.cuda.get_device_properties(q_c.device).multi_processor_count)
        if q_c.is_cuda
        else None
    )
    num_splits = select_gqa_num_splits(
        batch=batch,
        kv_heads=kv_heads,
        total_pages=kv_indices.numel(),
        num_cus=num_cus,
    )
    expected_shape = (batch, q_heads, num_splits, head_dim)
    if (
        workspace is not None
        and workspace[0].shape == expected_shape
        and workspace[0].device == q_c.device
    ):
        part_out, part_lse = workspace
    else:
        part_out, part_lse = allocate_splitk_gqa_workspace(q_c, num_splits)

    stage1 = _load_splitk_gqa_stage1()
    stage2 = _load_splitk_gqa_reduce()
    stage1[(batch, kv_heads, num_splits)](
        q_c,
        k_c,
        v_c,
        kv_indptr.contiguous(),
        kv_indices.contiguous(),
        kv_last_page_len.contiguous(),
        part_out,
        part_lse,
        group_size=group_size,
        head_dim=head_dim,
    )
    out = torch.empty_like(q_c)
    split_pad = 1 << (num_splits - 1).bit_length()
    stage2[(batch, q_heads)](
        part_out,
        part_lse,
        out,
        num_splits=num_splits,
        split_pad=split_pad,
        head_dim=head_dim,
    )
    return out


@cache
def _load_triton_paged_decode_kernel() -> TritonKernel:
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


def _build_moe_tile_schedule(
    gate_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build zero-sync GPU expert-grouped 16-row tile tables for Fused MoE."""
    probs = torch.softmax(gate_logits.float(), dim=-1)
    topk_w, topk_ids = torch.topk(probs, k=int(top_k), dim=-1)
    flat_weights = (
        (topk_w / topk_w.sum(dim=-1, keepdim=True)).reshape(-1).contiguous()
    )
    flat_experts = topk_ids.reshape(-1)
    order = torch.argsort(flat_experts).to(torch.int32)
    sorted_experts = flat_experts[order]
    num_experts = gate_logits.shape[1]
    counts = torch.bincount(
        sorted_experts.to(torch.int64), minlength=num_experts
    ).to(torch.int32)
    exp_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=gate_logits.device
    )
    torch.cumsum(counts, dim=0, out=exp_offsets[1:])
    tiles_per_exp = (counts + 15) // 16
    tile_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=gate_logits.device
    )
    torch.cumsum(tiles_per_exp, dim=0, out=tile_offsets[1:])
    n_pairs = flat_experts.numel()
    max_tiles = max(1, min(n_pairs, (n_pairs // 16) + num_experts))
    tile_ids = torch.arange(
        max_tiles, dtype=torch.int32, device=gate_logits.device
    )
    tile_expert = torch.searchsorted(tile_offsets[1:], tile_ids, right=True).to(
        torch.int32
    )
    clamped_e = tile_expert.clamp(max=num_experts - 1)
    local_tile = tile_ids - tile_offsets[clamped_e]
    row_offs = torch.arange(16, dtype=torch.int32, device=gate_logits.device)
    pos = (
        exp_offsets[clamped_e, None]
        + local_tile[:, None] * 16
        + row_offs[None, :]
    )
    exp_end = exp_offsets[clamped_e + 1, None]
    valid = (tile_expert[:, None] < num_experts) & (pos < exp_end)
    safe_pos = pos.clamp(min=0, max=max(0, n_pairs - 1))
    tile_pairs = torch.where(valid, order[safe_pos], -1).to(torch.int32)
    return flat_weights, tile_expert.contiguous(), tile_pairs.contiguous()


@cache
def _load_moe_gate_up_kernel() -> TritonKernel:
    """Compile and cache fused SwiGLU gate/up MoE kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _moe_gate_up_swiglu_kernel(
        tokens_ptr,  # noqa: ANN001 -- Triton JIT pointer
        gate_w_ptr,  # noqa: ANN001 -- Triton JIT pointer
        up_w_ptr,  # noqa: ANN001 -- Triton JIT pointer
        tile_expert_ptr,  # noqa: ANN001 -- Triton JIT pointer
        tile_pairs_ptr,  # noqa: ANN001 -- Triton JIT pointer
        inter_out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        hidden_dim: tl.constexpr,
        inter_dim: tl.constexpr,
        num_experts: tl.constexpr,
        top_k: tl.constexpr,
    ) -> None:
        tid = tl.program_id(0)
        n_block = tl.program_id(1)
        e = tl.load(tile_expert_ptr + tid)
        if e >= num_experts:
            return
        pairs = tl.load(tile_pairs_ptr + tid * 16 + tl.arange(0, 16))
        mask_m = pairs >= 0
        safe_pairs = tl.where(mask_m, pairs, 0)
        tok_ids = safe_pairs // top_k
        n_offs = n_block * 64 + tl.arange(0, 64)
        mask_n = n_offs < inter_dim
        acc_gate = tl.zeros([16, 64], dtype=tl.float32)
        acc_up = tl.zeros([16, 64], dtype=tl.float32)
        w_base = e * hidden_dim * inter_dim

        for k_start in range(0, hidden_dim, 64):
            k_offs = k_start + tl.arange(0, 64)
            mask_k = k_offs < hidden_dim
            a = tl.load(
                tokens_ptr + tok_ids[:, None] * hidden_dim + k_offs[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            w_off = w_base + k_offs[:, None] * inter_dim + n_offs[None, :]
            wg = tl.load(
                gate_w_ptr + w_off,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            wu = tl.load(
                up_w_ptr + w_off,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc_gate += tl.dot(a, wg)
            acc_up += tl.dot(a, wu)

        silu_g = acc_gate / (1.0 + tl.exp(-acc_gate))
        val = (silu_g * acc_up).to(inter_out_ptr.dtype.element_ty)
        out_off = safe_pairs[:, None] * inter_dim + n_offs[None, :]
        tl.store(
            inter_out_ptr + out_off,
            val,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    return cast(TritonKernel, _moe_gate_up_swiglu_kernel)


@cache
def _load_moe_down_kernel() -> TritonKernel:
    """Compile and cache weighted MoE down-projection kernel."""
    import triton
    import triton.language as tl

    @triton.jit
    def _moe_down_project_kernel(
        inter_in_ptr,  # noqa: ANN001 -- Triton JIT pointer
        down_w_ptr,  # noqa: ANN001 -- Triton JIT pointer
        weights_ptr,  # noqa: ANN001 -- Triton JIT pointer
        tile_expert_ptr,  # noqa: ANN001 -- Triton JIT pointer
        tile_pairs_ptr,  # noqa: ANN001 -- Triton JIT pointer
        pair_out_ptr,  # noqa: ANN001 -- Triton JIT pointer
        hidden_dim: tl.constexpr,
        inter_dim: tl.constexpr,
        num_experts: tl.constexpr,
    ) -> None:
        tid = tl.program_id(0)
        n_block = tl.program_id(1)
        e = tl.load(tile_expert_ptr + tid)
        if e >= num_experts:
            return
        pairs = tl.load(tile_pairs_ptr + tid * 16 + tl.arange(0, 16))
        mask_m = pairs >= 0
        safe_pairs = tl.where(mask_m, pairs, 0)
        rw = tl.load(weights_ptr + safe_pairs, mask=mask_m, other=0.0)
        n_offs = n_block * 64 + tl.arange(0, 64)
        mask_n = n_offs < hidden_dim
        acc = tl.zeros([16, 64], dtype=tl.float32)
        w_base = e * inter_dim * hidden_dim

        for k_start in range(0, inter_dim, 64):
            k_offs = k_start + tl.arange(0, 64)
            mask_k = k_offs < inter_dim
            a = tl.load(
                inter_in_ptr
                + safe_pairs[:, None] * inter_dim
                + k_offs[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            wd = tl.load(
                down_w_ptr
                + w_base
                + k_offs[:, None] * hidden_dim
                + n_offs[None, :],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            acc += tl.dot(a, wd)

        out_off = safe_pairs[:, None] * hidden_dim + n_offs[None, :]
        tl.store(
            pair_out_ptr + out_off,
            acc * rw[:, None],
            mask=mask_m[:, None] & mask_n[None, :],
        )

    return cast(TritonKernel, _moe_down_project_kernel)


def allocate_fused_moe_workspace(
    tokens: torch.Tensor,
    inter_dim: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-allocate Fused MoE intermediate activation and weighted output buffers."""
    n_pairs = tokens.shape[0] * int(top_k)
    hidden_dim = tokens.shape[1]
    out_dtype = (
        tokens.dtype
        if tokens.dtype in (torch.float16, torch.bfloat16)
        else torch.bfloat16
    )
    inter_buf = torch.empty(
        (n_pairs, inter_dim), dtype=out_dtype, device=tokens.device
    )
    pair_out = torch.empty(
        (n_pairs, hidden_dim), dtype=torch.float32, device=tokens.device
    )
    return inter_buf, pair_out


def run_fused_moe_triton(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    gate_logits: torch.Tensor,
    top_k: int,
    *,
    workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Execute zero-sync Triton Fused SwiGLU MoE without atomic contention."""
    import triton

    m, hidden_dim = tokens.shape
    num_experts, _, inter_dim = gate_weights.shape
    n_pairs = m * int(top_k)
    flat_weights, tile_expert, tile_pairs = _build_moe_tile_schedule(
        gate_logits, int(top_k)
    )
    max_tiles = tile_expert.shape[0]
    if (
        workspace is not None
        and workspace[0].shape == (n_pairs, inter_dim)
        and workspace[1].shape == (n_pairs, hidden_dim)
        and workspace[0].device == tokens.device
    ):
        inter_buf, pair_out = workspace
    else:
        inter_buf, pair_out = allocate_fused_moe_workspace(
            tokens, inter_dim, int(top_k)
        )

    k_gate_up = _load_moe_gate_up_kernel()
    k_down = _load_moe_down_kernel()
    k_gate_up[(max_tiles, triton.cdiv(inter_dim, 64))](
        tokens.contiguous(),
        gate_weights.contiguous(),
        up_weights.contiguous(),
        tile_expert,
        tile_pairs,
        inter_buf,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_experts=num_experts,
        top_k=int(top_k),
    )
    k_down[(max_tiles, triton.cdiv(hidden_dim, 64))](
        inter_buf,
        down_weights.contiguous(),
        flat_weights,
        tile_expert,
        tile_pairs,
        pair_out,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_experts=num_experts,
    )
    out_dtype = (
        tokens.dtype
        if tokens.dtype in (torch.float16, torch.bfloat16)
        else torch.bfloat16
    )
    return pair_out.view(m, int(top_k), hidden_dim).sum(dim=1).to(out_dtype)
