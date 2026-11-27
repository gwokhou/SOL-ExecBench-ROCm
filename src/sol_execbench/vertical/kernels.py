# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm software-stack kernels and PyTorch references for all 6 primitive families."""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import StrEnum

import torch

from sol_execbench.vertical.adapters import (
    PreparedRunner,
    try_make_aiter_gemm_runner,
    try_make_aiter_gqa_runner,
    try_make_aiter_mla_runner,
    try_make_aiter_moe_runner,
    try_make_aiter_rmsnorm_rope_runner,
    try_make_aiter_swiglu_quant_runner,
    try_make_vllm_gemm_runner,
    try_make_vllm_gqa_runner,
    try_make_vllm_mla_runner,
    try_make_vllm_moe_runner,
    try_make_vllm_rmsnorm_rope_runner,
    try_make_vllm_swiglu_quant_runner,
)
from sol_execbench.vertical.adapters.common import _as_int, _as_tensor
from sol_execbench.vertical.triton_kernels import (
    allocate_fused_moe_workspace,
    allocate_splitk_gqa_workspace,
    run_fused_moe_triton,
    run_gqa_paged_decode_splitk,
    run_gqa_paged_decode_triton,
    select_gqa_num_splits,
    supports_splitk_gqa,
)
from sol_execbench.vertical.triton_ops_ext import (
    allocate_splitk_mla_workspace,
    run_fused_rmsnorm_rope_triton,
    run_mla_paged_decode_splitk,
    run_swiglu_fp8_quant_triton,
    supports_splitk_mla,
)


class KernelBackend(StrEnum):
    """ROCm software-stack tier selection for vertical SOL diagnosis."""

    AUTO = "auto"
    TIER1_PORTABLE = "portable"
    TIER2_VLLM = "vllm"
    TIER3_AITER = "aiter"
    PYTORCH = "pytorch"


def _cpu_unavailable(prototype_args: list[object]) -> PreparedRunner:
    return PreparedRunner(
        fn=None,
        implementation_name="unavailable:cpu_device",
        candidate_args=prototype_args,
        unavailable_reason="cpu_device",
    )


def _dispatch_vendor_tiers(
    prototype_args: list[object],
    *,
    device: str,
    backend: KernelBackend,
    make_aiter: Callable[[], PreparedRunner],
    make_vllm: Callable[[], PreparedRunner],
) -> PreparedRunner | None:
    if backend is KernelBackend.TIER3_AITER:
        return (
            _cpu_unavailable(prototype_args)
            if device == "cpu"
            else make_aiter()
        )
    if backend is KernelBackend.TIER2_VLLM:
        return (
            _cpu_unavailable(prototype_args) if device == "cpu" else make_vllm()
        )
    if backend is KernelBackend.AUTO and device != "cpu":
        aiter_res = make_aiter()
        if aiter_res.fn is not None:
            return aiter_res
        vllm_res = make_vllm()
        if vllm_res.fn is not None:
            return vllm_res
    return None


def ref_skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference BF16 output for Skinny GEMM (also used for FP8 dequant ref)."""
    out = x.float() @ weight.float()
    if scale_a is not None:
        out = out * scale_a.float()
    if scale_b is not None:
        sb_f = scale_b.float()
        if (
            sb_f.ndim == 2
            and sb_f.shape == (out.shape[1], 1)
            and out.shape[0] != out.shape[1]
        ):
            sb_f = sb_f.t()
        out = out * sb_f
    return out.to(torch.bfloat16)


def _probe_fp8_candidate(
    cand_fn: Callable[..., torch.Tensor],
    prototype_args: list[object],
    expected_shape: tuple[int, int],
) -> bool:
    try:
        with torch.no_grad():
            out = cand_fn(*prototype_args)
        return (
            isinstance(out, torch.Tensor)
            and tuple(out.shape) == expected_shape
            and bool(torch.isfinite(out).all().item())
        )
    except Exception:  # noqa: BLE001 -- hipBLASLt shape probe
        return False


def _make_dequant_fp8_gemm(
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> Callable[..., torch.Tensor]:
    combined_scale = scale_a.float() * scale_b.float()

    def _run_dequant(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return (
            (x.to(torch.bfloat16) @ weight_t.to(torch.bfloat16)).float()
            * combined_scale
        ).to(torch.bfloat16)

    return _run_dequant


def _select_gpu_fp8_gemm_runner(
    prototype_args: list[object],
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> tuple[Callable[..., torch.Tensor], str]:
    x0, w_t0 = _as_tensor(prototype_args[0]), _as_tensor(prototype_args[1])
    m, k = int(x0.shape[0]), int(x0.shape[1])
    expected_shape = (m, int(w_t0.shape[1]))

    def _run_fp8(x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return torch._scaled_mm(
            x,
            weight_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    if _probe_fp8_candidate(_run_fp8, prototype_args, expected_shape):
        return _run_fp8, "torch._scaled_mm"

    if m % 16 != 0:
        m_pad = math.ceil(m / 16) * 16
        pad_x = torch.zeros((m_pad, k), dtype=x0.dtype, device=x0.device)

        def _run_fp8_pad16(
            x: torch.Tensor, weight_t: torch.Tensor
        ) -> torch.Tensor:
            rows = x.shape[0]
            pad_x[:rows].copy_(x)
            return torch._scaled_mm(
                pad_x,
                weight_t,
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )[:rows]

        if _probe_fp8_candidate(_run_fp8_pad16, prototype_args, expected_shape):
            return _run_fp8_pad16, "torch._scaled_mm_pad16"

    return (
        _make_dequant_fp8_gemm(scale_a, scale_b),
        "bf16_dequant_matmul_fallback",
    )


def make_skinny_gemm_runner(
    *,
    prototype_args: list[object],
    fp8: bool,
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    max_atol: float = 5e-1,
) -> PreparedRunner:
    """Build candidate runner for Tier 1..3 ROCm stacks or pure PyTorch."""
    vendor_res = _dispatch_vendor_tiers(
        prototype_args,
        device=device,
        backend=backend,
        make_aiter=lambda: try_make_aiter_gemm_runner(
            prototype_args,
            fp8=fp8,
            scale_a=scale_a,
            scale_b=scale_b,
            ref_fn=ref_skinny_gemm,
            max_atol=max_atol,
        ),
        make_vllm=lambda: try_make_vllm_gemm_runner(
            prototype_args,
            fp8=fp8,
            scale_a=scale_a,
            scale_b=scale_b,
            ref_fn=ref_skinny_gemm,
            max_atol=max_atol,
        ),
    )
    if vendor_res is not None:
        return vendor_res

    if not fp8:
        return PreparedRunner(
            fn=torch.matmul,
            implementation_name="torch.matmul",
            candidate_args=prototype_args,
        )
    if device == "cpu" or scale_a is None or scale_b is None:
        return PreparedRunner(
            fn=lambda x, w_t: ref_skinny_gemm(
                x, w_t, scale_a=scale_a, scale_b=scale_b
            ),
            implementation_name="cpu_fp8_ref",
            candidate_args=prototype_args,
        )
    if backend is KernelBackend.PYTORCH:
        return PreparedRunner(
            fn=_make_dequant_fp8_gemm(scale_a, scale_b),
            implementation_name="bf16_dequant_matmul",
            candidate_args=prototype_args,
        )
    fn, impl = _select_gpu_fp8_gemm_runner(
        prototype_args,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    return PreparedRunner(
        fn=fn, implementation_name=impl, candidate_args=prototype_args
    )


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
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
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
    tokens_bf = tokens.to(torch.bfloat16)
    gate_bf = gate_weights.to(torch.bfloat16)
    up_bf = up_weights.to(torch.bfloat16)
    down_bf = down_weights.to(torch.bfloat16)
    offset = 0
    for eid_t, count_t in zip(unique_e.tolist(), counts.tolist(), strict=True):
        sl = slice(offset, offset + count_t)
        t_idx = sorted_tokens[sl]
        w_e = sorted_weights[sl].unsqueeze(-1)
        x_e = tokens_bf[t_idx]
        hidden = torch.nn.functional.silu(x_e @ gate_bf[eid_t]) * (
            x_e @ up_bf[eid_t]
        )
        contrib = (hidden @ down_bf[eid_t]).float() * w_e.float()
        out.index_add_(0, t_idx, contrib)
        offset += count_t
    return out.to(torch.bfloat16)


def make_fused_moe_runner(
    *,
    prototype_args: list[object],
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 5e-2,
) -> PreparedRunner:
    """Select Tier 3 AITER, Tier 2 vLLM, Tier 1 Triton, or grouped PyTorch."""
    vendor_res = _dispatch_vendor_tiers(
        prototype_args,
        device=device,
        backend=backend,
        make_aiter=lambda: try_make_aiter_moe_runner(
            prototype_args, ref_fn=ref_fused_moe, max_atol=max_atol
        ),
        make_vllm=lambda: try_make_vllm_moe_runner(
            prototype_args, ref_fn=ref_fused_moe, max_atol=max_atol
        ),
    )
    if vendor_res is not None:
        return vendor_res

    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return PreparedRunner(
            fn=run_fused_moe_grouped,
            implementation_name="grouped_swiglu_moe",
            candidate_args=prototype_args,
        )

    tokens = _as_tensor(prototype_args[0])
    gate_w = _as_tensor(prototype_args[1])
    top_k = _as_int(prototype_args[5])
    workspace = allocate_fused_moe_workspace(tokens, gate_w.shape[2], top_k)

    def _run_triton_moe(
        tokens_in: torch.Tensor,
        gate_w_in: torch.Tensor,
        up_w_in: torch.Tensor,
        down_w_in: torch.Tensor,
        gate_logits_in: torch.Tensor,
        top_k_in: int,
    ) -> torch.Tensor:
        return run_fused_moe_triton(
            tokens_in,
            gate_w_in,
            up_w_in,
            down_w_in,
            gate_logits_in,
            top_k_in,
            workspace=workspace,
        )

    try:
        _run_triton_moe(
            tokens,
            gate_w,
            _as_tensor(prototype_args[2]),
            _as_tensor(prototype_args[3]),
            _as_tensor(prototype_args[4]),
            top_k,
        )
        return PreparedRunner(
            fn=_run_triton_moe,
            implementation_name="triton_fused_swiglu_moe",
            candidate_args=prototype_args,
        )
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton fails
        return PreparedRunner(
            fn=run_fused_moe_grouped,
            implementation_name="grouped_swiglu_moe",
            candidate_args=prototype_args,
        )


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


def _try_splitk_gqa(
    prototype_args: list[object],
) -> tuple[Callable[..., torch.Tensor], str] | None:
    q = _as_tensor(prototype_args[0])
    k_cache = _as_tensor(prototype_args[1])
    kv_indices = _as_tensor(prototype_args[4])
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

    def _run_splitk(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        indptr_in: torch.Tensor,
        indices_in: torch.Tensor,
        last_len_in: torch.Tensor,
    ) -> torch.Tensor:
        return run_gqa_paged_decode_splitk(
            q_in,
            k_in,
            v_in,
            indptr_in,
            indices_in,
            last_len_in,
            workspace=workspace,
        )

    try:
        _run_splitk(
            q,
            k_cache,
            _as_tensor(prototype_args[2]),
            _as_tensor(prototype_args[3]),
            kv_indices,
            _as_tensor(prototype_args[5]),
        )
        return _run_splitk, "triton_splitk_paged_gqa"
    except Exception:  # noqa: BLE001 -- degrade to single-split kernel
        return None


def make_gqa_paged_decode_runner(
    *,
    prototype_args: list[object],
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 2e-2,
) -> PreparedRunner:
    """Select Tier 3 AITER, Tier 2 vLLM, Split-KV Triton, or PyTorch ref."""
    vendor_res = _dispatch_vendor_tiers(
        prototype_args,
        device=device,
        backend=backend,
        make_aiter=lambda: try_make_aiter_gqa_runner(
            prototype_args, ref_fn=ref_gqa_paged_decode, max_atol=max_atol
        ),
        make_vllm=lambda: try_make_vllm_gqa_runner(
            prototype_args, ref_fn=ref_gqa_paged_decode, max_atol=max_atol
        ),
    )
    if vendor_res is not None:
        return vendor_res

    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return PreparedRunner(
            fn=ref_gqa_paged_decode,
            implementation_name="pytorch_ref",
            candidate_args=prototype_args,
        )
    splitk = _try_splitk_gqa(prototype_args)
    if splitk is not None:
        return PreparedRunner(
            fn=splitk[0],
            implementation_name=splitk[1],
            candidate_args=prototype_args,
        )
    try:
        run_gqa_paged_decode_triton(
            _as_tensor(prototype_args[0]),
            _as_tensor(prototype_args[1]),
            _as_tensor(prototype_args[2]),
            _as_tensor(prototype_args[3]),
            _as_tensor(prototype_args[4]),
            _as_tensor(prototype_args[5]),
        )
        return PreparedRunner(
            fn=run_gqa_paged_decode_triton,
            implementation_name="triton_paged_gqa",
            candidate_args=prototype_args,
        )
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton unavailable
        return PreparedRunner(
            fn=ref_gqa_paged_decode,
            implementation_name="pytorch_ref_fallback",
            candidate_args=prototype_args,
        )


def ref_mla_paged_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    d_ckv: int,
) -> torch.Tensor:
    """Reference FP32 absorbed MLA Paged Decode -> BF16."""
    batch, q_heads, d_total = q.shape
    page_size = kv_cache.shape[1]
    d_ckv_i = int(d_ckv)
    scale = 1.0 / math.sqrt(d_total)
    out = torch.empty((batch, q_heads, d_ckv_i), dtype=q.dtype, device=q.device)
    for b in range(batch):
        p0 = int(kv_indptr[b].item())
        p1 = int(kv_indptr[b + 1].item())
        pages = kv_indices[p0:p1]
        seq_len = (len(pages) - 1) * page_size + int(kv_last_page_len[b].item())
        kv_seq = kv_cache[pages].reshape(-1, d_total)[:seq_len].float()
        c_kv = kv_seq[:, :d_ckv_i]
        scores = (q[b].float() @ kv_seq.transpose(0, 1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out[b] = (probs @ c_kv).to(q.dtype)
    return out


def _try_splitk_mla(
    prototype_args: list[object],
) -> tuple[Callable[..., torch.Tensor], str] | None:
    q = _as_tensor(prototype_args[0])
    kv_cache = _as_tensor(prototype_args[1])
    kv_indices = _as_tensor(prototype_args[3])
    d_ckv = _as_int(prototype_args[5])
    if not supports_splitk_mla(q, kv_cache, d_ckv):
        return None
    num_cus = (
        int(torch.cuda.get_device_properties(q.device).multi_processor_count)
        if q.is_cuda
        else None
    )
    num_splits = select_gqa_num_splits(
        batch=q.shape[0],
        kv_heads=1,
        total_pages=kv_indices.numel(),
        num_cus=num_cus,
    )
    workspace = allocate_splitk_mla_workspace(q, int(d_ckv), num_splits)

    def _run_splitk(
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        indptr_in: torch.Tensor,
        indices_in: torch.Tensor,
        last_len_in: torch.Tensor,
        d_ckv_in: int,
    ) -> torch.Tensor:
        return run_mla_paged_decode_splitk(
            q_in,
            kv_in,
            indptr_in,
            indices_in,
            last_len_in,
            d_ckv_in,
            workspace=workspace,
        )

    try:
        _run_splitk(
            q,
            kv_cache,
            _as_tensor(prototype_args[2]),
            kv_indices,
            _as_tensor(prototype_args[4]),
            d_ckv,
        )
        return _run_splitk, "triton_splitk_paged_mla"
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton fails
        return None


def make_mla_paged_decode_runner(
    *,
    prototype_args: list[object],
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 2e-2,
) -> PreparedRunner:
    """Select Tier 3 AITER, Tier 2 vLLM, Split-KV Triton, or PyTorch MLA."""
    vendor_res = _dispatch_vendor_tiers(
        prototype_args,
        device=device,
        backend=backend,
        make_aiter=lambda: try_make_aiter_mla_runner(
            prototype_args, ref_fn=ref_mla_paged_decode, max_atol=max_atol
        ),
        make_vllm=lambda: try_make_vllm_mla_runner(
            prototype_args, ref_fn=ref_mla_paged_decode, max_atol=max_atol
        ),
    )
    if vendor_res is not None:
        return vendor_res
    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return PreparedRunner(
            fn=ref_mla_paged_decode,
            implementation_name="pytorch_mla_ref",
            candidate_args=prototype_args,
        )
    splitk = _try_splitk_mla(prototype_args)
    if splitk is not None:
        return PreparedRunner(
            fn=splitk[0],
            implementation_name=splitk[1],
            candidate_args=prototype_args,
        )
    return PreparedRunner(
        fn=ref_mla_paged_decode,
        implementation_name="pytorch_mla_ref_fallback",
        candidate_args=prototype_args,
    )


def ref_fused_rmsnorm_rope(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    cos_sin: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference FP32 Residual Add + RMSNorm + NeoX RoPE -> BF16."""
    h = x.float() + residual.float()
    rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = h * rms * weight.float()
    d2 = y.shape[-1] // 2
    u, v = y[..., :d2], y[..., d2:]
    cos_v = cos_sin[:, None, :d2].float()
    sin_v = cos_sin[:, None, d2:].float()
    return torch.cat([u * cos_v - v * sin_v, v * cos_v + u * sin_v], dim=-1).to(
        x.dtype
    )


def make_fused_rmsnorm_rope_runner(
    *,
    prototype_args: list[object],
    device: str,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 2e-2,
) -> PreparedRunner:
    """Select Tier 3 AITER, Tier 2 vLLM, Triton single-pass, or PyTorch."""
    vendor_res = _dispatch_vendor_tiers(
        prototype_args,
        device=device,
        backend=backend,
        make_aiter=lambda: try_make_aiter_rmsnorm_rope_runner(
            prototype_args, ref_fn=ref_fused_rmsnorm_rope, max_atol=max_atol
        ),
        make_vllm=lambda: try_make_vllm_rmsnorm_rope_runner(
            prototype_args, ref_fn=ref_fused_rmsnorm_rope, max_atol=max_atol
        ),
    )
    if vendor_res is not None:
        return vendor_res
    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return PreparedRunner(
            fn=ref_fused_rmsnorm_rope,
            implementation_name="pytorch_rmsnorm_rope",
            candidate_args=prototype_args,
        )
    try:
        run_fused_rmsnorm_rope_triton(
            _as_tensor(prototype_args[0]),
            _as_tensor(prototype_args[1]),
            _as_tensor(prototype_args[2]),
            _as_tensor(prototype_args[3]),
        )
        return PreparedRunner(
            fn=run_fused_rmsnorm_rope_triton,
            implementation_name="triton_fused_rmsnorm_rope",
            candidate_args=prototype_args,
        )
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton unavailable
        return PreparedRunner(
            fn=ref_fused_rmsnorm_rope,
            implementation_name="pytorch_rmsnorm_rope_fallback",
            candidate_args=prototype_args,
        )


def ref_swiglu_fp8_quant(gate_up: torch.Tensor) -> torch.Tensor:
    """Reference BF16 SwiGLU output prior to dynamic FP8 quantization."""
    half = gate_up.shape[-1] // 2
    g = gate_up[:, :half].float()
    u = gate_up[:, half:].float()
    return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)


def run_swiglu_fp8_quant_pytorch(
    gate_up: torch.Tensor,
    *,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch SwiGLU activation + dynamic per-token FP8 quantization."""
    half = gate_up.shape[-1] // 2
    g = gate_up[:, :half].float()
    u = gate_up[:, half:].float()
    act = torch.nn.functional.silu(g) * u
    fp8_max = float(torch.finfo(fp8_dtype).max)
    amax = act.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / fp8_max).to(torch.float32)
    y_fp8 = (act / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return y_fp8, scale


def make_swiglu_fp8_quant_runner(
    *,
    prototype_args: list[object],
    device: str,
    fp8_dtype: torch.dtype,
    backend: KernelBackend = KernelBackend.AUTO,
    max_atol: float = 2e-1,
) -> PreparedRunner:
    """Select Tier 3 AITER, Tier 2 vLLM, Triton fused, or PyTorch SwiGLU+FP8."""
    vendor_res = _dispatch_vendor_tiers(
        prototype_args,
        device=device,
        backend=backend,
        make_aiter=lambda: try_make_aiter_swiglu_quant_runner(
            prototype_args,
            fp8_dtype=fp8_dtype,
            ref_fn=ref_swiglu_fp8_quant,
            max_atol=max_atol,
        ),
        make_vllm=lambda: try_make_vllm_swiglu_quant_runner(
            prototype_args,
            fp8_dtype=fp8_dtype,
            ref_fn=ref_swiglu_fp8_quant,
            max_atol=max_atol,
        ),
    )
    if vendor_res is not None:
        return vendor_res

    def _run_pt(gu_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return run_swiglu_fp8_quant_pytorch(gu_in, fp8_dtype=fp8_dtype)

    if device == "cpu" or backend is KernelBackend.PYTORCH:
        return PreparedRunner(
            fn=_run_pt,
            implementation_name="pytorch_swiglu_fp8_quant",
            candidate_args=prototype_args,
        )

    def _run_triton(gu_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return run_swiglu_fp8_quant_triton(gu_in, fp8_dtype=fp8_dtype)

    try:
        _run_triton(_as_tensor(prototype_args[0]))
        return PreparedRunner(
            fn=_run_triton,
            implementation_name="triton_swiglu_fp8_quant",
            candidate_args=prototype_args,
        )
    except Exception:  # noqa: BLE001 -- degrade cleanly if triton unavailable
        return PreparedRunner(
            fn=_run_pt,
            implementation_name="pytorch_swiglu_fp8_quant_fallback",
            candidate_args=prototype_args,
        )
