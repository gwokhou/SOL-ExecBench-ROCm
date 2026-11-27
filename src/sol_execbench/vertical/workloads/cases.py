# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime tensor case materialization for the 6-family vertical SOL suite."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.vertical.analytical_roofline import (
    EmpiricalCeiling,
    HardwareCeiling,
    MVPOperator,
    PrecisionMode,
    bound_fused_moe,
    bound_fused_rmsnorm_rope,
    bound_gqa_paged_decode,
    bound_mla_paged_decode,
    bound_skinny_gemm,
    bound_swiglu_fp8_quant,
)
from sol_execbench.vertical.kernels import (
    KernelBackend,
    make_fused_moe_runner,
    make_fused_rmsnorm_rope_runner,
    make_gqa_paged_decode_runner,
    make_mla_paged_decode_runner,
    make_skinny_gemm_runner,
    make_swiglu_fp8_quant_runner,
    ref_fused_moe,
    ref_fused_rmsnorm_rope,
    ref_gqa_paged_decode,
    ref_mla_paged_decode,
    ref_skinny_gemm,
    ref_swiglu_fp8_quant,
)
from sol_execbench.vertical.workloads.specs import (
    RuntimeCase,
    WorkloadSpec,
    build_trace_definitions,
    to_trace_workload,
)


def build_runtime_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None = None,
    backend: KernelBackend = KernelBackend.AUTO,
    seed: int = 42,
) -> RuntimeCase:
    """Materialize concrete tensors, callables, and Roofline bound."""
    definitions = build_trace_definitions()
    trace_def = definitions[spec.definition_name]
    trace_wl = to_trace_workload(spec)
    gen = torch.Generator(device=device if device != "cpu" else "cpu")
    gen.manual_seed(seed)
    builders = {
        MVPOperator.SKINNY_GEMM: _build_gemm_case,
        MVPOperator.FUSED_MOE: _build_moe_case,
        MVPOperator.GQA_PAGED_DECODE: _build_gqa_case,
        MVPOperator.MLA_PAGED_DECODE: _build_mla_case,
        MVPOperator.FUSED_RMSNORM_ROPE: _build_rmsnorm_rope_case,
        MVPOperator.SWIGLU_FP8_QUANT: _build_swiglu_quant_case,
    }
    return builders[spec.operator](
        spec,
        device=device,
        ceiling=ceiling,
        empirical=empirical,
        backend=backend,
        gen=gen,
        trace_def=trace_def,
        trace_wl=trace_wl,
    )


def _build_gemm_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    backend: KernelBackend,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
    m, k, n = spec.axes["M"], spec.axes["K"], spec.axes["N"]
    scale = 1.0 / math.sqrt(k)
    x_bf = (
        torch.randn((m, k), dtype=torch.bfloat16, device=device, generator=gen)
        * scale
    )
    w_bf = (
        torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=gen)
        * scale
    )
    if spec.precision is PrecisionMode.FP8:
        fp8_dt = ceiling.fp8_torch_dtype()
        x = x_bf.to(fp8_dt)
        w = w_bf.to(fp8_dt).t()
        sa = torch.ones((), dtype=torch.float32, device=device)
        sb = torch.ones((), dtype=torch.float32, device=device)
        ref_fn: Callable[..., torch.Tensor] = partial(
            ref_skinny_gemm, scale_a=sa, scale_b=sb
        )
        prepared = make_skinny_gemm_runner(
            prototype_args=[x, w],
            fp8=True,
            device=device,
            backend=backend,
            scale_a=sa,
            scale_b=sb,
            max_atol=spec.tolerance.max_atol,
        )
    else:
        x = x_bf
        w = w_bf.t().contiguous()
        ref_fn = ref_skinny_gemm
        prepared = make_skinny_gemm_runner(
            prototype_args=[x, w],
            fp8=False,
            device=device,
            backend=backend,
            max_atol=spec.tolerance.max_atol,
        )
    bound = bound_skinny_gemm(
        m,
        n,
        k,
        spec.precision,
        ceiling,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=[x, w],
        candidate_args=prepared.candidate_args,
        reference_fn=ref_fn,
        candidate_fn=prepared.fn,
        implementation_name=prepared.implementation_name,
        unavailable_reason=prepared.unavailable_reason,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


def _sample_moe_inputs(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    gen: torch.Generator,
) -> tuple[list[object], int]:
    m = spec.axes["M"]
    h, inter, e, top_k = (
        spec.axes["H"],
        spec.axes["I"],
        spec.axes["E"],
        spec.axes["top_k"],
    )
    s_in = 1.0 / math.sqrt(h)
    s_down = 1.0 / math.sqrt(inter)
    tokens = (
        torch.randn((m, h), dtype=torch.bfloat16, device=device, generator=gen)
        * 0.5
    )
    gate_w = (
        torch.randn(
            (e, h, inter), dtype=torch.bfloat16, device=device, generator=gen
        )
        * s_in
    )
    up_w = (
        torch.randn(
            (e, h, inter), dtype=torch.bfloat16, device=device, generator=gen
        )
        * s_in
    )
    down_w = (
        torch.randn(
            (e, inter, h), dtype=torch.bfloat16, device=device, generator=gen
        )
        * s_down
    )
    logits = torch.randn(
        (m, e), dtype=torch.bfloat16, device=device, generator=gen
    )
    if spec.precision is PrecisionMode.FP8:
        fp8_dt = ceiling.fp8_torch_dtype()
        tokens = tokens.to(fp8_dt)
        gate_w = gate_w.to(fp8_dt)
        up_w = up_w.to(fp8_dt)
        down_w = down_w.to(fp8_dt)
    topk_ids = torch.topk(
        torch.softmax(logits.float(), dim=-1), k=top_k, dim=-1
    ).indices
    active_experts = int(torch.unique(topk_ids).numel())
    return [tokens, gate_w, up_w, down_w, logits, top_k], active_experts


def _build_moe_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    backend: KernelBackend,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
    proto_args, active_experts = _sample_moe_inputs(
        spec,
        device=device,
        ceiling=ceiling,
        gen=gen,
    )
    prepared = make_fused_moe_runner(
        prototype_args=proto_args,
        device=device,
        backend=backend,
        max_atol=spec.tolerance.max_atol,
    )
    bound = bound_fused_moe(
        spec.axes["M"],
        spec.axes["H"],
        spec.axes["I"],
        spec.axes["E"],
        spec.axes["top_k"],
        spec.precision,
        ceiling,
        active_experts=active_experts,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=proto_args,
        candidate_args=prepared.candidate_args,
        reference_fn=ref_fused_moe,
        candidate_fn=prepared.fn,
        implementation_name=prepared.implementation_name,
        unavailable_reason=prepared.unavailable_reason,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


def _build_paged_csr_metadata(
    b: int,
    kv_len: int,
    ps: int,
    *,
    device: str,
    gen: torch.Generator,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    per_seq = math.ceil(kv_len / ps)
    total_pages = b * per_seq
    n_pool = total_pages + 4
    kv_indptr = torch.arange(
        0, (b + 1) * per_seq, per_seq, dtype=torch.int32, device=device
    )
    kv_indices = torch.randperm(
        n_pool, dtype=torch.int32, device=device, generator=gen
    )[:total_pages].contiguous()
    tail = kv_len - (per_seq - 1) * ps
    kv_last_len = torch.full((b,), tail, dtype=torch.int32, device=device)
    return n_pool, kv_indptr, kv_indices, kv_last_len


def _make_gqa_paged_tensors(
    spec: WorkloadSpec,
    *,
    device: str,
    gen: torch.Generator,
) -> list[torch.Tensor]:
    b, hq, hk, d = (
        spec.axes["B"],
        spec.axes["HQ"],
        spec.axes["HK"],
        spec.axes["D"],
    )
    kv_len, ps = spec.axes["kv_len"], spec.axes["page_size"]
    n_pool, kv_indptr, kv_indices, kv_last_len = _build_paged_csr_metadata(
        b, kv_len, ps, device=device, gen=gen
    )
    q = (
        torch.randn(
            (b, hq, d), dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.5
    )
    k_cache = (
        torch.randn(
            (n_pool, ps, hk, d),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        * 0.5
    )
    v_cache = (
        torch.randn(
            (n_pool, ps, hk, d),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        * 0.5
    )
    return [q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_len]


def _build_gqa_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    backend: KernelBackend,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
    b, hq, hk, d = (
        spec.axes["B"],
        spec.axes["HQ"],
        spec.axes["HK"],
        spec.axes["D"],
    )
    kv_len, ps = spec.axes["kv_len"], spec.axes["page_size"]
    args: list[object] = list(
        _make_gqa_paged_tensors(spec, device=device, gen=gen)
    )
    prepared = make_gqa_paged_decode_runner(
        prototype_args=args,
        device=device,
        backend=backend,
        max_atol=spec.tolerance.max_atol,
    )
    bound = bound_gqa_paged_decode(
        b,
        hq,
        hk,
        d,
        kv_len,
        ps,
        spec.precision,
        ceiling,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=args,
        candidate_args=prepared.candidate_args,
        reference_fn=ref_gqa_paged_decode,
        candidate_fn=prepared.fn,
        implementation_name=prepared.implementation_name,
        unavailable_reason=prepared.unavailable_reason,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


def _build_mla_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    backend: KernelBackend,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
    b, hq, d_ckv, d_kpe = (
        spec.axes["B"],
        spec.axes["HQ"],
        spec.axes["DCKV"],
        spec.axes["DKPE"],
    )
    kv_len, ps = spec.axes["kv_len"], spec.axes["page_size"]
    d_total = d_ckv + d_kpe
    n_pool, kv_indptr, kv_indices, kv_last_len = _build_paged_csr_metadata(
        b, kv_len, ps, device=device, gen=gen
    )
    q = (
        torch.randn(
            (b, hq, d_total), dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.5
    )
    kv_cache = (
        torch.randn(
            (n_pool, ps, d_total),
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        * 0.5
    )
    proto_args: list[object] = [
        q,
        kv_cache,
        kv_indptr,
        kv_indices,
        kv_last_len,
        d_ckv,
    ]
    prepared = make_mla_paged_decode_runner(
        prototype_args=proto_args,
        device=device,
        backend=backend,
        max_atol=spec.tolerance.max_atol,
    )
    bound = bound_mla_paged_decode(
        b,
        hq,
        d_ckv,
        d_kpe,
        kv_len,
        ps,
        spec.precision,
        ceiling,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=proto_args,
        candidate_args=prepared.candidate_args,
        reference_fn=ref_mla_paged_decode,
        candidate_fn=prepared.fn,
        implementation_name=prepared.implementation_name,
        unavailable_reason=prepared.unavailable_reason,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


def _build_rmsnorm_rope_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    backend: KernelBackend,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
    m, h, d = spec.axes["M"], spec.axes["H"], spec.axes["D"]
    x = torch.randn(
        (m, h, d), dtype=torch.bfloat16, device=device, generator=gen
    )
    residual = torch.randn(
        (m, h, d), dtype=torch.bfloat16, device=device, generator=gen
    )
    weight = torch.randn(
        (d,), dtype=torch.bfloat16, device=device, generator=gen
    )
    angles = torch.randn(
        (m, d // 2), dtype=torch.float32, device=device, generator=gen
    )
    cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    proto_args: list[object] = [x, residual, weight, cos_sin]
    prepared = make_fused_rmsnorm_rope_runner(
        prototype_args=proto_args,
        device=device,
        backend=backend,
        max_atol=spec.tolerance.max_atol,
    )
    bound = bound_fused_rmsnorm_rope(
        m,
        h,
        d,
        spec.precision,
        ceiling,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=proto_args,
        candidate_args=prepared.candidate_args,
        reference_fn=ref_fused_rmsnorm_rope,
        candidate_fn=prepared.fn,
        implementation_name=prepared.implementation_name,
        unavailable_reason=prepared.unavailable_reason,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


def _build_swiglu_quant_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    backend: KernelBackend,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
    m, inter = spec.axes["M"], spec.axes["I"]
    gate_up = (
        torch.randn(
            (m, 2 * inter), dtype=torch.bfloat16, device=device, generator=gen
        )
        * 0.5
    )
    proto_args: list[object] = [gate_up]
    prepared = make_swiglu_fp8_quant_runner(
        prototype_args=proto_args,
        device=device,
        fp8_dtype=ceiling.fp8_torch_dtype(),
        backend=backend,
        max_atol=spec.tolerance.max_atol,
    )
    bound = bound_swiglu_fp8_quant(
        m,
        inter,
        ceiling,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=proto_args,
        candidate_args=prepared.candidate_args,
        reference_fn=ref_swiglu_fp8_quant,
        candidate_fn=prepared.fn,
        implementation_name=prepared.implementation_name,
        unavailable_reason=prepared.unavailable_reason,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )
