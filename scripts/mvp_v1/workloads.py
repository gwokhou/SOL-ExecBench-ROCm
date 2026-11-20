# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""3-operator MVP workloads in FlashInfer-Trace format + runtime builders."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from scripts.mvp_v1.analytical_roofline import (
    AnalyticalBound,
    EmpiricalCeiling,
    HardwareCeiling,
    MVPOperator,
    PrecisionMode,
    bound_fused_moe,
    bound_gqa_paged_decode,
    bound_skinny_gemm,
)
from scripts.mvp_v1.kernels import (
    KernelBackend,
    make_gqa_paged_decode_runner,
    make_skinny_gemm_runner,
    ref_fused_moe,
    ref_gqa_paged_decode,
    ref_skinny_gemm,
    run_fused_moe_grouped,
)
from sol_execbench.core.data.definition import (
    AxisConst,
    AxisExpr,
    AxisVar,
    Definition,
    TensorSpec,
)
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
)
from sol_execbench.core.data.workload import (
    CustomInput,
    NumericCheck,
    RandomInput,
    ScalarInput,
    ToleranceSpec,
    Workload,
)
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)

_PAGE_SIZE = 16


class SweepMode(StrEnum):
    """Benchmark workload scale presets."""

    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkloadSpec:
    """One concrete workload in the Phase 1 3-operator suite."""

    workload_id: str
    operator: MVPOperator
    precision: PrecisionMode
    definition_name: str
    axes: dict[str, int]
    tolerance: ToleranceSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeCase:
    """Executable tensors, reference/candidate callables, and Roofline bound."""

    spec: WorkloadSpec
    prototype_args: list[Any]
    reference_fn: Callable[..., torch.Tensor]
    candidate_fn: Callable[..., torch.Tensor]
    implementation_name: str
    bound: AnalyticalBound
    trace_definition: Definition
    trace_workload: Workload


_SKINNY_GEMM_REF = (
    "import torch\n\n"
    "def run(x, weight):\n"
    "    return (x.float() @ weight.float()).to(torch.bfloat16)\n"
)

_FUSED_MOE_REF = (
    "import torch\n\n"
    "def run(tokens, gate_weights, up_weights, down_weights, gate_logits, top_k):\n"
    "    probs = torch.softmax(gate_logits.float(), dim=-1)\n"
    "    weights, ids = torch.topk(probs, k=int(top_k), dim=-1)\n"
    "    weights = weights / weights.sum(dim=-1, keepdim=True)\n"
    "    out = torch.zeros_like(tokens, dtype=torch.float32)\n"
    "    for t in range(tokens.shape[0]):\n"
    "        for s in range(int(top_k)):\n"
    "            e = int(ids[t, s].item())\n"
    "            g = torch.nn.functional.silu(tokens[t:t+1].float() @ gate_weights[e].float())\n"
    "            u = tokens[t:t+1].float() @ up_weights[e].float()\n"
    "            out[t] += weights[t, s] * ((g * u) @ down_weights[e].float())[0]\n"
    "    return out.to(torch.bfloat16)\n"
)

_GQA_PAGED_REF = (
    "import math\nimport torch\n\n"
    "def gen_inputs(axes, device):\n"
    "    b = int(axes['B'])\n"
    "    total_pages = int(axes['NTOTALPAGES'])\n"
    "    per_seq = total_pages // b\n"
    "    return {\n"
    "        'kv_indptr': torch.arange(0, (b + 1) * per_seq, per_seq, dtype=torch.int32, device=device),\n"
    "        'kv_indices': torch.randperm(int(axes['NPAGES']), dtype=torch.int32, device=device)[:total_pages].contiguous(),\n"
    "        'kv_last_page_len': torch.full((b,), int(axes['PS']), dtype=torch.int32, device=device),\n"
    "    }\n\n"
    "def run(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len):\n"
    "    b, hq, d = q.shape\n"
    "    ps, hk = k_cache.shape[1], k_cache.shape[2]\n"
    "    rep = hq // hk\n"
    "    out = torch.empty_like(q)\n"
    "    for i in range(b):\n"
    "        pages = kv_indices[kv_indptr[i]:kv_indptr[i + 1]]\n"
    "        slen = (len(pages) - 1) * ps + int(kv_last_page_len[i].item())\n"
    "        k = k_cache[pages].reshape(-1, hk, d)[:slen].transpose(0, 1).repeat_interleave(rep, 0)\n"
    "        v = v_cache[pages].reshape(-1, hk, d)[:slen].transpose(0, 1).repeat_interleave(rep, 0)\n"
    "        s = torch.bmm(q[i].float().unsqueeze(1), k.float().transpose(1, 2)) / math.sqrt(d)\n"
    "        out[i] = torch.bmm(torch.softmax(s, dim=-1), v.float()).squeeze(1).to(q.dtype)\n"
    "    return out\n"
)


def build_trace_definitions() -> dict[str, Definition]:
    """Build standard FlashInfer-Trace Definition objects for all 3 ops."""
    gemm_axes = {"M": AxisVar(), "K": AxisVar(), "N": AxisVar()}
    gemm_out = {"output": TensorSpec(shape=["M", "N"], dtype=DType.BFLOAT16)}
    return {
        "skinny_gemm_bf16": Definition(
            name="skinny_gemm_bf16",
            op_type="linear",
            axes=gemm_axes,
            inputs={
                "x": TensorSpec(shape=["M", "K"], dtype=DType.BFLOAT16),
                "weight": TensorSpec(shape=["K", "N"], dtype=DType.BFLOAT16),
            },
            outputs=gemm_out,
            reference=_SKINNY_GEMM_REF,
            description="MVP Phase 1 Skinny GEMM (BF16).",
        ),
        "skinny_gemm_fp8": Definition(
            name="skinny_gemm_fp8",
            op_type="linear",
            axes=gemm_axes,
            inputs={
                "x": TensorSpec(shape=["M", "K"], dtype=DType.FLOAT8_E4M3FN),
                "weight": TensorSpec(
                    shape=["K", "N"], dtype=DType.FLOAT8_E4M3FN
                ),
            },
            outputs=gemm_out,
            reference=_SKINNY_GEMM_REF,
            description="MVP Phase 1 Skinny GEMM (8-bit FP8; E4M3FNUZ on CDNA3).",
        ),
        "fused_moe_bf16": _build_moe_definition(),
        "gqa_paged_decode_bf16": _build_gqa_definition(),
    }


def _build_moe_definition() -> Definition:
    return Definition(
        name="fused_moe_bf16",
        op_type="moe",
        axes={"M": AxisVar(), "H": AxisVar(), "I": AxisVar(), "E": AxisVar()},
        inputs={
            "tokens": TensorSpec(shape=["M", "H"], dtype=DType.BFLOAT16),
            "gate_weights": TensorSpec(
                shape=["E", "H", "I"], dtype=DType.BFLOAT16
            ),
            "up_weights": TensorSpec(
                shape=["E", "H", "I"], dtype=DType.BFLOAT16
            ),
            "down_weights": TensorSpec(
                shape=["E", "I", "H"], dtype=DType.BFLOAT16
            ),
            "gate_logits": TensorSpec(shape=["M", "E"], dtype=DType.BFLOAT16),
            "top_k": TensorSpec(shape=None, dtype=DType.INT64),
        },
        outputs={"output": TensorSpec(shape=["M", "H"], dtype=DType.BFLOAT16)},
        reference=_FUSED_MOE_REF,
        description="MVP Phase 1 Fused SwiGLU MoE.",
    )


def _build_gqa_definition() -> Definition:
    return Definition(
        name="gqa_paged_decode_bf16",
        op_type="attention",
        axes={
            "B": AxisVar(),
            "BP1": AxisExpr(expression="B + 1"),
            "NPAGES": AxisVar(),
            "PS": AxisConst(value=_PAGE_SIZE),
            "HQ": AxisVar(),
            "HK": AxisVar(),
            "D": AxisVar(),
            "NTOTALPAGES": AxisVar(),
        },
        custom_inputs_entrypoint="gen_inputs",
        inputs={
            "q": TensorSpec(shape=["B", "HQ", "D"], dtype=DType.BFLOAT16),
            "k_cache": TensorSpec(
                shape=["NPAGES", "PS", "HK", "D"], dtype=DType.BFLOAT16
            ),
            "v_cache": TensorSpec(
                shape=["NPAGES", "PS", "HK", "D"], dtype=DType.BFLOAT16
            ),
            "kv_indptr": TensorSpec(shape=["BP1"], dtype=DType.INT32),
            "kv_indices": TensorSpec(shape=["NTOTALPAGES"], dtype=DType.INT32),
            "kv_last_page_len": TensorSpec(shape=["B"], dtype=DType.INT32),
        },
        outputs={
            "output": TensorSpec(shape=["B", "HQ", "D"], dtype=DType.BFLOAT16)
        },
        reference=_GQA_PAGED_REF,
        description="MVP Phase 1 NHD GQA Paged Decode Attention.",
    )


def _smoke_specs() -> list[WorkloadSpec]:
    return [
        *[
            WorkloadSpec(
                workload_id=f"skinny_gemm_bf16_m{m}_n512_k256",
                operator=MVPOperator.SKINNY_GEMM,
                precision=PrecisionMode.BF16,
                definition_name="skinny_gemm_bf16",
                axes={"M": m, "K": 256, "N": 512},
                tolerance=ToleranceSpec(max_atol=5e-2, max_rtol=2e-2),
            )
            for m in (1, 16)
        ],
        WorkloadSpec(
            workload_id="skinny_gemm_fp8_m16_n512_k256",
            operator=MVPOperator.SKINNY_GEMM,
            precision=PrecisionMode.FP8,
            definition_name="skinny_gemm_fp8",
            axes={"M": 16, "K": 256, "N": 512},
            tolerance=ToleranceSpec(max_atol=2.5e-1, max_rtol=5e-2),
        ),
        *[
            WorkloadSpec(
                workload_id=f"fused_moe_bf16_m{m}_e8_k2",
                operator=MVPOperator.FUSED_MOE,
                precision=PrecisionMode.BF16,
                definition_name="fused_moe_bf16",
                axes={"M": m, "H": 128, "I": 64, "E": 8, "top_k": 2},
                tolerance=ToleranceSpec(max_atol=5e-2, max_rtol=2e-2),
            )
            for m in (1, 8)
        ],
        WorkloadSpec(
            workload_id="gqa_paged_decode_bf16_b2_kv64",
            operator=MVPOperator.GQA_PAGED_DECODE,
            precision=PrecisionMode.BF16,
            definition_name="gqa_paged_decode_bf16",
            axes={
                "B": 2,
                "HQ": 8,
                "HK": 2,
                "D": 64,
                "kv_len": 64,
                "page_size": _PAGE_SIZE,
            },
            tolerance=ToleranceSpec(max_atol=2e-2, max_rtol=2e-2),
        ),
    ]


def _full_specs() -> list[WorkloadSpec]:
    specs: list[WorkloadSpec] = []
    for prec, def_name, atol, rtol in (
        (PrecisionMode.BF16, "skinny_gemm_bf16", 5e-2, 2e-2),
        (PrecisionMode.FP8, "skinny_gemm_fp8", 5e-1, 1e-1),
    ):
        for m in (1, 4, 16, 64, 128):
            specs.append(
                WorkloadSpec(
                    workload_id=f"skinny_gemm_{prec.value}_m{m}_n4096_k7168",
                    operator=MVPOperator.SKINNY_GEMM,
                    precision=prec,
                    definition_name=def_name,
                    axes={"M": m, "K": 7168, "N": 4096},
                    tolerance=ToleranceSpec(max_atol=atol, max_rtol=rtol),
                )
            )
    for m in (1, 8, 32, 128):
        specs.append(
            WorkloadSpec(
                workload_id=f"fused_moe_bf16_m{m}_e64_k6",
                operator=MVPOperator.FUSED_MOE,
                precision=PrecisionMode.BF16,
                definition_name="fused_moe_bf16",
                axes={"M": m, "H": 2048, "I": 1024, "E": 64, "top_k": 6},
                tolerance=ToleranceSpec(max_atol=5e-2, max_rtol=2e-2),
            )
        )
    for b, kv_len in ((1, 4096), (8, 4096), (8, 32768), (16, 32768)):
        specs.append(
            WorkloadSpec(
                workload_id=f"gqa_paged_decode_bf16_b{b}_kv{kv_len}",
                operator=MVPOperator.GQA_PAGED_DECODE,
                precision=PrecisionMode.BF16,
                definition_name="gqa_paged_decode_bf16",
                axes={
                    "B": b,
                    "HQ": 32,
                    "HK": 4,
                    "D": 128,
                    "kv_len": kv_len,
                    "page_size": _PAGE_SIZE,
                },
                tolerance=ToleranceSpec(max_atol=2e-2, max_rtol=2e-2),
            )
        )
    return specs


def build_workload_specs(mode: SweepMode) -> list[WorkloadSpec]:
    """Return ordered MVP workload sweep for ``smoke`` or ``full`` runs."""
    return _smoke_specs() if mode is SweepMode.SMOKE else _full_specs()


def to_trace_workload(spec: WorkloadSpec) -> Workload:
    """Convert one ``WorkloadSpec`` to a validated FlashInfer-Trace ``Workload``."""
    check = NumericCheck(
        output="output",
        max_atol=spec.tolerance.max_atol,
        max_rtol=spec.tolerance.max_rtol,
    )
    if spec.operator is MVPOperator.SKINNY_GEMM:
        axes = {k: spec.axes[k] for k in ("M", "K", "N")}
        inputs: dict[str, Any] = {
            "x": RandomInput(),
            "weight": RandomInput(),
        }
    elif spec.operator is MVPOperator.FUSED_MOE:
        axes = {k: spec.axes[k] for k in ("M", "H", "I", "E")}
        inputs = {
            "tokens": RandomInput(),
            "gate_weights": RandomInput(),
            "up_weights": RandomInput(),
            "down_weights": RandomInput(),
            "gate_logits": RandomInput(),
            "top_k": ScalarInput(value=spec.axes["top_k"]),
        }
    else:
        b = spec.axes["B"]
        pages_per_seq = math.ceil(spec.axes["kv_len"] / spec.axes["page_size"])
        total_pages = b * pages_per_seq
        axes = {
            "B": b,
            "NPAGES": total_pages + 4,
            "HQ": spec.axes["HQ"],
            "HK": spec.axes["HK"],
            "D": spec.axes["D"],
            "NTOTALPAGES": total_pages,
        }
        inputs = {
            "q": RandomInput(),
            "k_cache": RandomInput(),
            "v_cache": RandomInput(),
            "kv_indptr": CustomInput(),
            "kv_indices": CustomInput(),
            "kv_last_page_len": CustomInput(),
        }
    return Workload(
        axes=axes,
        inputs=inputs,
        uuid=spec.workload_id,
        checks=[check],
    )


def materialize_trace_corpus(
    output_dir: Path,
    mode: SweepMode,
) -> dict[str, Path]:
    """Validate and write FlashInfer-Trace definition.json + workloads.jsonl."""
    definitions = build_trace_definitions()
    grouped: dict[str, list[Workload]] = {name: [] for name in definitions}
    for spec in build_workload_specs(mode):
        grouped[spec.definition_name].append(to_trace_workload(spec))

    written: dict[str, Path] = {}
    root = output_dir / "workloads"
    for name, definition in definitions.items():
        items = grouped[name]
        if not items:
            continue
        validate_problem_contract(definition, items)
        sub = root / name
        def_path = sub / "definition.json"
        wl_path = sub / "workloads.jsonl"
        atomic_write_json_value(
            def_path, definition.model_dump(mode="json"), sort_keys=False
        )
        atomic_write_jsonl_values(
            wl_path, [w.model_dump(mode="json") for w in items]
        )
        written[name] = wl_path
    return written


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

    if spec.operator is MVPOperator.SKINNY_GEMM:
        return _build_gemm_case(
            spec,
            device=device,
            ceiling=ceiling,
            empirical=empirical,
            gen=gen,
            trace_def=trace_def,
            trace_wl=trace_wl,
        )
    if spec.operator is MVPOperator.FUSED_MOE:
        return _build_moe_case(
            spec,
            device=device,
            ceiling=ceiling,
            empirical=empirical,
            gen=gen,
            trace_def=trace_def,
            trace_wl=trace_wl,
        )
    return _build_gqa_case(
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
        cand = make_skinny_gemm_runner(
            fp8=True, device=device, scale_a=sa, scale_b=sb
        )
        impl = "torch._scaled_mm" if device != "cpu" else "cpu_fp8_ref"
    else:
        x = x_bf
        w = w_bf.t().contiguous()
        cand = make_skinny_gemm_runner(fp8=False, device=device)
        impl = "torch.matmul"
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
        reference_fn=ref_skinny_gemm,
        candidate_fn=cand,
        implementation_name=impl,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


def _build_moe_case(
    spec: WorkloadSpec,
    *,
    device: str,
    ceiling: HardwareCeiling,
    empirical: EmpiricalCeiling | None,
    gen: torch.Generator,
    trace_def: Definition,
    trace_wl: Workload,
) -> RuntimeCase:
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
    topk_ids = torch.topk(
        torch.softmax(logits.float(), dim=-1), k=top_k, dim=-1
    ).indices
    active_experts = int(torch.unique(topk_ids).numel())
    bound = bound_fused_moe(
        m,
        h,
        inter,
        e,
        top_k,
        spec.precision,
        ceiling,
        active_experts=active_experts,
        empirical=empirical,
        workload_id=spec.workload_id,
    )
    return RuntimeCase(
        spec=spec,
        prototype_args=[tokens, gate_w, up_w, down_w, logits, top_k],
        reference_fn=ref_fused_moe,
        candidate_fn=run_fused_moe_grouped,
        implementation_name="grouped_swiglu_moe",
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )


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
    per_seq = math.ceil(kv_len / ps)
    total_pages = b * per_seq
    n_pool = total_pages + 4
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
    kv_indptr = torch.arange(
        0, (b + 1) * per_seq, per_seq, dtype=torch.int32, device=device
    )
    kv_indices = torch.randperm(
        n_pool, dtype=torch.int32, device=device, generator=gen
    )[:total_pages].contiguous()
    tail = kv_len - (per_seq - 1) * ps
    kv_last_len = torch.full((b,), tail, dtype=torch.int32, device=device)
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
    args = _make_gqa_paged_tensors(spec, device=device, gen=gen)
    cand_fn, impl = make_gqa_paged_decode_runner(device=device, backend=backend)
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
        reference_fn=ref_gqa_paged_decode,
        candidate_fn=cand_fn,
        implementation_name=impl,
        bound=bound,
        trace_definition=trace_def,
        trace_workload=trace_wl,
    )
