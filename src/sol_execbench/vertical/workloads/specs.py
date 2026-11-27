# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer-Trace workload definitions and rosters for the 6-family suite."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch

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
from sol_execbench.vertical.analytical_roofline import (
    AnalyticalBound,
    MVPOperator,
    PrecisionMode,
)

_PAGE_SIZE = 16


class SweepMode(StrEnum):
    """Benchmark workload scale presets."""

    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkloadSpec:
    """One concrete workload in the v0.1 6-family ROCm inference primitive suite."""

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
    prototype_args: list[object]
    candidate_args: list[object]
    reference_fn: Callable[..., torch.Tensor]
    candidate_fn: Callable[..., object] | None
    implementation_name: str
    unavailable_reason: str | None = None
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


_MLA_PAGED_REF = (
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
    "def run(q, kv_cache, kv_indptr, kv_indices, kv_last_page_len, d_ckv):\n"
    "    b, hq, d_tot = q.shape\n"
    "    ps = kv_cache.shape[1]\n"
    "    dc = int(d_ckv)\n"
    "    scale = 1.0 / math.sqrt(d_tot)\n"
    "    out = torch.empty((b, hq, dc), dtype=q.dtype, device=q.device)\n"
    "    for i in range(b):\n"
    "        pages = kv_indices[kv_indptr[i]:kv_indptr[i + 1]]\n"
    "        slen = (len(pages) - 1) * ps + int(kv_last_page_len[i].item())\n"
    "        kv = kv_cache[pages].reshape(-1, d_tot)[:slen].float()\n"
    "        probs = torch.softmax((q[i].float() @ kv.transpose(0, 1)) * scale, dim=-1)\n"
    "        out[i] = (probs @ kv[:, :dc]).to(q.dtype)\n"
    "    return out\n"
)

_RMSNORM_ROPE_REF = (
    "import torch\n\n"
    "def run(x, residual, weight, cos_sin):\n"
    "    h = x.float() + residual.float()\n"
    "    rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)\n"
    "    y = h * rms * weight.float()\n"
    "    d2 = y.shape[-1] // 2\n"
    "    u, v = y[..., :d2], y[..., d2:]\n"
    "    c, s = cos_sin[:, None, :d2].float(), cos_sin[:, None, d2:].float()\n"
    "    return torch.cat([u * c - v * s, v * c + u * s], dim=-1).to(x.dtype)\n"
)

_SWIGLU_QUANT_REF = (
    "import torch\n\n"
    "def run(gate_up):\n"
    "    half = gate_up.shape[-1] // 2\n"
    "    g, u = gate_up[:, :half].float(), gate_up[:, half:].float()\n"
    "    return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)\n"
)


def build_trace_definitions() -> dict[str, Definition]:
    """Build standard FlashInfer-Trace Definition objects for all 8 ops."""
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
        "fused_moe_bf16": _build_moe_definition(fp8=False),
        "fused_moe_fp8": _build_moe_definition(fp8=True),
        "gqa_paged_decode_bf16": _build_gqa_definition(),
        "mla_paged_decode_bf16": _build_mla_definition(),
        "fused_rmsnorm_rope_bf16": _build_rmsnorm_rope_definition(),
        "swiglu_fp8_quant": _build_swiglu_quant_definition(),
    }


def _build_moe_definition(*, fp8: bool) -> Definition:
    dt = DType.FLOAT8_E4M3FN if fp8 else DType.BFLOAT16
    return Definition(
        name="fused_moe_fp8" if fp8 else "fused_moe_bf16",
        op_type="moe",
        axes={"M": AxisVar(), "H": AxisVar(), "I": AxisVar(), "E": AxisVar()},
        inputs={
            "tokens": TensorSpec(shape=["M", "H"], dtype=dt),
            "gate_weights": TensorSpec(shape=["E", "H", "I"], dtype=dt),
            "up_weights": TensorSpec(shape=["E", "H", "I"], dtype=dt),
            "down_weights": TensorSpec(shape=["E", "I", "H"], dtype=dt),
            "gate_logits": TensorSpec(shape=["M", "E"], dtype=DType.BFLOAT16),
            "top_k": TensorSpec(shape=None, dtype=DType.INT64),
        },
        outputs={"output": TensorSpec(shape=["M", "H"], dtype=DType.BFLOAT16)},
        reference=_FUSED_MOE_REF,
        description=(
            "Fused SwiGLU MoE (8-bit FP8; E4M3FNUZ on CDNA3)."
            if fp8
            else "MVP Phase 1 Fused SwiGLU MoE."
        ),
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


def _build_mla_definition() -> Definition:
    return Definition(
        name="mla_paged_decode_bf16",
        op_type="attention",
        axes={
            "B": AxisVar(),
            "BP1": AxisExpr(expression="B + 1"),
            "NPAGES": AxisVar(),
            "PS": AxisConst(value=_PAGE_SIZE),
            "HQ": AxisVar(),
            "DCKV": AxisVar(),
            "DKPE": AxisVar(),
            "DTOTAL": AxisExpr(expression="DCKV + DKPE"),
            "NTOTALPAGES": AxisVar(),
        },
        custom_inputs_entrypoint="gen_inputs",
        inputs={
            "q": TensorSpec(shape=["B", "HQ", "DTOTAL"], dtype=DType.BFLOAT16),
            "kv_cache": TensorSpec(
                shape=["NPAGES", "PS", "DTOTAL"], dtype=DType.BFLOAT16
            ),
            "kv_indptr": TensorSpec(shape=["BP1"], dtype=DType.INT32),
            "kv_indices": TensorSpec(shape=["NTOTALPAGES"], dtype=DType.INT32),
            "kv_last_page_len": TensorSpec(shape=["B"], dtype=DType.INT32),
            "d_ckv": TensorSpec(shape=None, dtype=DType.INT64),
        },
        outputs={
            "output": TensorSpec(
                shape=["B", "HQ", "DCKV"], dtype=DType.BFLOAT16
            )
        },
        reference=_MLA_PAGED_REF,
        description="Absorbed DeepSeek MLA Paged Decode Attention.",
    )


def _build_rmsnorm_rope_definition() -> Definition:
    return Definition(
        name="fused_rmsnorm_rope_bf16",
        op_type="norm_rope",
        axes={"M": AxisVar(), "H": AxisVar(), "D": AxisVar()},
        inputs={
            "x": TensorSpec(shape=["M", "H", "D"], dtype=DType.BFLOAT16),
            "residual": TensorSpec(shape=["M", "H", "D"], dtype=DType.BFLOAT16),
            "weight": TensorSpec(shape=["D"], dtype=DType.BFLOAT16),
            "cos_sin": TensorSpec(shape=["M", "D"], dtype=DType.FLOAT32),
        },
        outputs={
            "output": TensorSpec(shape=["M", "H", "D"], dtype=DType.BFLOAT16)
        },
        reference=_RMSNORM_ROPE_REF,
        description="Fused Residual Add + RMSNorm + NeoX RoPE.",
    )


def _build_swiglu_quant_definition() -> Definition:
    return Definition(
        name="swiglu_fp8_quant",
        op_type="activation_quant",
        axes={
            "M": AxisVar(),
            "I": AxisVar(),
            "I2": AxisExpr(expression="2 * I"),
        },
        inputs={
            "gate_up": TensorSpec(shape=["M", "I2"], dtype=DType.BFLOAT16),
        },
        outputs={
            "output": TensorSpec(shape=["M", "I"], dtype=DType.BFLOAT16),
        },
        reference=_SWIGLU_QUANT_REF,
        description="Fused SwiGLU + Dynamic Per-Token FP8 Quantization.",
    )


def _stage2_smoke_specs() -> list[WorkloadSpec]:
    return [
        WorkloadSpec(
            workload_id="fused_moe_fp8_m8_e8_k2",
            operator=MVPOperator.FUSED_MOE,
            precision=PrecisionMode.FP8,
            definition_name="fused_moe_fp8",
            axes={"M": 8, "H": 128, "I": 64, "E": 8, "top_k": 2},
            tolerance=ToleranceSpec(max_atol=5e-1, max_rtol=1e-1),
        ),
        WorkloadSpec(
            workload_id="mla_paged_decode_bf16_b2_kv64",
            operator=MVPOperator.MLA_PAGED_DECODE,
            precision=PrecisionMode.BF16,
            definition_name="mla_paged_decode_bf16",
            axes={
                "B": 2,
                "HQ": 16,
                "DCKV": 64,
                "DKPE": 16,
                "kv_len": 64,
                "page_size": _PAGE_SIZE,
            },
            tolerance=ToleranceSpec(max_atol=2e-2, max_rtol=2e-2),
        ),
        WorkloadSpec(
            workload_id="fused_rmsnorm_rope_bf16_m16_h8_d64",
            operator=MVPOperator.FUSED_RMSNORM_ROPE,
            precision=PrecisionMode.BF16,
            definition_name="fused_rmsnorm_rope_bf16",
            axes={"M": 16, "H": 8, "D": 64},
            tolerance=ToleranceSpec(max_atol=8e-2, max_rtol=2e-2),
        ),
        WorkloadSpec(
            workload_id="swiglu_fp8_quant_m16_i128",
            operator=MVPOperator.SWIGLU_FP8_QUANT,
            precision=PrecisionMode.FP8,
            definition_name="swiglu_fp8_quant",
            axes={"M": 16, "I": 128},
            tolerance=ToleranceSpec(max_atol=2.5e-1, max_rtol=8e-2),
        ),
    ]


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
        *_stage2_smoke_specs(),
    ]


def _stage2_full_specs() -> list[WorkloadSpec]:
    specs: list[WorkloadSpec] = []
    for m in (1, 8, 32, 128):
        specs.append(
            WorkloadSpec(
                workload_id=f"fused_moe_fp8_m{m}_e64_k6",
                operator=MVPOperator.FUSED_MOE,
                precision=PrecisionMode.FP8,
                definition_name="fused_moe_fp8",
                axes={"M": m, "H": 2048, "I": 1024, "E": 64, "top_k": 6},
                tolerance=ToleranceSpec(max_atol=5e-1, max_rtol=1e-1),
            )
        )
    for b, kv_len in ((1, 4096), (8, 4096), (8, 32768), (16, 32768)):
        specs.append(
            WorkloadSpec(
                workload_id=f"mla_paged_decode_bf16_b{b}_kv{kv_len}",
                operator=MVPOperator.MLA_PAGED_DECODE,
                precision=PrecisionMode.BF16,
                definition_name="mla_paged_decode_bf16",
                axes={
                    "B": b,
                    "HQ": 16,
                    "DCKV": 512,
                    "DKPE": 64,
                    "kv_len": kv_len,
                    "page_size": _PAGE_SIZE,
                },
                tolerance=ToleranceSpec(max_atol=2e-2, max_rtol=2e-2),
            )
        )
    for m in (1, 16, 128, 1024):
        specs.append(
            WorkloadSpec(
                workload_id=f"fused_rmsnorm_rope_bf16_m{m}_h32_d128",
                operator=MVPOperator.FUSED_RMSNORM_ROPE,
                precision=PrecisionMode.BF16,
                definition_name="fused_rmsnorm_rope_bf16",
                axes={"M": m, "H": 32, "D": 128},
                tolerance=ToleranceSpec(max_atol=8e-2, max_rtol=2e-2),
            )
        )
    for m in (1, 16, 128, 1024):
        specs.append(
            WorkloadSpec(
                workload_id=f"swiglu_fp8_quant_m{m}_i4096",
                operator=MVPOperator.SWIGLU_FP8_QUANT,
                precision=PrecisionMode.FP8,
                definition_name="swiglu_fp8_quant",
                axes={"M": m, "I": 4096},
                tolerance=ToleranceSpec(max_atol=2.5e-1, max_rtol=8e-2),
            )
        )
    return specs


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
    specs.extend(_stage2_full_specs())
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
        inputs: dict[str, RandomInput | ScalarInput | CustomInput] = {
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
    elif spec.operator is MVPOperator.FUSED_RMSNORM_ROPE:
        axes = {k: spec.axes[k] for k in ("M", "H", "D")}
        inputs = {
            "x": RandomInput(),
            "residual": RandomInput(),
            "weight": RandomInput(),
            "cos_sin": RandomInput(),
        }
    elif spec.operator is MVPOperator.SWIGLU_FP8_QUANT:
        axes = {k: spec.axes[k] for k in ("M", "I")}
        inputs = {"gate_up": RandomInput()}
    elif spec.operator is MVPOperator.MLA_PAGED_DECODE:
        b = spec.axes["B"]
        pages_per_seq = math.ceil(spec.axes["kv_len"] / spec.axes["page_size"])
        total_pages = b * pages_per_seq
        axes = {
            "B": b,
            "NPAGES": total_pages + 4,
            "HQ": spec.axes["HQ"],
            "DCKV": spec.axes["DCKV"],
            "DKPE": spec.axes["DKPE"],
            "NTOTALPAGES": total_pages,
        }
        inputs = {
            "q": RandomInput(),
            "kv_cache": RandomInput(),
            "kv_indptr": CustomInput(),
            "kv_indices": CustomInput(),
            "kv_last_page_len": CustomInput(),
            "d_ckv": ScalarInput(value=spec.axes["DCKV"]),
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
