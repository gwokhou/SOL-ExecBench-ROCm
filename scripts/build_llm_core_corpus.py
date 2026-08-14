#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministically build the clean-room LLM Core V1 corpus."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.dtypes import dtype_storage_bits
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
)
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)
from sol_execbench.core.dataset.corpus import semantic_fingerprint
from sol_execbench.core.dataset.corpus_models import (
    CorpusCoveragePolicy,
    CorpusEntry,
    CorpusManifest,
    CorpusOperationFamily,
    CorpusProfile,
    CorpusReleaseState,
    CorpusWorkloadRecord,
    ModelSource,
    QuantizationScheme,
    ResourceEnvelope,
    ShapeTier,
    StaticCapability,
    StaticRequirements,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.integrity import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "problems" / "LLM_CORE"
MODEL_REGISTRY = CORPUS_ROOT / "registry" / "models.yaml"
SEMANTIC_REGISTRY = CORPUS_ROOT / "registry" / "semantics.yaml"
CANDIDATE_MANIFEST = CORPUS_ROOT / "candidates" / "manifest.yaml"
RELEASE_ROOT = CORPUS_ROOT / "releases" / "LLM_CORE_V1"
NAMESPACE = uuid.UUID("e5be2237-fddd-5f5f-9087-d35485ae51d5")
WORKLOADS_PER_PROBLEM = 15


@dataclass(frozen=True)
class ProblemSpec:
    """Declarative semantic specification consumed by family generators."""

    family: CorpusOperationFamily
    variant: str
    profiles: tuple[CorpusProfile, ...]
    source_ids: tuple[str, ...]
    quantization: QuantizationScheme | None = None
    capabilities: tuple[StaticCapability, ...] = ()

    @property
    def semantic_id(self) -> str:
        """Return the stable, source-independent semantic identifier."""
        return f"llm.{self.family.value}.{self.variant}.v1"

    @property
    def problem_name(self) -> str:
        """Return the filesystem-safe problem name."""
        return f"{self.family.value}_{self.variant}"


COMMON_SOURCES = ("qwen3_8", "deepseek_v4", "gemma_4", "ministral_3")
MOE_SOURCES = (
    "qwen3_8",
    "deepseek_v4",
    "kimi_k3",
    "glm_5_2",
    "minimax_m3",
    "llama_4",
)
LONG_SOURCES = ("kimi_k3", "glm_5_2", "minimax_m3", "llama_4")


def _specs() -> tuple[ProblemSpec, ...]:
    specs: list[ProblemSpec] = []
    specs.extend(
        _family_specs(
            CorpusOperationFamily.LINEAR,
            _LINEAR,
            (CorpusProfile.CORE,),
            COMMON_SOURCES,
        )
    )
    specs.extend(
        _family_specs(
            CorpusOperationFamily.NORM_ACTIVATION,
            _NORM,
            (CorpusProfile.CORE,),
            COMMON_SOURCES,
        )
    )
    specs.extend(
        _family_specs(
            CorpusOperationFamily.POSITION,
            _POSITION,
            (CorpusProfile.CORE, CorpusProfile.LONG_CONTEXT),
            COMMON_SOURCES,
        )
    )
    specs.extend(
        _family_specs(
            CorpusOperationFamily.ATTENTION,
            _ATTENTION,
            (CorpusProfile.CORE, CorpusProfile.LONG_CONTEXT),
            COMMON_SOURCES,
        )
    )
    specs.extend(
        _family_specs(
            CorpusOperationFamily.ADVANCED_ATTENTION,
            _ADVANCED,
            (CorpusProfile.ARCHITECTURE_SPECIFIC, CorpusProfile.LONG_CONTEXT),
            LONG_SOURCES,
            (StaticCapability.SPARSE_ATTENTION,),
        )
    )
    specs.extend(
        _family_specs(
            CorpusOperationFamily.KV_CACHE,
            _KV_CACHE,
            (CorpusProfile.KV_CACHE, CorpusProfile.LONG_CONTEXT),
            LONG_SOURCES,
            (StaticCapability.PAGED_MEMORY,),
        )
    )
    specs.extend(
        _family_specs(
            CorpusOperationFamily.MOE,
            _MOE,
            (CorpusProfile.MOE, CorpusProfile.ARCHITECTURE_SPECIFIC),
            MOE_SOURCES,
            (StaticCapability.GROUPED_GEMM,),
        )
    )
    specs.extend(_quantization_specs())
    specs.extend(
        _family_specs(
            CorpusOperationFamily.INDEXING_REDUCTION,
            _INDEXING,
            (CorpusProfile.CORE,),
            COMMON_SOURCES,
        )
    )
    return tuple(specs)


def _family_specs(
    family: CorpusOperationFamily,
    variants: tuple[str, ...],
    profiles: tuple[CorpusProfile, ...],
    sources: tuple[str, ...],
    capabilities: tuple[StaticCapability, ...] = (),
) -> list[ProblemSpec]:
    return [
        ProblemSpec(
            family=family,
            variant=variant,
            profiles=profiles,
            source_ids=sources,
            capabilities=capabilities,
        )
        for variant in variants
    ]


def _quantization_specs() -> list[ProblemSpec]:
    schemes = tuple(QuantizationScheme)
    sources = ("qwen3_8", "deepseek_v4", "kimi_k3", "minimax_m3", "ministral_3")
    return [
        ProblemSpec(
            family=CorpusOperationFamily.QUANTIZATION,
            variant=scheme.value,
            profiles=(CorpusProfile.QUANTIZED,),
            source_ids=sources,
            quantization=scheme,
            capabilities=(StaticCapability.PACKED_LOW_PRECISION,),
        )
        for scheme in schemes
    ]


_LINEAR = (
    "matmul_bias",
    "matmul_relu",
    "matmul_gelu",
    "matmul_silu",
    "matmul_tanh",
    "matmul_softcap",
    "matmul_scaled",
    "matmul_square_relu",
    "batched_matmul",
    "batched_bias",
    "grouped_matmul",
    "grouped_silu",
)
_NORM = (
    "rmsnorm",
    "layernorm",
    "l2norm",
    "silu",
    "gelu",
    "relu_squared",
    "swiglu",
    "geglu",
    "add_rmsnorm",
    "bias_gelu",
)
_POSITION = (
    "rope_interleaved",
    "rope_half",
    "rope_scaled",
    "rope_partial",
    "rope_xpos",
    "rope_yarn",
)
_ATTENTION = (
    "mha_causal",
    "mha_full",
    "gqa_causal",
    "gqa_full",
    "mqa_decode",
    "sliding_window",
    "alibi",
    "softcap",
    "local_global",
    "grouped_query_scale",
    "masked_softmax",
    "attention_sink",
)
_ADVANCED = (
    "mla_compressed",
    "delta_attention",
    "linear_attention",
    "indexed_sparse",
    "block_sparse",
    "topk_sparse",
    "recurrent_state",
    "gated_delta",
    "index_share",
    "sparse_sink",
)
_KV_CACHE = (
    "cache_write",
    "paged_write",
    "cache_gather",
    "cache_scatter",
    "cache_rotate",
    "cache_scale",
    "cache_compress",
    "cache_interleave",
)
_MOE = (
    "route_top1",
    "route_top2",
    "sigmoid_route",
    "softmax_route",
    "shared_expert",
    "expert_bias",
    "grouped_expert",
    "normalized_topk",
    "capacity_route",
    "expert_scale",
    "fused_gate_up",
    "routed_residual",
)
_INDEXING = (
    "embedding_lookup",
    "row_sum",
    "row_mean",
    "row_max",
    "row_l2",
    "logsumexp",
)


def _definition(spec: ProblemSpec) -> Definition:
    builders = {
        CorpusOperationFamily.LINEAR: _linear_definition,
        CorpusOperationFamily.NORM_ACTIVATION: _norm_definition,
        CorpusOperationFamily.POSITION: _position_definition,
        CorpusOperationFamily.ATTENTION: _attention_definition,
        CorpusOperationFamily.ADVANCED_ATTENTION: _advanced_definition,
        CorpusOperationFamily.KV_CACHE: _kv_cache_definition,
        CorpusOperationFamily.MOE: _moe_definition,
        CorpusOperationFamily.QUANTIZATION: _quantization_definition,
        CorpusOperationFamily.INDEXING_REDUCTION: _indexing_definition,
    }
    payload = builders[spec.family](spec)
    return Definition.model_validate(
        {
            "schema_version": BenchmarkArtifactSchema.DEFINITION,
            "name": spec.problem_name,
            "op_type": spec.family.value,
            "description": f"Clean-room LLM Core V1 semantic: {spec.semantic_id}.",
            **payload,
        },
    )


def _axes(*names: str) -> dict[str, dict[str, str]]:
    return {name: {"type": "var", "description": name} for name in names}


def _tensor(shape: list[str], dtype: str = "bfloat16") -> dict[str, Any]:
    return {"shape": shape, "dtype": dtype}


def _linear_definition(spec: ProblemSpec) -> dict[str, Any]:
    batched = any(word in spec.variant for word in ("batched", "grouped"))
    x_shape = ["B", "M", "K"] if batched else ["M", "K"]
    w_shape = ["B", "K", "N"] if batched else ["K", "N"]
    out_shape = ["B", "M", "N"] if batched else ["M", "N"]
    axes = _axes(*(("B", "M", "K", "N") if batched else ("M", "K", "N")))
    mode = _LINEAR.index(spec.variant)
    reference = _linear_reference(mode, batched)
    return {
        "axes": axes,
        "inputs": {
            "x": _tensor(x_shape),
            "weight": _tensor(w_shape),
            "bias": _tensor(["N"]),
        },
        "outputs": {"output": _tensor(out_shape)},
        "reference": reference,
    }


def _linear_reference(mode: int, batched: bool) -> str:
    product = (
        "torch.bmm(x.float(), weight.float())"
        if batched
        else "x.float() @ weight.float()"
    )
    transforms = (
        "y + bias.float()",
        "torch.relu(y + bias.float())",
        "torch.nn.functional.gelu(y + bias.float(), approximate='tanh')",
        "torch.nn.functional.silu(y + bias.float())",
        "torch.tanh(y + bias.float())",
        "torch.tanh((y + bias.float()) / 30.0) * 30.0",
        "(y + bias.float()) * 0.125",
        "torch.relu(y + bias.float()).square()",
        "y",
        "y + bias.float()",
        "y * 0.5",
        "torch.nn.functional.silu(y)",
    )
    return f"""import torch\n\ndef run(x, weight, bias):\n    y = {product}\n    return ({transforms[mode]}).to(torch.bfloat16)\n"""


def _norm_definition(spec: ProblemSpec) -> dict[str, Any]:
    mode = _NORM.index(spec.variant)
    reference = _norm_reference(mode)
    return {
        "axes": _axes("M", "N"),
        "inputs": {
            "x": _tensor(["M", "N"]),
            "weight": _tensor(["N"]),
            "bias": _tensor(["N"]),
        },
        "outputs": {"output": _tensor(["M", "N"])},
        "reference": reference,
    }


def _norm_reference(mode: int) -> str:
    expressions = (
        "x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w",
        "torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, 1e-5)",
        "torch.nn.functional.normalize(x, p=2.0, dim=-1) * w",
        "torch.nn.functional.silu(x) * w + b",
        "torch.nn.functional.gelu(x, approximate='tanh') * w + b",
        "torch.relu(x).square() * w + b",
        "torch.nn.functional.silu(x) * torch.sigmoid(x * 0.5) * w + b",
        "torch.nn.functional.gelu(x) * torch.tanh(x) * w + b",
        "(x + b) / torch.sqrt((x + b).square().mean(-1, keepdim=True) + 1e-6) * w",
        "torch.nn.functional.gelu(x + b, approximate='tanh') * w",
    )
    return f"""import torch\n\ndef run(x, weight, bias):\n    x = x.float()\n    w = weight.float()\n    b = bias.float()\n    return ({expressions[mode]}).to(torch.bfloat16)\n"""


def _position_definition(spec: ProblemSpec) -> dict[str, Any]:
    mode = _POSITION.index(spec.variant)
    return {
        "axes": _axes("B", "H", "S", "D"),
        "inputs": {
            "x": _tensor(["B", "H", "S", "D"]),
            "cos": _tensor(["S", "D"]),
            "sin": _tensor(["S", "D"]),
        },
        "outputs": {"output": _tensor(["B", "H", "S", "D"])},
        "reference": _position_reference(mode),
    }


def _position_reference(mode: int) -> str:
    scales = (1.0, 1.0, 0.5, 0.75, 1.01, 1.25)
    rotate = (
        "torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)"
        if mode != 1
        else "torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)"
    )
    return f"""import torch\n\ndef run(x, cos, sin):\n    x = x.float()\n    rotated = {rotate}\n    c = cos.float()[None, None]\n    s = sin.float()[None, None]\n    return (x * c + rotated * s * {scales[mode]}).to(torch.bfloat16)\n"""


def _attention_definition(spec: ProblemSpec) -> dict[str, Any]:
    return _attention_payload(_ATTENTION.index(spec.variant), advanced=False)


def _advanced_definition(spec: ProblemSpec) -> dict[str, Any]:
    return _attention_payload(_ADVANCED.index(spec.variant), advanced=True)


def _attention_payload(mode: int, *, advanced: bool) -> dict[str, Any]:
    return {
        "axes": _axes("B", "HQ", "HK", "S", "T", "D"),
        "inputs": {
            "q": _tensor(["B", "HQ", "S", "D"]),
            "k": _tensor(["B", "HK", "T", "D"]),
            "v": _tensor(["B", "HK", "T", "D"]),
        },
        "outputs": {"output": _tensor(["B", "HQ", "S", "D"])},
        "reference": _attention_reference(mode, advanced=advanced),
    }


def _attention_reference(mode: int, *, advanced: bool) -> str:
    causal = (mode % 2 == 0) and not advanced
    modifier = (
        _advanced_modifier(mode) if advanced else _attention_modifier(mode)
    )
    mask = (
        "scores = scores.masked_fill(torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1), float('-inf'))"
        if causal
        else "scores = scores"
    )
    return f"""import math\nimport torch\n\ndef run(q, k, v):\n    q = q.float()\n    repeat = q.shape[1] // k.shape[1]\n    k = k.float().repeat_interleave(repeat, dim=1)\n    v = v.float().repeat_interleave(repeat, dim=1)\n    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])\n    {modifier}\n    {mask}\n    probs = torch.softmax(scores, dim=-1)\n    return torch.matmul(probs, v).to(torch.bfloat16)\n"""


def _attention_modifier(mode: int) -> str:
    modifiers = (
        "scores = scores",
        "scores = scores",
        "scores = scores * 0.95",
        "scores = scores * 1.05",
        "scores = scores[:, :, -1:, :].expand_as(scores)",
        "scores = scores.masked_fill(torch.arange(scores.shape[-1], device=scores.device)[None, None, None, :] < max(0, scores.shape[-1] - 4), float('-inf'))",
        "scores = scores + torch.linspace(0.0, -1.0, scores.shape[-1], device=scores.device)",
        "scores = torch.tanh(scores / 30.0) * 30.0",
        "scores = scores + scores.mean(dim=-1, keepdim=True) * 0.01",
        "scores = scores / 1.1",
        "scores = scores - scores.amax(dim=-1, keepdim=True)",
        "scores[..., 0] = scores[..., 0] + 1.0",
    )
    return modifiers[mode]


def _advanced_modifier(mode: int) -> str:
    modifiers = (
        "scores = scores + scores.mean(dim=-1, keepdim=True) * 0.125",
        "scores = torch.cumsum(scores, dim=-1) / torch.arange(1, scores.shape[-1] + 1, device=scores.device)",
        "scores = torch.nn.functional.elu(scores) + 1.0",
        "scores[..., 1::2] = float('-inf')",
        "scores = scores.masked_fill((torch.arange(scores.shape[-1], device=scores.device) // 4)[None, None, None, :] % 2 == 1, float('-inf'))",
        "threshold = scores.topk(min(4, scores.shape[-1]), dim=-1).values[..., -1:]; scores = scores.masked_fill(scores < threshold, float('-inf'))",
        "scores = scores + torch.cumsum(scores, dim=-1) * 0.01",
        "scores = scores * torch.sigmoid(scores)",
        "scores = scores + scores[..., :1] * 0.05",
        "scores[..., 0] = scores[..., 0] + scores.mean(dim=-1)",
    )
    return modifiers[mode]


def _kv_cache_definition(spec: ProblemSpec) -> dict[str, Any]:
    mode = _KV_CACHE.index(spec.variant)
    return {
        "axes": _axes("P", "BS", "B", "S", "H", "D"),
        "inputs": {
            "cache": _tensor(["P", "BS", "H", "D"]),
            "update": _tensor(["B", "S", "H", "D"]),
            "slots": _tensor(["B", "S"], "int64"),
        },
        "outputs": {"output": _tensor(["P", "BS", "H", "D"])},
        "reference": _kv_cache_reference(mode),
    }


def _kv_cache_reference(mode: int) -> str:
    transforms = (
        "u",
        "u * 0.5",
        "u.flip(-1)",
        "u + 0.125",
        "u.roll(1, dims=-2)",
        "u * 1.01",
        "torch.round(u * 16.0) / 16.0",
        "u.roll(1, dims=-1)",
    )
    return f"""import torch\n\ndef run(cache, update, slots):\n    shape = cache.shape\n    flat = cache.float().reshape(-1, shape[-2], shape[-1]).clone()\n    u = update.float().reshape(-1, shape[-2], shape[-1])\n    u = {transforms[mode]}\n    index = slots.reshape(-1).remainder(flat.shape[0]).long()\n    flat.index_copy_(0, index, u)\n    return flat.reshape(shape).to(torch.bfloat16)\n"""


def _moe_definition(spec: ProblemSpec) -> dict[str, Any]:
    mode = _MOE.index(spec.variant)
    return {
        "axes": _axes("M", "D", "E", "O"),
        "inputs": {
            "tokens": _tensor(["M", "D"]),
            "experts": _tensor(["E", "D", "O"]),
            "gate": _tensor(["M", "E"]),
        },
        "outputs": {"output": _tensor(["M", "O"])},
        "reference": _moe_reference(mode),
    }


def _moe_reference(mode: int) -> str:
    topk = 1 if mode in (0, 2, 4, 8) else 2
    gate_expr = (
        "torch.sigmoid(gate.float())"
        if mode in (2, 4, 9)
        else "torch.softmax(gate.float(), dim=-1)"
    )
    post = (
        "out",
        "out * 1.001",
        "out * 1.002",
        "out * 1.003",
        "out + tokens.float().mean(-1, keepdim=True)",
        "out + 0.01",
        "torch.nn.functional.silu(out)",
        "out / weights.sum(-1, keepdim=True).clamp_min(1e-6)",
        "out.clamp(-8.0, 8.0)",
        "out * 1.1",
        "torch.nn.functional.gelu(out)",
        "out + tokens.float().mean(-1, keepdim=True) * 0.1",
    )[mode]
    return f"""import torch\n\ndef run(tokens, experts, gate):\n    probs = {gate_expr}\n    weights, ids = torch.topk(probs, k=min({topk}, probs.shape[-1]), dim=-1)\n    out = torch.zeros(tokens.shape[0], experts.shape[-1], dtype=torch.float32, device=tokens.device)\n    for choice in range(ids.shape[-1]):\n        selected = experts.float()[ids[:, choice]]\n        projected = torch.bmm(tokens.float().unsqueeze(1), selected).squeeze(1)\n        out = out + projected * weights[:, choice:choice + 1]\n    return ({post}).to(torch.bfloat16)\n"""


def _quantization_definition(spec: ProblemSpec) -> dict[str, Any]:
    mode = tuple(QuantizationScheme).index(spec.quantization)
    expressions = (
        "torch.round(x / scale).clamp(-448, 448) * scale",
        "torch.round(x / row_scale).clamp(-448, 448) * row_scale",
        "torch.round(x / block_scale).clamp(-448, 448) * block_scale",
        "torch.round(x / scale).clamp(-7, 7) * scale",
        "torch.round(x / block_scale).clamp(-7, 7) * block_scale",
        "torch.round(x / scale).clamp(-127, 127) * scale",
        "torch.round(x / row_scale).clamp(-127, 127) * row_scale",
        "torch.round(x * 16.0).clamp(-127, 127) / 16.0",
    )
    reference = f"""import torch\n\ndef run(x):\n    x = x.float()\n    scale = x.abs().amax().clamp_min(1e-6) / 127.0\n    row_scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0\n    block_scale = x.reshape(x.shape[0], -1, 16).abs().amax(-1, keepdim=True).repeat_interleave(16, -1).reshape_as(x).clamp_min(1e-6) / 127.0\n    return ({expressions[mode]}).to(torch.bfloat16)\n"""
    return {
        "axes": _axes("M", "N"),
        "inputs": {"x": _tensor(["M", "N"])},
        "outputs": {"output": _tensor(["M", "N"])},
        "reference": reference,
    }


def _indexing_definition(spec: ProblemSpec) -> dict[str, Any]:
    mode = _INDEXING.index(spec.variant)
    if mode == 0:
        reference = "import torch\n\ndef run(table, indices):\n    return table.index_select(0, indices.remainder(table.shape[0]).long())\n"
        return {
            "axes": _axes("V", "M", "D"),
            "inputs": {
                "table": _tensor(["V", "D"]),
                "indices": _tensor(["M"], "int64"),
            },
            "outputs": {"output": _tensor(["M", "D"])},
            "reference": reference,
        }
    expressions = (
        "x.sum(-1)",
        "x.mean(-1)",
        "x.amax(-1)",
        "torch.linalg.vector_norm(x, dim=-1)",
        "torch.logsumexp(x, dim=-1)",
    )
    reference = f"import torch\n\ndef run(x):\n    x = x.float()\n    return ({expressions[mode - 1]}).to(torch.bfloat16)\n"
    return {
        "axes": _axes("M", "N"),
        "inputs": {"x": _tensor(["M", "N"])},
        "outputs": {"output": _tensor(["M"])},
        "reference": reference,
    }


def _shape_rows(
    spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    factories = {
        CorpusOperationFamily.LINEAR: _linear_shapes,
        CorpusOperationFamily.NORM_ACTIVATION: _matrix_shapes,
        CorpusOperationFamily.POSITION: _position_shapes,
        CorpusOperationFamily.ATTENTION: _attention_shapes,
        CorpusOperationFamily.ADVANCED_ATTENTION: _attention_shapes,
        CorpusOperationFamily.KV_CACHE: _kv_shapes,
        CorpusOperationFamily.MOE: _moe_shapes,
        CorpusOperationFamily.QUANTIZATION: _quant_shapes,
        CorpusOperationFamily.INDEXING_REDUCTION: _indexing_shapes,
    }
    rows = factories[spec.family](spec)
    if len(rows) != WORKLOADS_PER_PROBLEM:
        raise ValueError(
            f"{spec.semantic_id} generated {len(rows)} workloads, "
            f"expected {WORKLOADS_PER_PROBLEM}",
        )
    return rows


def _tiered(
    values: dict[ShapeTier, tuple[dict[str, int], ...]],
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    return tuple((tier, axes) for tier, rows in values.items() for axes in rows)


def _linear_shapes(
    spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    batched = any(word in spec.variant for word in ("batched", "grouped"))
    rows = _tiered(
        {
            ShapeTier.MICRO: (
                {"M": 2, "K": 8, "N": 8},
                {"M": 3, "K": 16, "N": 12},
                {"M": 4, "K": 32, "N": 16},
            ),
            ShapeTier.EDGE: (
                {"M": 7, "K": 127, "N": 65},
                {"M": 13, "K": 257, "N": 129},
                {"M": 31, "K": 511, "N": 257},
            ),
            ShapeTier.DECODE: (
                {"M": 1, "K": 1024, "N": 4096},
                {"M": 2, "K": 2048, "N": 4096},
                {"M": 4, "K": 4096, "N": 4096},
            ),
            ShapeTier.PREFILL: (
                {"M": 128, "K": 2048, "N": 4096},
                {"M": 256, "K": 4096, "N": 4096},
                {"M": 512, "K": 4096, "N": 8192},
            ),
            ShapeTier.LONG_CONTEXT: (
                {"M": 1024, "K": 4096, "N": 4096},
                {"M": 2048, "K": 4096, "N": 8192},
                {"M": 4096, "K": 8192, "N": 4096},
            ),
        }
    )
    if not batched:
        return rows
    return tuple(
        (tier, {"B": 4 if tier is ShapeTier.MICRO else 8, **axes})
        for tier, axes in rows
    )


def _matrix_shapes(
    _spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    return _tiered(
        {
            ShapeTier.MICRO: (
                {"M": 2, "N": 16},
                {"M": 3, "N": 32},
                {"M": 4, "N": 64},
            ),
            ShapeTier.EDGE: (
                {"M": 7, "N": 127},
                {"M": 13, "N": 513},
                {"M": 31, "N": 1025},
            ),
            ShapeTier.DECODE: (
                {"M": 1, "N": 4096},
                {"M": 2, "N": 8192},
                {"M": 8, "N": 16384},
            ),
            ShapeTier.PREFILL: (
                {"M": 256, "N": 4096},
                {"M": 512, "N": 8192},
                {"M": 1024, "N": 4096},
            ),
            ShapeTier.LONG_CONTEXT: (
                {"M": 4096, "N": 4096},
                {"M": 8192, "N": 4096},
                {"M": 16384, "N": 8192},
            ),
        }
    )


def _position_shapes(
    _spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    return _tiered(
        {
            ShapeTier.MICRO: (
                {"B": 1, "H": 2, "S": 4, "D": 8},
                {"B": 1, "H": 4, "S": 8, "D": 16},
                {"B": 2, "H": 4, "S": 16, "D": 32},
            ),
            ShapeTier.EDGE: (
                {"B": 1, "H": 7, "S": 127, "D": 64},
                {"B": 2, "H": 5, "S": 257, "D": 128},
                {"B": 1, "H": 13, "S": 513, "D": 64},
            ),
            ShapeTier.DECODE: (
                {"B": 1, "H": 32, "S": 1, "D": 128},
                {"B": 8, "H": 32, "S": 1, "D": 128},
                {"B": 32, "H": 16, "S": 1, "D": 256},
            ),
            ShapeTier.PREFILL: (
                {"B": 1, "H": 32, "S": 512, "D": 128},
                {"B": 2, "H": 32, "S": 1024, "D": 128},
                {"B": 1, "H": 64, "S": 2048, "D": 128},
            ),
            ShapeTier.LONG_CONTEXT: (
                {"B": 1, "H": 32, "S": 8192, "D": 128},
                {"B": 1, "H": 16, "S": 32768, "D": 128},
                {"B": 1, "H": 8, "S": 131072, "D": 128},
            ),
        }
    )


def _attention_shapes(
    _spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    return _tiered(
        {
            ShapeTier.MICRO: (
                {"B": 1, "HQ": 2, "HK": 1, "S": 2, "T": 2, "D": 8},
                {"B": 1, "HQ": 4, "HK": 2, "S": 4, "T": 4, "D": 16},
                {"B": 2, "HQ": 4, "HK": 1, "S": 8, "T": 8, "D": 32},
            ),
            ShapeTier.EDGE: (
                {"B": 1, "HQ": 8, "HK": 2, "S": 7, "T": 127, "D": 64},
                {"B": 2, "HQ": 12, "HK": 3, "S": 13, "T": 257, "D": 80},
                {"B": 1, "HQ": 16, "HK": 4, "S": 31, "T": 513, "D": 96},
            ),
            ShapeTier.DECODE: (
                {"B": 1, "HQ": 32, "HK": 8, "S": 1, "T": 2048, "D": 128},
                {"B": 8, "HQ": 32, "HK": 8, "S": 1, "T": 8192, "D": 128},
                {"B": 32, "HQ": 64, "HK": 8, "S": 1, "T": 32768, "D": 128},
            ),
            ShapeTier.PREFILL: (
                {"B": 1, "HQ": 32, "HK": 8, "S": 256, "T": 256, "D": 128},
                {"B": 2, "HQ": 32, "HK": 8, "S": 512, "T": 512, "D": 128},
                {"B": 1, "HQ": 64, "HK": 8, "S": 1024, "T": 1024, "D": 128},
            ),
            ShapeTier.LONG_CONTEXT: (
                {"B": 1, "HQ": 32, "HK": 8, "S": 2048, "T": 2048, "D": 128},
                {"B": 1, "HQ": 32, "HK": 4, "S": 4096, "T": 4096, "D": 128},
                {"B": 1, "HQ": 16, "HK": 2, "S": 8192, "T": 8192, "D": 128},
            ),
        }
    )


def _kv_shapes(
    _spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    return _tiered(
        {
            ShapeTier.MICRO: (
                {"P": 2, "BS": 4, "B": 1, "S": 2, "H": 2, "D": 8},
                {"P": 4, "BS": 4, "B": 1, "S": 4, "H": 2, "D": 16},
                {"P": 4, "BS": 8, "B": 2, "S": 4, "H": 4, "D": 32},
            ),
            ShapeTier.EDGE: (
                {"P": 17, "BS": 7, "B": 1, "S": 7, "H": 5, "D": 64},
                {"P": 33, "BS": 13, "B": 2, "S": 13, "H": 7, "D": 80},
                {"P": 65, "BS": 17, "B": 1, "S": 31, "H": 9, "D": 96},
            ),
            ShapeTier.DECODE: (
                {"P": 128, "BS": 16, "B": 1, "S": 1, "H": 8, "D": 128},
                {"P": 512, "BS": 16, "B": 8, "S": 1, "H": 8, "D": 128},
                {"P": 2048, "BS": 16, "B": 32, "S": 1, "H": 8, "D": 128},
            ),
            ShapeTier.PREFILL: (
                {"P": 128, "BS": 16, "B": 1, "S": 256, "H": 8, "D": 128},
                {"P": 512, "BS": 16, "B": 2, "S": 512, "H": 8, "D": 128},
                {"P": 1024, "BS": 16, "B": 1, "S": 1024, "H": 16, "D": 128},
            ),
            ShapeTier.LONG_CONTEXT: (
                {"P": 2048, "BS": 16, "B": 1, "S": 2048, "H": 8, "D": 128},
                {"P": 4096, "BS": 16, "B": 1, "S": 4096, "H": 8, "D": 128},
                {"P": 8192, "BS": 16, "B": 1, "S": 8192, "H": 8, "D": 128},
            ),
        }
    )


def _moe_shapes(
    _spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    return _tiered(
        {
            ShapeTier.MICRO: (
                {"M": 2, "D": 8, "E": 4, "O": 8},
                {"M": 4, "D": 16, "E": 8, "O": 16},
                {"M": 8, "D": 32, "E": 8, "O": 24},
            ),
            ShapeTier.EDGE: (
                {"M": 7, "D": 127, "E": 17, "O": 65},
                {"M": 13, "D": 257, "E": 33, "O": 129},
                {"M": 31, "D": 513, "E": 65, "O": 257},
            ),
            ShapeTier.DECODE: (
                {"M": 1, "D": 2048, "E": 64, "O": 4096},
                {"M": 8, "D": 4096, "E": 256, "O": 4096},
                {"M": 32, "D": 4096, "E": 512, "O": 4096},
            ),
            ShapeTier.PREFILL: (
                {"M": 128, "D": 2048, "E": 64, "O": 4096},
                {"M": 256, "D": 4096, "E": 256, "O": 4096},
                {"M": 512, "D": 4096, "E": 512, "O": 4096},
            ),
            ShapeTier.LONG_CONTEXT: (
                {"M": 1024, "D": 4096, "E": 256, "O": 4096},
                {"M": 2048, "D": 4096, "E": 512, "O": 4096},
                {"M": 4096, "D": 8192, "E": 896, "O": 4096},
            ),
        }
    )


def _quant_shapes(
    spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    rows = _matrix_shapes(spec)
    return tuple(
        (tier, {"M": axes["M"], "N": max(16, math.ceil(axes["N"] / 16) * 16)})
        for tier, axes in rows
    )


def _indexing_shapes(
    spec: ProblemSpec,
) -> tuple[tuple[ShapeTier, dict[str, int]], ...]:
    if spec.variant != "embedding_lookup":
        return _matrix_shapes(spec)
    rows = _matrix_shapes(spec)
    return tuple(
        (
            tier,
            {
                "V": max(32, axes["N"]),
                "M": axes["M"],
                "D": 64 if tier is ShapeTier.MICRO else 128,
            },
        )
        for tier, axes in rows
    )


def _workloads(
    spec: ProblemSpec, definition: Definition
) -> tuple[list[Workload], tuple[CorpusWorkloadRecord, ...]]:
    workloads: list[Workload] = []
    records: list[CorpusWorkloadRecord] = []
    for index, (tier, axes) in enumerate(_shape_rows(spec)):
        workload = Workload.model_validate(
            {
                "schema_version": BenchmarkArtifactSchema.WORKLOAD,
                "axes": axes,
                "inputs": _workload_inputs(spec, definition),
                "uuid": str(
                    uuid.uuid5(
                        NAMESPACE,
                        f"{spec.semantic_id}:{tier.value}:{index}:{json.dumps(axes, sort_keys=True)}",
                    )
                ),
                "checks": [
                    {
                        "type": "numeric",
                        "output": "output",
                        "max_atol": 0.125,
                        "max_rtol": 0.125,
                        "required_matched_ratio": 1.0,
                    }
                ],
            }
        )
        requirements = _requirements(spec, definition, workload)
        workloads.append(workload)
        records.append(
            CorpusWorkloadRecord(
                uuid=workload.uuid,
                shape_tier=tier,
                source_ids=(spec.source_ids[index % len(spec.source_ids)],),
                requirements=requirements,
            )
        )
    validate_problem_contract(definition, workloads)
    return workloads, tuple(records)


def _workload_inputs(
    spec: ProblemSpec, definition: Definition
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        name: {"type": "random"} for name in definition.inputs
    }
    if spec.family is CorpusOperationFamily.KV_CACHE:
        inputs["slots"] = {
            "type": "generated",
            "generator": {"type": "integer", "low": 0, "high": "P * BS"},
        }
    if spec.variant == "embedding_lookup":
        inputs["indices"] = {
            "type": "generated",
            "generator": {"type": "integer", "low": 0, "high": "V"},
        }
    return inputs


def _requirements(
    spec: ProblemSpec, definition: Definition, workload: Workload
) -> StaticRequirements:
    input_sizes = _tensor_sizes(
        definition.inputs, definition.get_input_shapes(workload.axes)
    )
    output_sizes = _tensor_sizes(
        definition.outputs, definition.get_output_shapes(workload.axes)
    )
    input_bytes = sum(input_sizes)
    output_bytes = sum(output_sizes)
    temporary = _temporary_bytes(spec, workload.axes, output_bytes)
    dtypes = {
        tensor.dtype
        for tensor in (
            *definition.inputs.values(),
            *definition.outputs.values(),
        )
    }
    dtypes.update(_quantization_dtypes(spec.quantization))
    resources = ResourceEnvelope(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        max_tensor_bytes=max((*input_sizes, *output_sizes), default=0),
        reference_ipc_bytes=input_bytes + output_bytes,
        temporary_bytes=temporary,
        reference_peak_bytes=input_bytes + output_bytes + temporary,
    )
    return StaticRequirements(
        dtypes=tuple(sorted(dtypes, key=str)),
        quantization=(
            () if spec.quantization is None else (spec.quantization,)
        ),
        capabilities=(StaticCapability.DENSE_TENSOR, *spec.capabilities),
        resources=resources,
    )


def _tensor_sizes(
    tensors: Any, shapes: dict[str, tuple[int, ...] | None]
) -> list[int]:
    sizes: list[int] = []
    for name, tensor in tensors.items():
        shape = shapes[name]
        elements = 1 if shape is None else math.prod(shape)
        sizes.append((elements * dtype_storage_bits(tensor.dtype) + 7) // 8)
    return sizes


def _temporary_bytes(
    spec: ProblemSpec, axes: dict[str, int], output_bytes: int
) -> int:
    if spec.family is CorpusOperationFamily.ATTENTION:
        return axes["B"] * axes["HQ"] * axes["S"] * axes["T"] * 4
    if spec.family is CorpusOperationFamily.ADVANCED_ATTENTION:
        dense = axes["B"] * axes["HQ"] * axes["S"] * axes["T"] * 4
        return max(output_bytes, dense // 8)
    if spec.family is CorpusOperationFamily.MOE:
        return axes["M"] * axes["E"] * 4 + output_bytes * 2
    return output_bytes * 2


def _quantization_dtypes(scheme: QuantizationScheme | None) -> set[DType]:
    if scheme in (
        QuantizationScheme.FP8_PER_TENSOR,
        QuantizationScheme.FP8_PER_TOKEN,
    ):
        return {DType.FLOAT8_E4M3FN}
    if scheme is QuantizationScheme.MXFP8_BLOCK:
        return {DType.FLOAT8_E5M2}
    if scheme in (QuantizationScheme.FP4_GROUP, QuantizationScheme.MXFP4_BLOCK):
        return {DType.FLOAT4_E2M1FN_X2}
    if scheme is not None:
        return {DType.INT8}
    return set()


def _load_sources() -> tuple[ModelSource, ...]:
    raw = yaml.safe_load(MODEL_REGISTRY.read_text(encoding="utf-8")) or {}
    return tuple(ModelSource.model_validate(item) for item in raw["sources"])


def _coverage_policy() -> CorpusCoveragePolicy:
    return CorpusCoveragePolicy(
        minimum_definitions=80,
        minimum_workloads=1200,
        operation_minimum_definitions={
            CorpusOperationFamily.LINEAR: 12,
            CorpusOperationFamily.NORM_ACTIVATION: 10,
            CorpusOperationFamily.POSITION: 6,
            CorpusOperationFamily.ATTENTION: 12,
            CorpusOperationFamily.ADVANCED_ATTENTION: 10,
            CorpusOperationFamily.KV_CACHE: 8,
            CorpusOperationFamily.MOE: 12,
            CorpusOperationFamily.QUANTIZATION: 8,
            CorpusOperationFamily.INDEXING_REDUCTION: 6,
        },
        profile_minimum_workloads={
            CorpusProfile.CORE: 600,
            CorpusProfile.MOE: 180,
            CorpusProfile.KV_CACHE: 120,
            CorpusProfile.LONG_CONTEXT: 300,
            CorpusProfile.QUANTIZED: 120,
            CorpusProfile.ARCHITECTURE_SPECIFIC: 300,
        },
    )


def build(destination: Path, metadata_root: Path) -> CorpusManifest:
    """Build all release artifacts below an empty destination."""
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[CorpusEntry] = []
    semantic_records: list[dict[str, Any]] = []
    for spec in _specs():
        definition = _definition(spec)
        workloads, records = _workloads(spec, definition)
        problem_dir = (
            destination / "problems" / spec.family.value / spec.problem_name
        )
        definition_path = problem_dir / "definition.json"
        workload_path = problem_dir / "workload.jsonl"
        atomic_write_json_value(
            definition_path,
            definition.model_dump(mode="json"),
            sort_keys=False,
        )
        atomic_write_jsonl_values(workload_path, workloads)
        relative_definition = definition_path.relative_to(
            destination
        ).as_posix()
        relative_workload = workload_path.relative_to(destination).as_posix()
        entry_payload = {"operation_family": spec.family}
        fingerprint = semantic_fingerprint(definition, entry_payload)
        entry = CorpusEntry(
            semantic_id=spec.semantic_id,
            semantic_fingerprint=fingerprint,
            problem_name=spec.problem_name,
            operation_family=spec.family,
            profiles=spec.profiles,
            source_ids=spec.source_ids,
            definition_path=relative_definition,
            workload_path=relative_workload,
            definition_sha256=sha256_file(definition_path),
            workload_sha256=sha256_file(workload_path),
            workloads=records,
        )
        entries.append(entry)
        semantic_records.append(_semantic_record(spec, fingerprint))
    manifest = CorpusManifest(
        schema_version=DatasetArtifactSchema.CORPUS_MANIFEST,
        corpus_id="LLM_CORE",
        release_id="LLM_CORE_V1",
        release_state=CorpusReleaseState.FROZEN,
        license="Apache-2.0",
        source_freeze_date="2026-08-14",
        profiles=tuple(CorpusProfile),
        sources=_load_sources(),
        coverage_policy=_coverage_policy(),
        entries=tuple(entries),
    )
    _write_yaml(destination / "manifest.yaml", manifest.model_dump(mode="json"))
    _write_registry_outputs(metadata_root, semantic_records)
    return manifest


def _semantic_record(spec: ProblemSpec, fingerprint: str) -> dict[str, Any]:
    return {
        "semantic_id": spec.semantic_id,
        "semantic_fingerprint": fingerprint,
        "operation_family": spec.family.value,
        "variant": spec.variant,
        "profiles": [profile.value for profile in spec.profiles],
        "source_ids": list(spec.source_ids),
        "quantization": None
        if spec.quantization is None
        else spec.quantization.value,
        "capabilities": [capability.value for capability in spec.capabilities],
        "status": "frozen_in_llm_core_v1",
    }


def _write_registry_outputs(
    metadata_root: Path,
    records: list[dict[str, Any]],
) -> None:
    _write_yaml(
        metadata_root / "semantics.yaml",
        {"schema": "llm_core_semantic_registry.v1", "semantics": records},
    )
    _write_yaml(
        metadata_root / "candidates.yaml",
        {
            "schema": "llm_core_candidates.v1",
            "release_candidate": "LLM_CORE_V1",
            "source_freeze_date": "2026-08-14",
            "semantic_ids": [record["semantic_id"] for record in records],
            "next_release": "LLM_CORE_V2",
            "mutation_policy": "append candidates; never rewrite frozen V1",
        },
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _compare_trees(expected: Path, observed: Path) -> list[str]:
    expected_files = {
        path.relative_to(expected)
        for path in expected.rglob("*")
        if path.is_file()
    }
    observed_files = {
        path.relative_to(observed)
        for path in observed.rglob("*")
        if path.is_file()
    }
    differences = [
        f"missing generated file: {path}"
        for path in sorted(expected_files - observed_files)
    ]
    differences.extend(
        f"unexpected generated file: {path}"
        for path in sorted(observed_files - expected_files)
    )
    for relative in sorted(expected_files & observed_files):
        if (expected / relative).read_bytes() != (
            observed / relative
        ).read_bytes():
            differences.append(f"generated file differs: {relative}")
    return differences


def _write_release() -> None:
    with tempfile.TemporaryDirectory(prefix="llm-core-build-") as temp:
        generated = Path(temp) / "LLM_CORE_V1"
        metadata = Path(temp) / "metadata"
        build(generated, metadata)
        if RELEASE_ROOT.exists():
            shutil.rmtree(RELEASE_ROOT)
        shutil.copytree(generated, RELEASE_ROOT)
        SEMANTIC_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata / "semantics.yaml", SEMANTIC_REGISTRY)
        shutil.copyfile(metadata / "candidates.yaml", CANDIDATE_MANIFEST)


def _check_release() -> None:
    with tempfile.TemporaryDirectory(prefix="llm-core-check-") as temp:
        generated = Path(temp) / "LLM_CORE_V1"
        metadata = Path(temp) / "metadata"
        build(generated, metadata)
        differences = _compare_trees(generated, RELEASE_ROOT)
        metadata_targets = {
            metadata / "semantics.yaml": SEMANTIC_REGISTRY,
            metadata / "candidates.yaml": CANDIDATE_MANIFEST,
        }
        for expected, observed in metadata_targets.items():
            if (
                not observed.is_file()
                or expected.read_bytes() != observed.read_bytes()
            ):
                differences.append(
                    f"generated file differs: {observed.relative_to(CORPUS_ROOT)}"
                )
        if differences:
            raise SystemExit("\n".join(differences))


def main() -> None:
    """Build or verify the committed LLM Core corpus."""
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write:
        _write_release()
    else:
        _check_release()


if __name__ == "__main__":
    main()
