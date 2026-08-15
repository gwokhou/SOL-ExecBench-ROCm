#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Deterministically build the leaderboard-oriented LLM Core V2 corpus."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
)
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.dataset.corpus import semantic_fingerprint
from sol_execbench.core.dataset.corpus_models import (
    WORKLOAD_GENERATOR_VERSION,
    CorpusCoveragePolicy,
    CorpusEntry,
    CorpusGenerationPolicy,
    CorpusManifest,
    CorpusOperationFamily,
    CorpusProfile,
    CorpusReleaseState,
    GenerationSlotRule,
    ModelSource,
    QuantizationScheme,
    ServingPhase,
    ShapeBinding,
    StaticCapability,
    WorkloadGenerationRule,
    WorkloadRegime,
    WorkloadRole,
)
from sol_execbench.core.dataset.schema_versions import DatasetArtifactSchema
from sol_execbench.core.integrity import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPOSITORY_ROOT / "problems" / "LLM_CORE"
MODEL_REGISTRY = CORPUS_ROOT / "registry" / "models.yaml"
SEMANTIC_REGISTRY = CORPUS_ROOT / "registry" / "semantics.yaml"
RELEASE_ROOT = CORPUS_ROOT / "releases" / "LLM_CORE_V2"
PUBLIC_GFX_TARGETS = ("gfx1200", "gfx942")


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
        return f"llm.{self.family.value}.{self.variant}.v2"

    @property
    def problem_name(self) -> str:
        """Return the filesystem-safe problem name."""
        return f"{self.family.value}_{self.variant}"


COMMON_SOURCES = ("ministral_3",)
MOE_SOURCES = ("deepseek_v4",)
LONG_SOURCES = ("minimax_m3",)


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
    if len(specs) != 36:
        raise ValueError(
            f"LLM Core V2 requires 36 definitions, got {len(specs)}"
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
            source_ids=(_primary_source(family, variant),),
            capabilities=capabilities,
        )
        for variant in variants
    ]


def _quantization_specs() -> list[ProblemSpec]:
    schemes = (
        QuantizationScheme.FP8_PER_TOKEN,
        QuantizationScheme.MXFP8_BLOCK,
        QuantizationScheme.MXFP4_BLOCK,
        QuantizationScheme.INT8_WEIGHT_ONLY,
    )
    return [
        ProblemSpec(
            family=CorpusOperationFamily.QUANTIZATION,
            variant=scheme.value,
            profiles=(CorpusProfile.QUANTIZED,),
            source_ids=(_primary_quantization_source(scheme),),
            quantization=scheme,
            capabilities=(StaticCapability.PACKED_LOW_PRECISION,),
        )
        for scheme in schemes
    ]


def _primary_source(
    family: CorpusOperationFamily,
    variant: str,
) -> str:
    special = {
        (CorpusOperationFamily.POSITION, "rope_yarn"): "ministral_3",
        (CorpusOperationFamily.ATTENTION, "gqa_causal"): "qwen3_8",
        (CorpusOperationFamily.ATTENTION, "mqa_decode"): "deepseek_v4",
        (CorpusOperationFamily.ATTENTION, "sliding_window"): "gemma_4",
        (CorpusOperationFamily.ATTENTION, "local_global"): "gemma_4",
        (CorpusOperationFamily.ATTENTION, "masked_softmax"): "minimax_m3",
        (CorpusOperationFamily.ATTENTION, "attention_sink"): "deepseek_v4",
        (CorpusOperationFamily.ADVANCED_ATTENTION, "mla_compressed"): (
            "deepseek_v4"
        ),
        (CorpusOperationFamily.ADVANCED_ATTENTION, "indexed_sparse"): (
            "glm_5_2"
        ),
        (CorpusOperationFamily.KV_CACHE, "paged_write"): "llama_4",
        (CorpusOperationFamily.KV_CACHE, "cache_gather"): "qwen3_8",
        (CorpusOperationFamily.KV_CACHE, "cache_scatter"): "minimax_m3",
        (CorpusOperationFamily.KV_CACHE, "cache_compress"): "deepseek_v4",
        (CorpusOperationFamily.MOE, "route_top2"): "minimax_m3",
        (CorpusOperationFamily.MOE, "shared_expert"): "deepseek_v4",
        (CorpusOperationFamily.MOE, "grouped_expert"): "qwen3_8",
        (CorpusOperationFamily.MOE, "normalized_topk"): "glm_5_2",
        (CorpusOperationFamily.MOE, "fused_gate_up"): "kimi_k3",
        (CorpusOperationFamily.MOE, "routed_residual"): "llama_4",
    }
    defaults = {
        CorpusOperationFamily.LINEAR: "ministral_3",
        CorpusOperationFamily.NORM_ACTIVATION: "qwen3_8",
        CorpusOperationFamily.POSITION: "gemma_4",
    }
    return special.get((family, variant), defaults.get(family, "qwen3_8"))


def _primary_quantization_source(scheme: QuantizationScheme) -> str:
    return {
        QuantizationScheme.FP8_PER_TOKEN: "ministral_3",
        QuantizationScheme.MXFP8_BLOCK: "deepseek_v4",
        QuantizationScheme.MXFP4_BLOCK: "kimi_k3",
        QuantizationScheme.INT8_WEIGHT_ONLY: "llama_4",
    }[scheme]


_LINEAR = (
    "matmul_bias",
    "matmul_relu",
    "matmul_gelu",
    "matmul_silu",
    "batched_matmul",
    "batched_bias",
    "grouped_matmul",
    "grouped_silu",
)
_NORM = (
    "rmsnorm",
    "swiglu",
    "add_rmsnorm",
    "bias_gelu",
)
_POSITION = (
    "rope_interleaved",
    "rope_yarn",
)
_ATTENTION = (
    "gqa_causal",
    "mqa_decode",
    "sliding_window",
    "local_global",
    "masked_softmax",
    "attention_sink",
)
_ADVANCED = (
    "mla_compressed",
    "indexed_sparse",
)
_KV_CACHE = (
    "paged_write",
    "cache_gather",
    "cache_scatter",
    "cache_compress",
)
_MOE = (
    "route_top2",
    "shared_expert",
    "grouped_expert",
    "normalized_topk",
    "fused_gate_up",
    "routed_residual",
)
_LINEAR_MODE = dict(zip(_LINEAR, (0, 1, 2, 3, 8, 9, 10, 11), strict=True))
_NORM_MODE = dict(zip(_NORM, (0, 6, 8, 9), strict=True))
_POSITION_MODE = dict(zip(_POSITION, (0, 5), strict=True))
_ATTENTION_MODE = dict(zip(_ATTENTION, (2, 4, 5, 8, 10, 11), strict=True))
_ADVANCED_MODE = dict(zip(_ADVANCED, (0, 3), strict=True))
_KV_CACHE_MODE = dict(zip(_KV_CACHE, (1, 2, 3, 6), strict=True))
_MOE_MODE = dict(zip(_MOE, (1, 4, 6, 7, 10, 11), strict=True))
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
            "description": f"Production-derived LLM Core V2 semantic: {spec.semantic_id}.",
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
    mode = _LINEAR_MODE[spec.variant]
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
    mode = _NORM_MODE[spec.variant]
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
    mode = _POSITION_MODE[spec.variant]
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
    return _attention_payload(_ATTENTION_MODE[spec.variant], advanced=False)


def _advanced_definition(spec: ProblemSpec) -> dict[str, Any]:
    return _attention_payload(_ADVANCED_MODE[spec.variant], advanced=True)


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
    mode = _KV_CACHE_MODE[spec.variant]
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
    mode = _MOE_MODE[spec.variant]
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


def _eligible_gfx_targets(spec: ProblemSpec) -> tuple[str, ...]:
    if spec.quantization is QuantizationScheme.MXFP8_BLOCK:
        return ("gfx942",)
    if spec.quantization is QuantizationScheme.MXFP4_BLOCK:
        return ("gfx1200",)
    return PUBLIC_GFX_TARGETS


@cache
def _load_sources() -> tuple[ModelSource, ...]:
    raw = yaml.safe_load(MODEL_REGISTRY.read_text(encoding="utf-8")) or {}
    return tuple(ModelSource.model_validate(item) for item in raw["sources"])


def _coverage_policy() -> CorpusCoveragePolicy:
    return CorpusCoveragePolicy(
        definition_count=36,
        operation_definition_counts={
            CorpusOperationFamily.LINEAR: 8,
            CorpusOperationFamily.NORM_ACTIVATION: 4,
            CorpusOperationFamily.POSITION: 2,
            CorpusOperationFamily.ATTENTION: 6,
            CorpusOperationFamily.ADVANCED_ATTENTION: 2,
            CorpusOperationFamily.KV_CACHE: 4,
            CorpusOperationFamily.MOE: 6,
            CorpusOperationFamily.QUANTIZATION: 4,
            CorpusOperationFamily.INDEXING_REDUCTION: 0,
        },
        profile_minimum_generated_definitions={
            CorpusProfile.CORE: 20,
            CorpusProfile.MOE: 6,
            CorpusProfile.KV_CACHE: 4,
            CorpusProfile.LONG_CONTEXT: 14,
            CorpusProfile.QUANTIZED: 3,
            CorpusProfile.ARCHITECTURE_SPECIFIC: 8,
        },
    )


def _generation_slots() -> tuple[GenerationSlotRule, ...]:
    rows = (
        (
            "smoke",
            WorkloadRole.SMOKE,
            WorkloadRegime.LATENCY,
            ServingPhase.NOT_APPLICABLE,
            ShapeBinding.BOUNDARY,
            1,
            64,
            False,
        ),
        (
            "latency-low",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.LATENCY,
            ServingPhase.DECODE,
            ShapeBinding.MODEL,
            1,
            8,
            False,
        ),
        (
            "latency-high",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.LATENCY,
            ServingPhase.DECODE,
            ShapeBinding.MODEL,
            1,
            6,
            False,
        ),
        (
            "throughput-low",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.THROUGHPUT,
            ServingPhase.PREFILL,
            ShapeBinding.MODEL,
            1,
            3,
            False,
        ),
        (
            "throughput-high",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.THROUGHPUT,
            ServingPhase.PREFILL,
            ShapeBinding.MODEL,
            1,
            2,
            False,
        ),
        (
            "irregular-low",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.IRREGULAR,
            ServingPhase.NOT_APPLICABLE,
            ShapeBinding.BOUNDARY,
            1,
            4,
            True,
        ),
        (
            "irregular-high",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.IRREGULAR,
            ServingPhase.NOT_APPLICABLE,
            ShapeBinding.BOUNDARY,
            3,
            8,
            True,
        ),
        (
            "capacity-low",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.CAPACITY,
            ServingPhase.PREFILL,
            ShapeBinding.MODEL,
            3,
            4,
            False,
        ),
        (
            "capacity-high",
            WorkloadRole.DEVELOPMENT,
            WorkloadRegime.CAPACITY,
            ServingPhase.PREFILL,
            ShapeBinding.MODEL,
            1,
            1,
            False,
        ),
    )
    return tuple(
        GenerationSlotRule(
            slot_id=slot_id,
            role=role,
            regime=regime,
            serving_phase=phase,
            binding=binding,
            scale_numerator=numerator,
            scale_denominator=denominator,
            irregular=irregular,
        )
        for slot_id, role, regime, phase, binding, numerator, denominator, irregular in rows
    )


def _generation_rule(spec: ProblemSpec) -> WorkloadGenerationRule:
    return WorkloadGenerationRule(
        schema_version=DatasetArtifactSchema.WORKLOAD_GENERATION_RULE,
        semantic_id=spec.semantic_id,
        algorithm_version=WORKLOAD_GENERATOR_VERSION,
        operation_family=spec.family,
        variant=spec.variant,
        source_ids=spec.source_ids,
        eligible_gfx_targets=_eligible_gfx_targets(spec),
        quantization=spec.quantization,
        capabilities=spec.capabilities,
        slots=_generation_slots(),
    )


def build(destination: Path, metadata_root: Path) -> CorpusManifest:
    """Build all release artifacts below an empty destination."""
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[CorpusEntry] = []
    semantic_records: list[dict[str, Any]] = []
    for spec in _specs():
        definition = _definition(spec)
        rule = _generation_rule(spec)
        problem_dir = (
            destination / "problems" / spec.family.value / spec.problem_name
        )
        definition_path = problem_dir / "definition.json"
        rule_path = problem_dir / "generation-rule.yaml"
        atomic_write_json_value(
            definition_path,
            definition.model_dump(mode="json"),
            sort_keys=False,
        )
        _write_yaml(rule_path, rule.model_dump(mode="json"))
        relative_definition = definition_path.relative_to(
            destination
        ).as_posix()
        relative_rule = rule_path.relative_to(destination).as_posix()
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
            generation_rule_path=relative_rule,
            definition_sha256=sha256_file(definition_path),
            generation_rule_sha256=sha256_file(rule_path),
        )
        entries.append(entry)
        semantic_records.append(_semantic_record(spec, fingerprint))
    manifest = CorpusManifest(
        schema_version=DatasetArtifactSchema.CORPUS_MANIFEST,
        corpus_id="LLM_CORE",
        release_id="LLM_CORE_V2",
        release_state=CorpusReleaseState.FROZEN,
        license="Apache-2.0",
        source_freeze_date="2026-08-14",
        profiles=tuple(CorpusProfile),
        sources=_load_sources(),
        generation_policy=CorpusGenerationPolicy(
            algorithm_version=WORKLOAD_GENERATOR_VERSION,
            capacity_classes_gib=(
                1,
                2,
                4,
                6,
                8,
                12,
                16,
                24,
                32,
                48,
                64,
                80,
                96,
                128,
                160,
                192,
                256,
                384,
            ),
        ),
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
        "status": "frozen_in_llm_core_v2",
    }


def _write_registry_outputs(
    metadata_root: Path,
    records: list[dict[str, Any]],
) -> None:
    _write_yaml(
        metadata_root / "semantics.yaml",
        {"schema": "llm_core_semantic_registry.v2", "semantics": records},
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
        generated = Path(temp) / "LLM_CORE_V2"
        metadata = Path(temp) / "metadata"
        build(generated, metadata)
        if RELEASE_ROOT.exists():
            shutil.rmtree(RELEASE_ROOT)
        shutil.copytree(generated, RELEASE_ROOT)
        SEMANTIC_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata / "semantics.yaml", SEMANTIC_REGISTRY)


def _check_release() -> None:
    with tempfile.TemporaryDirectory(prefix="llm-core-check-") as temp:
        generated = Path(temp) / "LLM_CORE_V2"
        metadata = Path(temp) / "metadata"
        build(generated, metadata)
        differences = _compare_trees(generated, RELEASE_ROOT)
        metadata_targets = {metadata / "semantics.yaml": SEMANTIC_REGISTRY}
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
