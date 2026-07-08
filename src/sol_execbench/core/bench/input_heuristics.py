# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Heuristic tensor generation for benchmark inputs."""

from __future__ import annotations

import math
from typing import Optional

import torch

from sol_execbench.core.data.definition import Definition


def is_sampling_operation(definition: Definition) -> bool:
    return getattr(definition, "op_type", None) == "sampling"


def _is_weight_matrix(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) < 2:
        return False
    weight_suffixes = (
        "_weight",
        "_weights",
        "_proj",
        "_projs",
        "_proj_weight",
        "_proj_weights",
        "_weight_matrix",
    )
    if name.endswith(weight_suffixes) or name == "weight":
        return True
    stripped = name.rstrip("0123456789")
    if stripped and stripped in ("weight",) or stripped.endswith(weight_suffixes):
        return True
    return False


def _is_norm_weight(name: str) -> bool:
    if name == "norm_weight":
        return True
    if name.endswith("_weight"):
        prefix = name[: -len("_weight")]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_norm_bias(name: str) -> bool:
    if name == "norm_bias":
        return True
    if name.endswith("_bias"):
        prefix = name[: -len("_bias")]
        if prefix.endswith(("_norm", "_layernorm", "layernorm")):
            return True
        stripped = prefix.rstrip("0123456789")
        if stripped and stripped.endswith(("norm", "layernorm")):
            return True
    return False


def _is_causal_attention_mask(
    name: str, shape: tuple[int, ...], description: Optional[str]
) -> bool:
    if len(shape) < 2 or shape[-1] != shape[-2]:
        return False
    if name in ("attention_mask", "causal_mask"):
        return True
    if (
        description
        and "attention mask" in description.lower()
        and "causal" in description.lower()
    ):
        return True
    return False


def _is_binary_mask(name: str, description: Optional[str]) -> bool:
    mask_names = (
        "x_mask",
        "text_mask",
        "aspect_ratio_mask",
        "drop_mask",
        "attention_mask",
    )
    if name in mask_names:
        return True
    if name.endswith("_mask") and description:
        desc_lower = description.lower()
        if any(
            kw in desc_lower
            for kw in ("binary", "{0, 1}", "0 or 1", "1.0 for valid", "0.0 for masked")
        ):
            return True
    return False


def _is_rope_cos_sin(name: str) -> bool:
    return name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin")


def _is_positive_tensor(name: str, description: Optional[str]) -> bool:
    positive_names = ("rstd", "std", "variance", "var")
    if name in positive_names:
        return True
    if name.endswith(("_rstd", "_std", "_variance", "_var")):
        return True
    base = name.rstrip("0123456789")
    if base in positive_names or base.endswith(("_rstd", "_std", "_var")):
        return True
    return False


def _is_ssm_decay(name: str) -> bool:
    return name in ("A", "A_log", "A_cumsum", "g")


def _is_softmax_output(name: str, description: Optional[str]) -> bool:
    if name in ("attn_weights", "attention_weights", "routing_weights"):
        return True
    if (
        description
        and "softmax" in description.lower()
        and "output" in description.lower()
    ):
        return True
    return False


def _generate_heuristic_tensor(
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    description: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Generate a tensor using heuristics based on the input name.

    Returns None if no heuristic matches, falling back to _rand_tensor.
    """
    if not dtype.is_floating_point:
        return None

    # Low-precision floats (FP8, FP4) don't support ops like torch.randn/ones/empty.uniform_;
    # fall back to _rand_tensor() which generates in float32 then converts.
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        return None

    if _is_norm_weight(name):
        return torch.ones(shape, dtype=dtype, device=device)

    if _is_norm_bias(name):
        return torch.zeros(shape, dtype=dtype, device=device)

    if _is_causal_attention_mask(name, shape, description):
        seq_len = shape[-1]
        mask = torch.zeros(shape, dtype=dtype, device=device)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        mask[..., causal] = torch.finfo(dtype).min
        return mask

    if _is_binary_mask(name, description):
        return torch.randint(0, 2, shape, device=device).to(dtype)

    if _is_rope_cos_sin(name):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(
            -math.pi, math.pi
        )
        if name in ("cos", "cos_cached", "rope_cos"):
            return torch.cos(t).to(dtype)
        return torch.sin(t).to(dtype)

    if _is_positive_tensor(name, description):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1

    if _is_ssm_decay(name):
        if name == "A_cumsum":
            raw = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.1, 0.0)
            return raw.cumsum(dim=-1)
        if name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        if name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)

    if _is_softmax_output(name, description):
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)

    if _is_weight_matrix(name, shape):
        fan_in = shape[-1]
        return torch.randn(shape, dtype=dtype, device=device) / math.sqrt(fan_in)

    return None
