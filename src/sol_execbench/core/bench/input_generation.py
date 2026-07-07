# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Random and heuristic input generation for benchmark execution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import torch

from sol_execbench.core.bench.custom_inputs import gen_custom_inputs
from sol_execbench.core.data import (
    Definition,
    Workload,
)
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import CustomInput, SafetensorsInput, ScalarInput


def _cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Quantize a tensor to FP4 E2M1 and pack into uint8 (2 FP4 values per byte).

    Args:
        x: Input tensor of shape (..., cols) with values in range [-6, 6]

    Returns:
        uint8 tensor of shape (..., cols//2) with packed FP4 values
    """
    result = torch.zeros_like(x, dtype=torch.uint8)

    # Positive values
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    # Negative values
    result[(x >= -0.25) & (x < 0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    # Pack two FP4 values into one byte along cols dimension
    packed = result[..., ::2] + result[..., 1::2] * 16
    return packed.view(torch.float4_e2m1fn_x2)


def _rand_tensor(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, dtype=dtype, device=device)

    # low-precision floats
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)
    elif dtype == torch.float4_e2m1fn_x2:
        return _cast_to_fp4x2(torch.randn(shape, dtype=torch.float32, device=device))

    # booleans
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # integers
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        ranges = {
            torch.int8: (-128, 128),
            torch.int16: (-1024, 1024),
            torch.int32: (-1024, 1024),
            torch.int64: (-1024, 1024),
        }
        low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


# ── Heuristic tensor generation ──────────────────────────────────────────────


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
        else:
            return torch.sin(t).to(dtype)

    if _is_positive_tensor(name, description):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1

    if _is_ssm_decay(name):
        if name == "A_cumsum":
            raw = torch.empty(shape, dtype=dtype, device=device).uniform_(-0.1, 0.0)
            return raw.cumsum(dim=-1)
        elif name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        elif name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        else:  # g (decay gate)
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)

    if _is_softmax_output(name, description):
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)

    if _is_weight_matrix(name, shape):
        fan_in = shape[-1]
        return torch.randn(shape, dtype=dtype, device=device) / math.sqrt(fan_in)

    return None


# ── Safetensors loading ──────────────────────────────────────────────────────


def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: Optional[Dict[str, torch.Tensor]] = None,
    custom_inputs_fn: Optional[Any] = None,
    *,
    row_index: int | None = None,
) -> List[Any]:
    """Generate input tensors in definition order.

    Returns a list of input values (tensors or scalars) in the same order
    as definition.inputs.

    custom_inputs_fn: if provided, called each invocation to regenerate
        custom tensors instead of reusing *custom_tensors*.  Signature:
        ``(axes_and_scalars: dict, device: torch.device) -> dict[str, Tensor]``.
    """
    shapes = definition.get_input_shapes(workload.axes)
    dev = torch.device(device)
    out: List[Any] = []
    custom_tensors = None

    # Regenerate custom tensors on the fly when a factory is provided.
    if custom_inputs_fn is not None:
        custom_tensors, _provenance = gen_custom_inputs(
            definition,
            workload,
            dev,
            custom_inputs_fn,
            row_index=row_index,
        )

    for name, spec in definition.inputs.items():
        dtype = dtype_str_to_torch_dtype(spec.dtype)

        if name in workload.inputs and isinstance(
            workload.inputs[name], SafetensorsInput
        ):
            if safe_tensors is None or name not in safe_tensors:
                raise RuntimeError(f"Missing required safetensors input '{name}'")
            t_cpu = safe_tensors[name]
            out.append(t_cpu.to(device=dev, non_blocking=True))
        elif name in workload.inputs and isinstance(
            input_spec := workload.inputs[name], ScalarInput
        ):
            out.append(input_spec.value)
        elif custom_tensors is not None and name in custom_tensors:
            val = custom_tensors[name]
            if isinstance(val, torch.Tensor):
                out.append(val.to(device=dev, non_blocking=True))
            else:
                # Scalar values (int, float, bool) from custom_inputs_entrypoint
                out.append(val)
        elif name in workload.inputs and isinstance(workload.inputs[name], CustomInput):
            raise RuntimeError(
                f"CustomInput for '{name}' must be pre-generated by caller via "
                f"definition.custom_inputs_entrypoint"
            )
        else:  # random
            shape = shapes[name]

            if shape is None:
                value = _rand_tensor([], dtype, dev).item()
            else:
                value = _generate_heuristic_tensor(
                    name, tuple(shape), dtype, dev, spec.description
                )
                if value is None:
                    value = _rand_tensor(list(shape), dtype, dev)

                if is_sampling_operation(definition) and name == "probs":
                    value = torch.softmax(value, dim=-1)

            out.append(value)
    return out


# ── output generation ────────────────────────────────────────────────────────
