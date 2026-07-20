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

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch

from sol_execbench.core.bench.custom_inputs import gen_custom_inputs, isolated_torch_rng
from sol_execbench.core.bench.input_heuristics import (
    _generate_heuristic_tensor,
    _is_binary_mask,
    _is_causal_attention_mask,
    _is_norm_bias,
    _is_norm_weight,
    _is_positive_tensor,
    _is_rope_cos_sin,
    _is_softmax_output,
    _is_ssm_decay,
    _is_weight_matrix,
    is_sampling_operation,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import (
    CustomInput,
    SafetensorsInput,
    ScalarInput,
    Workload,
)

__all__ = [
    "_cast_to_fp4x2",
    "_generate_heuristic_tensor",
    "_is_binary_mask",
    "_is_causal_attention_mask",
    "_is_norm_bias",
    "_is_norm_weight",
    "_is_positive_tensor",
    "_is_rope_cos_sin",
    "_is_softmax_output",
    "_is_ssm_decay",
    "_is_weight_matrix",
    "_rand_tensor",
    "gen_inputs",
    "is_sampling_operation",
]


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


# ── Safetensors loading ──────────────────────────────────────────────────────


def gen_inputs(
    definition: Definition,
    workload: Workload,
    device: str,
    safe_tensors: Optional[Dict[str, torch.Tensor]] = None,
    custom_inputs_fn: Optional[Any] = None,
    *,
    row_index: int | None = None,
    seed: int | None = None,
) -> List[Any]:
    """Generate seeded input values in definition order."""
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
            seed=seed,
        )

    rng = isolated_torch_rng(seed) if seed is not None else nullcontext()
    with rng:
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
                    out.append(val)
            elif name in workload.inputs and isinstance(
                workload.inputs[name], CustomInput
            ):
                raise RuntimeError(
                    f"CustomInput for '{name}' must be pre-generated by caller via "
                    f"definition.custom_inputs_entrypoint"
                )
            else:
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
