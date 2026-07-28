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

"""Common utilities and base classes for data models."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _resolve_dtype[T](dtype_str: str, mapping: dict[str, T]) -> T:
    """Resolve a dtype string via *mapping*, raising a consistent error."""
    if not dtype_str:
        raise ValueError("dtype is None or empty")
    dtype = mapping.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{dtype_str}'")
    return dtype


@cache
def _get_dtype_str_to_torch_dtype() -> dict[str, torch.dtype]:
    """Lazily build dtype string to torch dtype mapping."""
    import torch

    return {
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float4_e2m1": torch.float4_e2m1fn_x2,
        "float4_e2m1fn_x2": torch.float4_e2m1fn_x2,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }


def dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Resolve a serialized dtype name to a Torch dtype."""
    return _resolve_dtype(dtype_str, _get_dtype_str_to_torch_dtype())


def dtype_storage_bits(dtype_str: str) -> int:
    """Return the physical bit width of one logical tensor element."""
    return _resolve_dtype(
        dtype_str,
        {
            "float64": 64,
            "float32": 32,
            "float16": 16,
            "bfloat16": 16,
            "float8_e4m3fn": 8,
            "float8_e5m2": 8,
            "float4_e2m1": 4,
            "float4_e2m1fn_x2": 4,
            "int64": 64,
            "int32": 32,
            "int16": 16,
            "int8": 8,
            "bool": 8,
        },
    )
