# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Precision normalization and storage widths used by SOLAR."""

# Default settings
DEFAULT_PRECISION = "fp16"

# Precision settings
BYTES_PER_ELEMENT = {
    "fp32": 4,
    "tf32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
    "fp64": 8,
    "fp8": 1,
    "nvfp4": 0.5,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "uint8": 1,
    "bool": 1,
}

_DTYPE_ALIASES = {
    "double": "fp64",
    "float64": "fp64",
    "float": "fp32",
    "float32": "fp32",
    "single": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "bfloat16": "bf16",
    "float8": "fp8",
    "float8_e4m3fn": "fp8",
    "float8_e4m3fnuz": "fp8",
    "float8_e5m2": "fp8",
    "float8_e5m2fnuz": "fp8",
    "float4_e2m1fn_x2": "nvfp4",
    "byte": "uint8",
    "char": "int8",
    "short": "int16",
    "int": "int32",
    "long": "int64",
}


def normalize_dtype(dtype: object, fallback: str | None = None) -> str:
    """Normalize torch and YAML dtype spellings to SOLAR precision names."""
    value = str(dtype or "").strip().lower()
    value = value.removeprefix("torch.")
    value = _DTYPE_ALIASES.get(value, value)
    if value in BYTES_PER_ELEMENT:
        return value
    if fallback is not None:
        return normalize_dtype(fallback)
    raise ValueError(f"Unknown tensor dtype {dtype!r}")


def dtype_bytes(dtype: object, fallback: str | None = None) -> float:
    """Return the storage width for one tensor element."""
    return float(BYTES_PER_ELEMENT[normalize_dtype(dtype, fallback)])


__all__ = [
    "BYTES_PER_ELEMENT",
    "DEFAULT_PRECISION",
    "dtype_bytes",
    "normalize_dtype",
]
