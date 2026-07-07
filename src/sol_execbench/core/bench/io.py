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

"""Input generation utilities for benchmark execution."""

from __future__ import annotations

from sol_execbench.core.bench.custom_inputs import (
    GEN_INPUTS_DEVICE_MISMATCH,
    GEN_INPUTS_ERROR,
    GEN_INPUTS_OOM_BLOCKED,
    GEN_INPUTS_SCHEMA_MISMATCH,
    GEN_INPUTS_TIMEOUT,
    CustomInputGenerationError,
    CustomInputProvenance,
    _classify_custom_generation_exception,
    _custom_input_provenance,
    _raise_custom_input_error,
    _validate_custom_tensors,
    derive_custom_input_seed,
    gen_custom_inputs,
    isolated_torch_rng,
)
from sol_execbench.core.bench.input_generation import (
    _cast_to_fp4x2,
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
    _rand_tensor,
    gen_inputs,
    is_sampling_operation,
)
from sol_execbench.core.bench.memory_pool import ShiftingMemoryPoolAllocator
from sol_execbench.core.bench.output_allocation import (
    allocate_outputs,
    normalize_outputs,
)
from sol_execbench.core.bench.safetensors_io import (
    FLASHINFER_TRACE_ENV,
    _resolve_blob_path,
    flashinfer_safetensors_env,
    load_safetensors,
)

__all__ = [
    "FLASHINFER_TRACE_ENV",
    "GEN_INPUTS_DEVICE_MISMATCH",
    "GEN_INPUTS_ERROR",
    "GEN_INPUTS_OOM_BLOCKED",
    "GEN_INPUTS_SCHEMA_MISMATCH",
    "GEN_INPUTS_TIMEOUT",
    "CustomInputGenerationError",
    "CustomInputProvenance",
    "ShiftingMemoryPoolAllocator",
    "_cast_to_fp4x2",
    "_classify_custom_generation_exception",
    "_custom_input_provenance",
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
    "_raise_custom_input_error",
    "_rand_tensor",
    "_resolve_blob_path",
    "_validate_custom_tensors",
    "allocate_outputs",
    "derive_custom_input_seed",
    "flashinfer_safetensors_env",
    "gen_custom_inputs",
    "gen_inputs",
    "isolated_torch_rng",
    "is_sampling_operation",
    "load_safetensors",
    "normalize_outputs",
]
