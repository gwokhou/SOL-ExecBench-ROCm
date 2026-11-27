# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Setup-time pre-packing ROCm vendor adapters (Tier 2 vLLM / Tier 3 AITER)."""

from __future__ import annotations

from sol_execbench.vertical.adapters.common import (
    PreparedRunner,
    materialize_comparable_output,
)
from sol_execbench.vertical.adapters.gemm import (
    try_make_aiter_gemm_runner,
    try_make_vllm_gemm_runner,
)
from sol_execbench.vertical.adapters.gqa import (
    build_paged_tables_and_seqlens,
    try_make_aiter_gqa_runner,
    try_make_vllm_gqa_runner,
)
from sol_execbench.vertical.adapters.mla import (
    try_make_aiter_mla_runner,
    try_make_vllm_mla_runner,
)
from sol_execbench.vertical.adapters.moe import (
    try_make_aiter_fp8_moe_runner,
    try_make_aiter_moe_runner,
    try_make_vllm_fp8_moe_runner,
    try_make_vllm_moe_runner,
)
from sol_execbench.vertical.adapters.rmsnorm_rope import (
    try_make_aiter_rmsnorm_rope_runner,
    try_make_vllm_rmsnorm_rope_runner,
)
from sol_execbench.vertical.adapters.swiglu import (
    try_make_aiter_swiglu_quant_runner,
    try_make_vllm_swiglu_quant_runner,
)

__all__ = [
    "PreparedRunner",
    "build_paged_tables_and_seqlens",
    "materialize_comparable_output",
    "try_make_aiter_fp8_moe_runner",
    "try_make_aiter_gemm_runner",
    "try_make_aiter_gqa_runner",
    "try_make_aiter_mla_runner",
    "try_make_aiter_moe_runner",
    "try_make_aiter_rmsnorm_rope_runner",
    "try_make_aiter_swiglu_quant_runner",
    "try_make_vllm_fp8_moe_runner",
    "try_make_vllm_gemm_runner",
    "try_make_vllm_gqa_runner",
    "try_make_vllm_mla_runner",
    "try_make_vllm_moe_runner",
    "try_make_vllm_rmsnorm_rope_runner",
    "try_make_vllm_swiglu_quant_runner",
]
