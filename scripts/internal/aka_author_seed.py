#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Author the AKA-derived seed problem set and its manifest.

This is the offline authoring tool for the problem set derived from AMD
AgentKernelArena (AKA). Each problem's PyTorch reference is AKA's own
correctness oracle (``module_fn``) lifted into a standalone ``def run(...)``;
axes, workloads, and dtypes are chosen per problem under the SOL-ExecBench
paper (arXiv 2603.19173) §3 methodology. Running this script regenerates the
committed problems under ``problems/AMD_AKA/`` and the manifest, recording
AKA per-task checksums when the AKA clone is present.

Usage:
    uv run python scripts/internal/aka_author_seed.py [--aka-root data/AgentKernelArena]
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_compatibility import (
    AKA_EXECUTION_TARGET_SPECS,
)
from sol_execbench.core.dataset.aka_contract import (
    AKA_MANIFEST_SCHEMA_VERSION,
    AKA_OFFICIAL_BASELINE_ID,
    AKA_TOLERANCE_CALIBRATION_FILENAME,
    AKAArtifactRole,
    AKACapability,
    AKACorpusRole,
    AKAFusionDepth,
    AKAOfficialScoringStatus,
    AKAOperation,
    AKAPassKind,
    AKASourceFamily,
    AKASuite,
)
from sol_execbench.core.dataset.aka_corpus import (
    AKA_LICENSE,
    AKA_PROVENANCE_CLASS,
    AKA_REPOSITORY,
    AKA_REVISION,
    FORMAL_ARCHITECTURE,
    FORMAL_ARCHITECTURE_SHA256,
    FORMAL_GFX_TARGET,
)
from sol_execbench.core.dataset.aka_task import (
    correctness_runner_path,
    functional_reference_path,
    read_task,
)
from sol_execbench.core.dataset.aka_tolerance import (
    calibration_checks,
    dtype_default_tolerance,
    load_tolerance_calibration,
    workload_contract_sha256,
)
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.runtime import resolve_tool_path
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"
PROBLEMS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"
CALIBRATION_PATH = PROBLEMS_ROOT / AKA_TOLERANCE_CALIBRATION_FILENAME


@dataclass(frozen=True)
class Spec:
    """Authored specification for one generated AKA problem."""

    name: str
    suite: AKASuite
    task_path: str
    op_type: AKAOperation
    dtype: DType
    pass_kind: AKAPassKind
    fusion_depth: AKAFusionDepth
    source_family: AKASourceFamily
    axes: dict[str, dict[str, Any]]
    inputs: dict[str, dict[str, Any]]
    outputs: dict[str, dict[str, Any]]
    reference: str
    workloads: list[dict[str, Any]]
    role: AKACorpusRole = AKACorpusRole.SCORED
    exclusion_reason_code: str = ""
    description: str = ""
    custom_inputs_entrypoint: str | None = None
    capabilities: tuple[AKACapability, ...] = ()


def _ax_var(desc: str) -> dict[str, Any]:
    return {"type": "var", "description": desc}


def _ax_const(value: int, desc: str = "") -> dict[str, Any]:
    return {"type": "const", "value": value, "description": desc}


def _ax_expr(expr: str, desc: str = "") -> dict[str, Any]:
    return {"type": "expr", "expression": expr, "description": desc}


def _wl(axes: dict[str, int], inputs: dict[str, Any]) -> dict[str, Any]:
    return {"axes": axes, "inputs": inputs}


SPECS: list[Spec] = [
    Spec(
        name="3267_doubled_matmul",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/3267_SimpleMatmulModule",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        description="Matrix multiply of a with (b + b): output = a @ (b + b). "
        "Derived from AKA torch2hip/gpumode/3267_SimpleMatmulModule module_fn.",
        axes={
            "M": _ax_var("Rows of a and the output."),
            "K": _ax_var("Inner dimension; columns of a and rows of b."),
            "N": _ax_var("Columns of b and the output."),
        },
        inputs={
            "a": {
                "shape": ["M", "K"],
                "dtype": "float32",
                "description": "Left-hand matrix (M, K).",
            },
            "b": {
                "shape": ["K", "N"],
                "dtype": "float32",
                "description": "Right-hand matrix (K, N), added to itself.",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "a @ (b + b).",
            },
        },
        reference="import torch\n\ndef run(a, b):\n    return torch.matmul(a, b + b)\n",
        workloads=[
            _wl({"M": 4, "K": 4, "N": 4}, {"a": "random", "b": "random"}),
            _wl({"M": 16, "K": 32, "N": 16}, {"a": "random", "b": "random"}),
            _wl({"M": 128, "K": 128, "N": 128}, {"a": "random", "b": "random"}),
            _wl({"M": 1, "K": 512, "N": 256}, {"a": "random", "b": "random"}),
        ],
    ),
    Spec(
        name="l1n1_square_matmul",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n1_Square_matrix_multiplication_",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Square matrix multiply C = A @ B. Derived from AKA "
        "torch2hip/kernelbench/level1/l1n1_Square_matrix_multiplication_ module_fn.",
        axes={
            "M": _ax_var("Rows of A."),
            "K": _ax_var("Inner dimension."),
            "N": _ax_var("Columns of B."),
        },
        inputs={
            "A": {
                "shape": ["M", "K"],
                "dtype": "float32",
                "description": "Left matrix (M, K).",
            },
            "B": {
                "shape": ["K", "N"],
                "dtype": "float32",
                "description": "Right matrix (K, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "A @ B.",
            },
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.matmul(A, B)\n",
        workloads=[
            _wl({"M": 64, "K": 64, "N": 64}, {"A": "random", "B": "random"}),
            _wl({"M": 128, "K": 128, "N": 128}, {"A": "random", "B": "random"}),
            _wl({"M": 256, "K": 256, "N": 256}, {"A": "random", "B": "random"}),
            _wl({"M": 512, "K": 512, "N": 512}, {"A": "random", "B": "random"}),
        ],
    ),
    Spec(
        name="l1n2_standard_matmul",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n2_Standard_matrix_multiplication_",
        op_type=AKAOperation.MATMUL,
        dtype=DType.BFLOAT16,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="General (non-square) BF16 matrix multiply C = A @ B. Derived "
        "from AKA torch2hip/kernelbench/level1/l1n2_Standard_matrix_multiplication_.",
        axes={
            "M": _ax_var("Rows of A."),
            "K": _ax_var("Inner dimension."),
            "N": _ax_var("Columns of B."),
        },
        inputs={
            "A": {
                "shape": ["M", "K"],
                "dtype": "bfloat16",
                "description": "Left BF16 matrix (M, K).",
            },
            "B": {
                "shape": ["K", "N"],
                "dtype": "bfloat16",
                "description": "Right BF16 matrix (K, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "bfloat16",
                "description": "A @ B.",
            },
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.matmul(A, B)\n",
        workloads=[
            _wl({"M": 128, "K": 256, "N": 64}, {"A": "random", "B": "random"}),
            _wl({"M": 256, "K": 128, "N": 512}, {"A": "random", "B": "random"}),
            _wl({"M": 64, "K": 512, "N": 128}, {"A": "random", "B": "random"}),
        ],
    ),
    Spec(
        name="l1n3_batched_matmul",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n3_Batched_matrix_multiplication",
        op_type=AKAOperation.MATMUL,
        dtype=DType.BFLOAT16,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Batched BF16 matrix multiply via torch.bmm. Derived from AKA "
        "torch2hip/kernelbench/level1/l1n3_Batched_matrix_multiplication.",
        axes={
            "Batch": _ax_var("Batch dimension."),
            "M": _ax_var("Rows of A."),
            "K": _ax_var("Inner dimension."),
            "N": _ax_var("Columns of B."),
        },
        inputs={
            "A": {
                "shape": ["Batch", "M", "K"],
                "dtype": "bfloat16",
                "description": "Batched LHS (Batch, M, K).",
            },
            "B": {
                "shape": ["Batch", "K", "N"],
                "dtype": "bfloat16",
                "description": "Batched RHS (Batch, K, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["Batch", "M", "N"],
                "dtype": "bfloat16",
                "description": "torch.bmm(A, B).",
            },
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.bmm(A, B)\n",
        workloads=[
            _wl(
                {"Batch": 4, "M": 64, "K": 64, "N": 64},
                {"A": "random", "B": "random"},
            ),
            _wl(
                {"Batch": 8, "M": 128, "K": 128, "N": 128},
                {"A": "random", "B": "random"},
            ),
            _wl(
                {"Batch": 2, "M": 256, "K": 64, "N": 256},
                {"A": "random", "B": "random"},
            ),
        ],
    ),
    Spec(
        name="l1n4_matrix_vector",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n4_Matrix_vector_multiplication_",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Matrix-vector product y = A @ B with B a column vector. "
        "Derived from AKA torch2hip/kernelbench/level1/l1n4_Matrix_vector_multiplication_.",
        axes={"M": _ax_var("Rows of A."), "K": _ax_var("Inner dimension.")},
        inputs={
            "A": {
                "shape": ["M", "K"],
                "dtype": "float32",
                "description": "Matrix (M, K).",
            },
            "B": {
                "shape": ["K", "1"],
                "dtype": "float32",
                "description": "Column vector (K, 1).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "1"],
                "dtype": "float32",
                "description": "A @ B of shape (M, 1).",
            },
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.matmul(A, B)\n",
        workloads=[
            _wl({"M": 1024, "K": 1024}, {"A": "random", "B": "random"}),
            _wl({"M": 2048, "K": 512}, {"A": "random", "B": "random"}),
            _wl({"M": 512, "K": 4096}, {"A": "random", "B": "random"}),
        ],
    ),
    Spec(
        name="l1n8_matmul_irregular",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n8_Matmul_with_irregular_shapes_",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT16,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="FP16 matrix multiply over irregular (non power-of-two) shapes. "
        "Derived from AKA torch2hip/kernelbench/level1/l1n8_Matmul_with_irregular_shapes_.",
        axes={
            "M": _ax_var("Rows of A."),
            "K": _ax_var("Inner dimension."),
            "N": _ax_var("Columns of B."),
        },
        inputs={
            "A": {
                "shape": ["M", "K"],
                "dtype": "float16",
                "description": "Left FP16 matrix (M, K).",
            },
            "B": {
                "shape": ["K", "N"],
                "dtype": "float16",
                "description": "Right FP16 matrix (K, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "A @ B.",
            },
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.matmul(A, B)\n",
        workloads=[
            _wl(
                {"M": 1823, "K": 781, "N": 511},
                {"A": "random", "B": "random"},
            ),
            _wl({"M": 359, "K": 127, "N": 211}, {"A": "random", "B": "random"}),
            _wl(
                {"M": 1024, "K": 333, "N": 717},
                {"A": "random", "B": "random"},
            ),
        ],
    ),
    Spec(
        name="l1n9_tall_skinny_matmul",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Tall-skinny matrix multiply (M >> K). Derived from AKA "
        "torch2hip/kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_.",
        axes={
            "M": _ax_var("Rows of A (large)."),
            "K": _ax_var("Inner dimension (small)."),
            "N": _ax_var("Columns of B."),
        },
        inputs={
            "A": {
                "shape": ["M", "K"],
                "dtype": "float32",
                "description": "Tall matrix (M, K).",
            },
            "B": {
                "shape": ["K", "N"],
                "dtype": "float32",
                "description": "Right matrix (K, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "A @ B.",
            },
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.matmul(A, B)\n",
        workloads=[
            _wl({"M": 8192, "K": 64, "N": 64}, {"A": "random", "B": "random"}),
            _wl({"M": 4096, "K": 32, "N": 128}, {"A": "random", "B": "random"}),
            _wl({"M": 16384, "K": 16, "N": 32}, {"A": "random", "B": "random"}),
        ],
    ),
    Spec(
        name="l1n23_softmax",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n23_Softmax",
        op_type=AKAOperation.SOFTMAX,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Row-wise softmax over the last dimension. Derived from AKA "
        "torch2hip/kernelbench/level1/l1n23_Softmax module_fn (dim=1).",
        axes={
            "M": _ax_var("Rows."),
            "N": _ax_var("Columns (softmax dimension)."),
        },
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Row-wise softmax.",
            },
        },
        reference="import torch\n\ndef run(x):\n    return torch.softmax(x, dim=-1)\n",
        workloads=[
            _wl({"M": 128, "N": 128}, {"x": "random"}),
            _wl({"M": 1, "N": 131072}, {"x": "random"}),
            _wl({"M": 1823, "N": 781}, {"x": "random"}),
            _wl({"M": 4096, "N": 8192}, {"x": "random"}),
        ],
    ),
    Spec(
        name="l1n26_gelu",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n26_GELU_",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT16,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="GELU activation. Derived from AKA torch2hip/kernelbench/level1/l1n26_GELU_ "
        "module_fn (F.gelu).",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "Input (M, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "F.gelu(x).",
            },
        },
        reference="import torch.nn.functional as F\n\ndef run(x):\n    return F.gelu(x)\n",
        workloads=[
            _wl({"M": 1024, "N": 1024}, {"x": "random"}),
            _wl({"M": 256, "N": 8192}, {"x": "random"}),
            _wl({"M": 4096, "N": 512}, {"x": "random"}),
        ],
    ),
    Spec(
        name="l1n36_rmsnorm",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n36_RMSNorm_",
        op_type=AKAOperation.NORM,
        dtype=DType.BFLOAT16,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Root-mean-square normalization over the last dimension. Derived "
        "from AKA torch2hip/kernelbench/level1/l1n36_RMSNorm_ module_fn.",
        axes={
            "B": _ax_var("Batch rows."),
            "F": _ax_var("Feature dimension (normalized)."),
        },
        inputs={
            "x": {
                "shape": ["B", "F"],
                "dtype": "bfloat16",
                "description": "Input (B, F).",
            },
            "eps": {
                "shape": None,
                "dtype": "float32",
                "description": "Numerical stability epsilon.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "F"],
                "dtype": "bfloat16",
                "description": "x / rms(x).",
            },
        },
        reference=(
            "import torch\n\ndef run(x, eps):\n"
            "    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)\n"
            "    return x / rms\n"
        ),
        workloads=[
            _wl({"B": 128, "F": 256}, {"x": "random", "eps": {"scalar": 1e-5}}),
            _wl({"B": 64, "F": 1024}, {"x": "random", "eps": {"scalar": 1e-5}}),
            _wl({"B": 512, "F": 128}, {"x": "random", "eps": {"scalar": 1e-5}}),
        ],
    ),
    Spec(
        name="l1n40_layernorm",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n40_LayerNorm",
        op_type=AKAOperation.NORM,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Layer normalization over the last dimension with affine "
        "weight/bias. Derived from AKA torch2hip/kernelbench/level1/l1n40_LayerNorm.",
        axes={
            "B": _ax_var("Batch rows."),
            "N": _ax_var("Feature dimension (normalized)."),
        },
        inputs={
            "x": {
                "shape": ["B", "N"],
                "dtype": "float32",
                "description": "Input (B, N).",
            },
            "weight": {
                "shape": ["N"],
                "dtype": "float32",
                "description": "Affine gain (N,).",
            },
            "bias": {
                "shape": ["N"],
                "dtype": "float32",
                "description": "Affine bias (N,).",
            },
            "eps": {
                "shape": None,
                "dtype": "float32",
                "description": "Numerical stability epsilon.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "N"],
                "dtype": "float32",
                "description": "layer_norm(x).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias, eps):\n"
            "    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)\n"
        ),
        workloads=[
            _wl(
                {"B": 128, "N": 256},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "eps": {"scalar": 1e-5},
                },
            ),
            _wl(
                {"B": 64, "N": 1024},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "eps": {"scalar": 1e-5},
                },
            ),
            _wl(
                {"B": 512, "N": 128},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "eps": {"scalar": 1e-5},
                },
            ),
        ],
    ),
    Spec(
        name="l1n47_sum_reduction",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n47_Sum_reduction_over_a_dimension",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Sum reduction over the last dimension with keepdim. Derived from "
        "AKA torch2hip/kernelbench/level1/l1n47_Sum_reduction_over_a_dimension.",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns (reduced).")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "1"],
                "dtype": "float32",
                "description": "Row sums (M, 1).",
            },
        },
        reference="import torch\n\ndef run(x):\n    return torch.sum(x, dim=-1, keepdim=True)\n",
        workloads=[
            _wl({"M": 128, "N": 256}, {"x": "random"}),
            _wl({"M": 1024, "N": 64}, {"x": "random"}),
            _wl({"M": 256, "N": 4096}, {"x": "random"}),
        ],
    ),
    Spec(
        name="l1n42_maxpool2d",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n42_Max_Pooling_2D",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="2x2 stride-2 max pooling. Derived from AKA "
        "torch2hip/kernelbench/level1/l1n42_Max_Pooling_2D module_fn.",
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_var("Channels."),
            "H": _ax_var("Input height (even)."),
            "W": _ax_var("Input width (even)."),
            "H_out": _ax_expr("H // 2", "Output height."),
            "W_out": _ax_expr("W // 2", "Output width."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input (B, C, H, W).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H_out", "W_out"],
                "dtype": "float32",
                "description": "Pooled output.",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x):\n    return F.max_pool2d(x, kernel_size=2, stride=2)\n"
        ),
        workloads=[
            _wl({"B": 4, "C": 16, "H": 64, "W": 64}, {"x": "random"}),
            _wl({"B": 8, "C": 32, "H": 128, "W": 128}, {"x": "random"}),
            _wl({"B": 2, "C": 64, "H": 256, "W": 256}, {"x": "random"}),
        ],
    ),
    Spec(
        name="l1n63_conv2d",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n63_conv_standard_2D__square_input__square_kernel",
        op_type=AKAOperation.CONV,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Standard 2D convolution (stride 1, no padding, 3x3 kernel). "
        "Derived from AKA torch2hip/kernelbench/level1/l1n63_conv_standard_2D.",
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_var("Input channels."),
            "H": _ax_var("Input height."),
            "W": _ax_var("Input width."),
            "O": _ax_var("Output channels."),
            "K": _ax_const(3, "Square kernel size."),
            "H_out": _ax_expr("H - K + 1", "Output height."),
            "W_out": _ax_expr("W - K + 1", "Output width."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input (B, C, H, W).",
            },
            "weight": {
                "shape": ["O", "C", "K", "K"],
                "dtype": "float32",
                "description": "Filters (O, C, K, K).",
            },
            "bias": {
                "shape": ["O"],
                "dtype": "float32",
                "description": "Bias (O,).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "O", "H_out", "W_out"],
                "dtype": "float32",
                "description": "Convolution output.",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias):\n"
            "    return F.conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=1)\n"
        ),
        workloads=[
            _wl(
                {"B": 4, "C": 8, "H": 32, "W": 32, "O": 16},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
            _wl(
                {"B": 2, "C": 16, "H": 64, "W": 64, "O": 32},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
            _wl(
                {"B": 8, "C": 4, "H": 48, "W": 48, "O": 8},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
        ],
    ),
    Spec(
        name="l1n82_conv_depthwise",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level1/l1n82_conv_depthwise_2D_square_input_square_kernel",
        op_type=AKAOperation.CONV,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Depthwise 2D convolution (groups = channels, 3x3 kernel, stride 1, "
        "no padding). Derived from AKA torch2hip/kernelbench/level1/l1n82_conv_depthwise_2D.",
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_var("Channels (input = output, depthwise)."),
            "H": _ax_var("Input height."),
            "W": _ax_var("Input width."),
            "K": _ax_const(3, "Square kernel size."),
            "H_out": _ax_expr("H - K + 1", "Output height."),
            "W_out": _ax_expr("W - K + 1", "Output width."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input (B, C, H, W).",
            },
            "weight": {
                "shape": ["C", "1", "K", "K"],
                "dtype": "float32",
                "description": "Depthwise filters (C, 1, K, K).",
            },
            "bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "Bias (C,).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H_out", "W_out"],
                "dtype": "float32",
                "description": "Depthwise conv output.",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias):\n"
            "    return F.conv2d(x, weight, bias, stride=1, padding=0, dilation=1, groups=x.shape[1])\n"
        ),
        workloads=[
            _wl(
                {"B": 4, "C": 8, "H": 32, "W": 32},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
            _wl(
                {"B": 2, "C": 16, "H": 64, "W": 64},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
            _wl(
                {"B": 8, "C": 4, "H": 48, "W": 48},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
        ],
    ),
    # ====================================================================
    # Expansion problems (friendliness categories, see
    # docs/internal/aka-expansion-friendliness.md). Cat1 = structurally
    # advantaged (scored); Cat2 = legal-but-fragile, mechanically included.
    # ====================================================================
    # --- Cat1: pointwise activation variants (gpumode) ---
    Spec(
        name="gpumode_silu",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/16636_SiLU",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.GPUMODE,
        description="SiLU activation x * sigmoid(x). Derived from AKA "
        "torch2hip/gpumode/16636_SiLU module_fn (silu_fn).",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "x * sigmoid(x).",
            },
        },
        reference="import torch\n\ndef run(x):\n    return x * torch.sigmoid(x)\n",
        workloads=[
            _wl({"M": 1024, "N": 1024}, {"x": "random"}),
            _wl({"M": 1, "N": 65536}, {"x": "random"}),
            _wl({"M": 1823, "N": 781}, {"x": "random"}),
            _wl({"M": 256, "N": 8192}, {"x": "random"}),
        ],
    ),
    Spec(
        name="gpumode_sigmoid",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/11184_Sigmoid",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.GPUMODE,
        description="Scaled sigmoid: sigmoid(a * x) * max. Derived from AKA "
        "torch2hip/gpumode/11184_Sigmoid module_fn.",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "v": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            },
            "a": {
                "shape": None,
                "dtype": "float32",
                "description": "Input scale.",
            },
            "max": {
                "shape": None,
                "dtype": "float32",
                "description": "Output scale.",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "sigmoid(a * v) * max.",
            },
        },
        reference="import torch\n\ndef run(v, a, max):\n    return torch.sigmoid(a * v) * max\n",
        workloads=[
            _wl(
                {"M": 1024, "N": 1024},
                {"v": "random", "a": {"scalar": 1.0}, "max": {"scalar": 10.0}},
            ),
            _wl(
                {"M": 1, "N": 65536},
                {"v": "random", "a": {"scalar": 2.0}, "max": {"scalar": 10.0}},
            ),
            _wl(
                {"M": 1823, "N": 781},
                {"v": "random", "a": {"scalar": 1.0}, "max": {"scalar": 10.0}},
            ),
            _wl(
                {"M": 256, "N": 8192},
                {"v": "random", "a": {"scalar": 1.0}, "max": {"scalar": 10.0}},
            ),
        ],
    ),
    Spec(
        name="gpumode_tanh",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/11178_TanH",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.GPUMODE,
        description="Scaled tanh: tanh(a * x) * max. Derived from AKA "
        "torch2hip/gpumode/11178_TanH module_fn.",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "v": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            },
            "a": {
                "shape": None,
                "dtype": "float32",
                "description": "Input scale.",
            },
            "max": {
                "shape": None,
                "dtype": "float32",
                "description": "Output scale.",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "tanh(a * v) * max.",
            },
        },
        reference="import torch\n\ndef run(v, a, max):\n    return torch.tanh(a * v) * max\n",
        workloads=[
            _wl(
                {"M": 1024, "N": 1024},
                {"v": "random", "a": {"scalar": 1.0}, "max": {"scalar": 10.0}},
            ),
            _wl(
                {"M": 1, "N": 65536},
                {"v": "random", "a": {"scalar": 2.0}, "max": {"scalar": 10.0}},
            ),
            _wl(
                {"M": 1823, "N": 781},
                {"v": "random", "a": {"scalar": 1.0}, "max": {"scalar": 10.0}},
            ),
            _wl(
                {"M": 256, "N": 8192},
                {"v": "random", "a": {"scalar": 1.0}, "max": {"scalar": 10.0}},
            ),
        ],
    ),
    # --- Cat1: fused matmul chains (kernelbench level2) ---
    Spec(
        name="l2n99_matmul_gelu_softmax",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Fused linear -> GELU -> softmax(dim=1). Derived from AKA "
        "torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax module_fn.",
        axes={
            "B": _ax_var("Batch rows."),
            "IN": _ax_const(8192, "Input features."),
            "OUT": _ax_const(8192, "Output features."),
        },
        inputs={
            "x": {
                "shape": ["B", "IN"],
                "dtype": "float32",
                "description": "Input (B, IN).",
            },
            "weight": {
                "shape": ["OUT", "IN"],
                "dtype": "float32",
                "description": "Linear weight (OUT, IN).",
            },
            "bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Linear bias (OUT,).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "OUT"],
                "dtype": "float32",
                "description": "softmax(gelu(linear(x))).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    x = F.gelu(x)\n"
            "    return F.softmax(x, dim=1)\n"
        ),
        workloads=[
            _wl(
                {"B": 256},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
            _wl(
                {"B": 1024},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
            _wl(
                {"B": 4096},
                {"x": "random", "weight": "random", "bias": "random"},
            ),
        ],
    ),
    Spec(
        name="l2n86_matmul_divide_gelu",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Fused linear -> divide -> GELU. Derived from AKA "
        "torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU module_fn.",
        axes={
            "B": _ax_var("Batch rows."),
            "IN": _ax_const(8192, "Input features."),
            "OUT": _ax_const(8192, "Output features."),
        },
        inputs={
            "x": {
                "shape": ["B", "IN"],
                "dtype": "float32",
                "description": "Input (B, IN).",
            },
            "weight": {
                "shape": ["OUT", "IN"],
                "dtype": "float32",
                "description": "Linear weight (OUT, IN).",
            },
            "bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Linear bias (OUT,).",
            },
            "divisor": {
                "shape": None,
                "dtype": "float32",
                "description": "Divisor scalar.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "OUT"],
                "dtype": "float32",
                "description": "gelu(linear(x) / divisor).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias, divisor):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    x = x / divisor\n"
            "    return F.gelu(x)\n"
        ),
        workloads=[
            _wl(
                {"B": 256},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "divisor": {"scalar": 10.0},
                },
            ),
            _wl(
                {"B": 1024},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "divisor": {"scalar": 10.0},
                },
            ),
            _wl(
                {"B": 4096},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "divisor": {"scalar": 10.0},
                },
            ),
        ],
    ),
    Spec(
        name="l2n40_matmul_scaling_residual",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Fused linear -> scale -> residual add. Derived from AKA "
        "torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd module_fn.",
        axes={
            "B": _ax_var("Batch rows."),
            "IN": _ax_const(4096, "Input features."),
            "OUT": _ax_const(4096, "Output features."),
        },
        inputs={
            "x": {
                "shape": ["B", "IN"],
                "dtype": "float32",
                "description": "Input (B, IN).",
            },
            "weight": {
                "shape": ["OUT", "IN"],
                "dtype": "float32",
                "description": "Linear weight (OUT, IN).",
            },
            "bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Linear bias (OUT,).",
            },
            "scaling_factor": {
                "shape": None,
                "dtype": "float32",
                "description": "Scaling factor.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "OUT"],
                "dtype": "float32",
                "description": "linear(x) * scaling + linear(x).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias, scaling_factor):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    return x * scaling_factor + x\n"
        ),
        workloads=[
            _wl(
                {"B": 2048},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "scaling_factor": {"scalar": 0.5},
                },
            ),
            _wl(
                {"B": 4096},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "scaling_factor": {"scalar": 0.5},
                },
            ),
            _wl(
                {"B": 16384},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "scaling_factor": {"scalar": 0.5},
                },
            ),
        ],
    ),
    # --- Cat1: norm variants (group/batch/instance) via fused chains ---
    Spec(
        name="l2n37_groupnorm_fused",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm",
        op_type=AKAOperation.NORM,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Fused linear -> swish -> add -> group_norm. Derived from AKA "
        "torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm module_fn.",
        axes={
            "B": _ax_var("Batch rows."),
            "IN": _ax_const(1024, "Input features."),
            "OUT": _ax_const(4096, "Output features (group_norm channels)."),
        },
        inputs={
            "x": {
                "shape": ["B", "IN"],
                "dtype": "float32",
                "description": "Input (B, IN).",
            },
            "weight": {
                "shape": ["OUT", "IN"],
                "dtype": "float32",
                "description": "Linear weight.",
            },
            "bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Linear bias.",
            },
            "extra_bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Additive bias.",
            },
            "gn_weight": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "GroupNorm gain.",
            },
            "gn_bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "GroupNorm bias.",
            },
            "num_groups": {
                "shape": None,
                "dtype": "float32",
                "description": "GroupNorm group count.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "OUT"],
                "dtype": "float32",
                "description": "group_norm(swish(linear(x)) + extra_bias).",
            },
        },
        reference=(
            "import torch\nimport torch.nn.functional as F\n\n"
            "def run(x, weight, bias, extra_bias, gn_weight, gn_bias, num_groups):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    x = torch.sigmoid(x) * x\n"
            "    x = x + extra_bias\n"
            "    return F.group_norm(x, int(num_groups), gn_weight, gn_bias)\n"
        ),
        workloads=[
            _wl(
                {"B": 4096},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "extra_bias": "random",
                    "gn_weight": "random",
                    "gn_bias": "random",
                    "num_groups": {"scalar": 64},
                },
            ),
            _wl(
                {"B": 8192},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "extra_bias": "random",
                    "gn_weight": "random",
                    "gn_bias": "random",
                    "num_groups": {"scalar": 64},
                },
            ),
            _wl(
                {"B": 16384},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "extra_bias": "random",
                    "gn_weight": "random",
                    "gn_bias": "random",
                    "num_groups": {"scalar": 64},
                },
            ),
        ],
    ),
    Spec(
        name="l2n17_instancenorm_fused",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide",
        op_type=AKAOperation.NORM,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Fused conv2d -> instance_norm -> divide. Derived from AKA "
        "torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide module_fn.",
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_const(64, "Input channels."),
            "H": _ax_var("Input height."),
            "W": _ax_var("Input width."),
            "O": _ax_const(128, "Output channels."),
            "K": _ax_const(3, "Kernel size."),
            "HO": _ax_expr("H - K + 1", "Output height."),
            "WO": _ax_expr("W - K + 1", "Output width."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input (B, C, H, W).",
            },
            "conv_weight": {
                "shape": ["O", "C", "K", "K"],
                "dtype": "float32",
                "description": "Conv filters.",
            },
            "conv_bias": {
                "shape": ["O"],
                "dtype": "float32",
                "description": "Conv bias.",
            },
            "divide_by": {
                "shape": None,
                "dtype": "float32",
                "description": "Divisor scalar.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "O", "HO", "WO"],
                "dtype": "float32",
                "description": "instance_norm(conv2d(x)) / divide_by.",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, conv_weight, conv_bias, divide_by):\n"
            "    x = F.conv2d(x, conv_weight, conv_bias)\n"
            "    x = F.instance_norm(x)\n"
            "    return x / divide_by\n"
        ),
        workloads=[
            _wl(
                {"B": 32, "H": 64, "W": 64},
                {
                    "x": "random",
                    "conv_weight": "random",
                    "conv_bias": "random",
                    "divide_by": {"scalar": 2.0},
                },
            ),
            _wl(
                {"B": 64, "H": 128, "W": 128},
                {
                    "x": "random",
                    "conv_weight": "random",
                    "conv_bias": "random",
                    "divide_by": {"scalar": 2.0},
                },
            ),
            _wl(
                {"B": 16, "H": 48, "W": 48},
                {
                    "x": "random",
                    "conv_weight": "random",
                    "conv_bias": "random",
                    "divide_by": {"scalar": 2.0},
                },
            ),
        ],
    ),
    # --- Cat1: attention (exact AKA functional oracle) ---
    Spec(
        name="dot_product_attention",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/1001_NormalAttention_dot",
        op_type=AKAOperation.ATTENTION,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        description=(
            "Dot-product non-local attention with ELU-normalized query/key energy, "
            "query/key/value 1x1 projections, and an output projection. Lifted "
            "exactly from AKA torch2hip/gpumode/1001_NormalAttention_dot module_fn."
        ),
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_const(4, "Channels fixed by the AKA task initializer."),
            "Q": _ax_const(1, "Reduced query/key channels (C // 4)."),
            "H": _ax_var("Spatial height."),
            "W": _ax_var("Spatial width."),
            "K": _ax_const(1, "Projection kernel size."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input feature map.",
            },
            "query_weight": {
                "shape": ["Q", "C", "K", "K"],
                "dtype": "float32",
                "description": "Query 1x1 convolution weight.",
            },
            "query_bias": {
                "shape": ["Q"],
                "dtype": "float32",
                "description": "Query convolution bias.",
            },
            "key_weight": {
                "shape": ["Q", "C", "K", "K"],
                "dtype": "float32",
                "description": "Key 1x1 convolution weight.",
            },
            "key_bias": {
                "shape": ["Q"],
                "dtype": "float32",
                "description": "Key convolution bias.",
            },
            "value_weight": {
                "shape": ["C", "C", "K", "K"],
                "dtype": "float32",
                "description": "Value 1x1 convolution weight.",
            },
            "value_bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "Value convolution bias.",
            },
            "gamma_weight": {
                "shape": ["C", "C", "K", "K"],
                "dtype": "float32",
                "description": "Output 1x1 convolution weight.",
            },
            "gamma_bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "Output convolution bias.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Attention output feature map.",
            },
        },
        reference=(
            "import torch\nimport torch.nn.functional as F\n\n"
            "def run(x, query_weight, query_bias, key_weight, key_bias, "
            "value_weight, value_bias, gamma_weight, gamma_bias):\n"
            "    b, c, h, w = x.shape\n"
            "    query = F.conv2d(x, query_weight, query_bias)\n"
            "    query = query.view(b, -1, h * w).permute(0, 2, 1)\n"
            "    key = F.conv2d(x, key_weight, key_bias).view(b, -1, h * w)\n"
            "    energy = torch.bmm(query, key)\n"
            "    energy = F.elu(energy) / (h * w)\n"
            "    value = F.conv2d(x, value_weight, value_bias).view(b, c, h * w)\n"
            "    out = torch.bmm(value, energy).view(b, c, h, w)\n"
            "    return F.conv2d(out, gamma_weight, gamma_bias)\n"
        ),
        workloads=[
            _wl(
                {"B": 2, "H": 4, "W": 4},
                {
                    "x": "random",
                    "query_weight": "random",
                    "query_bias": "random",
                    "key_weight": "random",
                    "key_bias": "random",
                    "value_weight": "random",
                    "value_bias": "random",
                    "gamma_weight": "random",
                    "gamma_bias": "random",
                },
            ),
            _wl(
                {"B": 4, "H": 8, "W": 8},
                {
                    "x": "random",
                    "query_weight": "random",
                    "query_bias": "random",
                    "key_weight": "random",
                    "key_bias": "random",
                    "value_weight": "random",
                    "value_bias": "random",
                    "gamma_weight": "random",
                    "gamma_bias": "random",
                },
            ),
            _wl(
                {"B": 2, "H": 16, "W": 16},
                {
                    "x": "random",
                    "query_weight": "random",
                    "query_bias": "random",
                    "key_weight": "random",
                    "key_bias": "random",
                    "value_weight": "random",
                    "value_bias": "random",
                    "gamma_weight": "random",
                    "gamma_bias": "random",
                },
            ),
            _wl(
                {"B": 1, "H": 32, "W": 32},
                {
                    "x": "random",
                    "query_weight": "random",
                    "query_bias": "random",
                    "key_weight": "random",
                    "key_bias": "random",
                    "value_weight": "random",
                    "value_bias": "random",
                    "gamma_weight": "random",
                    "gamma_bias": "random",
                },
            ),
        ],
    ),
    # --- Cat2 (mechanical): rank-split a variable-rank task
    # (C9/C10 -> Cat1).
    Spec(
        name="gpumode_gelu_4d",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/14539_GELU",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.GPUMODE,
        description="GELU pinned to 4D [B, C, H, W]. Rank-split from the variable-rank "
        "(1D-4D) AKA torch2hip/gpumode/14539_GELU module_fn; the 2D case is l1n26_gelu. "
        "The schema pins rank per Definition, so each rank becomes its own problem.",
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_var("Channels."),
            "H": _ax_var("Height."),
            "W": _ax_var("Width."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input (B, C, H, W).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "F.gelu(x).",
            },
        },
        reference="import torch.nn.functional as F\n\ndef run(x):\n    return F.gelu(x)\n",
        workloads=[
            _wl({"B": 8, "C": 64, "H": 32, "W": 32}, {"x": "random"}),
            _wl({"B": 16, "C": 128, "H": 64, "W": 64}, {"x": "random"}),
            _wl({"B": 4, "C": 32, "H": 16, "W": 16}, {"x": "random"}),
            _wl({"B": 32, "C": 64, "H": 48, "W": 48}, {"x": "random"}),
        ],
    ),
    # --- Cat2 (mechanical): FP8 compatibility sentinel (C8) ---
    Spec(
        name="fp8_cast_sentinel",
        suite=AKASuite.INSTRUCTION2TRITON,
        task_path="tasks/instruction2triton/rocmbench/test_chained_dot_fp8",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT8_E4M3FN,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.ROCMBENCH,
        role=AKACorpusRole.COMPATIBILITY_SENTINEL,
        description="FP8 compatibility sentinel: per-tensor cast to float8_e4m3fn. "
        "Probes the harness's ability to materialize, compare, and time an FP8 output "
        "tensor end-to-end. Provenance-bound to AKA instruction2triton/rocmbench/"
        "test_chained_dot_fp8 (a portable pure-torch cast stands in for the arch-coupled "
        "FP8 chained-dot reference so it executes on gfx1200). Not scored for SOL.",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "FP32 input (M, N).",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float8_e4m3fn",
                "description": "x cast to float8_e4m3fn.",
            },
        },
        reference="import torch\n\ndef run(x):\n    return x.to(torch.float8_e4m3fn)\n",
        workloads=[
            _wl({"M": 128, "N": 128}, {"x": "random"}),
            _wl({"M": 256, "N": 512}, {"x": "random"}),
            _wl({"M": 64, "N": 1024}, {"x": "random"}),
        ],
    ),
    # --- Cat2 (mechanical): backward pass via instruction2triton (C13) ---
    Spec(
        name="rmsnorm_bwd",
        suite=AKASuite.INSTRUCTION2TRITON,
        task_path="tasks/instruction2triton/rocmbench/rmsnorm_bwd",
        op_type=AKAOperation.NORM,
        dtype=DType.FLOAT16,
        pass_kind=AKAPassKind.BACKWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.ROCMBENCH,
        description="RMSNorm backward (gradient w.r.t. input and weight) via autograd. "
        "Derived from and cross-checked against the PyTorch oracle in AKA "
        "instruction2triton/rocmbench/rmsnorm_bwd.",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Feature dimension.")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "Forward input (M, N).",
            },
            "g": {
                "shape": ["1", "N"],
                "dtype": "float16",
                "description": "RMSNorm gain (1, N).",
            },
            "grad_output": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "Upstream gradient (M, N).",
            },
        },
        outputs={
            "grad_x": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "Gradient w.r.t. x (M, N).",
            },
            "grad_g": {
                "shape": ["1", "N"],
                "dtype": "float16",
                "description": "Gradient w.r.t. g (1, N).",
            },
        },
        reference=(
            "import torch\n\n"
            "def run(x, g, grad_output):\n"
            "    xr = x.clone().detach().requires_grad_()\n"
            "    gr = g.clone().detach().requires_grad_()\n"
            "    rms = torch.sqrt(torch.sum(xr.float() ** 2, dim=-1, keepdim=True) * (1.0 / xr.shape[-1]))\n"
            "    y = (xr.float() / rms * gr.float()).to(x.dtype)\n"
            "    y.backward(grad_output)\n"
            "    return xr.grad.to(x.dtype), gr.grad.to(x.dtype)\n"
        ),
        workloads=[
            _wl(
                {"M": 256, "N": 4096},
                {"x": "random", "g": "random", "grad_output": "random"},
            ),
            _wl(
                {"M": 1, "N": 31744},
                {"x": "random", "g": "random", "grad_output": "random"},
            ),
            _wl(
                {"M": 873, "N": 1245},
                {"x": "random", "g": "random", "grad_output": "random"},
            ),
            _wl(
                {"M": 64, "N": 1024},
                {"x": "random", "g": "random", "grad_output": "random"},
            ),
        ],
    ),
    # --- Cat2 (mechanical): clean torch2flydsl elementwise (bf16) ---
    Spec(
        name="silu_and_mul_bf16",
        suite=AKASuite.TORCH2FLYDSL,
        task_path="tasks/torch2flydsl/silu_and_mul_kernel",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.BFLOAT16,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.FLYDSL,
        description="Fused SiLU-and-multiply (SwIGLU): silu(x) * y over the two halves "
        "of the last dimension, computed in FP32 then cast to BF16. Derived from AKA "
        "torch2flydsl/silu_and_mul_kernel Model.forward (the suite's FlyDSL target is "
        "irrelevant here; we lift the PyTorch oracle).",
        axes={
            "M": _ax_var("Rows."),
            "D": _ax_var("Last dimension (even)."),
            "d": _ax_expr("D // 2", "Half of the last dimension."),
        },
        inputs={
            "input": {
                "shape": ["M", "D"],
                "dtype": "bfloat16",
                "description": "Input (M, D), D even.",
            },
        },
        outputs={
            "output": {
                "shape": ["M", "d"],
                "dtype": "bfloat16",
                "description": "silu(x) * y (M, d).",
            },
        },
        reference=(
            "import torch\nimport torch.nn.functional as F\n\n"
            "def run(input):\n"
            "    d = input.shape[-1] // 2\n"
            "    x, y = input.split([d, d], dim=-1)\n"
            "    return (F.silu(x.float()) * y.float()).to(torch.bfloat16)\n"
        ),
        workloads=[
            _wl({"M": 512, "D": 8192}, {"input": "random"}),
            _wl({"M": 1, "D": 4096}, {"input": "random"}),
            _wl({"M": 2048, "D": 8192}, {"input": "random"}),
            _wl({"M": 256, "D": 16384}, {"input": "random"}),
        ],
    ),
    # --- Cat1 (batch 2): more gpumode pointwise / fused blocks ---
    Spec(
        name="gpumode_fused_leaky_relu",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/10190_FusedLeakyReLU",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        description="Fused bias-add -> leaky_relu -> scale. Derived from AKA "
        "torch2hip/gpumode/10190_FusedLeakyReLU module_fn (fused_leaky_relu_fn).",
        axes={
            "B": _ax_var("Batch."),
            "C": _ax_var("Channels."),
            "H": _ax_var("Height."),
            "W": _ax_var("Width."),
        },
        inputs={
            "x": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "Input (B, C, H, W).",
            },
            "bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "Channel bias (>= C, sliced to x's channels).",
            },
            "negative_slope": {
                "shape": None,
                "dtype": "float32",
                "description": "Leaky-ReLU negative slope.",
            },
            "scale": {
                "shape": None,
                "dtype": "float32",
                "description": "Output scale.",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "scale * leaky_relu(x + bias).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, bias, negative_slope, scale):\n"
            "    x = x + bias.reshape(1, -1, 1, 1)[:, : x.shape[1]]\n"
            "    x = F.leaky_relu(x, negative_slope=negative_slope)\n"
            "    return x * scale\n"
        ),
        workloads=[
            _wl(
                {"B": 8, "C": 4, "H": 32, "W": 32},
                {
                    "x": "random",
                    "bias": "random",
                    "negative_slope": {"scalar": 0.2},
                    "scale": {"scalar": 1.4142135623730951},
                },
            ),
            _wl(
                {"B": 16, "C": 64, "H": 64, "W": 64},
                {
                    "x": "random",
                    "bias": "random",
                    "negative_slope": {"scalar": 0.2},
                    "scale": {"scalar": 1.4142135623730951},
                },
            ),
            _wl(
                {"B": 8, "C": 128, "H": 48, "W": 48},
                {
                    "x": "random",
                    "bias": "random",
                    "negative_slope": {"scalar": 0.2},
                    "scale": {"scalar": 1.4142135623730951},
                },
            ),
            _wl(
                {"B": 32, "C": 16, "H": 56, "W": 56},
                {
                    "x": "random",
                    "bias": "random",
                    "negative_slope": {"scalar": 0.2},
                    "scale": {"scalar": 1.4142135623730951},
                },
            ),
        ],
    ),
    Spec(
        name="gpumode_transpose",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/1067_Transpose",
        op_type=AKAOperation.ELEMENTWISE,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.GPUMODE,
        description="Transpose two dimensions and return a contiguous tensor. Derived "
        "from AKA torch2hip/gpumode/1067_Transpose module_fn (dims 1, 2).",
        axes={
            "A": _ax_var("Dim 0 size."),
            "B": _ax_var("Dim 1 size (swapped to dim 2)."),
            "C": _ax_var("Dim 2 size (swapped to dim 1)."),
            "D": _ax_var("Dim 3 size."),
        },
        inputs={
            "input": {
                "shape": ["A", "B", "C", "D"],
                "dtype": "float32",
                "description": "Input (A, B, C, D).",
            },
            "dim1": {
                "shape": None,
                "dtype": "float32",
                "description": "First transpose dimension.",
            },
            "dim2": {
                "shape": None,
                "dtype": "float32",
                "description": "Second transpose dimension.",
            },
        },
        outputs={
            "output": {
                "shape": ["A", "C", "B", "D"],
                "dtype": "float32",
                "description": "input.transpose(dim1, dim2).",
            },
        },
        reference=(
            "def run(input, dim1, dim2):\n"
            "    return input.transpose(int(dim1), int(dim2)).contiguous()\n"
        ),
        workloads=[
            _wl(
                {"A": 4, "B": 8, "C": 16, "D": 32},
                {
                    "input": "random",
                    "dim1": {"scalar": 1},
                    "dim2": {"scalar": 2},
                },
            ),
            _wl(
                {"A": 8, "B": 16, "C": 32, "D": 64},
                {
                    "input": "random",
                    "dim1": {"scalar": 1},
                    "dim2": {"scalar": 2},
                },
            ),
            _wl(
                {"A": 2, "B": 64, "C": 128, "D": 16},
                {
                    "input": "random",
                    "dim1": {"scalar": 1},
                    "dim2": {"scalar": 2},
                },
            ),
            _wl(
                {"A": 16, "B": 32, "C": 48, "D": 24},
                {
                    "input": "random",
                    "dim1": {"scalar": 1},
                    "dim2": {"scalar": 2},
                },
            ),
        ],
    ),
    Spec(
        name="gpumode_softmax_3d",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/10082_SoftmaxModule",
        op_type=AKAOperation.SOFTMAX,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.SINGLE,
        source_family=AKASourceFamily.GPUMODE,
        description="Softmax over the last axis of a 3D cube. Derived from AKA "
        "torch2hip/gpumode/10082_SoftmaxModule module_fn (axis=2).",
        axes={"N": _ax_var("Cube edge (all three dims equal).")},
        inputs={
            "v": {
                "shape": ["N", "N", "N"],
                "dtype": "float32",
                "description": "Input cube (N, N, N).",
            },
            "axis": {
                "shape": None,
                "dtype": "float32",
                "description": "Softmax axis.",
            },
        },
        outputs={
            "output": {
                "shape": ["N", "N", "N"],
                "dtype": "float32",
                "description": "softmax(v, dim=axis).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(v, axis):\n    return F.softmax(v, dim=int(axis))\n"
        ),
        workloads=[
            _wl({"N": 16}, {"v": "random", "axis": {"scalar": 2}}),
            _wl({"N": 64}, {"v": "random", "axis": {"scalar": 2}}),
            _wl({"N": 128}, {"v": "random", "axis": {"scalar": 2}}),
            _wl({"N": 91}, {"v": "random", "axis": {"scalar": 2}}),
        ],
    ),
    Spec(
        name="gpumode_feedforward",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/10024_Feedforward",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        description="Fused two-input feedforward: vstack(x,y) -> linear -> ReLU -> "
        "linear -> sigmoid. Derived from AKA torch2hip/gpumode/10024_Feedforward module_fn.",
        axes={
            "B": _ax_var("Per-input batch (output batch is 2*B)."),
            "C": _ax_const(4, "Input features."),
            "H": _ax_const(100, "Hidden features."),
            "OB": _ax_expr("2 * B", "Output batch (vstack doubles the batch)."),
        },
        inputs={
            "x": {
                "shape": ["B", "C"],
                "dtype": "float32",
                "description": "First input (B, C).",
            },
            "y": {
                "shape": ["B", "C"],
                "dtype": "float32",
                "description": "Second input (B, C).",
            },
            "fc1_weight": {
                "shape": ["H", "C"],
                "dtype": "float32",
                "description": "First linear weight (H, C).",
            },
            "fc1_bias": {
                "shape": ["H"],
                "dtype": "float32",
                "description": "First linear bias (H,).",
            },
            "fc2_weight": {
                "shape": ["1", "H"],
                "dtype": "float32",
                "description": "Second linear weight (1, H).",
            },
            "fc2_bias": {
                "shape": ["1"],
                "dtype": "float32",
                "description": "Second linear bias (1,).",
            },
        },
        outputs={
            "output": {
                "shape": ["OB", "1"],
                "dtype": "float32",
                "description": "sigmoid(linear2(relu(linear1(vstack(x,y))))).",
            },
        },
        reference=(
            "import torch\nimport torch.nn.functional as F\n\n"
            "def run(x, y, fc1_weight, fc1_bias, fc2_weight, fc2_bias):\n"
            "    inp = torch.vstack([x, y])\n"
            "    hidden = F.linear(inp, fc1_weight, fc1_bias)\n"
            "    relu = F.relu(hidden)\n"
            "    output = F.linear(relu, fc2_weight, fc2_bias)\n"
            "    return torch.sigmoid(output)\n"
        ),
        workloads=[
            _wl(
                {"B": 1},
                {
                    "x": "random",
                    "y": "random",
                    "fc1_weight": "random",
                    "fc1_bias": "random",
                    "fc2_weight": "random",
                    "fc2_bias": "random",
                },
            ),
            _wl(
                {"B": 4},
                {
                    "x": "random",
                    "y": "random",
                    "fc1_weight": "random",
                    "fc1_bias": "random",
                    "fc2_weight": "random",
                    "fc2_bias": "random",
                },
            ),
            _wl(
                {"B": 8},
                {
                    "x": "random",
                    "y": "random",
                    "fc1_weight": "random",
                    "fc1_bias": "random",
                    "fc2_weight": "random",
                    "fc2_bias": "random",
                },
            ),
            _wl(
                {"B": 16},
                {
                    "x": "random",
                    "y": "random",
                    "fc1_weight": "random",
                    "fc1_bias": "random",
                    "fc2_weight": "random",
                    "fc2_bias": "random",
                },
            ),
        ],
    ),
    Spec(
        name="gpumode_positionwise_ffn",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/14044_PositionWiseFeedForward",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        description="Position-wise feedforward block: linear -> ReLU -> linear -> "
        "dropout(eval) -> residual -> layer_norm. Derived from AKA torch2hip/gpumode/"
        "14044_PositionWiseFeedForward module_fn (eval mode, deterministic).",
        axes={
            "B": _ax_var("Batch."),
            "S": _ax_var("Sequence length."),
            "C": _ax_const(4, "Model dimension."),
        },
        inputs={
            "x": {
                "shape": ["B", "S", "C"],
                "dtype": "float32",
                "description": "Input (B, S, C).",
            },
            "W_1_weight": {
                "shape": ["C", "C"],
                "dtype": "float32",
                "description": "First linear weight.",
            },
            "W_1_bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "First linear bias.",
            },
            "W_2_weight": {
                "shape": ["C", "C"],
                "dtype": "float32",
                "description": "Second linear weight.",
            },
            "W_2_bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "Second linear bias.",
            },
            "layer_norm_weight": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "LayerNorm gain.",
            },
            "layer_norm_bias": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "LayerNorm bias.",
            },
            "dropout_p": {
                "shape": None,
                "dtype": "float32",
                "description": "Dropout probability.",
            },
            "training": {
                "shape": None,
                "dtype": "float32",
                "description": "Dropout training flag (false for determinism).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "S", "C"],
                "dtype": "float32",
                "description": "FFN block output (B, S, C).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, W_1_weight, W_1_bias, W_2_weight, W_2_bias, layer_norm_weight, layer_norm_bias, dropout_p, training):\n"
            "    out = F.linear(x, W_1_weight, W_1_bias)\n"
            "    out = F.relu(out)\n"
            "    out = F.linear(out, W_2_weight, W_2_bias)\n"
            "    out = F.dropout(out, p=dropout_p, training=bool(training))\n"
            "    out = out + x\n"
            "    return F.layer_norm(out, out.shape[-1:], layer_norm_weight, layer_norm_bias)\n"
        ),
        workloads=[
            _wl(
                {"B": 4, "S": 4},
                {
                    "x": "random",
                    "W_1_weight": "random",
                    "W_1_bias": "random",
                    "W_2_weight": "random",
                    "W_2_bias": "random",
                    "layer_norm_weight": "random",
                    "layer_norm_bias": "random",
                    "dropout_p": {"scalar": 0.5},
                    "training": {"scalar": False},
                },
            ),
            _wl(
                {"B": 16, "S": 16},
                {
                    "x": "random",
                    "W_1_weight": "random",
                    "W_1_bias": "random",
                    "W_2_weight": "random",
                    "W_2_bias": "random",
                    "layer_norm_weight": "random",
                    "layer_norm_bias": "random",
                    "dropout_p": {"scalar": 0.5},
                    "training": {"scalar": False},
                },
            ),
            _wl(
                {"B": 64, "S": 64},
                {
                    "x": "random",
                    "W_1_weight": "random",
                    "W_1_bias": "random",
                    "W_2_weight": "random",
                    "W_2_bias": "random",
                    "layer_norm_weight": "random",
                    "layer_norm_bias": "random",
                    "dropout_p": {"scalar": 0.5},
                    "training": {"scalar": False},
                },
            ),
            _wl(
                {"B": 8, "S": 128},
                {
                    "x": "random",
                    "W_1_weight": "random",
                    "W_1_bias": "random",
                    "W_2_weight": "random",
                    "W_2_bias": "random",
                    "layer_norm_weight": "random",
                    "layer_norm_bias": "random",
                    "dropout_p": {"scalar": 0.5},
                    "training": {"scalar": False},
                },
            ),
        ],
    ),
    Spec(
        name="gpumode_mlp",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/gpumode/1178_MLP_model",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.GPUMODE,
        description="7-layer MLP (4 -> 4096 -> 2048 -> 512 -> 128 -> 64 -> 32 -> 4) "
        "with ReLU. Derived from AKA torch2hip/gpumode/1178_MLP_model module_fn.",
        axes={
            "B": _ax_var("Batch."),
            "IN4": _ax_const(4, "Input/output size."),
            "H1": _ax_const(4096, "Hidden 1."),
            "H2": _ax_const(2048, "Hidden 2."),
            "H3": _ax_const(512, "Hidden 3."),
            "H4": _ax_const(128, "Hidden 4."),
            "H5": _ax_const(64, "Hidden 5."),
            "H6": _ax_const(32, "Hidden 6."),
        },
        inputs={
            "xb": {
                "shape": ["B", "IN4"],
                "dtype": "float32",
                "description": "Input (B, 4).",
            },
            "linear1_weight": {"shape": ["H1", "IN4"], "dtype": "float32"},
            "linear1_bias": {"shape": ["H1"], "dtype": "float32"},
            "linear2_weight": {"shape": ["H2", "H1"], "dtype": "float32"},
            "linear2_bias": {"shape": ["H2"], "dtype": "float32"},
            "linear3_weight": {"shape": ["H3", "H2"], "dtype": "float32"},
            "linear3_bias": {"shape": ["H3"], "dtype": "float32"},
            "linear4_weight": {"shape": ["H4", "H3"], "dtype": "float32"},
            "linear4_bias": {"shape": ["H4"], "dtype": "float32"},
            "linear5_weight": {"shape": ["H5", "H4"], "dtype": "float32"},
            "linear5_bias": {"shape": ["H5"], "dtype": "float32"},
            "linear6_weight": {"shape": ["H6", "H5"], "dtype": "float32"},
            "linear6_bias": {"shape": ["H6"], "dtype": "float32"},
            "linear7_weight": {"shape": ["IN4", "H6"], "dtype": "float32"},
            "linear7_bias": {"shape": ["IN4"], "dtype": "float32"},
        },
        outputs={
            "output": {
                "shape": ["B", "IN4"],
                "dtype": "float32",
                "description": "MLP output (B, 4).",
            },
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(xb, linear1_weight, linear1_bias, linear2_weight, linear2_bias, linear3_weight, linear3_bias, linear4_weight, linear4_bias, linear5_weight, linear5_bias, linear6_weight, linear6_bias, linear7_weight, linear7_bias):\n"
            "    xb = xb.view(xb.size(0), -1)\n"
            "    out = F.relu(F.linear(xb, linear1_weight, linear1_bias))\n"
            "    out = F.relu(F.linear(out, linear2_weight, linear2_bias))\n"
            "    out = F.relu(F.linear(out, linear3_weight, linear3_bias))\n"
            "    out = F.relu(F.linear(out, linear4_weight, linear4_bias))\n"
            "    out = F.relu(F.linear(out, linear5_weight, linear5_bias))\n"
            "    out = F.relu(F.linear(out, linear6_weight, linear6_bias))\n"
            "    return F.linear(out, linear7_weight, linear7_bias)\n"
        ),
        workloads=[
            _wl(
                {"B": 4},
                {
                    "xb": "random",
                    "linear1_weight": "random",
                    "linear1_bias": "random",
                    "linear2_weight": "random",
                    "linear2_bias": "random",
                    "linear3_weight": "random",
                    "linear3_bias": "random",
                    "linear4_weight": "random",
                    "linear4_bias": "random",
                    "linear5_weight": "random",
                    "linear5_bias": "random",
                    "linear6_weight": "random",
                    "linear6_bias": "random",
                    "linear7_weight": "random",
                    "linear7_bias": "random",
                },
            ),
            _wl(
                {"B": 16},
                {
                    "xb": "random",
                    "linear1_weight": "random",
                    "linear1_bias": "random",
                    "linear2_weight": "random",
                    "linear2_bias": "random",
                    "linear3_weight": "random",
                    "linear3_bias": "random",
                    "linear4_weight": "random",
                    "linear4_bias": "random",
                    "linear5_weight": "random",
                    "linear5_bias": "random",
                    "linear6_weight": "random",
                    "linear6_bias": "random",
                    "linear7_weight": "random",
                    "linear7_bias": "random",
                },
            ),
            _wl(
                {"B": 64},
                {
                    "xb": "random",
                    "linear1_weight": "random",
                    "linear1_bias": "random",
                    "linear2_weight": "random",
                    "linear2_bias": "random",
                    "linear3_weight": "random",
                    "linear3_bias": "random",
                    "linear4_weight": "random",
                    "linear4_bias": "random",
                    "linear5_weight": "random",
                    "linear5_bias": "random",
                    "linear6_weight": "random",
                    "linear6_bias": "random",
                    "linear7_weight": "random",
                    "linear7_bias": "random",
                },
            ),
            _wl(
                {"B": 128},
                {
                    "xb": "random",
                    "linear1_weight": "random",
                    "linear1_bias": "random",
                    "linear2_weight": "random",
                    "linear2_bias": "random",
                    "linear3_weight": "random",
                    "linear3_bias": "random",
                    "linear4_weight": "random",
                    "linear4_bias": "random",
                    "linear5_weight": "random",
                    "linear5_bias": "random",
                    "linear6_weight": "random",
                    "linear6_bias": "random",
                    "linear7_weight": "random",
                    "linear7_bias": "random",
                },
            ),
        ],
    ),
    Spec(
        name="l2n55_matmul_maxpool_sum_scale",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        role=AKACorpusRole.TARGET_INCOMPATIBLE,
        exclusion_reason_code="reference_ipc_payload_limit",
        description="Fused linear -> max_pool1d -> sum -> scale, reducing to a 1D "
        "per-batch output. Derived from AKA torch2hip/kernelbench/level2/"
        "l2n55_Matmul_MaxPool_Sum_Scale. Retained for provenance but excluded "
        "from the gfx1200 scoring denominator because its 32768x32768 FP32 "
        "weight alone exceeds the bounded trusted-reference IPC payload.",
        axes={
            "B": _ax_var("Batch."),
            "IN": _ax_const(32768, "Input features."),
            "OUT": _ax_const(32768, "Output features."),
        },
        inputs={
            "x": {
                "shape": ["B", "IN"],
                "dtype": "float32",
                "description": "Input (B, IN).",
            },
            "weight": {
                "shape": ["OUT", "IN"],
                "dtype": "float32",
                "description": "Linear weight.",
            },
            "bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Linear bias.",
            },
            "kernel_size": {
                "shape": None,
                "dtype": "float32",
                "description": "Max-pool kernel size.",
            },
            "scale_factor": {
                "shape": None,
                "dtype": "float32",
                "description": "Output scale.",
            },
        },
        outputs={
            "output": {
                "shape": ["B"],
                "dtype": "float32",
                "description": "Reduced output (B,).",
            },
        },
        reference=(
            "import torch\nimport torch.nn.functional as F\n\n"
            "def run(x, weight, bias, kernel_size, scale_factor):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    x = F.max_pool1d(x.unsqueeze(1), int(kernel_size)).squeeze(1)\n"
            "    x = torch.sum(x, dim=1)\n"
            "    return x * scale_factor\n"
        ),
        workloads=[
            _wl(
                {"B": 32},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "kernel_size": {"scalar": 2},
                    "scale_factor": {"scalar": 0.5},
                },
            ),
            _wl(
                {"B": 64},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "kernel_size": {"scalar": 2},
                    "scale_factor": {"scalar": 0.5},
                },
            ),
            _wl(
                {"B": 128},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "kernel_size": {"scalar": 2},
                    "scale_factor": {"scalar": 0.5},
                },
            ),
        ],
    ),
    Spec(
        name="l2n98_matmul_avgpool_gelu_scale_max",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max",
        op_type=AKAOperation.MATMUL,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Fused linear -> avg_pool1d -> GELU -> scale -> max-reduce to a "
        "1D per-batch output. Derived from AKA torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max.",
        axes={
            "B": _ax_var("Batch."),
            "IN": _ax_const(8192, "Input features."),
            "OUT": _ax_const(8192, "Output features."),
        },
        inputs={
            "x": {
                "shape": ["B", "IN"],
                "dtype": "float32",
                "description": "Input (B, IN).",
            },
            "weight": {
                "shape": ["OUT", "IN"],
                "dtype": "float32",
                "description": "Linear weight.",
            },
            "bias": {
                "shape": ["OUT"],
                "dtype": "float32",
                "description": "Linear bias.",
            },
            "pool_kernel_size": {
                "shape": None,
                "dtype": "float32",
                "description": "Avg-pool kernel size.",
            },
            "scale_factor": {
                "shape": None,
                "dtype": "float32",
                "description": "Output scale.",
            },
        },
        outputs={
            "output": {
                "shape": ["B"],
                "dtype": "float32",
                "description": "Reduced output (B,).",
            },
        },
        reference=(
            "import torch\nimport torch.nn.functional as F\n\n"
            "def run(x, weight, bias, pool_kernel_size, scale_factor):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    x = F.avg_pool1d(x.unsqueeze(1), int(pool_kernel_size)).squeeze(1)\n"
            "    x = F.gelu(x)\n"
            "    x = x * scale_factor\n"
            "    return torch.max(x, dim=1).values\n"
        ),
        workloads=[
            _wl(
                {"B": 256},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "pool_kernel_size": {"scalar": 16},
                    "scale_factor": {"scalar": 2.0},
                },
            ),
            _wl(
                {"B": 1024},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "pool_kernel_size": {"scalar": 16},
                    "scale_factor": {"scalar": 2.0},
                },
            ),
            _wl(
                {"B": 4096},
                {
                    "x": "random",
                    "weight": "random",
                    "bias": "random",
                    "pool_kernel_size": {"scalar": 16},
                    "scale_factor": {"scalar": 2.0},
                },
            ),
        ],
    ),
    Spec(
        name="l3n44_mingpt_block",
        suite=AKASuite.TORCH2HIP,
        task_path="tasks/torch2hip/kernelbench/level3/l3n44_MiniGPTBlock",
        op_type=AKAOperation.ATTENTION,
        dtype=DType.FLOAT32,
        pass_kind=AKAPassKind.FORWARD,
        fusion_depth=AKAFusionDepth.FUSED,
        source_family=AKASourceFamily.KERNELBENCH,
        description="Full MiniGPT transformer block: layernorm -> causal self-attention "
        "-> residual -> layernorm -> MLP(new-GELU) -> residual. Derived from AKA "
        "torch2hip/kernelbench/level3/l3n44_MiniGPTBlock module_fn.",
        axes={
            "B": _ax_var("Batch."),
            "S": _ax_var("Sequence length (<= 1024)."),
            "C": _ax_const(768, "Model dimension."),
            "C3": _ax_expr("3 * C", "QKV projection output (3*C)."),
            "C4": _ax_expr("4 * C", "MLP hidden (4*C)."),
            "MT": _ax_const(
                1024,
                "Max sequence length (causal-mask buffer edge).",
            ),
        },
        inputs={
            "x": {
                "shape": ["B", "S", "C"],
                "dtype": "float32",
                "description": "Input (B, S, C).",
            },
            "ln1_w": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "LayerNorm 1 gain.",
            },
            "ln1_b": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "LayerNorm 1 bias.",
            },
            "attn_c_attn_w": {
                "shape": ["C3", "C"],
                "dtype": "float32",
                "description": "QKV projection weight.",
            },
            "attn_c_attn_b": {
                "shape": ["C3"],
                "dtype": "float32",
                "description": "QKV projection bias.",
            },
            "attn_c_proj_w": {
                "shape": ["C", "C"],
                "dtype": "float32",
                "description": "Attention output projection weight.",
            },
            "attn_c_proj_b": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "Attention output projection bias.",
            },
            "attn_bias": {
                "shape": ["1", "1", "MT", "MT"],
                "dtype": "float32",
                "description": "Causal mask buffer (1, 1, MT, MT).",
            },
            "n_head": {
                "shape": None,
                "dtype": "float32",
                "description": "Attention head count.",
            },
            "ln2_w": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "LayerNorm 2 gain.",
            },
            "ln2_b": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "LayerNorm 2 bias.",
            },
            "mlp_cfc_w": {
                "shape": ["C4", "C"],
                "dtype": "float32",
                "description": "MLP fc weight.",
            },
            "mlp_cfc_b": {
                "shape": ["C4"],
                "dtype": "float32",
                "description": "MLP fc bias.",
            },
            "mlp_cproj_w": {
                "shape": ["C", "C4"],
                "dtype": "float32",
                "description": "MLP proj weight.",
            },
            "mlp_cproj_b": {
                "shape": ["C"],
                "dtype": "float32",
                "description": "MLP proj bias.",
            },
            "n_embd": {
                "shape": None,
                "dtype": "float32",
                "description": "Model dimension (== C).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "S", "C"],
                "dtype": "float32",
                "description": "Block output (B, S, C).",
            },
        },
        reference=(
            "import math\n\nimport torch\nimport torch.nn.functional as F\n\n"
            "def _new_gelu(z):\n"
            "    return 0.5 * z * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * torch.pow(z, 3.0))))\n\n"
            "def run(x, ln1_w, ln1_b, attn_c_attn_w, attn_c_attn_b, attn_c_proj_w, attn_c_proj_b, attn_bias, n_head, ln2_w, ln2_b, mlp_cfc_w, mlp_cfc_b, mlp_cproj_w, mlp_cproj_b, n_embd):\n"
            "    a = F.layer_norm(x, (int(n_embd),), ln1_w, ln1_b)\n"
            "    B, T, C = a.size()\n"
            "    nh = int(n_head)\n"
            "    qkv = F.linear(a, attn_c_attn_w, attn_c_attn_b)\n"
            "    q, k, v = qkv.split(C, dim=2)\n"
            "    k = k.view(B, T, nh, C // nh).transpose(1, 2)\n"
            "    q = q.view(B, T, nh, C // nh).transpose(1, 2)\n"
            "    v = v.view(B, T, nh, C // nh).transpose(1, 2)\n"
            "    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))\n"
            "    att = att.masked_fill(attn_bias[:, :, :T, :T] == 0, float('-inf'))\n"
            "    att = F.softmax(att, dim=-1)\n"
            "    y = att @ v\n"
            "    y = y.transpose(1, 2).contiguous().view(B, T, C)\n"
            "    x = x + F.linear(y, attn_c_proj_w, attn_c_proj_b)\n"
            "    m = F.layer_norm(x, (int(n_embd),), ln2_w, ln2_b)\n"
            "    h = F.linear(m, mlp_cfc_w, mlp_cfc_b)\n"
            "    h = _new_gelu(h)\n"
            "    h = F.linear(h, mlp_cproj_w, mlp_cproj_b)\n"
            "    return x + h\n"
        ),
        workloads=[
            _wl(
                {"B": 16, "S": 256},
                {
                    "x": "random",
                    "ln1_w": "random",
                    "ln1_b": "random",
                    "attn_c_attn_w": "random",
                    "attn_c_attn_b": "random",
                    "attn_c_proj_w": "random",
                    "attn_c_proj_b": "random",
                    "attn_bias": "random",
                    "n_head": {"scalar": 8},
                    "ln2_w": "random",
                    "ln2_b": "random",
                    "mlp_cfc_w": "random",
                    "mlp_cfc_b": "random",
                    "mlp_cproj_w": "random",
                    "mlp_cproj_b": "random",
                    "n_embd": {"scalar": 768},
                },
            ),
            _wl(
                {"B": 32, "S": 256},
                {
                    "x": "random",
                    "ln1_w": "random",
                    "ln1_b": "random",
                    "attn_c_attn_w": "random",
                    "attn_c_attn_b": "random",
                    "attn_c_proj_w": "random",
                    "attn_c_proj_b": "random",
                    "attn_bias": "random",
                    "n_head": {"scalar": 8},
                    "ln2_w": "random",
                    "ln2_b": "random",
                    "mlp_cfc_w": "random",
                    "mlp_cfc_b": "random",
                    "mlp_cproj_w": "random",
                    "mlp_cproj_b": "random",
                    "n_embd": {"scalar": 768},
                },
            ),
            _wl(
                {"B": 16, "S": 512},
                {
                    "x": "random",
                    "ln1_w": "random",
                    "ln1_b": "random",
                    "attn_c_attn_w": "random",
                    "attn_c_attn_b": "random",
                    "attn_c_proj_w": "random",
                    "attn_c_proj_b": "random",
                    "attn_bias": "random",
                    "n_head": {"scalar": 8},
                    "ln2_w": "random",
                    "ln2_b": "random",
                    "mlp_cfc_w": "random",
                    "mlp_cfc_b": "random",
                    "mlp_cproj_w": "random",
                    "mlp_cproj_b": "random",
                    "n_embd": {"scalar": 768},
                },
            ),
            _wl(
                {"B": 8, "S": 512},
                {
                    "x": "random",
                    "ln1_w": "random",
                    "ln1_b": "random",
                    "attn_c_attn_w": "random",
                    "attn_c_attn_b": "random",
                    "attn_c_proj_w": "random",
                    "attn_c_proj_b": "random",
                    "attn_bias": "random",
                    "n_head": {"scalar": 8},
                    "ln2_w": "random",
                    "ln2_b": "random",
                    "mlp_cfc_w": "random",
                    "mlp_cfc_b": "random",
                    "mlp_cproj_w": "random",
                    "mlp_cproj_b": "random",
                    "n_embd": {"scalar": 768},
                },
            ),
        ],
    ),
]


def _generated(generator: dict[str, Any]) -> dict[str, Any]:
    return {"type": "generated", "generator": generator}


def _normal(mean: float = 0.0, std: float = 1.0) -> dict[str, Any]:
    return _generated({"type": "normal", "mean": mean, "std": std})


def _constant(value: float | bool) -> dict[str, Any]:
    return _generated({"type": "constant", "value": value})


def _numeric_normalized(output: str, threshold: float) -> dict[str, Any]:
    return {
        "type": "numeric",
        "output": output,
        "mode": "normalized_max",
        "max_atol": 0.0,
        "max_rtol": threshold,
        "required_matched_ratio": 1.0,
        "max_error_cap": None,
        "allow_negative_inf": False,
    }


_CROSS_ENTROPY_SHAPES = [
    (8192, 1024),
    (16384, 2048),
    (32768, 4096),
    (16384, 8192),
    (8192, 16384),
]
_BATCH_NORM_SHAPES = [
    (16, 64, 64),
    (32, 64, 64),
    (64, 64, 64),
    (32, 128, 128),
    (64, 128, 128),
]
_KD_LOSS_SHAPES = [
    (8, 10, 32, 32),
    (16, 10, 32, 32),
    (8, 10, 64, 64),
    (16, 10, 64, 64),
]
_RMS_SHAPES = [
    (1, 4096),
    (8, 4096),
    (32, 8192),
    (128, 4096),
    (256, 8192),
    (64, 16384),
]
_I8_SHAPES = [(1, 4096), (16, 8192), (128, 4096), (256, 8192), (1024, 8192)]
_MXFP8_SHAPES = [
    (1, 32),
    (8, 64),
    (16, 128),
    (32, 256),
    (64, 512),
    (128, 1024),
    (137, 64),
    (256, 32),
]


SPECS.extend(
    [
        Spec(
            name="l1n95_cross_entropy",
            suite=AKASuite.TORCH2HIP,
            task_path="tasks/torch2hip/kernelbench/level1/l1n95_CrossEntropyLoss",
            op_type=AKAOperation.LOSS,
            dtype=DType.FLOAT32,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.SINGLE,
            source_family=AKASourceFamily.KERNELBENCH,
            axes={"B": _ax_var("Batch size."), "C": _ax_var("Class count.")},
            inputs={
                "predictions": {"shape": ["B", "C"], "dtype": "float32"},
                "targets": {"shape": ["B"], "dtype": "int64"},
            },
            outputs={"loss": {"shape": [], "dtype": "float32"}},
            reference=(
                "import torch.nn.functional as F\n\n"
                "def run(predictions, targets):\n"
                "    return F.cross_entropy(predictions, targets)\n"
            ),
            workloads=[
                _wl(
                    {"B": batch, "C": classes},
                    {
                        "predictions": _generated(
                            {"type": "uniform", "low": 0.0, "high": 1.0},
                        ),
                        "targets": _generated(
                            {"type": "integer", "low": 0, "high": "C"},
                        ),
                    },
                )
                for batch, classes in _CROSS_ENTROPY_SHAPES
            ],
            capabilities=(
                AKACapability.BOUNDED_INTEGER_INPUT,
                AKACapability.SCALAR_TENSOR_OUTPUT,
            ),
            description="Cross entropy with class-bounded integer labels.",
        ),
        Spec(
            name="l2n52_conv_activation_batchnorm",
            suite=AKASuite.TORCH2HIP,
            task_path=(
                "tasks/torch2hip/kernelbench/level2/"
                "l2n52_Conv2d_Activation_BatchNorm"
            ),
            op_type=AKAOperation.CONV,
            dtype=DType.FLOAT32,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.FUSED,
            source_family=AKASourceFamily.KERNELBENCH,
            axes={
                "B": _ax_var("Batch size."),
                "H": _ax_var("Input height."),
                "W": _ax_var("Input width."),
                "IC": _ax_const(64),
                "OC": _ax_const(128),
                "K": _ax_const(3),
                "OH": _ax_expr("H - K + 1"),
                "OW": _ax_expr("W - K + 1"),
            },
            inputs={
                "x": {"shape": ["B", "IC", "H", "W"], "dtype": "float32"},
                "conv_weight": {
                    "shape": ["OC", "IC", "K", "K"],
                    "dtype": "float32",
                },
                "conv_bias": {"shape": ["OC"], "dtype": "float32"},
                "bn_weight": {"shape": ["OC"], "dtype": "float32"},
                "bn_bias": {"shape": ["OC"], "dtype": "float32"},
                "bn_mean": {"shape": ["OC"], "dtype": "float32"},
                "bn_var": {"shape": ["OC"], "dtype": "float32"},
                "bn_eps": {"shape": None, "dtype": "float32"},
            },
            outputs={
                "output": {"shape": ["B", "OC", "OH", "OW"], "dtype": "float32"}
            },
            reference=(
                "import torch\n"
                "import torch.nn.functional as F\n\n"
                "def run(x, conv_weight, conv_bias, bn_weight, bn_bias, "
                "bn_mean, bn_var, bn_eps):\n"
                "    value = F.conv2d(x, conv_weight, conv_bias)\n"
                "    value = torch.multiply(torch.tanh(F.softplus(value)), value)\n"
                "    return F.batch_norm(value, bn_mean, bn_var, bn_weight, "
                "bn_bias, training=False, eps=bn_eps)\n"
            ),
            workloads=[
                _wl(
                    {"B": batch, "H": height, "W": width},
                    {
                        "x": _generated(
                            {"type": "uniform", "low": 0.0, "high": 1.0},
                        ),
                        "conv_weight": "random",
                        "conv_bias": "random",
                        "bn_weight": _constant(1.0),
                        "bn_bias": _constant(0.0),
                        "bn_mean": _constant(0.0),
                        "bn_var": _constant(1.0),
                        "bn_eps": {"scalar": 1e-5},
                    },
                )
                for batch, height, width in _BATCH_NORM_SHAPES
            ],
            capabilities=(AKACapability.POSITIVE_INPUT,),
            description="Conv2d, Mish activation, and eval-mode BatchNorm.",
        ),
        Spec(
            name="14007_kd_loss",
            suite=AKASuite.TORCH2HIP,
            task_path="tasks/torch2hip/gpumode/14007_KDLoss",
            op_type=AKAOperation.LOSS,
            dtype=DType.FLOAT32,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.FUSED,
            source_family=AKASourceFamily.GPUMODE,
            axes={
                "N": _ax_var("Batch size."),
                "C": _ax_var("Channel count."),
                "H": _ax_var("Height."),
                "W": _ax_var("Width."),
            },
            inputs={
                "input": {"shape": ["N", "C", "H", "W"], "dtype": "float32"},
                "target": {"shape": ["N", "C", "H", "W"], "dtype": "float32"},
                "temperature": {"shape": None, "dtype": "float32"},
            },
            outputs={"loss": {"shape": [], "dtype": "float32"}},
            reference=(
                "import torch.nn.functional as F\n\n"
                "def run(input, target, temperature):\n"
                "    log_p = F.log_softmax(input / temperature, dim=1)\n"
                "    return F.kl_div(log_p, target, reduction='sum') * "
                "(temperature * temperature) / input.size(0)\n"
            ),
            workloads=[
                _wl(
                    {"N": n, "C": c, "H": h, "W": w},
                    {
                        "input": _normal(),
                        "target": _generated(
                            {"type": "simplex", "axis": 1, "temperature": 1.0},
                        ),
                        "temperature": {"scalar": 4.0},
                    },
                )
                for n, c, h, w in _KD_LOSS_SHAPES
            ],
            capabilities=(
                AKACapability.SIMPLEX_INPUT,
                AKACapability.SCALAR_TENSOR_OUTPUT,
            ),
            description="Knowledge-distillation KL loss with probability targets.",
        ),
        Spec(
            name="fused_add_rmsnorm_bf16",
            suite=AKASuite.TORCH2FLYDSL,
            task_path="tasks/torch2flydsl/fused_add_rmsnorm_kernel",
            op_type=AKAOperation.NORM,
            dtype=DType.BFLOAT16,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.FUSED,
            source_family=AKASourceFamily.FLYDSL,
            axes={"M": _ax_var("Rows."), "N": _ax_var("Hidden size.")},
            inputs={
                "input": {"shape": ["M", "N"], "dtype": "bfloat16"},
                "weight": {"shape": ["N"], "dtype": "bfloat16"},
                "residual": {"shape": ["M", "N"], "dtype": "bfloat16"},
                "eps": {"shape": None, "dtype": "float32"},
            },
            outputs={
                "output": {"shape": ["M", "N"], "dtype": "bfloat16"},
                "residual_out": {"shape": ["M", "N"], "dtype": "bfloat16"},
            },
            reference=(
                "import torch\n\n"
                "def run(input, weight, residual, eps):\n"
                "    residual_out = input + residual\n"
                "    value = residual_out.float()\n"
                "    rstd = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)\n"
                "    output = value * rstd * weight.float()\n"
                "    return output.to(input.dtype), residual_out.to(input.dtype)\n"
            ),
            workloads=[
                {
                    **_wl(
                        {"M": m, "N": n},
                        {
                            "input": _normal(),
                            "weight": _normal(),
                            "residual": _normal(),
                            "eps": {"scalar": 1e-5},
                        },
                    ),
                    "checks": [
                        _numeric_normalized("output", 1e-2),
                        _numeric_normalized("residual_out", 1e-2),
                    ],
                }
                for m, n in _RMS_SHAPES
            ],
            capabilities=(AKACapability.MULTI_OUTPUT,),
            description="BF16 fused residual add and RMSNorm with two outputs.",
        ),
        Spec(
            name="per_token_i8_quant",
            suite=AKASuite.TORCH2FLYDSL,
            task_path="tasks/torch2flydsl/per_token_i8_quant_kernel",
            op_type=AKAOperation.QUANTIZATION,
            dtype=DType.BFLOAT16,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.FUSED,
            source_family=AKASourceFamily.FLYDSL,
            axes={"M": _ax_var("Rows."), "N": _ax_var("Hidden size.")},
            inputs={"input": {"shape": ["M", "N"], "dtype": "bfloat16"}},
            outputs={
                "codes": {"shape": ["M", "N"], "dtype": "int8"},
                "scale": {"shape": ["M", "1"], "dtype": "float32"},
            },
            reference=(
                "import torch\n\n"
                "def run(input):\n"
                "    value = input.float()\n"
                "    scale = value.abs().amax(dim=-1, keepdim=True) / 127.0\n"
                "    scale = torch.where(scale == 0, torch.ones_like(scale), scale)\n"
                "    codes = torch.clamp(value / scale, -128.0, 127.0).to(torch.int8)\n"
                "    return codes, scale.float()\n"
            ),
            workloads=[
                {
                    **_wl({"M": m, "N": n}, {"input": _normal()}),
                    "checks": [
                        {
                            "type": "code_distance",
                            "output": "codes",
                            "mode": "value",
                            "max_distance": 1,
                            "required_matched_ratio": 1.0,
                        },
                        _numeric_normalized("scale", 1e-3),
                    ],
                }
                for m, n in _I8_SHAPES
            ],
            capabilities=(
                AKACapability.CODE_DISTANCE,
                AKACapability.MIXED_OUTPUT_DTYPE,
                AKACapability.MULTI_OUTPUT,
            ),
            description="Dynamic per-token INT8 quantization with FP32 scale.",
        ),
        Spec(
            name="rope_thd_fwd_bf16",
            suite=AKASuite.TORCH2FLYDSL,
            task_path="tasks/torch2flydsl/rope_thd_fwd_kernel",
            op_type=AKAOperation.POSITION_ENCODING,
            dtype=DType.BFLOAT16,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.SINGLE,
            source_family=AKASourceFamily.FLYDSL,
            axes={
                "T": _ax_var("Packed token count."),
                "H": _ax_var("Head count."),
                "D": _ax_var("Head dimension."),
                "S1": _ax_var("Number of sequence offsets."),
                "HALF": _ax_expr("D // 2"),
                "ONE": _ax_const(1),
            },
            inputs={
                "input": {"shape": ["T", "H", "D"], "dtype": "bfloat16"},
                "cu_seqlens": {"shape": ["S1"], "dtype": "int32"},
                "freqs": {
                    "shape": ["T", "ONE", "ONE", "HALF"],
                    "dtype": "bfloat16",
                },
            },
            outputs={"output": {"shape": ["T", "H", "D"], "dtype": "bfloat16"}},
            reference=(
                "import torch\n\n"
                "def gen_structured_inputs(values, device):\n"
                "    cases = {(1024, 6): [0, 100, 228, 484, 712, 1024], "
                "(1024, 5): [0, 233, 456, 711, 1024], "
                "(1024, 9): [0, 100, 102, 128, 233, 456, 460, 711, 1024], "
                "(2048, 4): [0, 512, 1024, 2048]}\n"
                "    cu = torch.tensor(cases[(values['T'], values['S1'])], "
                "dtype=torch.int32, device=device)\n"
                "    half = values['D'] // 2\n"
                "    inv = 1.0 / (10000 ** "
                "(torch.arange(half, device=device).float() / half))\n"
                "    pos = torch.arange(values['T'], device=device).float()\n"
                "    freqs = torch.einsum('i,j->ij', pos, inv).view("
                "values['T'], 1, 1, half).to(torch.bfloat16)\n"
                "    return {'cu_seqlens': cu, 'freqs': freqs}\n\n"
                "def run(input, cu_seqlens, freqs):\n"
                "    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()\n"
                "    outputs = []\n"
                "    for value in torch.split(input, lengths):\n"
                "        x = value.float().unsqueeze(1)\n"
                "        angles = freqs[:value.shape[0]].float().repeat(1, 1, 1, 2)\n"
                "        first, second = x.chunk(2, dim=-1)\n"
                "        rotated = torch.cat((-second, first), dim=-1)\n"
                "        outputs.append((x * angles.cos() + rotated * angles.sin()).squeeze(1))\n"
                "    return torch.cat(outputs).to(input.dtype)\n"
            ),
            workloads=[
                _wl(
                    {"T": t, "H": h, "D": d, "S1": s1},
                    {
                        "input": _normal(),
                        "cu_seqlens": {"type": "custom"},
                        "freqs": {"type": "custom"},
                    },
                )
                for t, h, d, s1 in [
                    (1024, 8, 128, 6),
                    (1024, 16, 128, 5),
                    (1024, 8, 64, 9),
                    (2048, 32, 128, 4),
                ]
            ],
            custom_inputs_entrypoint="gen_structured_inputs",
            capabilities=(
                AKACapability.PARTIAL_CUSTOM_INPUT,
                AKACapability.STRUCTURED_OFFSETS,
            ),
            description="Variable-length packed BF16 RoPE with int32 prefix offsets.",
        ),
        Spec(
            name="dynamic_mxfp8_quant",
            suite=AKASuite.TORCH2FLYDSL,
            task_path="tasks/torch2flydsl/dynamic_mxfp8_quant_kernel",
            op_type=AKAOperation.QUANTIZATION,
            dtype=DType.BFLOAT16,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.FUSED,
            source_family=AKASourceFamily.FLYDSL,
            axes={
                "M": _ax_var("Rows."),
                "K": _ax_var("Columns."),
                "G": _ax_expr("K // 32"),
            },
            inputs={"input": {"shape": ["M", "K"], "dtype": "bfloat16"}},
            outputs={
                "codes": {"shape": ["M", "K"], "dtype": "float8_e4m3fn"},
                "scale": {"shape": ["M", "G"], "dtype": "uint8"},
            },
            reference=(
                "import torch\n\n"
                "def run(input):\n"
                "    value = input.float()\n"
                "    m, k = value.shape\n"
                "    blocks = value.reshape(m, k // 32, 32)\n"
                "    amax = blocks.abs().amax(dim=-1, keepdim=True)\n"
                "    bits = (amax.contiguous().view(torch.int32) + 0x200000) & -8388608\n"
                "    power = bits.view(torch.float32)\n"
                "    exponent = torch.clamp(power.log2().floor() - 8, -127, 127)\n"
                "    scale = (exponent.to(torch.int32) + 127).to(torch.uint8)\n"
                "    codes = (blocks * torch.exp2(-exponent)).reshape(m, k)\n"
                "    return codes.to(torch.float8_e4m3fn), scale.reshape(m, k // 32)\n"
            ),
            workloads=[
                {
                    **_wl({"M": m, "K": k}, {"input": _normal(std=4.0)}),
                    "checks": [
                        {
                            "type": "code_distance",
                            "output": "codes",
                            "mode": "raw_bits",
                            "max_distance": 1,
                            "required_matched_ratio": 1.0,
                        },
                        {"type": "exact", "output": "scale"},
                    ],
                }
                for m, k in _MXFP8_SHAPES
            ],
            capabilities=(
                AKACapability.CODE_DISTANCE,
                AKACapability.FP8_OUTPUT,
                AKACapability.MIXED_OUTPUT_DTYPE,
                AKACapability.MULTI_OUTPUT,
                AKACapability.RAW_CODE_DISTANCE,
                AKACapability.UINT8_OUTPUT,
            ),
            description="Per-1x32 MXFP8 quantization with E8M0 byte scales.",
        ),
        Spec(
            name="moe_topk_softmax",
            suite=AKASuite.TORCH2FLYDSL,
            task_path="tasks/torch2flydsl/moe_topk_softmax_kernel",
            op_type=AKAOperation.ROUTING,
            dtype=DType.BFLOAT16,
            pass_kind=AKAPassKind.FORWARD,
            fusion_depth=AKAFusionDepth.FUSED,
            source_family=AKASourceFamily.FLYDSL,
            axes={
                "T": _ax_var("Token count."),
                "E": _ax_var("Expert count."),
                "K": _ax_var("Selected experts."),
            },
            inputs={
                "gating": {"shape": ["T", "E"], "dtype": "bfloat16"},
                "bias": {"shape": ["E"], "dtype": "float32"},
                "topk": {"shape": None, "dtype": "int32"},
                "route_scale": {"shape": None, "dtype": "float32"},
            },
            outputs={
                "weights": {"shape": ["T", "K"], "dtype": "float32"},
                "ids": {"shape": ["T", "K"], "dtype": "int32"},
            },
            reference=(
                "import torch\n\n"
                "def gen_gating(values, device):\n"
                "    experts, tokens = values['E'], values['T']\n"
                "    base = torch.arange(-1, 1, 2.0 / experts, device=device)[:experts]\n"
                "    gating = base.repeat(tokens, 1).to(torch.bfloat16)\n"
                "    perm = torch.argsort(torch.rand(gating.shape, device=device), dim=-1)\n"
                "    return {'gating': torch.gather(gating, -1, perm).contiguous()}\n\n"
                "def run(gating, bias, topk, route_scale):\n"
                "    scores = torch.softmax(gating.float(), dim=-1)\n"
                "    ids = (scores + bias.float()).topk(topk, dim=-1, sorted=False).indices\n"
                "    weights = scores.gather(1, ids) * route_scale\n"
                "    return weights.float(), ids.to(torch.int32)\n"
            ),
            workloads=[
                {
                    **_wl(
                        {"T": t, "E": e, "K": k},
                        {
                            "gating": {"type": "custom"},
                            "bias": _normal(std=0.1)
                            if use_bias
                            else _constant(0.0),
                            "topk": {"scalar": k},
                            "route_scale": {"scalar": 1.0},
                        },
                    ),
                    "checks": [
                        {
                            "type": "topk_routing",
                            "ids_output": "ids",
                            "weights_output": "weights",
                            "gating_input": "gating",
                            "bias_input": "bias",
                            "topk": k,
                            "tie_atol": 1e-4,
                            "weight_atol": 1e-2,
                            "max_mismatch_ratio": 0.05 if use_bias else 0.0,
                        },
                    ],
                }
                for t, e, k, use_bias in [
                    (64, 256, 8, True),
                    (1024, 256, 8, True),
                    (256, 128, 4, True),
                    (64, 64, 2, False),
                ]
            ],
            custom_inputs_entrypoint="gen_gating",
            capabilities=(
                AKACapability.COUPLED_TOPK,
                AKACapability.MIXED_OUTPUT_DTYPE,
                AKACapability.MULTI_OUTPUT,
                AKACapability.PARTIAL_CUSTOM_INPUT,
            ),
            description="Tie-aware softmax MoE top-k routing with IDs and weights.",
        ),
    ],
)


def _artifact_record(
    task_root: Path,
    role: AKAArtifactRole,
    path: Path,
) -> dict[str, str]:
    return {
        "role": str(role),
        "path": path.relative_to(task_root).as_posix(),
        "sha256": sha256_file(path),
    }


def _aka_artifacts(aka_root: Path, spec: Spec) -> list[dict[str, str]]:
    task = read_task(aka_root, spec.task_path)
    runner = correctness_runner_path(task)
    if spec.suite is AKASuite.TORCH2HIP:
        semantic_reference = functional_reference_path(task)
    elif spec.suite is AKASuite.TORCH2FLYDSL:
        semantic_reference = task.root / "model.py"
    elif spec.suite is AKASuite.INSTRUCTION2TRITON:
        semantic_reference = runner
    else:
        raise ValueError(f"unsupported AKA suite for provenance: {spec.suite}")
    return [
        _artifact_record(
            task.root,
            AKAArtifactRole.CONFIG,
            task.root / "config.yaml",
        ),
        _artifact_record(
            task.root,
            AKAArtifactRole.SEMANTIC_REFERENCE,
            semantic_reference,
        ),
        _artifact_record(
            task.root,
            AKAArtifactRole.CORRECTNESS_RUNNER,
            runner,
        ),
    ]


def _workload_checks(
    spec: Spec,
    workload: dict[str, Any],
    uuid: str,
    calibrated: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if calibrated is None or spec.role is AKACorpusRole.TARGET_INCOMPATIBLE:
        checks = [dict(check) for check in workload.get("checks", [])]
    else:
        try:
            checks = [
                check.model_dump(mode="json") for check in calibrated[uuid]
            ]
        except KeyError as exc:
            raise ValueError(f"missing calibrated checks for {uuid}") from exc
    if checks:
        return checks
    return [
        {
            "type": "numeric",
            "output": output,
            **dtype_default_tolerance(output_spec["dtype"]).model_dump(
                mode="json",
            ),
        }
        for output, output_spec in spec.outputs.items()
    ]


def _workload_records(
    spec: Spec,
    calibrated: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records = []
    for index, workload in enumerate(spec.workloads):
        uuid = f"aka-{spec.name}-w{index}"
        inputs = {
            name: (
                {"type": "scalar", "value": meta["scalar"]}
                if isinstance(meta, dict) and "scalar" in meta
                else dict(meta)
                if isinstance(meta, dict) and "type" in meta
                else {"type": "random"}
            )
            for name, meta in workload["inputs"].items()
        }
        record = {
            "axes": workload["axes"],
            "inputs": inputs,
            "checks": _workload_checks(spec, workload, uuid, calibrated),
            "uuid": uuid,
        }
        Workload.model_validate(record)
        records.append(record)
    return records


def _definition_payload(spec: Spec) -> dict[str, Any]:
    payload = {
        "name": spec.name,
        "op_type": spec.op_type,
        "description": spec.description,
        "axes": spec.axes,
        "inputs": spec.inputs,
        "outputs": spec.outputs,
        "reference": spec.reference,
    }
    if spec.custom_inputs_entrypoint is not None:
        payload["custom_inputs_entrypoint"] = spec.custom_inputs_entrypoint
    Definition.model_validate(payload)
    return payload


def _write_problem(
    spec: Spec,
    calibrated: dict[str, Any] | None,
    *,
    problems_root: Path = PROBLEMS_ROOT,
) -> dict[str, str]:
    problem_dir = problems_root / spec.suite / spec.name
    problem_dir.mkdir(parents=True, exist_ok=True)
    definition_payload = _definition_payload(spec)
    workload_records = _workload_records(spec, calibrated)
    definition_path = problem_dir / "definition.json"
    workload_path = problem_dir / "workload.jsonl"
    reference_path = problem_dir / "reference.py"
    definition_path.write_text(json.dumps(definition_payload, indent=2) + "\n")
    workload_path.write_text(
        "".join(
            json.dumps(item, sort_keys=True) + "\n" for item in workload_records
        ),
    )
    reference_path.write_text(
        f'"""Standalone PyTorch reference for {spec.name} (debug mirror)."""\n'
        + spec.reference,
    )
    return {
        "path": f"{spec.suite}/{spec.name}",
        "definition_sha256": sha256_file(definition_path),
        "workload_sha256": sha256_file(workload_path),
    }


def _format_authored_references(
    specs: list[Spec],
    records: list[dict[str, str]],
    *,
    problems_root: Path,
) -> None:
    """Ruff-format debug mirrors and bind the same source into Definitions."""
    ruff = resolve_tool_path("ruff")
    if ruff is None:
        raise RuntimeError(
            "Ruff is required to author canonical AKA references",
        )
    reference_paths = [
        problems_root / spec.suite / spec.name / "reference.py"
        for spec in specs
    ]
    completed = run_in_process_group_bounded(
        [str(ruff), "format", *map(str, reference_paths)],
        cwd=REPO_ROOT,
        timeout=120,
        max_capture_bytes=16 * 1024,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "Ruff failed").strip()
        raise RuntimeError(f"could not format AKA references: {detail}")
    for spec, record, reference_path in zip(
        specs,
        records,
        reference_paths,
        strict=True,
    ):
        header = f'"""Standalone PyTorch reference for {spec.name} (debug mirror)."""'
        mirror = reference_path.read_text(encoding="utf-8")
        if not mirror.startswith(header):
            raise ValueError(
                f"formatted AKA reference lost its header: {spec.name}",
            )
        reference = mirror[len(header) :].lstrip("\n")
        if ast.dump(ast.parse(reference)) != ast.dump(
            ast.parse(spec.reference),
        ):
            raise ValueError(
                f"Ruff changed AKA reference semantics: {spec.name}",
            )
        definition_path = reference_path.with_name("definition.json")
        payload = json.loads(definition_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"AKA definition must be an object: {spec.name}")
        payload["reference"] = reference
        Definition.model_validate(payload)
        definition_path.write_text(json.dumps(payload, indent=2) + "\n")
        record["definition_sha256"] = sha256_file(definition_path)


def _rebind_format_only_calibration(
    calibration_path: Path,
    *,
    problems_root: Path,
) -> None:
    """Rebind calibration contracts after AST-equivalent reference formatting."""
    payload = load_tolerance_calibration(calibration_path)
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise ValueError("calibration records must be a list")
    records = {
        str(record["workload_uuid"]): record
        for record in raw_records
        if isinstance(record, dict)
    }
    if len(records) != len(raw_records):
        raise ValueError("calibration records must be unique objects")
    observed: set[str] = set()
    for spec in SPECS:
        problem_dir = problems_root / spec.suite / spec.name
        definition = Definition.model_validate_json(
            (problem_dir / "definition.json").read_text(encoding="utf-8"),
        )
        workloads = [
            Workload.model_validate_json(line)
            for line in (problem_dir / "workload.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        for workload in workloads:
            record = records.get(workload.uuid)
            if record is None:
                raise ValueError(f"missing calibration record: {workload.uuid}")
            record["contract_sha256"] = workload_contract_sha256(
                definition,
                workload,
            )
            observed.add(workload.uuid)
    if observed != set(records):
        raise ValueError("calibration contains records outside the AKA corpus")
    atomic_write_json_value(calibration_path, payload)


def _coverage_axes(specs: list[Spec]) -> dict[str, dict[str, int]]:
    def _count(field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in specs:
            value = str(getattr(s, field))
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items()))

    axes = {
        "operation": _count("op_type"),
        "pass_kind": _count("pass_kind"),
        "fusion_depth": _count("fusion_depth"),
        "source_family": _count("source_family"),
        "suite": _count("suite"),
    }
    for name, values in (
        (
            "input_dtype",
            (
                sorted({str(item["dtype"]) for item in spec.inputs.values()})
                for spec in specs
            ),
        ),
        (
            "output_dtype",
            (
                sorted({str(item["dtype"]) for item in spec.outputs.values()})
                for spec in specs
            ),
        ),
        (
            "capability",
            (
                [str(item) for item in _spec_capabilities(spec)]
                for spec in specs
            ),
        ),
    ):
        counts: dict[str, int] = {}
        for items in values:
            for item in items:
                counts[item] = counts.get(item, 0) + 1
        axes[name] = dict(sorted(counts.items()))
    return axes


def _spec_capabilities(spec: Spec) -> tuple[AKACapability, ...]:
    capabilities = set(spec.capabilities)
    output_dtypes = {str(item["dtype"]) for item in spec.outputs.values()}
    if len(spec.outputs) > 1:
        capabilities.add(AKACapability.MULTI_OUTPUT)
    if len(output_dtypes) > 1:
        capabilities.add(AKACapability.MIXED_OUTPUT_DTYPE)
    if any(item.get("shape") == [] for item in spec.outputs.values()):
        capabilities.add(AKACapability.SCALAR_TENSOR_OUTPUT)
    if "uint8" in output_dtypes:
        capabilities.add(AKACapability.UINT8_OUTPUT)
    if any(item.startswith("float8") for item in output_dtypes):
        capabilities.add(AKACapability.FP8_OUTPUT)
    return tuple(sorted(capabilities))


def _manifest_entries(
    specs: list[Spec],
    aka_artifacts: dict[str, list[dict[str, str]]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for spec in specs:
        entry: dict[str, object] = {
            "slot": spec.name,
            "task_path": spec.task_path,
            "problem_name": spec.name,
            "operation": str(spec.op_type),
            "input_dtypes": sorted(
                {str(item["dtype"]) for item in spec.inputs.values()},
            ),
            "output_dtypes": sorted(
                {str(item["dtype"]) for item in spec.outputs.values()},
            ),
            "capabilities": [
                str(capability) for capability in _spec_capabilities(spec)
            ],
            "pass_kind": str(spec.pass_kind),
            "fusion_depth": str(spec.fusion_depth),
            "source_family": str(spec.source_family),
            "suite": str(spec.suite),
            "role": str(spec.role),
            "workload_uuids": [
                f"aka-{spec.name}-w{i}" for i in range(len(spec.workloads))
            ],
            "aka_artifacts": aka_artifacts[spec.task_path],
            "golden": {},
        }
        if spec.exclusion_reason_code:
            entry["exclusion_reason_code"] = spec.exclusion_reason_code
        entries.append(entry)
    return entries


def _base_coverage_combinations() -> list[dict[str, object]]:
    combinations = [
        {
            "operation": str(AKAOperation.MATMUL),
            "input_dtype": str(DType.FLOAT32),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.MATMUL),
            "input_dtype": str(DType.BFLOAT16),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.SOFTMAX),
            "input_dtype": str(DType.FLOAT32),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.NORM),
            "input_dtype": str(DType.BFLOAT16),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.CONV),
            "input_dtype": str(DType.FLOAT32),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.ELEMENTWISE),
            "input_dtype": str(DType.FLOAT16),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.ATTENTION),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {
            "operation": str(AKAOperation.NORM),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 2,
        },
        {"pass": str(AKAPassKind.BACKWARD), "min_count": 1},
        {
            "output_dtype": str(DType.FLOAT8_E4M3FN),
            "pass": str(AKAPassKind.FORWARD),
            "min_count": 1,
        },
        {"fusion_depth": str(AKAFusionDepth.FUSED), "min_count": 1},
    ]
    return combinations


def _formal_coverage_combinations() -> list[dict[str, object]]:
    combinations = _base_coverage_combinations()
    combinations.extend(
        [
            {
                "capability": str(capability),
                "role": str(AKACorpusRole.SCORED),
                "min_problems": problems,
                "min_workloads": workloads,
            }
            for capability, problems, workloads in (
                (AKACapability.BOUNDED_INTEGER_INPUT, 1, 5),
                (AKACapability.POSITIVE_INPUT, 1, 5),
                (AKACapability.SIMPLEX_INPUT, 1, 4),
                (AKACapability.SCALAR_TENSOR_OUTPUT, 2, 9),
                (AKACapability.MULTI_OUTPUT, 4, 23),
                (AKACapability.MIXED_OUTPUT_DTYPE, 3, 17),
                (AKACapability.CODE_DISTANCE, 2, 13),
                (AKACapability.PARTIAL_CUSTOM_INPUT, 2, 8),
                (AKACapability.STRUCTURED_OFFSETS, 1, 4),
                (AKACapability.FP8_OUTPUT, 1, 8),
                (AKACapability.UINT8_OUTPUT, 1, 8),
                (AKACapability.RAW_CODE_DISTANCE, 1, 8),
                (AKACapability.COUPLED_TOPK, 1, 4),
            )
        ],
    )
    combinations.extend(
        [
            {
                "operation": str(operation),
                "role": str(AKACorpusRole.SCORED),
                "min_problems": problems,
                "min_workloads": workloads,
            }
            for operation, problems, workloads in (
                (AKAOperation.LOSS, 2, 9),
                (AKAOperation.QUANTIZATION, 2, 13),
                (AKAOperation.ROUTING, 1, 4),
                (AKAOperation.POSITION_ENCODING, 1, 4),
            )
        ],
    )
    return combinations


def _manifest_payload(
    specs: list[Spec],
    records: list[dict[str, str]],
    entries: list[dict[str, object]],
    aka_commit: str,
    *,
    problems_root: Path,
    calibration_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": AKA_MANIFEST_SCHEMA_VERSION,
        "source": {
            "repository": AKA_REPOSITORY,
            "revision": AKA_REVISION,
            "license": AKA_LICENSE,
            "provenance_class": AKA_PROVENANCE_CLASS,
            "aka_commit_sha256": aka_commit,
        },
        "execution_targets": {
            gfx_target: {
                "generation": str(spec["generation"]),
                "supported_tensor_dtypes": [
                    str(dtype) for dtype in spec["supported_tensor_dtypes"]
                ],
            }
            for gfx_target, spec in AKA_EXECUTION_TARGET_SPECS.items()
        },
        "formal_analysis": {
            "architecture_profile": FORMAL_ARCHITECTURE,
            "formal_gfx_target": FORMAL_GFX_TARGET,
            "architecture_profile_sha256": FORMAL_ARCHITECTURE_SHA256,
        },
        "tolerance_calibration": {
            "path": calibration_path.relative_to(problems_root).as_posix(),
            "sha256": sha256_file(calibration_path),
        },
        "official_scoring": {
            "status": str(AKAOfficialScoringStatus.UNAVAILABLE),
            "baseline_id": AKA_OFFICIAL_BASELINE_ID,
            "reason_code": "baseline_v2_release_evidence_pending",
        },
        "formal_coverage_requirements": {
            "axes": _coverage_axes(specs),
            "combinations": _formal_coverage_combinations(),
        },
        "materialized_problems": [
            {
                "path": record["path"],
                "task_path": spec.task_path,
                "definition_sha256": record["definition_sha256"],
                "workload_sha256": record["workload_sha256"],
            }
            for spec, record in zip(specs, records, strict=True)
        ],
        "entries": entries,
    }


def _write_manifest(
    specs: list[Spec],
    records: list[dict[str, str]],
    aka_artifacts: dict[str, list[dict[str, str]]],
    aka_commit: str,
    *,
    problems_root: Path = PROBLEMS_ROOT,
    manifest_path: Path = MANIFEST_PATH,
    calibration_path: Path = CALIBRATION_PATH,
) -> None:
    entries = _manifest_entries(specs, aka_artifacts)
    payload = _manifest_payload(
        specs,
        records,
        entries,
        aka_commit,
        problems_root=problems_root,
        calibration_path=calibration_path,
    )
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    """Materialize the authored AKA corpus and its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aka-root",
        type=Path,
        default=REPO_ROOT / "data" / "AgentKernelArena",
    )
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=PROBLEMS_ROOT,
        help="destination root for authored problems and manifest",
    )
    parser.add_argument(
        "--bootstrap-calibration",
        action="store_true",
        help="write provisional problems for the runtime probe, but not the manifest",
    )
    parser.add_argument(
        "--rebind-format-only-calibration",
        action="store_true",
        help=(
            "migrate calibration contract hashes after AST-equivalent Ruff "
            "formatting; preserves all measured values"
        ),
    )
    args = parser.parse_args()
    aka_root = args.aka_root.resolve()
    problems_root = args.problems_root.resolve()
    manifest_path = problems_root / "manifest.yaml"
    calibration_path = problems_root / AKA_TOLERANCE_CALIBRATION_FILENAME
    if not aka_root.is_dir():
        raise FileNotFoundError(
            "the pinned AKA clone is required to author provenance bindings: "
            f"{aka_root}",
        )

    if args.bootstrap_calibration:
        calibrated = None
    else:
        if not calibration_path.is_file():
            raise FileNotFoundError(
                "run scripts/internal/aka_calibrate_tolerances.py first",
            )
        calibrated = calibration_checks(calibration_path)

    records = []
    aka_artifacts: dict[str, list[dict[str, str]]] = {}
    for spec in SPECS:
        record = _write_problem(spec, calibrated, problems_root=problems_root)
        records.append(record)
        aka_artifacts[spec.task_path] = _aka_artifacts(aka_root, spec)
        print(f"authored {record['path']} ({spec.op_type}/{spec.dtype})")
    _format_authored_references(SPECS, records, problems_root=problems_root)
    if args.bootstrap_calibration:
        print("bootstrap complete; manifest intentionally left unchanged")
        return
    if args.rebind_format_only_calibration:
        _rebind_format_only_calibration(
            calibration_path,
            problems_root=problems_root,
        )
    head_file = aka_root / ".aka-head"
    aka_commit = head_file.read_text().strip() if head_file.is_file() else ""
    _write_manifest(
        SPECS,
        records,
        aka_artifacts,
        aka_commit,
        problems_root=problems_root,
        manifest_path=manifest_path,
        calibration_path=calibration_path,
    )
    print(f"wrote {manifest_path} ({len(SPECS)} problems)")


if __name__ == "__main__":
    main()
