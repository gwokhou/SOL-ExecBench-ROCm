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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.aka_corpus import (
    AKA_LICENSE,
    AKA_PROVENANCE_CLASS,
    AKA_REPOSITORY,
    AKA_REVISION,
    FORMAL_ARCHITECTURE,
    FORMAL_ARCHITECTURE_SHA256,
    FORMAL_GFX_TARGET,
)
from sol_execbench.core.dataset.aka_compatibility import AKA_EXECUTION_TARGET_SPECS
from sol_execbench.core.dataset.aka_tolerance import dtype_default_tolerance
from sol_execbench.core.integrity import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "problems" / "AMD_AKA" / "manifest.yaml"
PROBLEMS_ROOT = REPO_ROOT / "problems" / "AMD_AKA"


@dataclass(frozen=True)
class Spec:
    name: str
    suite: str
    task_path: str
    op_type: str
    dtype: str
    pass_kind: str
    fusion_depth: str
    source_family: str
    axes: dict[str, dict[str, Any]]
    inputs: dict[str, dict[str, Any]]
    outputs: dict[str, dict[str, Any]]
    reference: str
    workloads: list[dict[str, Any]]
    role: str = "scored"
    description: str = ""


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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/3267_SimpleMatmulModule",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n1_Square_matrix_multiplication_",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            "output": {"shape": ["M", "N"], "dtype": "float32", "description": "A @ B."}
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n2_Standard_matrix_multiplication_",
        op_type="matmul",
        dtype="bfloat16",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n3_Batched_matrix_multiplication",
        op_type="matmul",
        dtype="bfloat16",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            }
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.bmm(A, B)\n",
        workloads=[
            _wl(
                {"Batch": 4, "M": 64, "K": 64, "N": 64}, {"A": "random", "B": "random"}
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n4_Matrix_vector_multiplication_",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n8_Matmul_with_irregular_shapes_",
        op_type="matmul",
        dtype="float16",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            "output": {"shape": ["M", "N"], "dtype": "float16", "description": "A @ B."}
        },
        reference="import torch\n\ndef run(A, B):\n    return torch.matmul(A, B)\n",
        workloads=[
            _wl({"M": 1823, "K": 781, "N": 511}, {"A": "random", "B": "random"}),
            _wl({"M": 359, "K": 127, "N": 211}, {"A": "random", "B": "random"}),
            _wl({"M": 1024, "K": 333, "N": 717}, {"A": "random", "B": "random"}),
        ],
    ),
    Spec(
        name="l1n9_tall_skinny_matmul",
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            "output": {"shape": ["M", "N"], "dtype": "float32", "description": "A @ B."}
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n23_Softmax",
        op_type="softmax",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
        description="Row-wise softmax over the last dimension. Derived from AKA "
        "torch2hip/kernelbench/level1/l1n23_Softmax module_fn (dim=1).",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns (softmax dimension).")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            }
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Row-wise softmax.",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n26_GELU_",
        op_type="elementwise",
        dtype="float16",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
        description="GELU activation. Derived from AKA torch2hip/kernelbench/level1/l1n26_GELU_ "
        "module_fn (F.gelu).",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "Input (M, N).",
            }
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float16",
                "description": "F.gelu(x).",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n36_RMSNorm_",
        op_type="norm",
        dtype="bfloat16",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n40_LayerNorm",
        op_type="norm",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n47_Sum_reduction_over_a_dimension",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
        description="Sum reduction over the last dimension with keepdim. Derived from "
        "AKA torch2hip/kernelbench/level1/l1n47_Sum_reduction_over_a_dimension.",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns (reduced).")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            }
        },
        outputs={
            "output": {
                "shape": ["M", "1"],
                "dtype": "float32",
                "description": "Row sums (M, 1).",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n42_Max_Pooling_2D",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            }
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H_out", "W_out"],
                "dtype": "float32",
                "description": "Pooled output.",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n63_conv_standard_2D__square_input__square_kernel",
        op_type="conv",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            "bias": {"shape": ["O"], "dtype": "float32", "description": "Bias (O,)."},
        },
        outputs={
            "output": {
                "shape": ["B", "O", "H_out", "W_out"],
                "dtype": "float32",
                "description": "Convolution output.",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level1/l1n82_conv_depthwise_2D_square_input_square_kernel",
        op_type="conv",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="kernelbench",
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
            "bias": {"shape": ["C"], "dtype": "float32", "description": "Bias (C,)."},
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H_out", "W_out"],
                "dtype": "float32",
                "description": "Depthwise conv output.",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/16636_SiLU",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="gpumode",
        description="SiLU activation x * sigmoid(x). Derived from AKA "
        "torch2hip/gpumode/16636_SiLU module_fn (silu_fn).",
        axes={"M": _ax_var("Rows."), "N": _ax_var("Columns.")},
        inputs={
            "x": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "Input (M, N).",
            }
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float32",
                "description": "x * sigmoid(x).",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/11184_Sigmoid",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/11178_TanH",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
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
            }
        },
        reference=(
            "import torch.nn.functional as F\n\n"
            "def run(x, weight, bias):\n"
            "    x = F.linear(x, weight, bias)\n"
            "    x = F.gelu(x)\n"
            "    return F.softmax(x, dim=1)\n"
        ),
        workloads=[
            _wl({"B": 256}, {"x": "random", "weight": "random", "bias": "random"}),
            _wl({"B": 1024}, {"x": "random", "weight": "random", "bias": "random"}),
            _wl({"B": 4096}, {"x": "random", "weight": "random", "bias": "random"}),
        ],
    ),
    Spec(
        name="l2n86_matmul_divide_gelu",
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm",
        op_type="norm",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide",
        op_type="norm",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
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
            }
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
    # --- Cat1: attention (scaled-dot-product core) ---
    Spec(
        name="sdp_attention",
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/10456_MultiHeadAttention",
        op_type="attention",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
        description="Scaled-dot-product attention core (single-head): "
        "softmax(Q @ K^T / sqrt(d)) @ V. Representative of AKA's attention tasks; "
        "provenance-bound to torch2hip/gpumode/10456_MultiHeadAttention (whose "
        "module_fn embeds this Q*K^T -> softmax -> *V core).",
        axes={
            "B": _ax_var("Batch."),
            "S": _ax_var("Sequence length."),
            "D": _ax_var("Head dimension."),
        },
        inputs={
            "q": {
                "shape": ["B", "S", "D"],
                "dtype": "float32",
                "description": "Queries (B, S, D).",
            },
            "k": {
                "shape": ["B", "S", "D"],
                "dtype": "float32",
                "description": "Keys (B, S, D).",
            },
            "v": {
                "shape": ["B", "S", "D"],
                "dtype": "float32",
                "description": "Values (B, S, D).",
            },
        },
        outputs={
            "output": {
                "shape": ["B", "S", "D"],
                "dtype": "float32",
                "description": "Attention output (B, S, D).",
            }
        },
        reference=(
            "import math\nimport torch\nimport torch.nn.functional as F\n\n"
            "def run(q, k, v):\n"
            "    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])\n"
            "    return torch.matmul(F.softmax(scores, dim=-1), v)\n"
        ),
        workloads=[
            _wl(
                {"B": 2, "S": 256, "D": 64},
                {"q": "random", "k": "random", "v": "random"},
            ),
            _wl(
                {"B": 4, "S": 512, "D": 64},
                {"q": "random", "k": "random", "v": "random"},
            ),
            _wl(
                {"B": 2, "S": 1024, "D": 128},
                {"q": "random", "k": "random", "v": "random"},
            ),
            _wl(
                {"B": 8, "S": 256, "D": 64},
                {"q": "random", "k": "random", "v": "random"},
            ),
        ],
    ),
    # --- Cat2 (mechanical): rank-split of a variable-rank task (C9/C10 -> Cat1) ---
    Spec(
        name="gpumode_gelu_4d",
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/14539_GELU",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="gpumode",
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
            }
        },
        outputs={
            "output": {
                "shape": ["B", "C", "H", "W"],
                "dtype": "float32",
                "description": "F.gelu(x).",
            }
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
        suite="instruction2triton",
        task_path="tasks/instruction2triton/rocmbench/test_chained_dot_fp8",
        op_type="elementwise",
        dtype="float8_e4m3fn",
        pass_kind="forward",
        fusion_depth="single",
        source_family="rocmbench",
        role="compatibility_sentinel",
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
            }
        },
        outputs={
            "output": {
                "shape": ["M", "N"],
                "dtype": "float8_e4m3fn",
                "description": "x cast to float8_e4m3fn.",
            }
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
        suite="instruction2triton",
        task_path="tasks/instruction2triton/rocmbench/rmsnorm_bwd",
        op_type="norm",
        dtype="float16",
        pass_kind="backward",
        fusion_depth="single",
        source_family="rocmbench",
        description="RMSNorm backward (gradient w.r.t. input and weight) via autograd. "
        "Derived from AKA instruction2triton/rocmbench/rmsnorm_bwd; provenance-weakened "
        "(no module_fn cross-check; correctness rests on the lifted reference).",
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
        suite="torch2flydsl",
        task_path="tasks/torch2flydsl/silu_and_mul_kernel",
        op_type="elementwise",
        dtype="bfloat16",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="flydsl",
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
            }
        },
        outputs={
            "output": {
                "shape": ["M", "d"],
                "dtype": "bfloat16",
                "description": "silu(x) * y (M, d).",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/10190_FusedLeakyReLU",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/1067_Transpose",
        op_type="elementwise",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="gpumode",
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
            }
        },
        reference=(
            "def run(input, dim1, dim2):\n"
            "    return input.transpose(int(dim1), int(dim2)).contiguous()\n"
        ),
        workloads=[
            _wl(
                {"A": 4, "B": 8, "C": 16, "D": 32},
                {"input": "random", "dim1": {"scalar": 1}, "dim2": {"scalar": 2}},
            ),
            _wl(
                {"A": 8, "B": 16, "C": 32, "D": 64},
                {"input": "random", "dim1": {"scalar": 1}, "dim2": {"scalar": 2}},
            ),
            _wl(
                {"A": 2, "B": 64, "C": 128, "D": 16},
                {"input": "random", "dim1": {"scalar": 1}, "dim2": {"scalar": 2}},
            ),
            _wl(
                {"A": 16, "B": 32, "C": 48, "D": 24},
                {"input": "random", "dim1": {"scalar": 1}, "dim2": {"scalar": 2}},
            ),
        ],
    ),
    Spec(
        name="gpumode_softmax_3d",
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/10082_SoftmaxModule",
        op_type="softmax",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="single",
        source_family="gpumode",
        description="Softmax over the last axis of a 3D cube. Derived from AKA "
        "torch2hip/gpumode/10082_SoftmaxModule module_fn (axis=2).",
        axes={"N": _ax_var("Cube edge (all three dims equal).")},
        inputs={
            "v": {
                "shape": ["N", "N", "N"],
                "dtype": "float32",
                "description": "Input cube (N, N, N).",
            },
            "axis": {"shape": None, "dtype": "float32", "description": "Softmax axis."},
        },
        outputs={
            "output": {
                "shape": ["N", "N", "N"],
                "dtype": "float32",
                "description": "softmax(v, dim=axis).",
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/10024_Feedforward",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/14044_PositionWiseFeedForward",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/gpumode/1178_MLP_model",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="gpumode",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
        description="Fused linear -> max_pool1d -> sum -> scale, reducing to a 1D "
        "per-batch output. Derived from AKA torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale.",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max",
        op_type="matmul",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
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
            }
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
        suite="torch2hip",
        task_path="tasks/torch2hip/kernelbench/level3/l3n44_MiniGPTBlock",
        op_type="attention",
        dtype="float32",
        pass_kind="forward",
        fusion_depth="fused",
        source_family="kernelbench",
        description="Full MiniGPT transformer block: layernorm -> causal self-attention "
        "-> residual -> layernorm -> MLP(new-GELU) -> residual. Derived from AKA "
        "torch2hip/kernelbench/level3/l3n44_MiniGPTBlock module_fn.",
        axes={
            "B": _ax_var("Batch."),
            "S": _ax_var("Sequence length (<= 1024)."),
            "C": _ax_const(768, "Model dimension."),
            "C3": _ax_expr("3 * C", "QKV projection output (3*C)."),
            "C4": _ax_expr("4 * C", "MLP hidden (4*C)."),
            "MT": _ax_const(1024, "Max sequence length (causal-mask buffer edge)."),
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
            }
        },
        reference=(
            "import math\nimport torch\nimport torch.nn.functional as F\n\n"
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


def _aka_checksums(aka_root: Path | None, task_path: str) -> dict[str, str]:
    if aka_root is None or not aka_root.is_dir():
        return {
            "aka_config_sha256": "",
            "aka_source_sha256": "",
            "aka_runner_sha256": "",
        }
    root = aka_root / task_path
    config = root / "config.yaml"
    func_dir = root / "pytorch_code_functional"
    func_files = sorted(func_dir.glob("*.py")) if func_dir.is_dir() else []
    runner = root / "eval_tools" / "correctness_check.py"
    return {
        "aka_config_sha256": sha256_file(config) if config.is_file() else "",
        "aka_source_sha256": sha256_file(func_files[0]) if func_files else "",
        "aka_runner_sha256": sha256_file(runner) if runner.is_file() else "",
    }


def _write_problem(spec: Spec) -> dict[str, str]:
    problem_dir = PROBLEMS_ROOT / spec.suite / spec.name
    problem_dir.mkdir(parents=True, exist_ok=True)
    tolerance = dtype_default_tolerance(spec.dtype)
    tolerance_payload = {
        "max_atol": tolerance.max_atol,
        "max_rtol": tolerance.max_rtol,
        "required_matched_ratio": tolerance.required_matched_ratio,
    }
    workload_records = []
    for idx, wl in enumerate(spec.workloads):
        inputs_payload: dict[str, Any] = {}
        for name, meta in wl["inputs"].items():
            if isinstance(meta, dict) and "scalar" in meta:
                inputs_payload[name] = {"type": "scalar", "value": meta["scalar"]}
            else:
                inputs_payload[name] = {"type": "random"}
        record = {
            "axes": wl["axes"],
            "inputs": inputs_payload,
            "tolerance": tolerance_payload,
            "uuid": f"aka-{spec.name}-w{idx}",
        }
        Workload.model_validate(record)
        workload_records.append(record)

    definition_payload = {
        "name": spec.name,
        "op_type": spec.op_type,
        "description": spec.description,
        "axes": spec.axes,
        "inputs": spec.inputs,
        "outputs": spec.outputs,
        "reference": spec.reference,
    }
    Definition.model_validate(definition_payload)

    definition_path = problem_dir / "definition.json"
    workload_path = problem_dir / "workload.jsonl"
    reference_path = problem_dir / "reference.py"
    definition_path.write_text(json.dumps(definition_payload, indent=2) + "\n")
    workload_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in workload_records)
    )
    reference_path.write_text(
        f'"""Standalone PyTorch reference for {spec.name} (debug mirror)."""\n'
        + spec.reference
    )
    return {
        "path": f"{spec.suite}/{spec.name}",
        "definition_sha256": sha256_file(definition_path),
        "workload_sha256": sha256_file(workload_path),
    }


def _coverage_axes(specs: list[Spec]) -> dict[str, dict[str, int]]:
    def _count(field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in specs:
            out[getattr(s, field)] = out.get(getattr(s, field), 0) + 1
        return dict(sorted(out.items()))

    return {
        "operation": _count("op_type"),
        "dtype": _count("dtype"),
        "pass_kind": _count("pass_kind"),
        "fusion_depth": _count("fusion_depth"),
        "source_family": _count("source_family"),
        "suite": _count("suite"),
    }


def _write_manifest(
    specs: list[Spec],
    records: list[dict[str, str]],
    aka_checksums: dict[str, dict[str, str]],
    aka_commit: str,
) -> None:
    entries = []
    for spec, record in zip(specs, records, strict=True):
        entry = {
            "slot": spec.name,
            "task_path": spec.task_path,
            "problem_name": spec.name,
            "operation": spec.op_type,
            "dtype": spec.dtype,
            "pass_kind": spec.pass_kind,
            "fusion_depth": spec.fusion_depth,
            "source_family": spec.source_family,
            "suite": spec.suite,
            "role": spec.role,
            "workload_uuids": [
                f"aka-{spec.name}-w{i}" for i in range(len(spec.workloads))
            ],
            **aka_checksums[spec.task_path],
            "golden": {},
        }
        entries.append(entry)

    payload = {
        "schema_version": 4,
        "source": {
            "repository": AKA_REPOSITORY,
            "revision": AKA_REVISION,
            "license": AKA_LICENSE,
            "provenance_class": AKA_PROVENANCE_CLASS,
            "aka_commit_sha256": aka_commit,
        },
        "execution_targets": {
            gfx_target: {
                "generation": spec["generation"],
                "supported_tensor_dtypes": list(spec["supported_tensor_dtypes"]),
            }
            for gfx_target, spec in AKA_EXECUTION_TARGET_SPECS.items()
        },
        "formal_analysis": {
            "architecture_profile": FORMAL_ARCHITECTURE,
            "formal_gfx_target": FORMAL_GFX_TARGET,
            "architecture_profile_sha256": FORMAL_ARCHITECTURE_SHA256,
        },
        "official_scoring": {
            "status": "unavailable",
            "reason_code": "release_authority_not_published",
            "required_evidence": [
                "content_addressed_release_baseline",
                "independent_rerun_verification",
                "trusted_candidate_execution_attestation",
                "pinned_solar_manifests",
            ],
        },
        "formal_coverage_requirements": {
            "axes": _coverage_axes(specs),
            "combinations": [
                {
                    "operation": "matmul",
                    "dtype": "float32",
                    "pass": "forward",
                    "min_count": 1,
                },
                {
                    "operation": "matmul",
                    "dtype": "bfloat16",
                    "pass": "forward",
                    "min_count": 1,
                },
                {
                    "operation": "softmax",
                    "dtype": "float32",
                    "pass": "forward",
                    "min_count": 1,
                },
                {
                    "operation": "norm",
                    "dtype": "bfloat16",
                    "pass": "forward",
                    "min_count": 1,
                },
                {
                    "operation": "conv",
                    "dtype": "float32",
                    "pass": "forward",
                    "min_count": 1,
                },
                {
                    "operation": "elementwise",
                    "dtype": "float16",
                    "pass": "forward",
                    "min_count": 1,
                },
                # --- Expansion floor constraints (friendliness categories) ---
                # Cat1 coverage breadth: the attention op family is present.
                {
                    "operation": "attention",
                    "pass": "forward",
                    "min_count": 1,
                },
                # At least two norm problems so a norm variant (group/batch/instance)
                # is represented alongside the baseline layernorm/rmsnorm.
                {
                    "operation": "norm",
                    "pass": "forward",
                    "min_count": 2,
                },
                # Cat2 backward pass (instruction2triton rmsnorm_bwd).
                {"pass": "backward", "min_count": 1},
                # Cat2 FP8 compatibility sentinel (float8_e4m3fn).
                {
                    "dtype": "float8_e4m3fn",
                    "pass": "forward",
                    "min_count": 1,
                },
                # Fused-op depth is represented.
                {"fusion_depth": "fused", "min_count": 1},
            ],
        },
        "materialized_problems": [
            {
                "path": r["path"],
                "task_path": s.task_path,
                "definition_sha256": r["definition_sha256"],
                "workload_sha256": r["workload_sha256"],
            }
            for s, r in zip(specs, records, strict=True)
        ],
        "entries": entries,
    }
    MANIFEST_PATH.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aka-root", type=Path, default=REPO_ROOT / "data" / "AgentKernelArena"
    )
    args = parser.parse_args()
    aka_root = args.aka_root if args.aka_root.is_dir() else None

    records = []
    aka_checksums: dict[str, dict[str, str]] = {}
    for spec in SPECS:
        record = _write_problem(spec)
        records.append(record)
        aka_checksums[spec.task_path] = _aka_checksums(aka_root, spec.task_path)
        print(f"authored {record['path']} ({spec.op_type}/{spec.dtype})")
    aka_commit = ""
    if aka_root is not None:
        head_file = aka_root / ".aka-head"
        if head_file.is_file():
            aka_commit = head_file.read_text().strip()
    _write_manifest(SPECS, records, aka_checksums, aka_commit)
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(SPECS)} problems)")


if __name__ == "__main__":
    main()
