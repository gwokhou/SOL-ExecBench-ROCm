#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Author the AKA-derived seed problem set and its manifest.

This is the offline authoring tool for the problem set derived from AMD
AgentKernelArena (AKA). Each problem's PyTorch reference is AKA's own
correctness oracle (``module_fn``) lifted into a standalone ``def run(...)``;
axes, workloads, and dtypes are chosen per problem under the SOL-ExecBench
paper (arXiv 2603.19173) §3 methodology. Running this script regenerates the
committed problems under ``problems/RX_9060_XT/`` and the manifest, recording
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
from sol_execbench.core.dataset.aka_tolerance import dtype_default_tolerance
from sol_execbench.core.integrity import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "problems" / "RX_9060_XT" / "manifest.yaml"
PROBLEMS_ROOT = REPO_ROOT / "problems" / "RX_9060_XT"


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
        "schema_version": 3,
        "source": {
            "repository": AKA_REPOSITORY,
            "revision": AKA_REVISION,
            "license": AKA_LICENSE,
            "provenance_class": AKA_PROVENANCE_CLASS,
            "aka_commit_sha256": aka_commit,
        },
        "target": {
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
