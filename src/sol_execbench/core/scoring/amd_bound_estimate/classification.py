# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Operation-family classification helpers for AMD bound graph extraction."""

from __future__ import annotations

from dataclasses import dataclass

from sol_execbench.core.scoring.confidence import EstimateConfidence


@dataclass(frozen=True)
class CallClassification:
    """Classification result for a reference-code call target."""

    op_family: str
    confidence: EstimateConfidence
    rationale: str


_CALL_CLASSIFIERS: tuple[tuple[set[str], CallClassification], ...] = (
    (
        {"matmul", "mm", "bmm"},
        CallClassification(
            "gemm", EstimateConfidence.SUPPORTED, "recognized matrix multiply"
        ),
    ),
    (
        {"linear", "in_proj", "out_proj"},
        CallClassification(
            "linear_projection",
            EstimateConfidence.SUPPORTED,
            "recognized linear projection",
        ),
    ),
    (
        {"conv1d", "conv2d", "conv3d", "depthwise_conv"},
        CallClassification(
            "convolution",
            EstimateConfidence.SUPPORTED,
            "recognized convolution operation",
        ),
    ),
    (
        {"embedding", "gather", "index_select", "take"},
        CallClassification(
            "embedding_positional",
            EstimateConfidence.SUPPORTED,
            "recognized embedding or gather memory-bound operation",
        ),
    ),
    (
        {"router", "topk", "top_k", "dispatch_and_combine", "dispatch_dynamic"},
        CallClassification(
            "moe",
            EstimateConfidence.INEXACT,
            "recognized visible MoE routing primitive",
        ),
    ),
    (
        {"sum", "mean", "amax", "max", "amin", "min", "var", "std"},
        CallClassification(
            "reduction",
            EstimateConfidence.INEXACT,
            "recognized reduction with conservative later-modeling semantics",
        ),
    ),
    (
        {"layer_norm", "group_norm", "rms_norm", "norm", "rsqrt"},
        CallClassification(
            "normalization",
            EstimateConfidence.INEXACT,
            "recognized normalization-like operation",
        ),
    ),
    (
        {"softmax", "log_softmax"},
        CallClassification(
            "softmax",
            EstimateConfidence.INEXACT,
            "recognized softmax-like operation",
        ),
    ),
    (
        {"relu", "gelu", "silu", "sigmoid", "tanh", "exp", "sqrt", "gate"},
        CallClassification(
            "mlp_activation",
            EstimateConfidence.INEXACT,
            "recognized activation operation",
        ),
    ),
    (
        {
            "abs",
            "clamp",
            "clamp_min",
            "clamp_max",
            "clip",
            "maximum",
            "minimum",
            "where",
            "one_hot",
            "logical_and",
            "logical_or",
            "logical_not",
        },
        CallClassification(
            "elementwise",
            EstimateConfidence.INEXACT,
            "recognized pointwise operation with shape-dependent work",
        ),
    ),
    (
        {"selective_scan", "mamba_scan", "ssm_scan"},
        CallClassification(
            "ssm_mamba",
            EstimateConfidence.INEXACT,
            "recognized SSM/Mamba selective scan primitive",
        ),
    ),
    (
        {
            "t",
            "transpose",
            "permute",
            "view",
            "reshape",
            "flatten",
            "contiguous",
            "unsqueeze",
            "squeeze",
            "expand",
            "broadcast_to",
        },
        CallClassification(
            "data_movement",
            EstimateConfidence.INEXACT,
            "recognized logical view or broadcast operation",
        ),
    ),
    (
        {
            "zeros",
            "zeros_like",
            "cat",
            "concat",
            "stack",
            "copy_",
        },
        CallClassification(
            "data_movement",
            EstimateConfidence.SUPPORTED,
            "recognized materialized data-movement operation",
        ),
    ),
    (
        {"to", "type", "float", "half", "bfloat16", "double", "bool", "int", "long"},
        CallClassification(
            "dtype_conversion",
            EstimateConfidence.INEXACT,
            "recognized dtype conversion operation",
        ),
    ),
)

_LOGICAL_VIEW_NAMES = {
    "t",
    "transpose",
    "permute",
    "view",
    "reshape",
    "flatten",
    "unsqueeze",
    "squeeze",
}
_BROADCAST_VIEW_NAMES = {"expand", "broadcast_to"}
_MATERIALIZED_MOVEMENT_NAMES = {"contiguous"}
_MATERIALIZED_MOVEMENT_NAMES.update(
    {"zeros", "zeros_like", "cat", "concat", "stack", "copy_", "getitem"}
)
_DTYPE_METHOD_TARGETS = {
    "float": "float32",
    "half": "float16",
    "bfloat16": "bfloat16",
    "double": "float64",
    "bool": "bool",
    "int": "int32",
    "long": "int64",
}


def classify_call(func_name: str) -> CallClassification | None:
    """Classify a fully qualified or leaf call name into an operation family."""
    leaf_name = func_name.rsplit(".", maxsplit=1)[-1]
    for names, classification in _CALL_CLASSIFIERS:
        if leaf_name in names or func_name in names:
            return classification
    return None


def movement_kind_for_name(leaf_name: str) -> str | None:
    """Return movement-kind evidence for view/copy-like call names."""
    if leaf_name in _BROADCAST_VIEW_NAMES:
        return "broadcast_view"
    if leaf_name in _MATERIALIZED_MOVEMENT_NAMES:
        return "materialized"
    if leaf_name in _LOGICAL_VIEW_NAMES:
        return "logical_view"
    return None


def dtype_method_target(leaf_name: str) -> str | None:
    """Return the dtype implied by PyTorch dtype conversion method names."""
    return _DTYPE_METHOD_TARGETS.get(leaf_name)
