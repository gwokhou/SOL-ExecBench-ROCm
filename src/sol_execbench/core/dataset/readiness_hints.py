# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Static hint extraction for ROCm readiness classification."""

from __future__ import annotations

from pathlib import Path

from .inventory import ProblemInventoryRecord, WorkloadInventoryRecord

LOW_PRECISION_DTYPES = {
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "float4_e2m1fn_x2",
}
BLACKWELL_LOW_PRECISION_DTYPES = {
    "float4_e2m1",
    "float4_e2m1fn_x2",
    "nvfp4",
    "mxfp4",
}

def _reference_has_nvidia_blocker(problem: ProblemInventoryRecord) -> bool:
    return bool(problem.definition and problem.definition.reference_runtime_hints)


def _low_precision_or_quant(
    problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord
) -> bool:
    """Check if a workload uses low-precision dtypes (excludes Quant category check)."""
    dtypes = set(workload.input_dtypes.values()) | set(workload.output_dtypes.values())
    return bool(dtypes & LOW_PRECISION_DTYPES)


def _blackwell_low_precision(
    problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord
) -> bool:
    """Check if a workload uses Blackwell-specific formats or identity tokens."""
    dtypes = set(workload.input_dtypes.values()) | set(workload.output_dtypes.values())
    identity = f"{problem.problem_id} {problem.problem_path}".lower()
    return (
        bool(dtypes & BLACKWELL_LOW_PRECISION_DTYPES)
        or "blackwell" in identity
        or "nvfp4" in identity
        or "mxfp4" in identity
    )


def _solution_runtime_hints(
    problem: ProblemInventoryRecord, dataset_root: Path
) -> set[str]:
    hints: set[str] = set()
    problem_dir = Path(dataset_root) / problem.problem_path
    for filename in problem.solution_files:
        lowered = filename.lower()
        if lowered.endswith((".cu", ".cuh", ".so")) or "cuda" in lowered:
            hints.add("cuda_kernel")
        if "cute" in lowered or "cutile" in lowered or "cutlass" in lowered:
            hints.add("nvidia_dsl")
        path = problem_dir / filename
        if not path.is_file() or path.suffix.lower() not in {
            ".json",
            ".py",
            ".cu",
            ".cuh",
            ".cpp",
            ".hip",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(token in text for token in ("cuda", "cublas", "cutlass", "nvrtc")):
            hints.add("cuda_kernel")
        if any(
            token in text for token in ("flashinfer", "paged_decode", "paged attention")
        ):
            hints.add("flashinfer_runtime")
        if any(token in text for token in ("cute_dsl", "cutile", "cutlass")):
            hints.add("nvidia_dsl")
        if any(
            token in text for token in ("nvfp4", "mxfp4", "float4_e2m1", "blackwell")
        ):
            hints.add("blackwell_low_precision")
    return hints


FLASHINFER_SIMPLE_REFERENCE_TOKENS = (
    "fused_add_rmsnorm",
    "gemm",
    "rmsnorm",
)
FLASHINFER_RUNTIME_REFERENCE_HINTS = (
    "import flashinfer",
    "from flashinfer",
    "flashinfer.",
)
FLASHINFER_RUNTIME_BUCKETS = {
    # More specific buckets must come before more general ones to avoid
    # premature substring matches (e.g. "mla_paged_decode" contains both
    # "mla_paged" and "paged_decode" — mla_paged must win).
    "mla_paged": {
        "mla_paged",
    },
    "moe_fp8_block_scale": {
        "moe_fp8_block_scale",
    },
    "ragged_prefill": {
        "ragged_prefill",
    },
    "paged_prefill": {
        "paged_prefill",
    },
    "paged_decode": {
        "paged_decode",
        "gqa_paged_decode",
    },
}
FLASHINFER_RUNTIME_BUCKET_TO_REASON = {
    "paged_decode": "flashinfer_runtime_paged_decode",
    "paged_prefill": "flashinfer_runtime_paged_prefill",
    "ragged_prefill": "flashinfer_runtime_ragged_prefill",
    "mla_paged": "flashinfer_runtime_mla_paged",
    "moe_fp8_block_scale": "flashinfer_runtime_moe_fp8_block_scale",
    "unknown_flashinfer_runtime": "flashinfer_runtime_unknown",
}


def _read_reference_text(problem: ProblemInventoryRecord, dataset_root: Path) -> str:
    problem_dir = Path(dataset_root) / problem.problem_path
    reference_path = problem_dir / "reference.py"
    if not reference_path.is_file():
        return ""
    try:
        return reference_path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return ""


def _flashinfer_reference_is_runtime_dependent(
    problem: ProblemInventoryRecord, dataset_root: Path
) -> bool:
    reference_text = _read_reference_text(problem, dataset_root)
    return any(token in reference_text for token in FLASHINFER_RUNTIME_REFERENCE_HINTS)


def _flashinfer_semantic_bucket(problem: ProblemInventoryRecord) -> str:
    lowered = problem.problem_path.lower()
    if any(token in lowered for token in FLASHINFER_SIMPLE_REFERENCE_TOKENS):
        return "pytorch_compatible"
    for bucket, tokens in FLASHINFER_RUNTIME_BUCKETS.items():
        if any(token in lowered for token in tokens):
            return bucket
    return "unknown_flashinfer_runtime"


def _unsupported_dtype_failure(
    problem: ProblemInventoryRecord, workload: WorkloadInventoryRecord
) -> str | None:
    failure = " ".join(
        item
        for item in (problem.schema_failure, workload.schema_failure)
        if item is not None
    ).lower()
    if "unsupported dtype" in failure or (
        "dtype" in failure and "unsupported" in failure
    ):
        return problem.schema_failure or workload.schema_failure or "unsupported dtype"
    return None
