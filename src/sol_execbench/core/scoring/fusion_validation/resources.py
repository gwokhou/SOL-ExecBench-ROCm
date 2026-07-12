"""Kernel resource evidence collection."""

from __future__ import annotations

import hashlib

from sol_execbench.core.bench.static_kernel.amdgpu_metadata import (
    extract_amdgpu_kernel_metadata,
)

from .models import KernelResourceEvidence


def kernel_resource_from_code_object(
    *,
    code_object: bytes,
    source: bytes,
    kernel_name: str,
    compile_command: tuple[str, ...],
    architecture: str,
    dynamic_lds_bytes: int,
    lds_limit_bytes: int,
    active_blocks_per_multiprocessor: int,
    launch_passed: bool,
    correctness_passed: bool,
) -> KernelResourceEvidence:
    """Build authority evidence after an exact, unique metadata match."""
    matches = [
        item
        for item in extract_amdgpu_kernel_metadata(
            code_object, target_architecture=architecture
        )
        if item.name == kernel_name or item.symbol == kernel_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"kernel {kernel_name!r} must uniquely match gfx code-object metadata"
        )
    metadata = matches[0]
    if (
        metadata.vgpr_count is None
        or metadata.sgpr_count is None
        or metadata.private_segment_bytes is None
        or metadata.group_segment_bytes is None
    ):
        raise ValueError(f"kernel {kernel_name!r} has incomplete resource metadata")
    return KernelResourceEvidence(
        metadata.name,
        hashlib.sha256(code_object).hexdigest(),
        hashlib.sha256(source).hexdigest(),
        compile_command,
        metadata.architecture,
        metadata.vgpr_count,
        metadata.sgpr_count,
        metadata.vgpr_spill_count,
        metadata.sgpr_spill_count,
        metadata.private_segment_bytes,
        metadata.group_segment_bytes,
        dynamic_lds_bytes,
        lds_limit_bytes,
        active_blocks_per_multiprocessor,
        launch_passed,
        correctness_passed,
    )


__all__ = ["kernel_resource_from_code_object"]
