# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Publication-device policy for the SOLAR bridge."""

from __future__ import annotations

import gc
from functools import partial
from pathlib import Path

from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.bench.gpu_lock import (
    gpu_lock_path,
)
from sol_execbench.core.bench.gpu_lock_paths import configured_gpu_locks
from sol_execbench.core.process.termination import defer_termination
from solar.contracts import AnalysisExecutionPolicy

# Canonical architecture id passed to solar.api (no namespace prefix).
# aka_corpus.FORMAL_ARCHITECTURE carries the manifest label "solar:RX_9060_XT";
# the two are deliberately different views of the same target.
FORMAL_ARCHITECTURE, FORMAL_GFX_TARGET = "RX_9060_XT", "gfx1200"


def require_formal_device(device: str) -> None:
    """Require the release's publication-grade ROCm target."""
    import torch

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("formal SOLAR analysis requires a ROCm device")
    selected = torch.device(device)
    index = (
        selected.index
        if selected.index is not None
        else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(index)
    gfx_target = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if gfx_target != FORMAL_GFX_TARGET:
        raise RuntimeError(
            f"formal SOLAR analysis requires {FORMAL_GFX_TARGET}, got "
            f"{gfx_target or 'unknown'}; other AMD devices remain diagnostic "
            "evaluation targets",
        )


def release_formal_device_memory(device: str) -> None:
    """Release cached ROCm allocations before CPU-only formal analysis."""
    import torch

    selected = torch.device(device)
    with defer_termination(), torch.cuda.device(selected):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def device_execution_policy(
    device: str,
    lock_path: str | Path | None,
    timeout: float,
) -> AnalysisExecutionPolicy:
    """Share the native GPU lock; an explicit extra lock cannot replace it."""
    extra = Path(lock_path) if lock_path is not None else None
    gpu = device.split(":", 1)[0] in {"cuda", "hip"}
    if (
        gpu
        and extra is not None
        and extra.resolve()
        in {
            gpu_lock_path().resolve(),
            configured_gpu_locks().execution.resolve(),
        }
    ):
        extra = None
    return AnalysisExecutionPolicy(
        device_stage_lock_path=extra,
        device_stage_lock_timeout_seconds=timeout,
        device_stage_context=partial(
            native_gpu_execution,
            device,
            timeout_seconds=timeout,
        )
        if gpu
        else None,
        device_stage_cleanup=partial(release_formal_device_memory, device)
        if gpu
        else None,
    )


__all__ = [
    "FORMAL_ARCHITECTURE",
    "FORMAL_GFX_TARGET",
    "release_formal_device_memory",
    "require_formal_device",
]
