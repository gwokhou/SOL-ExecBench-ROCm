# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment construction for untrusted compile and evaluation processes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

ENV_HIP_VISIBLE_DEVICES = "HIP_VISIBLE_DEVICES"
ENV_ROCR_VISIBLE_DEVICES = "ROCR_VISIBLE_DEVICES"
ENV_SOL_EXECBENCH_CLOCKS_LOCKED = "SOL_EXECBENCH_CLOCKS_LOCKED"
ENV_SOL_EXECBENCH_DEVICE = "SOL_EXECBENCH_DEVICE"
ENV_SOL_EXECBENCH_GRACEFUL_EXIT = "SOL_EXECBENCH_GRACEFUL_EXIT"
ENV_SOL_EXECBENCH_SANDBOXED = "SOL_EXECBENCH_SANDBOXED"
ENV_SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION = (
    "SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION"
)

_PASSTHROUGH_NAMES = frozenset(
    {
        "DEVICE_LIB_PATH",
        "FLASHINFER_TRACE_DIR",
        "GPU_DEVICE_ORDINAL",
        "HIP_CLANG_PATH",
        "HIP_PATH",
        "HIP_PLATFORM",
        ENV_HIP_VISIBLE_DEVICES,
        "HSA_OVERRIDE_GFX_VERSION",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "MIOPEN_USER_DB_PATH",
        "OMP_NUM_THREADS",
        "PATH",
        "PYTHONPATH",
        "PYTORCH_ROCM_ARCH",
        "ROCM_PATH",
        ENV_ROCR_VISIBLE_DEVICES,
        "SOL_EXECBENCH_ALLOW_CPU_TIMING",
        ENV_SOL_EXECBENCH_CLOCKS_LOCKED,
        ENV_SOL_EXECBENCH_DEVICE,
        "SOL_EXECBENCH_GPU_LOCK_DIR",
        ENV_SOL_EXECBENCH_GRACEFUL_EXIT,
        ENV_SOL_EXECBENCH_SANDBOXED,
        ENV_SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION,
        "VIRTUAL_ENV",
    },
)


def sanitized_subprocess_env(
    base: Mapping[str, str],
    *,
    staging_dir: Path,
) -> dict[str, str]:
    """Return the minimum runtime environment, excluding credentials and proxies."""
    result = {name: base[name] for name in _PASSTHROUGH_NAMES if name in base}
    result.update(
        {
            "HOME": str(staging_dir),
            "TMPDIR": str(staging_dir / ".tmp"),
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
    )
    return result


__all__ = [
    "ENV_HIP_VISIBLE_DEVICES",
    "ENV_ROCR_VISIBLE_DEVICES",
    "ENV_SOL_EXECBENCH_CLOCKS_LOCKED",
    "ENV_SOL_EXECBENCH_DEVICE",
    "ENV_SOL_EXECBENCH_GRACEFUL_EXIT",
    "ENV_SOL_EXECBENCH_SANDBOXED",
    "ENV_SOL_EXECBENCH_UNSAFE_LOCAL_EXECUTION",
    "sanitized_subprocess_env",
]
