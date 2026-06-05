# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Known benign ROCm stderr noise filtering."""

from __future__ import annotations

BENIGN_ROCM_STDERR_FIXTURES = (
    "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory",
)


def filter_benign_rocm_stderr(text: str | bytes | None) -> str:
    """Remove fixed ROCm userspace noise lines from diagnostic stderr text."""
    if text is None:
        return ""
    if isinstance(text, bytes):
        value = text.decode(errors="replace")
    else:
        value = text
    lines = [
        line
        for line in value.splitlines()
        if line.strip() not in BENIGN_ROCM_STDERR_FIXTURES
    ]
    if not lines:
        return ""
    return "\n".join(lines) + ("\n" if value.endswith("\n") else "")
