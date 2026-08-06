# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Publication-device policy for the SOLAR bridge."""

from __future__ import annotations

import gc

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
    with torch.cuda.device(selected):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


__all__ = [
    "FORMAL_ARCHITECTURE",
    "FORMAL_GFX_TARGET",
    "release_formal_device_memory",
    "require_formal_device",
]
