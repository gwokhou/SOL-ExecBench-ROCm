# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Publication-device policy for the SOLAR bridge."""

from __future__ import annotations

import gc

# Canonical architecture id passed to solar.api (no namespace prefix).
# aka_corpus.FORMAL_ARCHITECTURE carries the manifest label "solar:RX_9060_XT";
# the two are deliberately different views of the same target.
FORMAL_ARCHITECTURE, FORMAL_GFX_TARGET = "RX_9060_XT", "gfx1200"
_ARCHITECTURE_BY_GFX_TARGET = {
    "gfx1200": FORMAL_ARCHITECTURE,
    "gfx942": "MI300X",
}


def formal_producer_readiness() -> tuple[bool, str]:
    """Report whether SOLAR can produce publication-grade formal bounds."""
    from solar.api import formal_producer_readiness as solar_readiness

    readiness = solar_readiness()
    return readiness.ready, readiness.reason_code


def formal_architecture_profile_hash(
    architecture: str = FORMAL_ARCHITECTURE,
) -> str:
    """Return the canonical hash of a packaged SOLAR architecture profile."""
    from solar.api import architecture_profile_sha256

    return architecture_profile_sha256(architecture)


def solar_architecture_for_gfx_target(gfx_target: str) -> str:
    """Return the packaged SOLAR profile for one supported gfx target."""
    architecture = _ARCHITECTURE_BY_GFX_TARGET.get(gfx_target)
    if architecture is None:
        raise ValueError(f"unsupported_solar_architecture:{gfx_target}")
    return architecture


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
    "formal_architecture_profile_hash",
    "formal_producer_readiness",
    "release_formal_device_memory",
    "require_formal_device",
    "solar_architecture_for_gfx_target",
]
