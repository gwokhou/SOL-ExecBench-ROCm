# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Publication-device policy for the SOLAR bridge."""

from __future__ import annotations

import gc

from sol_execbench.core.platform.hardware import (
    HardwareConfiguration,
    HardwareConfigurationKind,
    HardwareFactConfidence,
    HardwareNominalProfile,
)

# Canonical architecture id passed to solar.api (no namespace prefix).
# aka_corpus.FORMAL_ARCHITECTURE carries the manifest label "solar:RX_9060_XT";
# the two are deliberately different views of the same target.
FORMAL_ARCHITECTURE, FORMAL_GFX_TARGET = "RX_9060_XT", "gfx1200"
_ARCHITECTURE_BY_DEVICE_MODEL = {
    "amd instinct mi300x": "MI300X",
    "amd radeon rx 9060 xt": FORMAL_ARCHITECTURE,
    "radeon rx 9060 xt": FORMAL_ARCHITECTURE,
}
_DEVICE_MODEL_BY_ARCHITECTURE = {
    "MI300X": "AMD Instinct MI300X",
    FORMAL_ARCHITECTURE: "AMD Radeon RX 9060 XT",
}
_PRODUCT_SKU_BY_ARCHITECTURE = {
    "MI300X": "mi300x-oam",
    FORMAL_ARCHITECTURE: "rx9060xt-standard",
}
_PROFILE_REQUIREMENTS = {
    "MI300X": (
        "gfx942",
        192 * 1024**3,
        frozenset({None, "mi300x-oam"}),
    ),
    FORMAL_ARCHITECTURE: (
        "gfx1200",
        16 * 1024**3,
        frozenset({None, "rx9060xt-standard"}),
    ),
}
_RESOLVED_HARDWARE_KINDS = frozenset(
    {
        HardwareConfigurationKind.OBSERVED_DEVICE,
        HardwareConfigurationKind.PHYSICAL_DEVICE,
        HardwareConfigurationKind.VIRTUAL_DEVICE,
        HardwareConfigurationKind.PARTITION,
    }
)


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


def solar_architecture_for_configuration(
    hardware: HardwareConfiguration,
) -> str:
    """Select a SOLAR profile by device configuration, never ISA alone."""
    model = " ".join((hardware.device_model or "").lower().split())
    architecture = _ARCHITECTURE_BY_DEVICE_MODEL.get(model)
    if architecture is None:
        raise ValueError(
            "unsupported_solar_hardware:"
            f"{hardware.gfx_target}:{model or 'unknown-model'}"
        )
    expected_gfx, expected_memory, supported_skus = _PROFILE_REQUIREMENTS[
        architecture
    ]
    if (
        hardware.kind not in _RESOLVED_HARDWARE_KINDS
        or hardware.gfx_target != expected_gfx
        or hardware.visible_memory_bytes != expected_memory
        or hardware.product_sku not in supported_skus
    ):
        raise ValueError(
            "unsupported_solar_hardware_configuration:"
            f"{hardware.gfx_target}:{model}:"
            f"{hardware.product_sku or 'unknown-sku'}:"
            f"{hardware.visible_memory_bytes or 'unknown-memory'}"
        )
    return architecture


def nominal_hardware_profile(
    architecture: str = FORMAL_ARCHITECTURE,
) -> HardwareNominalProfile:
    """Project one packaged SOLAR profile into the canonical nominal model."""
    from solar.rocm.architecture import ArchitectureProfile

    profile = ArchitectureProfile.load(architecture)
    confidence = (
        HardwareFactConfidence.CALIBRATED
        if profile.audit_evidence.get("status") == "verified"
        else HardwareFactConfidence.SPEC_DERIVED
    )
    return HardwareNominalProfile(
        vendor=profile.vendor,
        device_model=_DEVICE_MODEL_BY_ARCHITECTURE[architecture],
        product_sku=_PRODUCT_SKU_BY_ARCHITECTURE[architecture],
        gfx_target=profile.gfx_target,
        compute_units=profile.compute_units,
        memory_capacity_bytes=profile.memory_capacity_bytes,
        memory_bandwidth_bytes_per_second=(
            profile.memory_bandwidth_bytes_per_second
        ),
        l2_cache_bytes=profile.l2_bytes,
        last_level_cache_bytes=profile.last_level_cache_bytes,
        peak_ops_per_second=profile.peak_ops_per_second,
        profile_revision=profile.profile_revision,
        source=profile.source or profile.profile_revision,
        confidence=confidence,
    )


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
    "nominal_hardware_profile",
    "release_formal_device_memory",
    "require_formal_device",
    "solar_architecture_for_configuration",
]
