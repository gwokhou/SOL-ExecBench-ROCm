from __future__ import annotations

import pytest

from sol_execbench.core.bench.performance_model.vram_policy import (
    GIB,
    MIB,
    DiagnosticVRAMWorkingSetPolicy,
    select_vram_working_set_policy,
)


@pytest.mark.parametrize(
    ("observed", "nominal", "probe"),
    [
        (8 * GIB, 8 * GIB, 256 * MIB),
        (17_095_983_104, 16 * GIB, 512 * MIB),
    ],
)
def test_policy_selects_supported_rdna4_capacity_class(
    observed: int,
    nominal: int,
    probe: int,
) -> None:
    policy = select_vram_working_set_policy(
        gpu_architecture="gfx1200",
        gpu_id="gpu-id",
        total_memory_bytes=observed,
        source_revision="a" * 40,
        created_at="2026-08-10T00:00:00+00:00",
    )

    assert policy.nominal_total_memory_bytes == nominal
    assert policy.probe_working_set_bytes == probe
    assert policy.applicability_max_bytes == probe


def test_policy_selects_cdna3_capacity_class() -> None:
    policy = select_vram_working_set_policy(
        gpu_architecture="gfx942",
        gpu_id="gpu-id",
        total_memory_bytes=192 * GIB,
        source_revision="a" * 40,
        created_at="2026-08-10T00:00:00+00:00",
    )

    assert policy.gpu_architecture == "gfx942"
    assert policy.nominal_total_memory_bytes == 192 * GIB
    assert policy.probe_working_set_bytes == 8 * GIB
    assert policy.applicability_max_bytes == 8 * GIB
    assert policy.selection_algorithm == "cdna3_total_memory_class.v1"


@pytest.mark.parametrize(
    ("architecture", "capacity", "reason"),
    [
        ("gfx942", 256 * GIB, "unsupported_total_vram_class"),
        ("gfx1200", 192 * GIB, "unsupported_total_vram_class"),
        ("gfx1200", 12 * GIB, "unsupported_total_vram_class"),
        ("gfx1100", 8 * GIB, "unsupported_vram_policy_architecture:gfx1100"),
    ],
)
def test_policy_rejects_unvalidated_architecture_or_capacity(
    architecture: str,
    capacity: int,
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        select_vram_working_set_policy(
            gpu_architecture=architecture,
            gpu_id="gpu-id",
            total_memory_bytes=capacity,
            source_revision="a" * 40,
        )


def test_policy_rejects_noncanonical_tier() -> None:
    with pytest.raises(ValueError, match="probe tier differs"):
        DiagnosticVRAMWorkingSetPolicy(
            gpu_architecture="gfx1200",
            gpu_id="gpu-id",
            observed_total_memory_bytes=16 * GIB,
            nominal_total_memory_bytes=16 * GIB,
            probe_working_set_bytes=256 * MIB,
            applicability_max_bytes=256 * MIB,
            source_revision="a" * 40,
            created_at="2026-08-10T00:00:00+00:00",
        )
