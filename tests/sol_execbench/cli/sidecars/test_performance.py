from __future__ import annotations

from typing import Literal

from sol_execbench.cli.sidecars.performance import _gpu_identity
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.platform.runtime import (
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)


def _topology(width: int = 8) -> PCIeTopologyIdentity:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=width,
        max_width=16,
    )
    return PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )


def _snapshot(
    phase: Literal["pre", "post"],
    *,
    gpu_id: str | None = "gpu-0",
    gpu_bdf: str | None = "0000:03:00.0",
    include_topology: bool = True,
    topology_width: int = 8,
) -> RuntimeGPUTelemetry:
    return RuntimeGPUTelemetry.model_validate(
        {
            "phase": phase,
            "gpu_id": gpu_id,
            "gpu_bdf": gpu_bdf,
            "pcie_topology": (
                _topology(topology_width) if include_topology else None
            ),
        },
    )


def test_gpu_identity_binds_matching_pre_and_post_snapshots() -> None:
    identity = _gpu_identity((_snapshot("pre"), _snapshot("post")))

    assert identity == ("gpu-0", "0000:03:00.0", _topology(), [])


def test_gpu_identity_rejects_snapshot_drift() -> None:
    identity = _gpu_identity(
        (
            _snapshot("pre"),
            _snapshot("post", gpu_id="gpu-1"),
        ),
    )

    assert identity == (None, None, None, ["gpu_id_snapshot_invalid"])


def test_gpu_identity_rejects_incomplete_pcie_topology() -> None:
    identity = _gpu_identity(
        (
            _snapshot("pre"),
            _snapshot("post", include_topology=False),
        ),
    )

    assert identity == (
        "gpu-0",
        "0000:03:00.0",
        None,
        ["pcie_topology_snapshot_incomplete"],
    )


def test_gpu_identity_rejects_pcie_topology_drift() -> None:
    identity = _gpu_identity(
        (
            _snapshot("pre"),
            _snapshot("post", topology_width=4),
        ),
    )

    assert identity == (
        "gpu-0",
        "0000:03:00.0",
        None,
        ["pcie_topology_snapshot_changed"],
    )
