from __future__ import annotations

import pytest

from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceRunIdentity,
)
from sol_execbench.core.bench.performance_model.lifecycle.collection_identity import (
    require_consistent_collection_gpu_identity,
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


def _run(
    run_id: str,
    topology: PCIeTopologyIdentity | None,
) -> PerformanceRunIdentity:
    return PerformanceRunIdentity(
        run_id=run_id,
        definition="fixture",
        definition_sha256="a" * 64,
        workload_uuid=run_id,
        workload_sha256="b" * 64,
        solution_sha256="c" * 64,
        candidate_sha256="d" * 64,
        gpu_architecture="gfx1200",
        gpu_id="gpu-0",
        gpu_bdf="0000:03:00.0",
        pcie_topology=topology,
        rocm_version="7.2.0",
        compiler_version="HIP 7.2",
        clock_mode="locked",
        power_profile="stable_peak",
        timing_protocol="device_event_v1",
    )


def test_collection_identity_requires_one_stable_pcie_path() -> None:
    topology = _topology()

    identity = require_consistent_collection_gpu_identity(
        [_run("1" * 64, topology), _run("2" * 64, topology)]
    )

    assert identity.pcie_topology == topology


def test_collection_identity_rejects_cross_case_pcie_drift() -> None:
    with pytest.raises(ValueError, match="different GPU identities"):
        require_consistent_collection_gpu_identity(
            [_run("1" * 64, _topology(8)), _run("2" * 64, _topology(4))]
        )


def test_collection_identity_rejects_missing_pcie_path() -> None:
    with pytest.raises(ValueError, match="pcie_topology"):
        require_consistent_collection_gpu_identity([_run("1" * 64, None)])
