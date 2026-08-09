from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from sol_execbench.core.evidence.runtime_evidence import collectors
from sol_execbench.core.evidence.runtime_evidence.collectors import (
    build_dependency_observation,
    collect_gpu_evidence,
    collect_runtime_gpu_telemetry,
    collect_visible_device_environment,
)
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.platform.dependency_matrix import (
    PytorchDependencyObservation,
)
from sol_execbench.core.platform.runtime import (
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)


def test_dependency_overrides_preserve_local_collection() -> None:
    calls = 0

    def collect() -> PytorchDependencyObservation:
        nonlocal calls
        calls += 1
        return PytorchDependencyObservation(
            torch_version="2.11.0+rocm7.2",
            torch_device_available=True,
        )

    result = build_dependency_observation(
        PytorchDependencyObservation(
            container_rocm_user_space_version="7.2.0",
            toolchain_rocm_version="7.2.0",
        ),
        collect_observation=collect,
    )

    assert calls == 1
    assert result.torch_version == "2.11.0+rocm7.2"
    assert result.torch_device_available is True
    assert result.container_rocm_user_space_version == "7.2.0"
    assert result.toolchain_rocm_version == "7.2.0"


def test_explicit_dependency_observation_bypasses_local_collection() -> None:
    overrides = PytorchDependencyObservation(
        torch_version="explicit",
        torch_device_available=False,
        hipcc_version="HIP 7.2.0",
    )

    def collect() -> PytorchDependencyObservation:
        raise AssertionError(
            "explicit observations must bypass local collection",
        )

    result = build_dependency_observation(
        overrides,
        collect_observation=collect,
    )

    assert result is overrides


def test_gpu_collector_discovers_rocm_device_and_normalizes_architecture(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _index: "AMD Fixture",
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(gcnArchName="gfx1200:sramecc+"),
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")

    evidence = collect_gpu_evidence()

    assert evidence.device_count == 2
    assert evidence.device_name == "AMD Fixture"
    assert evidence.gfx_architecture == "gfx1200"
    assert evidence.visible_device_environment == {"HIP_VISIBLE_DEVICES": "1"}


def test_gpu_collector_tolerates_runtime_probe_failure(monkeypatch) -> None:
    def fail_probe() -> int:
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr(torch.cuda, "device_count", fail_probe)

    evidence = collect_gpu_evidence(visible_device_environment={})

    assert evidence.device_count is None
    assert evidence.device_name is None
    assert evidence.gfx_architecture is None


def test_visible_device_collector_ignores_unrelated_variables() -> None:
    assert collect_visible_device_environment(
        {"ROCR_VISIBLE_DEVICES": "0", "UNRELATED": "ignored"},
    ) == {"ROCR_VISIBLE_DEVICES": "0"}


def test_runtime_gpu_telemetry_captures_pcie_topology(monkeypatch) -> None:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=8,
        max_width=16,
    )
    topology = PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )
    responses = {
        "list": '[{"gpu":0,"bdf":"0000:03:00.0","uuid":"gpu-uuid"}]',
        "metric": (
            '{"gpu_data":[{"gpu":0,'
            '"perf_level":"AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK",'
            '"clock":{"gfx_clock":"3200 MHz","mem_clock":"1250 MHz"},'
            '"temperature":{"hotspot_temperature":"52.5 C"},'
            '"power":{"current_socket_power":"101 W",'
            '"power_cap":"182 W","power_profile":"COMPUTE"}}]}'
        ),
        "process": "[]",
    }
    observed_bdfs: list[str] = []

    monkeypatch.setattr(
        collectors,
        "resolve_rocm_tool",
        lambda _name: Path("/opt/rocm/bin/amd-smi"),
    )
    monkeypatch.setattr(
        collectors,
        "_amd_smi_json",
        lambda _executable, command: responses[command],
    )

    def collect(bdf: str) -> PCIeTopologyIdentity:
        observed_bdfs.append(bdf)
        return topology

    monkeypatch.setattr(collectors, "collect_pcie_topology", collect)

    telemetry = collect_runtime_gpu_telemetry(phase="pre")

    assert observed_bdfs == ["0000:03:00.0"]
    assert telemetry.pcie_topology == topology


def test_runtime_gpu_telemetry_rejects_topology_for_another_device() -> None:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=8,
        max_width=16,
    )
    topology = PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )

    with pytest.raises(ValidationError, match="terminate at gpu_bdf"):
        RuntimeGPUTelemetry(
            phase="pre",
            gpu_bdf="0000:04:00.0",
            pcie_topology=topology,
        )
