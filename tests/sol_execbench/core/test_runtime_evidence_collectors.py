from __future__ import annotations

from types import SimpleNamespace

import torch

from sol_execbench.core.evidence.runtime_evidence.collectors import (
    build_dependency_observation,
    collect_gpu_evidence,
    collect_visible_device_environment,
)
from sol_execbench.core.platform.dependency_matrix import (
    PytorchDependencyObservation,
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
        raise AssertionError("explicit observations must bypass local collection")

    result = build_dependency_observation(
        overrides,
        collect_observation=collect,
    )

    assert result is overrides


def test_gpu_collector_discovers_rocm_device_and_normalizes_architecture(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "AMD Fixture")
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
        {"ROCR_VISIBLE_DEVICES": "0", "UNRELATED": "ignored"}
    ) == {"ROCR_VISIBLE_DEVICES": "0"}
