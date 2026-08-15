from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from sol_execbench.core.platform import memory_quota
from sol_execbench.core.platform.memory_quota import (
    ALLOCATION_QUANTUM_BYTES,
    GPUMemoryQuotaEvidence,
    capacity_probe_digest,
    collect_gpu_memory_quota,
    collect_gpu_memory_quota_isolated,
    derive_usable_budget,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema

GIB = 1024**3
MIB = 1024**2


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def get_device_properties(self, _index: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="fake MI300X",
            gcnArchName="gfx942:sramecc+",
            total_memory=12 * GIB,
            L2_cache_size=16 * MIB,
        )

    def mem_get_info(self, _index: int) -> tuple[int, int]:
        return 10 * GIB, 12 * GIB

    def memory_reserved(self, _index: int) -> int:
        return 128 * MIB

    def empty_cache(self) -> None:
        return None

    def synchronize(self, _device: str) -> None:
        return None


class _FakeTorch:
    __version__ = "test-torch"
    version = SimpleNamespace(hip="test-hip")
    cuda = _FakeCuda()  # cuda=SimpleNamespace mock torch ROCm namespace
    uint8 = object()

    @staticmethod
    def device(value: str) -> SimpleNamespace:
        kind, _, raw_index = value.partition(":")
        return SimpleNamespace(
            type=kind,
            index=int(raw_index) if raw_index else None,
        )

    @staticmethod
    def empty(size: int, **_kwargs: object) -> object:
        if size > 6 * GIB:
            raise RuntimeError("HIP out of memory")
        return object()


def _evidence() -> GPUMemoryQuotaEvidence:
    payload: dict[str, Any] = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": "cuda:0",
        "device_index": 0,
        "gpu_name": "fake MI300X",
        "gfx_target": "gfx942",
        "torch_version": "test-torch",
        "hip_version": "test-hip",
        "collected_at": datetime(2026, 8, 15, tzinfo=UTC),
        "runtime_free_bytes": 10 * GIB,
        "runtime_total_bytes": 12 * GIB,
        "environment_quota_bytes": 8 * GIB,
        "stable_allocatable_bytes": 6 * GIB,
        "harness_reserve_bytes": 128 * MIB,
        "safety_percent": 85,
        "usable_budget_bytes": derive_usable_budget(
            runtime_free_bytes=10 * GIB,
            environment_quota_bytes=8 * GIB,
            stable_allocatable_bytes=6 * GIB,
            harness_reserve_bytes=128 * MIB,
        ),
        "capacity_probe_digest": "0" * 64,
    }
    provisional = GPUMemoryQuotaEvidence.model_construct(**payload)
    payload["capacity_probe_digest"] = capacity_probe_digest(provisional)
    return GPUMemoryQuotaEvidence.model_validate(payload)


def test_collect_quota_uses_measured_allocation_and_environment_limit() -> None:
    evidence = collect_gpu_memory_quota(
        "cuda:0",
        environment_quota_bytes=8 * GIB,
        torch_module=_FakeTorch(),
        now=datetime(2026, 8, 15, tzinfo=UTC),
    )

    assert evidence.gfx_target == "gfx942"
    assert evidence.stable_allocatable_bytes <= 6 * GIB
    assert evidence.stable_allocatable_bytes % ALLOCATION_QUANTUM_BYTES == 0
    assert evidence.usable_budget_bytes == (
        evidence.stable_allocatable_bytes * 85 // 100 - 128 * MIB
    )
    assert evidence.capacity_probe_digest == capacity_probe_digest(evidence)


def test_quota_evidence_rejects_noncanonical_budget() -> None:
    payload = _evidence().model_dump(mode="json")
    payload["usable_budget_bytes"] += 1

    with pytest.raises(ValidationError, match="not canonically derived"):
        GPUMemoryQuotaEvidence.model_validate(payload)


def test_isolated_probe_parses_child_artifact(monkeypatch) -> None:
    evidence = _evidence()
    observed: dict[str, Any] = {}

    def run(command, *, timeout):
        observed.update(command=command, timeout=timeout)
        return subprocess.CompletedProcess(
            command,
            0,
            evidence.model_dump_json(),
            "",
        )

    monkeypatch.setattr(memory_quota, "run_in_process_group_bounded", run)

    result = collect_gpu_memory_quota_isolated(
        "cuda:3",
        environment_quota_bytes=8 * GIB,
        timeout_seconds=12.0,
    )

    assert result == evidence
    assert observed["timeout"] == 12.0
    assert "--environment-quota" in observed["command"]
