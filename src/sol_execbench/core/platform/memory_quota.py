# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated ROCm memory-quota measurement for corpus target views."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from enum import StrEnum
from operator import attrgetter
from typing import Any, Literal

from pydantic import Field, model_validator

from sol_execbench.core.data.base_model import (
    CurrentFrozenSchemaModel,
    NonEmptyString,
    NonNegativeInt,
)
from sol_execbench.core.integrity import stable_json_checksum
from sol_execbench.core.platform.runtime import detect_rocm_device
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

MIB = 1024**2
DEFAULT_SAFETY_PERCENT = 85
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0
ALLOCATION_QUANTUM_BYTES = 64 * MIB
MAX_ALLOCATION_PROBE_STEPS = 8
ALLOCATION_PROBE_CEILING_PERCENT = 90


class MemoryQuotaProbeMethod(StrEnum):
    """Closed vocabulary for memory-quota measurement methods."""

    TORCH_ROCM_ALLOCATION_SEARCH = "torch_rocm_allocation_search.v1"


CAPACITY_PROBE_ALGORITHM = (
    MemoryQuotaProbeMethod.TORCH_ROCM_ALLOCATION_SEARCH.value
)


class GPUMemoryQuotaEvidence(CurrentFrozenSchemaModel):
    """Measured memory inputs and the canonically derived usable budget."""

    current_schema_version = PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE

    schema_version: Literal[PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE]
    probe_method: Literal[
        MemoryQuotaProbeMethod.TORCH_ROCM_ALLOCATION_SEARCH
    ] = MemoryQuotaProbeMethod.TORCH_ROCM_ALLOCATION_SEARCH
    device: NonEmptyString
    device_index: NonNegativeInt
    gpu_name: NonEmptyString
    gfx_target: NonEmptyString
    torch_version: NonEmptyString
    hip_version: NonEmptyString
    collected_at: datetime
    runtime_free_bytes: int = Field(gt=0)
    runtime_total_bytes: int = Field(gt=0)
    environment_quota_bytes: int | None = Field(default=None, gt=0)
    stable_allocatable_bytes: int = Field(gt=0)
    harness_reserve_bytes: NonNegativeInt
    safety_percent: int = Field(
        default=DEFAULT_SAFETY_PERCENT,
        ge=1,
        le=99,
    )
    usable_budget_bytes: int = Field(gt=0)
    capacity_probe_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_derived_budget(self) -> GPUMemoryQuotaEvidence:
        expected_budget = derive_usable_budget(
            runtime_free_bytes=self.runtime_free_bytes,
            environment_quota_bytes=self.environment_quota_bytes,
            stable_allocatable_bytes=self.stable_allocatable_bytes,
            harness_reserve_bytes=self.harness_reserve_bytes,
            safety_percent=self.safety_percent,
        )
        if self.usable_budget_bytes != expected_budget:
            raise ValueError("usable memory budget is not canonically derived")
        if self.runtime_free_bytes > self.runtime_total_bytes:
            raise ValueError("runtime free memory exceeds total memory")
        if self.capacity_probe_digest != capacity_probe_digest(self):
            raise ValueError("capacity probe digest does not match evidence")
        return self


def derive_usable_budget(
    *,
    runtime_free_bytes: int,
    environment_quota_bytes: int | None,
    stable_allocatable_bytes: int,
    harness_reserve_bytes: int,
    safety_percent: int = DEFAULT_SAFETY_PERCENT,
) -> int:
    """Derive the usable budget from measured bounds and reserved overhead."""
    bounds = [runtime_free_bytes, stable_allocatable_bytes]
    if environment_quota_bytes is not None:
        bounds.append(environment_quota_bytes)
    safe_bytes = min(bounds) * safety_percent // 100
    usable = safe_bytes - harness_reserve_bytes
    if usable <= 0:
        raise ValueError("measured memory quota leaves no usable budget")
    return usable


def _digest_payload(evidence: GPUMemoryQuotaEvidence) -> dict[str, object]:
    return {
        "probe_method": evidence.probe_method,
        "device": evidence.device,
        "device_index": evidence.device_index,
        "gpu_name": evidence.gpu_name,
        "gfx_target": evidence.gfx_target,
        "torch_version": evidence.torch_version,
        "hip_version": evidence.hip_version,
        "runtime_free_bytes": evidence.runtime_free_bytes,
        "runtime_total_bytes": evidence.runtime_total_bytes,
        "environment_quota_bytes": evidence.environment_quota_bytes,
        "stable_allocatable_bytes": evidence.stable_allocatable_bytes,
        "harness_reserve_bytes": evidence.harness_reserve_bytes,
        "safety_percent": evidence.safety_percent,
        "usable_budget_bytes": evidence.usable_budget_bytes,
    }


def capacity_probe_digest(evidence: GPUMemoryQuotaEvidence) -> str:
    """Return the stable quota identity, excluding collection wall-clock time."""
    return stable_json_checksum(_digest_payload(evidence))


def _is_out_of_memory(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "hiperroroutofmemory" in message


def _round_down(value: int, quantum: int) -> int:
    return value // quantum * quantum


def _stable_allocation_probe(
    torch_module: Any,
    device: str,
    runtime_free_bytes: int,
) -> int:
    rocm_runtime = attrgetter("cuda")(torch_module)
    rocm_runtime.empty_cache()
    ceiling = _round_down(
        runtime_free_bytes * ALLOCATION_PROBE_CEILING_PERCENT // 100,
        ALLOCATION_QUANTUM_BYTES,
    )
    if ceiling < ALLOCATION_QUANTUM_BYTES:
        raise RuntimeError("insufficient free memory for allocation probe")
    low = 0
    high = ceiling
    for _ in range(MAX_ALLOCATION_PROBE_STEPS):
        if high - low < ALLOCATION_QUANTUM_BYTES:
            break
        candidate = _round_down(
            (low + high + ALLOCATION_QUANTUM_BYTES) // 2,
            ALLOCATION_QUANTUM_BYTES,
        )
        allocation = None
        try:
            allocation = torch_module.empty(
                candidate,
                dtype=torch_module.uint8,
                device=device,
            )
            rocm_runtime.synchronize(device)
            low = candidate
        except RuntimeError as error:
            if not _is_out_of_memory(error):
                raise
            high = candidate - ALLOCATION_QUANTUM_BYTES
        finally:
            del allocation
            rocm_runtime.empty_cache()
    if low <= 0:
        raise RuntimeError("ROCm allocation probe found no stable capacity")
    return low


def collect_gpu_memory_quota(
    device: str = "cuda:0",
    *,
    environment_quota_bytes: int | None = None,
    safety_percent: int = DEFAULT_SAFETY_PERCENT,
    torch_module: Any | None = None,
    now: datetime | None = None,
) -> GPUMemoryQuotaEvidence:
    """Measure one ROCm device's current allocatable memory quota."""
    if torch_module is None:
        import torch as torch_module

    info = detect_rocm_device(device, torch_module=torch_module)
    rocm_runtime = attrgetter("cuda")(torch_module)
    runtime_free, runtime_total = rocm_runtime.mem_get_info(info.index)
    harness_reserve = int(rocm_runtime.memory_reserved(info.index))
    stable_allocatable = _stable_allocation_probe(
        torch_module,
        info.device,
        int(runtime_free),
    )
    usable = derive_usable_budget(
        runtime_free_bytes=int(runtime_free),
        environment_quota_bytes=environment_quota_bytes,
        stable_allocatable_bytes=stable_allocatable,
        harness_reserve_bytes=harness_reserve,
        safety_percent=safety_percent,
    )
    payload: dict[str, Any] = {
        "schema_version": PlatformArtifactSchema.GPU_MEMORY_QUOTA_EVIDENCE,
        "device": info.device,
        "device_index": info.index,
        "gpu_name": info.name,
        "gfx_target": info.gfx_target,
        "torch_version": info.torch_version,
        "hip_version": info.hip_version,
        "collected_at": now or datetime.now(UTC),
        "runtime_free_bytes": int(runtime_free),
        "runtime_total_bytes": int(runtime_total),
        "environment_quota_bytes": environment_quota_bytes,
        "stable_allocatable_bytes": stable_allocatable,
        "harness_reserve_bytes": harness_reserve,
        "safety_percent": safety_percent,
        "usable_budget_bytes": usable,
        "capacity_probe_digest": "0" * 64,
    }
    provisional = GPUMemoryQuotaEvidence.model_construct(**payload)
    payload["capacity_probe_digest"] = capacity_probe_digest(provisional)
    return GPUMemoryQuotaEvidence.model_validate(payload)


def collect_gpu_memory_quota_isolated(
    device: str = "cuda:0",
    *,
    environment_quota_bytes: int | None = None,
    safety_percent: int = DEFAULT_SAFETY_PERCENT,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> GPUMemoryQuotaEvidence:
    """Measure memory in a bounded child process so allocations are reclaimed."""
    command = [
        sys.executable,
        "-m",
        "sol_execbench.core.platform.memory_quota",
        "--device",
        device,
        "--safety-percent",
        str(safety_percent),
    ]
    if environment_quota_bytes is not None:
        command.extend(("--environment-quota", str(environment_quota_bytes)))
    completed = run_in_process_group_bounded(command, timeout=timeout_seconds)
    if completed.returncode != 0:
        detail = (completed.stderr or "memory quota probe failed").strip()
        raise RuntimeError(detail[-4000:])
    return GPUMemoryQuotaEvidence.model_validate_json(completed.stdout)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--environment-quota", type=int)
    parser.add_argument(
        "--safety-percent",
        type=int,
        default=DEFAULT_SAFETY_PERCENT,
    )
    args = parser.parse_args()
    evidence = collect_gpu_memory_quota(
        args.device,
        environment_quota_bytes=args.environment_quota,
        safety_percent=args.safety_percent,
    )
    print(evidence.model_dump_json())


if __name__ == "__main__":
    _main()


__all__ = [
    "CAPACITY_PROBE_ALGORITHM",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_SAFETY_PERCENT",
    "GPUMemoryQuotaEvidence",
    "MemoryQuotaProbeMethod",
    "capacity_probe_digest",
    "collect_gpu_memory_quota",
    "collect_gpu_memory_quota_isolated",
    "derive_usable_budget",
]
