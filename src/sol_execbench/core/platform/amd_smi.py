# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed parsers for AMD SMI JSON used by benchmark isolation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, RootModel


class AMDSmiPerformanceLevel(BaseModel):
    """Performance-level record for one GPU."""

    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    perf_level: str


class AMDSmiPerformancePayload(BaseModel):
    """Top-level AMD SMI performance-level response."""

    model_config = ConfigDict(extra="ignore")

    gpu_data: list[AMDSmiPerformanceLevel]


class AMDSmiProcess(BaseModel):
    """Process record reported by AMD SMI."""

    model_config = ConfigDict(extra="ignore")

    pid: int | None = None
    name: str | None = None
    process_name: str | None = None


class AMDSmiGPUProcesses(BaseModel):
    """Processes associated with one GPU."""

    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    process_list: list[AMDSmiProcess]


class AMDSmiProcessPayload(RootModel[list[AMDSmiGPUProcesses]]):
    """Root list of per-GPU process records."""


class AMDSmiGPUIdentity(BaseModel):
    """Minimal identity record for one GPU."""

    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    bdf: str | None = None
    uuid: str | None = None


class AMDSmiListPayload(RootModel[list[AMDSmiGPUIdentity]]):
    """Root list returned by the AMD SMI list command."""


def parse_performance_levels(raw: str) -> tuple[str, ...]:
    """Return normalized, non-empty levels for every reported GPU."""
    payload = AMDSmiPerformancePayload.model_validate_json(raw)
    if not payload.gpu_data:
        raise ValueError("amd-smi returned no GPU performance-level data")
    levels = tuple(
        entry.perf_level.strip().upper() for entry in payload.gpu_data
    )
    if any(not level for level in levels):
        raise ValueError("amd-smi returned an empty GPU performance level")
    return levels


def parse_processes(raw: str) -> list[dict[str, int | str]]:
    """Return the stable process fields used by isolation snapshots."""
    payload = AMDSmiProcessPayload.model_validate_json(raw)
    processes: list[dict[str, int | str]] = []
    for gpu_entry in payload.root:
        for process in gpu_entry.process_list:
            if process.pid is None:
                continue
            processes.append(
                {
                    "pid": process.pid,
                    "device": str(gpu_entry.gpu),
                    "name": process.name or process.process_name or "unknown",
                },
            )
    return processes


def parse_gpu_count(raw: str) -> int:
    """Return the number of unique GPU identifiers in ``amd-smi list`` JSON."""
    payload = AMDSmiListPayload.model_validate_json(raw)
    return len({str(entry.gpu) for entry in payload.root})


def parse_gpu_identity(raw: str, gpu_index: int) -> AMDSmiGPUIdentity:
    """Return one exact GPU list identity with required UUID and BDF."""
    payload = AMDSmiListPayload.model_validate_json(raw)
    matches = [
        identity
        for identity in payload.root
        if str(identity.gpu) == str(gpu_index)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"amd-smi returned no unique identity for GPU {gpu_index}"
        )
    identity = matches[0]
    if not identity.uuid or not identity.bdf:
        raise ValueError("amd-smi GPU identity lacks UUID or BDF")
    return identity
