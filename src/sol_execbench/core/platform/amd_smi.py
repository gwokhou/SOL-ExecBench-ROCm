# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Typed parsers for AMD SMI JSON used by benchmark isolation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, RootModel


class AmdSmiPerformanceLevel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    perf_level: str


class AmdSmiPerformancePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu_data: list[AmdSmiPerformanceLevel]


class AmdSmiProcess(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pid: int | None = None
    name: str | None = None
    process_name: str | None = None


class AmdSmiGpuProcesses(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu: int | str
    process_list: list[AmdSmiProcess]


class AmdSmiProcessPayload(RootModel[list[AmdSmiGpuProcesses]]):
    pass


class AmdSmiGpuIdentity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gpu: int | str


class AmdSmiListPayload(RootModel[list[AmdSmiGpuIdentity]]):
    pass


def parse_performance_levels(raw: str) -> tuple[str, ...]:
    """Return normalized, non-empty levels for every reported GPU."""
    payload = AmdSmiPerformancePayload.model_validate_json(raw)
    if not payload.gpu_data:
        raise ValueError("amd-smi returned no GPU performance-level data")
    levels = tuple(entry.perf_level.strip().upper() for entry in payload.gpu_data)
    if any(not level for level in levels):
        raise ValueError("amd-smi returned an empty GPU performance level")
    return levels


def parse_processes(raw: str) -> list[dict[str, int | str]]:
    """Return the stable process fields used by isolation snapshots."""
    payload = AmdSmiProcessPayload.model_validate_json(raw)
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
                }
            )
    return processes


def parse_gpu_count(raw: str) -> int:
    """Return the number of unique GPU identifiers in ``amd-smi list`` JSON."""
    payload = AmdSmiListPayload.model_validate_json(raw)
    return len({str(entry.gpu) for entry in payload.root})
